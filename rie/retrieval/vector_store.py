import json
import hashlib
import math
from pathlib import Path
import re

import faiss
import numpy as np

from rie.state import Chunk


class FaissVectorStore:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.encoder = None
        self._idf: np.ndarray | None = None
        self._lexical_dimensions = 8192
        if model_name != "lexical":
            try:
                from sentence_transformers import SentenceTransformer
            except ModuleNotFoundError as exc:
                if exc.name != "sentence_transformers":
                    raise
                # Keep the lightweight API usable when its optional embedding
                # package is not installed; lexical embeddings need no model download.
                self.model_name = "lexical"
            else:
                self.encoder = SentenceTransformer(model_name)
        self.index = None
        self.chunks: list[Chunk] = []

    def build(self, chunks: list[Chunk]) -> None:
        if not chunks:
            self.index = None
            self.chunks = []
            return
        if self.encoder is None:
            document_buckets = [self._lexical_features(chunk.text) for chunk in chunks]
            frequencies = np.zeros(self._lexical_dimensions, dtype="float32")
            for features in document_buckets:
                for bucket in features:
                    frequencies[bucket] += 1
            self._idf = np.log((1 + len(chunks)) / (1 + frequencies)) + 1
        vectors = self._embed([chunk.text for chunk in chunks])
        faiss.normalize_L2(vectors)
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.chunks = chunks

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        if self.index is None or not self.chunks:
            return []
        vector = self._embed([query])
        faiss.normalize_L2(vector)
        scores, indices = self.index.search(vector, min(k, len(self.chunks)))
        return [
            (self.chunks[index], float(score))
            for score, index in zip(scores[0], indices[0])
            if index >= 0
        ]

    def save(self, index_path: str | Path, chunks_path: str | Path) -> None:
        if self.index is None:
            raise ValueError("Build the index before saving")
        index_path, chunks_path = Path(index_path), Path(chunks_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        chunks_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        with chunks_path.open("w", encoding="utf-8") as handle:
            for chunk in self.chunks:
                handle.write(json.dumps(chunk.model_dump()) + "\n")

    def load(self, index_path: str | Path, chunks_path: str | Path) -> None:
        self.index = faiss.read_index(str(index_path))
        with Path(chunks_path).open("r", encoding="utf-8") as handle:
            self.chunks = [Chunk.model_validate_json(line) for line in handle if line.strip()]

    def _embed(self, texts: list[str]) -> np.ndarray:
        if self.encoder is not None:
            vectors = self.encoder.encode(texts, convert_to_numpy=True, normalize_embeddings=False)
            return np.asarray(vectors, dtype="float32")

        vectors = np.zeros((len(texts), self._lexical_dimensions), dtype="float32")
        idf = self._idf if self._idf is not None else np.ones(self._lexical_dimensions, dtype="float32")
        for row, text in enumerate(texts):
            for bucket, count in self._lexical_features(text).items():
                vectors[row, bucket] = (1 + math.log(count)) * idf[bucket]
        return vectors

    def _lexical_features(self, text: str) -> dict[int, int]:
        tokens = re.findall(r"[\w]+", text.casefold())
        terms = tokens + [f"{left}_{right}" for left, right in zip(tokens, tokens[1:])]
        buckets: dict[int, int] = {}
        for term in terms:
            digest = hashlib.blake2b(term.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest, "little") % self._lexical_dimensions
            buckets[bucket] = buckets.get(bucket, 0) + 1
        return buckets
