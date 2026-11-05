from rie.config import get_settings
from rie.retrieval.http import polite_get
from rie.state import Paper


class OpenAlexClient:
    base_url = "https://api.openalex.org"

    def __init__(self, api_key: str | None = None, mailto: str = ""):
        self.api_key = api_key if api_key is not None else get_settings().openalex_api_key
        self.mailto = mailto

    def search(self, query: str, limit: int = 25) -> list[Paper]:
        params = {"search": query, "per-page": min(limit, 200), "select": "id,title,abstract_inverted_index,authorships,publication_year,doi,cited_by_count,primary_location"}
        if self.api_key:
            params["api_key"] = self.api_key
        if self.mailto:
            params["mailto"] = self.mailto
        response = polite_get(
            "openalex",
            f"{self.base_url}/works",
            params=params,
            headers={"User-Agent": "ResearchIntelligenceEngine/0.3"},
            timeout=30,
        )
        if not response.ok:
            raise RuntimeError(f"OpenAlex request failed with HTTP {response.status_code}.")
        return [self._convert(work) for work in response.json().get("results", [])]

    def _convert(self, work: dict) -> Paper:
        inverted = work.get("abstract_inverted_index") or {}
        abstract = "".join(self._restore_abstract(inverted))
        authors = [entry["author"]["display_name"] for entry in work.get("authorships", []) if entry.get("author")]
        location = work.get("primary_location") or {}
        return Paper(
            paper_id=work.get("id", "").split("/")[-1],
            title=work.get("title") or "Untitled",
            abstract=abstract,
            authors=authors,
            year=work.get("publication_year"),
            doi=work.get("doi"),
            url=location.get("landing_page_url"),
            cited_by_count=work.get("cited_by_count") or 0,
            source="openalex",
        )

    @staticmethod
    def _restore_abstract(inverted: dict[str, list[int]]) -> list[str]:
        positions: dict[int, str] = {}
        for word, indices in inverted.items():
            for index in indices:
                positions[index] = word
        return [positions[index] + (" " if index + 1 in positions else "") for index in sorted(positions)]
