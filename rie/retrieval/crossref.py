"""Crossref metadata search, used as a resilient paper-source fallback."""

from __future__ import annotations

import html
import re

from rie.config import get_settings
from rie.retrieval.http import polite_get
from rie.state import Paper


class CrossrefClient:
    base_url = "https://api.crossref.org/works"

    def __init__(self, mailto: str | None = None):
        self.mailto = mailto if mailto is not None else get_settings().crossref_mailto

    def search(self, query: str, limit: int = 25) -> list[Paper]:
        params = {
            "query": query,
            "rows": min(limit, 100),
            "select": "DOI,title,author,published,published-print,abstract,URL,is-referenced-by-count",
        }
        if self.mailto:
            params["mailto"] = self.mailto
        response = polite_get(
            "crossref",
            self.base_url,
            params=params,
            headers={"User-Agent": "ResearchIntelligenceEngine/0.3"},
            timeout=30,
        )
        if not response.ok:
            raise RuntimeError(f"Crossref request failed with HTTP {response.status_code}.")
        return [self._convert(item) for item in response.json().get("message", {}).get("items", [])]

    @staticmethod
    def _convert(item: dict) -> Paper:
        authors = []
        for author in item.get("author", []):
            name = " ".join(part for part in (author.get("given"), author.get("family")) if part)
            if name:
                authors.append(name)
        published = item.get("published-print") or item.get("published") or {}
        date_parts = published.get("date-parts") or []
        year = date_parts[0][0] if date_parts and date_parts[0] else None
        doi = item.get("DOI")
        url = item.get("URL") or (f"https://doi.org/{doi}" if doi else None)
        abstract = html.unescape(re.sub(r"<[^>]+>", " ", item.get("abstract") or ""))
        abstract = " ".join(abstract.split())
        return Paper(
            paper_id=doi or url or "crossref-unknown",
            title=(item.get("title") or ["Untitled"])[0],
            abstract=abstract,
            authors=authors,
            year=year,
            doi=doi,
            url=url,
            cited_by_count=item.get("is-referenced-by-count") or 0,
            source="crossref",
        )
