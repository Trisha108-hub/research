from rie.retrieval.http import polite_get
from rie.state import Paper


class SemanticScholarClient:
    base_url = "https://api.semanticscholar.org/graph/v1"

    def __init__(self, api_key: str = ""):
        self.api_key = api_key

    def search(self, query: str, limit: int = 25) -> list[Paper]:
        headers = {"x-api-key": self.api_key} if self.api_key else {}
        params = {
            "query": query,
            "limit": min(limit, 100),
            "fields": "title,abstract,authors,year,externalIds,citationCount,url",
        }
        response = polite_get(
            "semantic_scholar",
            f"{self.base_url}/paper/search",
            params=params,
            headers=headers,
            timeout=30,
        )
        if not response.ok:
            raise RuntimeError(f"Semantic Scholar request failed with HTTP {response.status_code}.")
        return [self._convert(item) for item in response.json().get("data", [])]

    def _convert(self, item: dict) -> Paper:
        external_ids = item.get("externalIds") or {}
        return Paper(
            paper_id=item.get("paperId", ""),
            title=item.get("title") or "Untitled",
            abstract=item.get("abstract") or "",
            authors=[author["name"] for author in item.get("authors", [])],
            year=item.get("year"),
            doi=external_ids.get("DOI"),
            url=item.get("url"),
            cited_by_count=item.get("citationCount") or 0,
            source="semantic_scholar",
        )
