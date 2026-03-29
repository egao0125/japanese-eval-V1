"""GitHub repository search."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)


@dataclass
class RepoResult:
    """A single GitHub repository from search results."""

    name: str
    full_name: str
    description: str
    url: str
    stars: int
    language: str
    updated: str
    topics: list[str] = field(default_factory=list)

    @property
    def citation(self) -> str:
        return f"{self.full_name} ({self.stars} stars). {self.url}"


class GitHubSource:
    """Search GitHub repositories via the public search API (no auth required)."""

    BASE_URL = "https://api.github.com/search/repositories"

    async def search(self, query: str, max_results: int = 10) -> list[RepoResult]:
        """Search GitHub for repositories matching *query*.

        Uses the unauthenticated search endpoint (rate-limited to 10 req/min).
        """
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    self.BASE_URL,
                    params={
                        "q": query,
                        "sort": "stars",
                        "per_page": min(max_results, 30),
                    },
                    headers={"Accept": "application/vnd.github.v3+json"},
                    timeout=15.0,
                )
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as exc:
            logger.error("GitHub search HTTP error for %r: %s", query, exc)
            return []
        except httpx.RequestError as exc:
            logger.error("GitHub search request failed for %r: %s", query, exc)
            return []

        results: list[RepoResult] = []
        for item in data.get("items", [])[:max_results]:
            results.append(
                RepoResult(
                    name=item["name"],
                    full_name=item["full_name"],
                    description=item.get("description") or "",
                    url=item["html_url"],
                    stars=item["stargazers_count"],
                    language=item.get("language") or "",
                    updated=item.get("updated_at", ""),
                    topics=item.get("topics", []),
                )
            )
        return results
