"""GitHub repository search."""
from __future__ import annotations

import asyncio
import logging
import os
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
    """Search GitHub repositories via the public search API.

    Uses GITHUB_TOKEN from env if available (5000 req/hr vs 10 req/min).
    Retries on 403 rate-limit errors with backoff.
    """

    BASE_URL = "https://api.github.com/search/repositories"
    MAX_RETRIES = 3
    RETRY_DELAY = 15.0

    async def search(self, query: str, max_results: int = 10) -> list[RepoResult]:
        """Search GitHub for repositories matching *query*."""
        headers = {"Accept": "application/vnd.github.v3+json"}
        token = os.environ.get("GITHUB_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(
                        self.BASE_URL,
                        params={
                            "q": query,
                            "sort": "stars",
                            "per_page": min(max_results, 30),
                        },
                        headers=headers,
                        timeout=15.0,
                    )
                    if resp.status_code in (403, 429):
                        delay = self.RETRY_DELAY * attempt
                        logger.warning(
                            "GitHub rate-limited for %r (HTTP %d) -- retry %d/%d in %.0fs",
                            query, resp.status_code, attempt, self.MAX_RETRIES, delay,
                        )
                        await asyncio.sleep(delay)
                        continue
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

        logger.warning("GitHub search exhausted retries for %r", query)
        return []
