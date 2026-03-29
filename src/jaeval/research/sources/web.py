"""Web search source using Brave Search API."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)


@dataclass
class WebResult:
    """A single web search result."""

    title: str
    url: str
    description: str

    @property
    def citation(self) -> str:
        return f'"{self.title}". {self.url}'


class WebSearchSource:
    """Search the web via the Brave Search API.

    Requires BRAVE_API_KEY environment variable.
    Falls back gracefully (returns []) if API key is missing.
    """

    BASE_URL = "https://api.search.brave.com/res/v1/web/search"

    async def search(self, query: str, max_results: int = 10) -> list[WebResult]:
        """Search the web for *query* and return structured results."""
        api_key = os.environ.get("BRAVE_API_KEY")
        if not api_key:
            logger.debug("BRAVE_API_KEY not set, skipping web search")
            return []

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    self.BASE_URL,
                    params={"q": query, "count": min(max_results, 20)},
                    headers={
                        "Accept": "application/json",
                        "X-Subscription-Token": api_key,
                    },
                    timeout=15.0,
                )
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as exc:
            logger.error("Web search HTTP error for %r: %s", query, exc)
            return []
        except httpx.RequestError as exc:
            logger.error("Web search request failed for %r: %s", query, exc)
            return []

        results: list[WebResult] = []
        for item in data.get("web", {}).get("results", [])[:max_results]:
            results.append(
                WebResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    description=item.get("description", ""),
                )
            )
        return results
