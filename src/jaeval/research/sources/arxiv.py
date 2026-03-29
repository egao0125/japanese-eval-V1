"""ArXiv paper search and retrieval."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Lazy-initialized lock to serialize arxiv API calls (avoids 429 rate limits).
_arxiv_lock: asyncio.Lock | None = None


def _get_arxiv_lock() -> asyncio.Lock:
    """Return the module-level asyncio lock, creating it lazily."""
    global _arxiv_lock
    if _arxiv_lock is None:
        _arxiv_lock = asyncio.Lock()
    return _arxiv_lock


@dataclass
class PaperResult:
    """A single paper from ArXiv search results."""

    title: str
    authors: list[str]
    abstract: str
    arxiv_id: str
    url: str
    published: str
    categories: list[str]

    @property
    def citation(self) -> str:
        authors_str = ", ".join(self.authors[:3])
        if len(self.authors) > 3:
            authors_str += " et al."
        return f'{authors_str}. "{self.title}" ({self.published[:4]}). {self.url}'


class ArxivSource:
    """Search arxiv for papers using the arxiv Python package."""

    MAX_RETRIES = 3
    RETRY_DELAY = 30.0  # seconds between retries on 429

    async def search(self, query: str, max_results: int = 10) -> list[PaperResult]:
        """Search arxiv and return structured results.

        Serializes calls via a module-level lock to respect arxiv rate limits.
        Retries on HTTP 429 with exponential backoff.
        """
        async with _get_arxiv_lock():
            loop = asyncio.get_running_loop()
            for attempt in range(1, self.MAX_RETRIES + 1):
                results = await loop.run_in_executor(
                    None, self._search_sync, query, max_results
                )
                if results is not None:
                    return results
                # _search_sync returns None on 429 -- retry after delay
                delay = self.RETRY_DELAY * attempt
                logger.info(
                    "ArXiv 429 for %r -- retry %d/%d in %.0fs",
                    query, attempt, self.MAX_RETRIES, delay,
                )
                await asyncio.sleep(delay)
            logger.warning("ArXiv search exhausted retries for %r", query)
            return []

    def _search_sync(self, query: str, max_results: int) -> list[PaperResult] | None:
        """Synchronous arxiv search (run in executor).

        Returns None on HTTP 429 (signals caller to retry), [] on other errors.
        """
        try:
            import arxiv
        except ImportError:
            logger.warning("arxiv package not installed. Install with: pip install arxiv>=2.0.0")
            return []

        try:
            # Use page_size matching max_results to avoid over-fetching.
            # num_retries=5 with delay_seconds=10 gives the arxiv client
            # time to clear rate limits before giving up.
            client = arxiv.Client(
                page_size=max_results, delay_seconds=10.0, num_retries=5
            )
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance,
            )
            results: list[PaperResult] = []
            for paper in client.results(search):
                results.append(
                    PaperResult(
                        title=paper.title,
                        authors=[a.name for a in paper.authors],
                        abstract=paper.summary,
                        arxiv_id=paper.entry_id.split("/")[-1],
                        url=paper.entry_id,
                        published=paper.published.isoformat(),
                        categories=paper.categories,
                    )
                )
            return results
        except Exception as exc:
            exc_str = str(exc)
            if "429" in exc_str:
                logger.warning("ArXiv rate-limited for query %r", query)
                return None  # signal retry
            logger.error("ArXiv search failed for query %r: %s", query, exc)
            return []
