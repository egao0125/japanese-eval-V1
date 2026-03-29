"""ArXiv paper search and retrieval."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


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

    async def search(self, query: str, max_results: int = 10) -> list[PaperResult]:
        """Search arxiv and return structured results.

        Runs the synchronous arxiv client in a thread executor to avoid
        blocking the event loop.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._search_sync, query, max_results)

    def _search_sync(self, query: str, max_results: int) -> list[PaperResult]:
        """Synchronous arxiv search (run in executor)."""
        try:
            import arxiv
        except ImportError:
            logger.warning("arxiv package not installed. Install with: pip install arxiv>=2.0.0")
            return []

        try:
            client = arxiv.Client()
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
            logger.error("ArXiv search failed for query %r: %s", query, exc)
            return []
