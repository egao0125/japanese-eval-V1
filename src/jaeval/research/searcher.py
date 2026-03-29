"""Parallel search across multiple sources."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from .planner import ResearchPlan
from .sources import (
    ArxivSource,
    GitHubSource,
    HuggingFaceSource,
    PaperResult,
    RepoResult,
    HFModelResult,
    HFDatasetResult,
)

logger = logging.getLogger(__name__)


@dataclass
class SearchResults:
    """Aggregated, deduplicated results from all sources."""

    papers: list[PaperResult] = field(default_factory=list)
    repos: list[RepoResult] = field(default_factory=list)
    models: list[HFModelResult] = field(default_factory=list)
    datasets: list[HFDatasetResult] = field(default_factory=list)


class ParallelSearcher:
    """Execute search queries from a ResearchPlan across all sources in parallel."""

    def __init__(self, *, max_per_query: int = 5):
        self.max_per_query = max_per_query
        self.arxiv = ArxivSource()
        self.github = GitHubSource()
        self.huggingface = HuggingFaceSource()

    async def search(self, plan: ResearchPlan) -> SearchResults:
        """Execute all searches in parallel, then deduplicate."""
        # Build all tasks
        tasks: list[asyncio.Task] = []
        task_labels: list[str] = []

        for q in plan.arxiv_queries:
            tasks.append(asyncio.ensure_future(self.arxiv.search(q, self.max_per_query)))
            task_labels.append(f"arxiv:{q}")

        for q in plan.github_queries:
            tasks.append(asyncio.ensure_future(self.github.search(q, self.max_per_query)))
            task_labels.append(f"github:{q}")

        for q in plan.huggingface_queries:
            tasks.append(asyncio.ensure_future(self.huggingface.search_models(q, self.max_per_query)))
            task_labels.append(f"hf_model:{q}")

        for q in plan.huggingface_queries:
            tasks.append(asyncio.ensure_future(self.huggingface.search_datasets(q, self.max_per_query)))
            task_labels.append(f"hf_dataset:{q}")

        # Run all concurrently
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect and deduplicate
        papers: dict[str, PaperResult] = {}
        repos: dict[str, RepoResult] = {}
        models: dict[str, HFModelResult] = {}
        datasets: dict[str, HFDatasetResult] = {}

        for label, result in zip(task_labels, raw_results):
            if isinstance(result, BaseException):
                logger.warning("Search task %s failed: %s", label, result)
                continue

            for item in result:
                if isinstance(item, PaperResult):
                    papers.setdefault(item.arxiv_id, item)
                elif isinstance(item, RepoResult):
                    repos.setdefault(item.full_name, item)
                elif isinstance(item, HFModelResult):
                    models.setdefault(item.model_id, item)
                elif isinstance(item, HFDatasetResult):
                    datasets.setdefault(item.dataset_id, item)

        return SearchResults(
            papers=list(papers.values()),
            repos=list(repos.values()),
            models=list(models.values()),
            datasets=list(datasets.values()),
        )
