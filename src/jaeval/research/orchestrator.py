"""Research pipeline orchestrator -- Plan -> Search -> Read -> Synthesize."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .planner import ResearchPlanner
from .searcher import ParallelSearcher, SearchResults
from .reader import DocumentReader
from .synthesizer import ReportSynthesizer

logger = logging.getLogger(__name__)


@dataclass
class ResearchConfig:
    """Configuration for the research pipeline."""

    model: str = "claude-sonnet-4-20250514"
    max_papers: int = 10
    max_repos: int = 5
    max_models: int = 5
    max_per_query: int = 5
    output_dir: Path = field(default_factory=lambda: Path("reports"))


class ResearchOrchestrator:
    """Top-level coordinator: Plan -> Search -> Read -> Synthesize."""

    def __init__(self, config: ResearchConfig | None = None):
        self.config = config or ResearchConfig()
        self.planner = ResearchPlanner(model=self.config.model)
        self.searcher = ParallelSearcher(max_per_query=self.config.max_per_query)
        self.reader = DocumentReader(model=self.config.model)
        self.synthesizer = ReportSynthesizer(model=self.config.model)

    async def run(self, topic: str) -> str:
        """Execute full research pipeline: Plan -> Search -> Read -> Synthesize.

        Returns the generated Markdown report.
        """
        # 1. Plan
        print(f"[1/4] Planning research on: {topic}")
        plan = await self.planner.plan(topic)
        print(
            f"  Generated {len(plan.questions)} questions, "
            f"{len(plan.arxiv_queries)} arxiv queries, "
            f"{len(plan.github_queries)} github queries, "
            f"{len(plan.huggingface_queries)} HF queries"
        )

        # 2. Search
        print("[2/4] Searching across sources...")
        results = await self.searcher.search(plan)
        print(
            f"  Found {len(results.papers)} papers, "
            f"{len(results.repos)} repos, "
            f"{len(results.models)} models, "
            f"{len(results.datasets)} datasets, "
            f"{len(results.web_results)} web results"
        )

        # 3. Read & Summarize
        print("[3/4] Summarizing findings...")
        summaries = await self._summarize_all(results)
        print(f"  Summarized {len(summaries)} items")

        # 4. Synthesize
        print("[4/4] Synthesizing report...")
        report = await self.synthesizer.synthesize(plan, results, summaries)

        # Save report
        report_path = self._save_report(topic, report)
        print(f"\nReport saved: {report_path}")

        return report

    async def _summarize_all(self, results: SearchResults) -> dict[str, str]:
        """Summarize papers, repos, and models concurrently (with limits)."""
        summaries: dict[str, str] = {}
        tasks: list[tuple[str, asyncio.Task]] = []

        # Papers (limited)
        for paper in results.papers[: self.config.max_papers]:
            task = asyncio.ensure_future(
                self.reader.summarize_paper(paper.title, paper.abstract)
            )
            tasks.append((paper.arxiv_id, task))

        # Repos (limited)
        for repo in results.repos[: self.config.max_repos]:
            task = asyncio.ensure_future(
                self.reader.summarize_repo(repo.full_name, repo.description)
            )
            tasks.append((repo.full_name, task))

        # Models (limited)
        for model in results.models[: self.config.max_models]:
            task = asyncio.ensure_future(
                self.reader.summarize_model(model.model_id, model.pipeline_tag, model.downloads)
            )
            tasks.append((model.model_id, task))

        # Await all
        if tasks:
            keys, coros = zip(*tasks)
            raw_results = await asyncio.gather(*coros, return_exceptions=True)
            for key, result in zip(keys, raw_results):
                if isinstance(result, BaseException):
                    logger.warning("Summarization failed for %s: %s", key, result)
                    summaries[key] = f"(Summarization failed: {result})"
                else:
                    summaries[key] = result

        return summaries

    def _save_report(self, topic: str, report: str) -> Path:
        """Write report to disk and return its path."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now().strftime("%Y%m%d")
        # Sanitize topic for filename
        safe_topic = topic[:50].replace(" ", "_").replace("/", "_").replace("\\", "_")
        # Remove any other problematic chars
        safe_topic = "".join(c for c in safe_topic if c.isalnum() or c in ("_", "-"))
        report_path = self.config.output_dir / f"{date_str}_{safe_topic}.md"
        report_path.write_text(report, encoding="utf-8")
        return report_path


def run_research(topic: str, **kwargs) -> str:
    """Convenience function to run research synchronously.

    Accepts any ResearchConfig fields as keyword arguments.
    """
    config = ResearchConfig(**kwargs)
    orchestrator = ResearchOrchestrator(config)
    return asyncio.run(orchestrator.run(topic))
