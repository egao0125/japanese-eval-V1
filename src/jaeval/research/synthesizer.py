"""Synthesize research findings into structured Markdown reports."""
from __future__ import annotations

import logging

from .planner import ResearchPlan
from .searcher import SearchResults

logger = logging.getLogger(__name__)


class ReportSynthesizer:
    """Generate a structured research report with citations using LLM."""

    def __init__(self, *, model: str = "claude-sonnet-4-20250514"):
        self.model = model

    async def synthesize(
        self,
        plan: ResearchPlan,
        results: SearchResults,
        summaries: dict[str, str],
    ) -> str:
        """Generate a structured research report with citations.

        Args:
            plan: The research plan (topic + questions).
            results: Aggregated search results from all sources.
            summaries: Mapping of item key -> LLM summary text.

        Returns:
            A Markdown report string.
        """
        import anthropic

        context = self._build_context(results, summaries)

        questions_block = "\n".join(f"- {q}" for q in plan.questions)

        prompt = f"""Based on the following research findings, write a comprehensive report on: "{plan.topic}"

Research Questions:
{questions_block}

Findings:
{context}

Write a structured Markdown report with these sections:
1. Executive Summary (2-3 sentences)
2. Key Findings (organized by research question)
3. Notable Papers (with full citations)
4. Notable Tools & Models (with links and download counts)
5. Available Datasets
6. Recommendations for Japanese Voice AI Evaluation
7. References (numbered list)

Use proper citations. Be specific and technical. If information is insufficient for a question, note the gap."""

        try:
            client = anthropic.Anthropic()
            response = client.messages.create(
                model=self.model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            return "".join(b.text for b in response.content if hasattr(b, "text"))
        except Exception as exc:
            logger.error("Report synthesis failed: %s", exc)
            # Fall back to a raw dump so the user still gets something
            return self._fallback_report(plan, results, summaries)

    def _build_context(self, results: SearchResults, summaries: dict[str, str]) -> str:
        """Build a structured context string from search results and summaries."""
        sections: list[str] = []

        # Papers
        if results.papers:
            lines = ["## Papers"]
            for p in results.papers:
                summary = summaries.get(p.arxiv_id, "")
                lines.append(f"- **{p.title}**")
                lines.append(f"  Citation: {p.citation}")
                lines.append(f"  Categories: {', '.join(p.categories)}")
                if summary:
                    lines.append(f"  Summary: {summary}")
            sections.append("\n".join(lines))

        # Repos
        if results.repos:
            lines = ["## GitHub Repositories"]
            for r in results.repos:
                summary = summaries.get(r.full_name, "")
                lines.append(f"- **{r.full_name}** ({r.stars} stars, {r.language})")
                lines.append(f"  {r.description}")
                lines.append(f"  URL: {r.url}")
                if summary:
                    lines.append(f"  Summary: {summary}")
            sections.append("\n".join(lines))

        # Models
        if results.models:
            lines = ["## HuggingFace Models"]
            for m in results.models:
                summary = summaries.get(m.model_id, "")
                lines.append(f"- **{m.model_id}** (pipeline: {m.pipeline_tag}, {m.downloads} downloads)")
                lines.append(f"  URL: {m.url}")
                if summary:
                    lines.append(f"  Summary: {summary}")
            sections.append("\n".join(lines))

        # Datasets
        if results.datasets:
            lines = ["## HuggingFace Datasets"]
            for d in results.datasets:
                lines.append(f"- **{d.dataset_id}** ({d.downloads} downloads)")
                lines.append(f"  URL: {d.url}")
            sections.append("\n".join(lines))

        # Web results
        if results.web_results:
            lines = ["## Web Search Results"]
            for w in results.web_results:
                lines.append(f"- **{w.title}**")
                lines.append(f"  URL: {w.url}")
                if w.description:
                    lines.append(f"  {w.description}")
            sections.append("\n".join(lines))

        if not sections:
            return "(No results found across any source.)"

        return "\n\n".join(sections)

    def _fallback_report(
        self,
        plan: ResearchPlan,
        results: SearchResults,
        summaries: dict[str, str],
    ) -> str:
        """Generate a simple Markdown report without LLM (fallback on API failure)."""
        lines = [
            f"# Research Report: {plan.topic}",
            "",
            "**Note:** LLM synthesis failed. Raw results follow.",
            "",
        ]
        lines.append(self._build_context(results, summaries))
        return "\n".join(lines)
