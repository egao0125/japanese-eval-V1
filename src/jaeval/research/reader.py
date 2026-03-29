"""Fetch and summarize documents using LLM."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class DocumentReader:
    """Summarize papers, repos, and models using the Anthropic API."""

    def __init__(self, *, model: str = "claude-sonnet-4-20250514"):
        self.model = model

    async def summarize_paper(self, title: str, abstract: str) -> str:
        """Summarize a paper's key findings in 2-3 sentences."""
        if not abstract.strip():
            return f"No abstract available for: {title}"

        prompt = (
            "Summarize this paper for a Japanese voice AI researcher in 2-3 sentences. "
            "Focus on methodology, key results, and relevance to Japanese speech evaluation.\n\n"
            f"Title: {title}\n"
            f"Abstract: {abstract}"
        )
        return await self._call_llm(prompt, max_tokens=256)

    async def summarize_repo(self, name: str, description: str) -> str:
        """Summarize a repo's relevance in 1-2 sentences."""
        if not description.strip():
            return f"Repository {name} -- no description available."

        prompt = (
            "Summarize this GitHub repository's relevance for Japanese voice AI evaluation "
            "in 1-2 sentences.\n\n"
            f"Repository: {name}\n"
            f"Description: {description}"
        )
        return await self._call_llm(prompt, max_tokens=128)

    async def summarize_model(self, model_id: str, pipeline_tag: str, downloads: int) -> str:
        """Summarize a HuggingFace model's relevance in 1-2 sentences."""
        prompt = (
            "Summarize this HuggingFace model's relevance for Japanese voice AI evaluation "
            "in 1-2 sentences.\n\n"
            f"Model: {model_id}\n"
            f"Pipeline: {pipeline_tag}\n"
            f"Downloads: {downloads}"
        )
        return await self._call_llm(prompt, max_tokens=128)

    async def _call_llm(self, prompt: str, max_tokens: int = 256) -> str:
        """Make a single LLM call and return the text response."""
        import anthropic

        try:
            client = anthropic.Anthropic()
            response = client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return "".join(b.text for b in response.content if hasattr(b, "text"))
        except Exception as exc:
            logger.error("LLM summarization failed: %s", exc)
            return f"(Summarization failed: {exc})"
