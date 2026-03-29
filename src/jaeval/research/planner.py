"""Research planner -- generates search queries from a research topic using LLM."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ResearchPlan:
    """Structured research plan with topic, questions, and per-source queries."""

    topic: str
    questions: list[str] = field(default_factory=list)
    arxiv_queries: list[str] = field(default_factory=list)
    github_queries: list[str] = field(default_factory=list)
    huggingface_queries: list[str] = field(default_factory=list)
    web_queries: list[str] = field(default_factory=list)


def _extract_json(text: str) -> dict:
    """Extract a JSON object from LLM output, handling code fences and preamble.

    Tries multiple strategies:
    1. Direct parse (output is pure JSON)
    2. Code-fence extraction (```json ... ```)
    3. First { ... last } extraction (greedy brace matching)
    """
    text = text.strip()

    # Strategy 1: direct parse
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    # Strategy 2: code fence
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Strategy 3: greedy brace extraction
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        try:
            return json.loads(text[first_brace : last_brace + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract JSON from LLM response: {text[:200]!r}")


class ResearchPlanner:
    """Generate research questions and multi-source search queries via LLM."""

    def __init__(self, *, model: str = "claude-sonnet-4-20250514"):
        self.model = model

    async def plan(self, topic: str) -> ResearchPlan:
        """Generate a research plan for *topic* using the Anthropic API."""
        import anthropic

        client = anthropic.Anthropic()

        prompt = f"""You are a research planner for Japanese voice/speech AI evaluation.

Given the research topic: "{topic}"

Generate a structured research plan as JSON:
{{
    "questions": ["3-5 research questions to investigate"],
    "arxiv_queries": ["3-5 arxiv search queries (use technical terms, e.g. 'Japanese ASR evaluation CER')"],
    "github_queries": ["3-5 GitHub search queries (e.g. 'japanese speech recognition benchmark')"],
    "huggingface_queries": ["3-5 HuggingFace model/dataset search queries (e.g. 'japanese asr whisper')"],
    "web_queries": ["3-5 general web search queries (e.g. 'Japanese speech recognition benchmark comparison 2024')"]
}}

Focus on: Japanese speech recognition, TTS, voice agents, evaluation methods, telephony.
Return ONLY the JSON object, no other text."""

        response = client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = "".join(b.text for b in response.content if hasattr(b, "text"))
        logger.debug("Planner raw response: %s", raw[:500])

        try:
            data = _extract_json(raw)
        except ValueError:
            logger.warning("Failed to parse planner JSON, using fallback queries for topic: %s", topic)
            return self._fallback_plan(topic)

        plan = ResearchPlan(
            topic=topic,
            questions=data.get("questions", []),
            arxiv_queries=data.get("arxiv_queries", []),
            github_queries=data.get("github_queries", []),
            huggingface_queries=data.get("huggingface_queries", []),
            web_queries=data.get("web_queries", []),
        )
        # Merge in baseline queries that are known to return results,
        # ensuring we always get some data even if LLM queries are too specific.
        plan = self._merge_baseline_queries(plan)
        return plan

    # Baseline queries known to return results across sources.
    _BASELINE_HF = [
        "japanese asr whisper",
        "japanese speech recognition",
        "japanese tts",
    ]
    _BASELINE_ARXIV = [
        "Japanese speech recognition",
        "Japanese text to speech evaluation",
    ]
    _BASELINE_GITHUB = [
        "japanese speech",
        "whisper japanese",
    ]

    @staticmethod
    def _merge_baseline_queries(plan: ResearchPlan) -> ResearchPlan:
        """Add baseline queries to the plan, avoiding duplicates."""
        def _merge(existing: list[str], baseline: list[str]) -> list[str]:
            existing_lower = {q.lower() for q in existing}
            for q in baseline:
                if q.lower() not in existing_lower:
                    existing.append(q)
                    existing_lower.add(q.lower())
            return existing

        plan.huggingface_queries = _merge(
            plan.huggingface_queries, ResearchPlanner._BASELINE_HF
        )
        plan.arxiv_queries = _merge(
            plan.arxiv_queries, ResearchPlanner._BASELINE_ARXIV
        )
        plan.github_queries = _merge(
            plan.github_queries, ResearchPlanner._BASELINE_GITHUB
        )
        return plan

    @staticmethod
    def _fallback_plan(topic: str) -> ResearchPlan:
        """Produce a minimal plan when LLM parsing fails."""
        return ResearchPlan(
            topic=topic,
            questions=[f"What is the current state of the art for: {topic}?"],
            arxiv_queries=[topic, f"Japanese {topic}"] + ResearchPlanner._BASELINE_ARXIV,
            github_queries=[topic] + ResearchPlanner._BASELINE_GITHUB,
            huggingface_queries=[topic] + ResearchPlanner._BASELINE_HF,
            web_queries=[topic, f"{topic} benchmark comparison"],
        )
