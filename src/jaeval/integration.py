"""High-level integration API for post-call auto-evaluation.

Provides a single entry point that voice-fullduplex (or any voice AI system)
can call after a call ends to get automated quality scores.

The flow is:
    1. Build a Tier 1 scorecard from structured turn data
    2. Run Tier 2 LLM judge on the scorecard
    3. Return a combined result with both tiers

Usage::

    from jaeval.integration import evaluate_call

    turns = [
        {"user_text": "Recoの料金を教えてください", "bot_text": "はい、..."},
        ...
    ]
    result = evaluate_call("CA001", turns, duration_sec=120.0)
    print(result.grade)           # "B"
    print(result.weighted_score)  # 3.7
    print(result.production_ready)  # True
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .harness.evaluators.scorecard import ScorecardMetrics, build_scorecard


@dataclass
class CallEvalResult:
    """Combined evaluation result from Tier 1 + Tier 2."""

    call_sid: str

    # Tier 1: Scorecard
    grade: str
    task_completion: str
    stt_error_count: int
    hallucination_count: int
    banned_words_used: list[str]
    avg_latency_sec: float

    # Tier 2: LLM Judge (None if skipped)
    weighted_score: float | None = None
    production_ready: bool | None = None
    dimension_scores: dict[str, Any] = field(default_factory=dict)
    biggest_issue: str = ""
    recommendations: list[str] = field(default_factory=list)

    # Raw scorecard for downstream use
    scorecard: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-safe dict."""
        return {
            "call_sid": self.call_sid,
            "tier1": {
                "grade": self.grade,
                "task_completion": self.task_completion,
                "stt_error_count": self.stt_error_count,
                "hallucination_count": self.hallucination_count,
                "banned_words_used": self.banned_words_used,
                "avg_latency_sec": round(self.avg_latency_sec, 3),
            },
            "tier2": {
                "weighted_score": self.weighted_score,
                "production_ready": self.production_ready,
                "dimension_scores": self.dimension_scores,
                "biggest_issue": self.biggest_issue,
                "recommendations": self.recommendations,
            } if self.weighted_score is not None else None,
            "scorecard": self.scorecard,
        }


def evaluate_call(
    call_sid: str,
    turns: list[dict[str, Any]],
    *,
    duration_sec: float = 0.0,
    run_judge: bool = True,
    judge_model: str = "claude-sonnet-4-20250514",
    judge_config_yaml: str | Path | None = None,
) -> CallEvalResult:
    """Evaluate a completed call with Tier 1 scorecard + optional Tier 2 LLM judge.

    Args:
        call_sid: Call session identifier.
        turns: List of turn dicts. Each must have ``user_text`` and ``bot_text``.
            Optional: ``confidence``, ``has_uncertainty``, ``duration_s``, ``latency_ms``.
        duration_sec: Total call duration in seconds.
        run_judge: Whether to run Tier 2 LLM judge (requires anthropic API key).
        judge_model: Model to use for LLM judge.
        judge_config_yaml: Path to judge config YAML. Uses defaults if None.

    Returns:
        CallEvalResult with both tier scores.
    """
    # Tier 1: Build scorecard
    scorecard: ScorecardMetrics = build_scorecard(
        call_sid, turns, duration_sec=duration_sec
    )

    result = CallEvalResult(
        call_sid=call_sid,
        grade=scorecard.overall_grade,
        task_completion=scorecard.task_completion,
        stt_error_count=scorecard.stt_error_count,
        hallucination_count=scorecard.hallucination_count,
        banned_words_used=scorecard.banned_words_used,
        avg_latency_sec=scorecard.avg_latency_sec,
        scorecard=scorecard.to_dict(),
    )

    # Tier 2: LLM Judge (optional)
    if run_judge:
        try:
            from .harness.evaluators.llm_judge import JudgeConfig, LLMJudge

            if judge_config_yaml:
                config = JudgeConfig.from_yaml(Path(judge_config_yaml))
            else:
                config = JudgeConfig()
            config.model = judge_model

            judge = LLMJudge(config)
            judge_result = judge.evaluate_scorecard(scorecard.to_dict())

            result.weighted_score = judge_result.weighted_score
            result.production_ready = judge_result.production_ready
            result.dimension_scores = judge_result.scores
            result.biggest_issue = judge_result.biggest_issue
            result.recommendations = judge_result.recommendations
        except ImportError:
            # anthropic not installed — skip Tier 2
            pass
        except Exception as e:
            # Judge failed — record error but don't crash
            result.biggest_issue = f"Judge error: {e}"

    return result
