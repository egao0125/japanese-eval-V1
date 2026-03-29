"""Automated call scorecard evaluator.

Simplified, decoupled version of the voice-fullduplex call_scorecard.py.
Builds quality metrics from structured turn data without log parsing dependencies.

Usage:
    from jaeval.harness.evaluators.scorecard import build_scorecard, grade_call

    turns = [
        {"user_text": "Recoの料金を教えてください", "bot_text": "はい、...", ...},
        ...
    ]
    metrics = build_scorecard("CA001", turns, duration_sec=120.0)
    print(metrics.overall_grade, metrics.turn_count)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BANNED_WORDS: list[str] = [
    "もちろん",
    "かしこまりました",
    "承知しました",
    "申し訳ありません",
    "申し訳ございません",
    "いたします",
    "させていただきます",
    "できかねます",
    "させていただ",
    "お手伝い",
    "何かご質問は",
    "他にありますか",
    "何かあれば",
    "いつでもご連絡ください",
    "お気軽に",
    "遠慮なく",
    "くださいね",
    "くださいませ",
    "詳しく教えてください",
]

# Repeat-request phrases (bot asking user to repeat)
REPEAT_REQUEST_PHRASES: list[str] = [
    "もう一度お願いできますか",
    "もう一度",
    "聞き取れなかった",
    "聞き取れませんでした",
    "もう一度おっしゃって",
    "もう一度お伺い",
]

# Hallucination indicators (bot claims audio problems that don't exist)
HALLUCINATION_PHRASES: list[str] = [
    "聞こえない",
    "聞き取れない",
    "音声が途切れ",
    "音が切れ",
    "声が聞こえない",
]

# Confidence thresholds
CONF_GARBLE: float = -0.15


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ScorecardMetrics:
    """Aggregated quality metrics for a single call."""

    call_sid: str
    duration_sec: float
    turn_count: int
    task_completion: str  # "completed" | "partial" | "failed"
    stt_error_count: int
    hallucination_count: int
    avg_latency_sec: float
    banned_words_used: list[str] = field(default_factory=list)
    overall_grade: str = "C"  # "A" | "B" | "C" | "D" | "F"
    turns: list[dict[str, Any]] = field(default_factory=list)
    call_outcome: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict (JSON-safe)."""
        return {
            "call_sid": self.call_sid,
            "duration_sec": self.duration_sec,
            "turn_count": self.turn_count,
            "task_completion": self.task_completion,
            "stt_error_count": self.stt_error_count,
            "hallucination_count": self.hallucination_count,
            "avg_latency_sec": round(self.avg_latency_sec, 3),
            "banned_words_used": self.banned_words_used,
            "overall_grade": self.overall_grade,
            "call_outcome": self.call_outcome,
            "turns": self.turns,
        }


# ---------------------------------------------------------------------------
# Turn classification
# ---------------------------------------------------------------------------


def _classify_turn(turn: dict[str, Any]) -> str:
    """Classify a turn as CLEAN_UNDERSTOOD, GARBLED_RECOVERED, REPEAT_REQUEST,
    TRUNCATED, or NEEDS_REVIEW based on confidence, uncertainty, and bot response.
    """
    user_text = turn.get("user_text", "")
    bot_text = turn.get("bot_text", "")
    confidence = turn.get("confidence")
    has_uncertainty = turn.get("has_uncertainty", False)
    duration = turn.get("duration_s", 1.0)

    # Very short audio with very short transcript
    if duration < 0.5 and len(user_text) < 3:
        return "TRUNCATED"

    # Bot asks user to repeat
    for phrase in REPEAT_REQUEST_PHRASES:
        if phrase in bot_text:
            return "REPEAT_REQUEST"

    # Garbled: low confidence or uncertainty marker
    is_garbled = has_uncertainty or (confidence is not None and confidence < CONF_GARBLE)
    if is_garbled:
        if confidence is not None and confidence < -0.25:
            return "NEEDS_REVIEW"
        return "GARBLED_RECOVERED"

    return "CLEAN_UNDERSTOOD"


def _detect_banned_words(bot_text: str) -> list[str]:
    """Return list of banned words found in bot text."""
    found: list[str] = []
    for word in BANNED_WORDS:
        if word in bot_text:
            # Special case: "ございます" is OK inside "ありがとうございます"
            if word == "ございます":
                # Remove all occurrences of the exception, then check
                cleaned = bot_text.replace("ありがとうございます", "")
                if word in cleaned:
                    found.append(word)
            else:
                found.append(word)
    return found


def _detect_hallucinations(bot_text: str) -> int:
    """Count hallucination indicators in bot text."""
    count = 0
    for phrase in HALLUCINATION_PHRASES:
        if phrase in bot_text:
            count += 1
    return count


def _infer_task_completion(turns: list[dict[str, Any]]) -> str:
    """Heuristic task completion inference from turn data.

    Checks for booking confirmations, information delivery, and call outcome markers.
    """
    if not turns:
        return "failed"

    all_bot_text = " ".join(t.get("bot_text", "") for t in turns)

    # Positive signals
    completion_signals = [
        "予約",
        "承りました",
        "ご案内",
        "お伝え",
        "デモ",
        "ありがとうございます",
    ]
    signal_count = sum(1 for s in completion_signals if s in all_bot_text)

    # Negative signals
    failure_signals = ["お力になれず", "対応できません", "分かりかねます"]
    failure_count = sum(1 for s in failure_signals if s in all_bot_text)

    if failure_count > 0:
        return "failed"
    if signal_count >= 2:
        return "completed"
    if signal_count >= 1:
        return "partial"
    return "partial"


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------


def build_scorecard(
    call_sid: str,
    turns: list[dict[str, Any]],
    *,
    duration_sec: float = 0.0,
) -> ScorecardMetrics:
    """Build a scorecard from structured turn data.

    Each turn dict should have at minimum:
        - ``user_text``: str
        - ``bot_text``: str

    Optional fields: ``confidence`` (float), ``has_uncertainty`` (bool),
    ``duration_s`` (float), ``latency_ms`` (float).

    Args:
        call_sid: Call session identifier.
        turns: List of turn dicts.
        duration_sec: Total call duration in seconds (0 to auto-estimate).

    Returns:
        :class:`ScorecardMetrics` with computed grades and metrics.
    """
    classified_turns: list[dict[str, Any]] = []
    stt_error_count = 0
    hallucination_count = 0
    all_banned: list[str] = []
    latencies: list[float] = []

    for turn in turns:
        classification = _classify_turn(turn)
        bot_text = turn.get("bot_text", "")

        # Count STT errors
        if classification in ("GARBLED_RECOVERED", "NEEDS_REVIEW", "REPEAT_REQUEST"):
            stt_error_count += 1

        # Detect hallucinations
        hallucination_count += _detect_hallucinations(bot_text)

        # Detect banned words
        banned_in_turn = _detect_banned_words(bot_text)
        all_banned.extend(banned_in_turn)

        # Collect latency
        latency_ms = turn.get("latency_ms", 0)
        if latency_ms > 0:
            latencies.append(latency_ms / 1000.0)

        classified_turns.append({
            **turn,
            "classification": classification,
            "banned_words": banned_in_turn,
        })

    # Auto-estimate duration if not provided
    if duration_sec <= 0 and latencies:
        duration_sec = sum(latencies) * 2  # rough estimate

    # Average latency
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    # Task completion
    task_completion = _infer_task_completion(turns)

    # Deduplicate banned words
    unique_banned = sorted(set(all_banned))

    metrics = ScorecardMetrics(
        call_sid=call_sid,
        duration_sec=duration_sec,
        turn_count=len(turns),
        task_completion=task_completion,
        stt_error_count=stt_error_count,
        hallucination_count=hallucination_count,
        avg_latency_sec=avg_latency,
        banned_words_used=unique_banned,
        turns=classified_turns,
    )

    metrics.overall_grade = grade_call(metrics)
    return metrics


def grade_call(metrics: ScorecardMetrics) -> str:
    """Assign A-F grade based on metrics.

    Grading rubric:
        A: No errors, no banned words, no hallucinations, task completed
        B: Minor issues (<=2 STT errors, task completed/partial)
        C: Moderate issues (some STT errors, partial completion)
        D: Significant issues (many errors, banned words, or hallucinations)
        F: Critical failure (task failed, or hallucinations + banned words)
    """
    # Automatic F conditions
    if metrics.task_completion == "failed" and metrics.hallucination_count > 0:
        return "F"
    if metrics.hallucination_count >= 3:
        return "F"

    # Automatic D conditions
    if metrics.banned_words_used and metrics.hallucination_count > 0:
        return "D"
    if metrics.task_completion == "failed":
        return "D"

    # Score-based grading
    score = 100

    # STT errors: -5 per error
    score -= metrics.stt_error_count * 5

    # Banned words: -15 per word
    score -= len(metrics.banned_words_used) * 15

    # Hallucinations: -20 per occurrence
    score -= metrics.hallucination_count * 20

    # Task completion penalty
    if metrics.task_completion == "partial":
        score -= 10
    elif metrics.task_completion == "failed":
        score -= 30

    # Latency penalty (>2s average is concerning)
    if metrics.avg_latency_sec > 2.0:
        score -= 10
    elif metrics.avg_latency_sec > 3.0:
        score -= 20

    if score >= 90:
        return "A"
    if score >= 75:
        return "B"
    if score >= 60:
        return "C"
    if score >= 40:
        return "D"
    return "F"
