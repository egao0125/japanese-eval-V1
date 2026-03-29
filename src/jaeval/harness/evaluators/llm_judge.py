"""LLM-as-Judge evaluator for Japanese voice AI conversation quality.

Generalized from voice-fullduplex's Tier 2 LLM judge. Supports configurable
dimensions, weights, system prompts, and judge models via YAML or dataclass config.

Usage:
    from jaeval.harness.evaluators.llm_judge import LLMJudge, JudgeConfig

    judge = LLMJudge()  # defaults to 6-dim Japanese business phone eval
    result = judge.evaluate(transcript_text)
    print(result.weighted_score, result.production_ready)

    # From YAML config
    config = JudgeConfig.from_yaml("tasks/conversation/llm_judge_6dim.yaml")
    judge = LLMJudge(config)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class JudgeDimension:
    """A single evaluation dimension with name, weight, and optional description."""

    name: str
    weight: float
    description: str = ""


@dataclass
class JudgeConfig:
    """Configuration for the LLM judge."""

    model: str = "claude-sonnet-4-20250514"
    dimensions: list[JudgeDimension] = field(
        default_factory=lambda: [
            JudgeDimension("task_completion", 0.30, "タスク完遂度"),
            JudgeDimension("natural_flow", 0.20, "自然さ"),
            JudgeDimension("stt_error_handling", 0.20, "音声認識エラー対応"),
            JudgeDimension("prompt_compliance", 0.15, "プロンプト準拠"),
            JudgeDimension("information_accuracy", 0.10, "情報の正確さ"),
            JudgeDimension("caller_experience", 0.05, "発信者体験"),
        ]
    )
    system_prompt: str = ""  # Empty means use DEFAULT_SYSTEM_PROMPT
    max_tokens: int = 4096
    production_ready_threshold: float = 3.5
    min_dimension_score: int = 2

    # -- Helpers ---------------------------------------------------------------

    @property
    def dimension_names(self) -> list[str]:
        return [d.name for d in self.dimensions]

    @property
    def weight_map(self) -> dict[str, float]:
        return {d.name: d.weight for d in self.dimensions}

    # -- Factory from YAML -----------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> JudgeConfig:
        """Load judge config from a YAML task file.

        Expected keys under ``judge:``:
            model, dimensions (name->weight map), production_ready_threshold,
            min_dimension_score.
        """
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        judge_raw = raw.get("judge", {})
        dims: list[JudgeDimension] = []
        for name, weight in judge_raw.get("dimensions", {}).items():
            dims.append(JudgeDimension(name=name, weight=float(weight)))

        return cls(
            model=judge_raw.get("model", cls.model),
            dimensions=dims if dims else cls().dimensions,
            system_prompt=judge_raw.get("system_prompt", ""),
            max_tokens=judge_raw.get("max_tokens", cls.max_tokens),
            production_ready_threshold=float(
                judge_raw.get("production_ready_threshold", cls.production_ready_threshold)
            ),
            min_dimension_score=int(
                judge_raw.get("min_dimension_score", cls.min_dimension_score)
            ),
        )


@dataclass
class JudgeResult:
    """Result returned by the LLM judge."""

    scores: dict[str, dict[str, Any]]  # dim -> {"score": int, "justification": str}
    weighted_score: float
    production_ready: bool
    biggest_issue: str
    recommendations: list[str]
    raw_response: str


# ---------------------------------------------------------------------------
# Default Japanese business phone system prompt
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = """\
あなたは日本語ビジネス電話AIの品質評価者です。
StepAIの電話エージェント「レコ」の通話品質を6つの指標で評価してください。

## 評価指標（各1-5点）

1. **Task Completion**（タスク完遂度、重み: 0.30）
   発信者のニーズに対応できたか？問い合わせに回答したか、デモ予約につなげたか。
   5 = 完全達成、4 = ほぼ達成、3 = 部分的、2 = 不十分、1 = 未達成

2. **Natural Flow**（自然さ、重み: 0.20）
   人間のオペレーターのように自然に聞こえるか？不自然な繰り返しや唐突な応答はないか。
   5 = 人間と区別不可、4 = ほぼ自然、3 = やや不自然、2 = 明らかにAI、1 = 破綻

3. **STT Error Handling**（音声認識エラー対応、重み: 0.20）
   音声認識の誤りに対して適切に対応できたか？
   5 = エラーなし or 完璧に対応、4 = ほぼ適切、3 = 一部対応ミス、2 = 対応不良、1 = エラーに引きずられて破綻

4. **Prompt Compliance**（プロンプト準拠、重み: 0.15）
   禁止ワードを使用していないか？応答の長さ、ペルソナは適切か？
   禁止ワードが1つでもあれば即座に1点。
   5 = 完全準拠、4 = 軽微な逸脱、3 = 一部逸脱、2 = 複数逸脱、1 = 禁止ワード使用

5. **Information Accuracy**（情報の正確さ、重み: 0.10）
   提供した情報は全て正確か？StepAIやRecoについての誤情報はないか。
   5 = 全て正確、4 = ほぼ正確、3 = 軽微な不正確、2 = 重大な不正確あり、1 = 虚偽情報

6. **Caller Experience**（発信者体験、重み: 0.05）
   この通話の後、発信者はStepAIを推薦するか？満足感はあるか。
   5 = 大変満足、4 = 満足、3 = 普通、2 = 不満、1 = 非常に不満

## 評価の注意点
- [conf=X]はSTT(音声認識)の信頼度スコア。数値が低いほど認識精度が低い。
- [uncertain]は音声認識の精度が低い発話を示すマーカー。
- STTが間違えたこと自体はボットの責任ではない。ペナルティにしない。
- ペナルティにすべきは：STTが間違えた内容にそのまま応答し、間違った情報を伝えたとき。
- STTが不確実でも文脈から正しく推測して応答した場合は高評価。
- 意味が分からない場合に聞き返すのは適切な対応。ただし聞き返しが多すぎるとユーザー体験が悪化する。
- 日本語の漢字とひらがなの読みは同一視すること。例：「受電」と「じゅでん」は同じ意味。ボットが漢字をひらがなで表記しても誤認識ではない。TTS（音声合成）の都合で漢字をひらがなに変換するのは正常な動作。

## 禁止ワード（1つでも使ったらPrompt Complianceは1点）
もちろん、かしこまりました、承知しました、申し訳ありません、申し訳ございません、
いたします、させていただきます、できかねます、させていただ（全活用形）、
お手伝い、何かご質問は、他にありますか、何かあれば、いつでもご連絡ください、
お気軽に、遠慮なく、くださいね、くださいませ、詳しく教えてください
注意：「ありがとうございます」は禁止ワードではない。

## 出力形式
必ず以下のJSON形式で出力してください。他のテキストは一切不要です。

```json
{
    "task_completion": {"score": <1-5>, "justification": "<日本語で理由>"},
    "natural_flow": {"score": <1-5>, "justification": "<日本語で理由>"},
    "stt_error_handling": {"score": <1-5>, "justification": "<日本語で理由>"},
    "prompt_compliance": {"score": <1-5>, "justification": "<日本語で理由>"},
    "information_accuracy": {"score": <1-5>, "justification": "<日本語で理由>"},
    "caller_experience": {"score": <1-5>, "justification": "<日本語で理由>"},
    "weighted_score": <加重平均スコア>,
    "production_ready": <true/false (加重平均3.5以上かつ全指標2以上でtrue)>,
    "biggest_issue": "<最大の問題点を1文で>",
    "recommendations": ["<改善提案1>", "<改善提案2>", "..."]
}
```"""


# ---------------------------------------------------------------------------
# JSON extraction helpers
# ---------------------------------------------------------------------------


def extract_json(text: str) -> dict | None:
    """Extract JSON from judge response with multi-strategy fallback.

    Strategies (in order):
      1. Direct parse (response is pure JSON)
      2. Markdown code fence extraction
      3. Outermost brace search
    """
    text = text.strip()

    # Strategy 1: direct parse
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    # Strategy 2: code fence
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Strategy 3: outermost braces
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end > brace_start:
        try:
            return json.loads(text[brace_start : brace_end + 1])
        except json.JSONDecodeError:
            pass

    return None


# ---------------------------------------------------------------------------
# Validation & scoring
# ---------------------------------------------------------------------------


def validate_scores(
    scores: dict, dimensions: list[JudgeDimension]
) -> tuple[bool, list[str]]:
    """Validate judge output structure and score ranges.

    Returns ``(is_valid, list_of_errors)``.
    """
    errors: list[str] = []

    for dim in dimensions:
        name = dim.name
        if name not in scores:
            errors.append(f"Missing dimension: {name}")
            continue
        entry = scores[name]
        if not isinstance(entry, dict):
            errors.append(f"{name}: expected dict, got {type(entry).__name__}")
            continue
        if "score" not in entry:
            errors.append(f"{name}: missing 'score'")
        elif not isinstance(entry["score"], (int, float)):
            errors.append(f"{name}: score must be numeric, got {type(entry['score']).__name__}")
        elif not (1 <= entry["score"] <= 5):
            errors.append(f"{name}: score {entry['score']} out of range [1, 5]")
        if "justification" not in entry:
            errors.append(f"{name}: missing 'justification'")

    for fld in ("weighted_score", "production_ready", "biggest_issue", "recommendations"):
        if fld not in scores:
            errors.append(f"Missing field: {fld}")

    return len(errors) == 0, errors


def compute_weighted_score(scores: dict, dimensions: list[JudgeDimension]) -> float:
    """Recompute weighted score from individual dimension scores."""
    total = 0.0
    for dim in dimensions:
        entry = scores.get(dim.name, {})
        score = entry.get("score", 0) if isinstance(entry, dict) else 0
        total += score * dim.weight
    return round(total, 2)


def compute_production_ready(scores: dict, config: JudgeConfig) -> bool:
    """Determine production readiness.

    ``True`` when weighted_score >= threshold AND every dimension >= min_dimension_score.
    """
    weighted = compute_weighted_score(scores, config.dimensions)
    if weighted < config.production_ready_threshold:
        return False
    for dim in config.dimensions:
        entry = scores.get(dim.name, {})
        score = entry.get("score", 0) if isinstance(entry, dict) else 0
        if score < config.min_dimension_score:
            return False
    return True


# ---------------------------------------------------------------------------
# Transcript formatting
# ---------------------------------------------------------------------------


def format_transcript(scorecard: dict) -> str:
    """Format a scorecard dict into a conversation transcript for the judge.

    Expects ``scorecard`` to have keys: ``call_sid``, ``duration_sec``,
    ``turn_count``, ``overall_grade``, ``task_completion``, ``call_outcome``,
    and ``turns`` (list of turn dicts with ``user_text``, ``bot_text``,
    ``confidence``, ``has_uncertainty``).
    """
    lines: list[str] = []

    # Metadata header
    lines.append("[通話メタデータ]")
    lines.append(f"call_sid: {scorecard.get('call_sid', 'unknown')}")
    duration = scorecard.get("duration_sec", 0)
    lines.append(f"通話時間: {duration:.0f}秒 ({duration / 60:.1f}分)")
    lines.append(f"発話数: {scorecard.get('turn_count', 0)}")
    lines.append(f"Tier1グレード: {scorecard.get('overall_grade', 'N/A')}")
    lines.append(f"タスク完了: {scorecard.get('task_completion', 'N/A')}")
    lines.append(f"通話結果: {scorecard.get('call_outcome', 'N/A')}")
    lines.append("")
    lines.append("[会話]")

    turns = scorecard.get("turns", [])

    # If the first turn has both user and bot text, prepend standard greeting
    if turns:
        first = turns[0]
        if first.get("bot_text") and first.get("user_text"):
            lines.append(
                "BOT: お電話ありがとうございます、StepAIのレコです。"
                "Recoについてのお問い合わせですか？"
            )

    for turn in turns:
        user_text = turn.get("user_text", "")
        bot_text = turn.get("bot_text", "")
        confidence = turn.get("confidence")
        has_uncertainty = turn.get("has_uncertainty", False)

        # Build annotation string
        annotations: list[str] = []
        if confidence is not None:
            annotations.append(f"conf={confidence:.3f}")
        if has_uncertainty:
            annotations.append("uncertain")
        annotation_str = f" [{', '.join(annotations)}]" if annotations else ""

        if user_text:
            lines.append(f"USER{annotation_str}: {user_text}")
        if bot_text:
            lines.append(f"BOT: {bot_text}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLMJudge
# ---------------------------------------------------------------------------


class LLMJudge:
    """LLM-as-Judge evaluator.

    Calls the Anthropic API with a configurable system prompt and dimensions.
    Parses, validates, and recomputes the judge's JSON response.
    """

    def __init__(self, config: JudgeConfig | None = None) -> None:
        self.config = config or JudgeConfig()
        if not self.config.system_prompt:
            self.config.system_prompt = DEFAULT_SYSTEM_PROMPT

    # -- Core API --------------------------------------------------------------

    def evaluate(self, transcript: str, *, dry_run: bool = False) -> JudgeResult:
        """Evaluate a transcript string.

        Args:
            transcript: Formatted conversation text.
            dry_run: If ``True``, return an empty result without calling the API.

        Returns:
            :class:`JudgeResult` with dimension scores, weighted score, etc.

        Raises:
            RuntimeError: If the API response cannot be parsed as JSON.
        """
        user_message = f"以下の通話を評価してください。\n\n{transcript}"

        if dry_run:
            return JudgeResult(
                scores={},
                weighted_score=0.0,
                production_ready=False,
                biggest_issue="",
                recommendations=[],
                raw_response="DRY RUN",
            )

        import anthropic

        client = anthropic.Anthropic()

        try:
            response = client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                system=self.config.system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
        except anthropic.APIError as exc:
            raise RuntimeError(f"Anthropic API error: {exc}") from exc

        raw_text = "".join(b.text for b in response.content if hasattr(b, "text"))

        if not raw_text.strip():
            raise RuntimeError("Empty response from judge API")

        scores = extract_json(raw_text)
        if scores is None:
            raise RuntimeError(f"Failed to parse judge response: {raw_text[:500]}")

        # Validate
        is_valid, errors = validate_scores(scores, self.config.dimensions)
        if not is_valid:
            import sys

            print(f"  WARNING: Judge output validation issues: {errors}", file=sys.stderr)

        # Recompute weighted_score and production_ready for consistency
        scores["weighted_score"] = compute_weighted_score(scores, self.config.dimensions)
        scores["production_ready"] = compute_production_ready(scores, self.config)

        # Extract dimension scores into structured dict
        dim_scores: dict[str, dict[str, Any]] = {}
        for dim in self.config.dimensions:
            entry = scores.get(dim.name, {})
            if isinstance(entry, dict):
                dim_scores[dim.name] = {
                    "score": entry.get("score", 0),
                    "justification": entry.get("justification", ""),
                }
            else:
                dim_scores[dim.name] = {"score": 0, "justification": ""}

        return JudgeResult(
            scores=dim_scores,
            weighted_score=scores["weighted_score"],
            production_ready=scores["production_ready"],
            biggest_issue=scores.get("biggest_issue", ""),
            recommendations=scores.get("recommendations", []),
            raw_response=raw_text,
        )

    # -- Convenience -----------------------------------------------------------

    def evaluate_scorecard(self, scorecard: dict, *, dry_run: bool = False) -> JudgeResult:
        """Evaluate from a scorecard dict (output of :func:`build_scorecard`).

        Formats the scorecard into a conversation transcript, then calls
        :meth:`evaluate`.
        """
        transcript = format_transcript(scorecard)
        return self.evaluate(transcript, dry_run=dry_run)
