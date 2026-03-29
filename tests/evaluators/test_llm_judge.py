"""Tests for the LLM-as-Judge evaluator."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from jaeval.harness.evaluators.llm_judge import (
    JudgeConfig,
    JudgeDimension,
    JudgeResult,
    LLMJudge,
    compute_production_ready,
    compute_weighted_score,
    extract_json,
    format_transcript,
    validate_scores,
)


# ---------------------------------------------------------------------------
# extract_json
# ---------------------------------------------------------------------------


class TestExtractJson:
    def test_direct_json(self):
        data = extract_json('{"a": 1}')
        assert data == {"a": 1}

    def test_code_fence(self):
        text = "Here is the result:\n```json\n{\"x\": 2}\n```\nDone."
        assert extract_json(text) == {"x": 2}

    def test_code_fence_no_language(self):
        text = "```\n{\"y\": 3}\n```"
        assert extract_json(text) == {"y": 3}

    def test_brace_extraction(self):
        text = "Some preamble {\"key\": \"value\"} trailing text"
        assert extract_json(text) == {"key": "value"}

    def test_returns_none_for_garbage(self):
        assert extract_json("no json here") is None

    def test_nested_json(self):
        data = extract_json('{"a": {"b": [1, 2]}}')
        assert data == {"a": {"b": [1, 2]}}


# ---------------------------------------------------------------------------
# validate_scores
# ---------------------------------------------------------------------------

_DIMS_2 = [
    JudgeDimension("dim_a", 0.6),
    JudgeDimension("dim_b", 0.4),
]


class TestValidateScores:
    def test_valid(self):
        scores = {
            "dim_a": {"score": 4, "justification": "good"},
            "dim_b": {"score": 3, "justification": "ok"},
            "weighted_score": 3.6,
            "production_ready": True,
            "biggest_issue": "none",
            "recommendations": [],
        }
        is_valid, errors = validate_scores(scores, _DIMS_2)
        assert is_valid
        assert errors == []

    def test_missing_dimension(self):
        scores = {
            "dim_b": {"score": 3, "justification": "ok"},
            "weighted_score": 0,
            "production_ready": False,
            "biggest_issue": "",
            "recommendations": [],
        }
        is_valid, errors = validate_scores(scores, _DIMS_2)
        assert not is_valid
        assert any("dim_a" in e for e in errors)

    def test_score_out_of_range(self):
        scores = {
            "dim_a": {"score": 6, "justification": "too high"},
            "dim_b": {"score": 3, "justification": "ok"},
            "weighted_score": 0,
            "production_ready": False,
            "biggest_issue": "",
            "recommendations": [],
        }
        is_valid, errors = validate_scores(scores, _DIMS_2)
        assert not is_valid
        assert any("out of range" in e for e in errors)

    def test_missing_justification(self):
        scores = {
            "dim_a": {"score": 4},
            "dim_b": {"score": 3, "justification": "ok"},
            "weighted_score": 0,
            "production_ready": False,
            "biggest_issue": "",
            "recommendations": [],
        }
        is_valid, errors = validate_scores(scores, _DIMS_2)
        assert not is_valid
        assert any("justification" in e for e in errors)

    def test_missing_top_level_fields(self):
        scores = {
            "dim_a": {"score": 4, "justification": "ok"},
            "dim_b": {"score": 3, "justification": "ok"},
        }
        is_valid, errors = validate_scores(scores, _DIMS_2)
        assert not is_valid
        assert len(errors) == 4  # weighted_score, production_ready, biggest_issue, recommendations


# ---------------------------------------------------------------------------
# compute_weighted_score
# ---------------------------------------------------------------------------


class TestComputeWeightedScore:
    def test_basic(self):
        scores = {
            "dim_a": {"score": 5},
            "dim_b": {"score": 3},
        }
        result = compute_weighted_score(scores, _DIMS_2)
        expected = 5 * 0.6 + 3 * 0.4  # 3.0 + 1.2 = 4.2
        assert result == expected

    def test_missing_dimension(self):
        scores = {"dim_a": {"score": 5}}
        result = compute_weighted_score(scores, _DIMS_2)
        assert result == 5 * 0.6  # dim_b contributes 0

    def test_all_fives(self):
        scores = {
            "dim_a": {"score": 5},
            "dim_b": {"score": 5},
        }
        assert compute_weighted_score(scores, _DIMS_2) == 5.0


# ---------------------------------------------------------------------------
# compute_production_ready
# ---------------------------------------------------------------------------


class TestComputeProductionReady:
    def test_ready(self):
        config = JudgeConfig(
            dimensions=_DIMS_2,
            production_ready_threshold=3.5,
            min_dimension_score=2,
        )
        scores = {
            "dim_a": {"score": 4},
            "dim_b": {"score": 4},
        }
        assert compute_production_ready(scores, config) is True

    def test_not_ready_low_weighted(self):
        config = JudgeConfig(
            dimensions=_DIMS_2,
            production_ready_threshold=3.5,
            min_dimension_score=2,
        )
        scores = {
            "dim_a": {"score": 2},
            "dim_b": {"score": 2},
        }
        # weighted = 2*0.6 + 2*0.4 = 2.0 < 3.5
        assert compute_production_ready(scores, config) is False

    def test_not_ready_low_dimension(self):
        config = JudgeConfig(
            dimensions=_DIMS_2,
            production_ready_threshold=3.0,
            min_dimension_score=3,
        )
        scores = {
            "dim_a": {"score": 5},
            "dim_b": {"score": 2},  # below min_dimension_score=3
        }
        assert compute_production_ready(scores, config) is False


# ---------------------------------------------------------------------------
# format_transcript
# ---------------------------------------------------------------------------


class TestFormatTranscript:
    def test_basic_format(self):
        scorecard = {
            "call_sid": "TEST001",
            "duration_sec": 120.0,
            "turn_count": 2,
            "overall_grade": "B",
            "task_completion": "completed",
            "call_outcome": "info_provided",
            "turns": [
                {
                    "user_text": "Recoについて教えてください",
                    "bot_text": "はい、Recoは電話AIです。",
                    "confidence": 0.95,
                    "has_uncertainty": False,
                },
                {
                    "user_text": "料金は？",
                    "bot_text": "料金についてご案内します。",
                    "confidence": 0.88,
                    "has_uncertainty": False,
                },
            ],
        }
        text = format_transcript(scorecard)
        assert "[通話メタデータ]" in text
        assert "TEST001" in text
        assert "120秒" in text
        assert "USER [conf=0.950]: Recoについて教えてください" in text
        assert "BOT: はい、Recoは電話AIです。" in text

    def test_uncertainty_annotation(self):
        scorecard = {
            "call_sid": "TEST002",
            "duration_sec": 60.0,
            "turn_count": 1,
            "overall_grade": "C",
            "task_completion": "partial",
            "call_outcome": "unknown",
            "turns": [
                {
                    "user_text": "カデニンを...",
                    "bot_text": "架電についてですね。",
                    "confidence": -0.2,
                    "has_uncertainty": True,
                },
            ],
        }
        text = format_transcript(scorecard)
        assert "uncertain" in text
        assert "conf=-0.200" in text

    def test_empty_turns(self):
        scorecard = {
            "call_sid": "EMPTY",
            "duration_sec": 0,
            "turn_count": 0,
            "turns": [],
        }
        text = format_transcript(scorecard)
        assert "[会話]" in text
        assert "USER" not in text

    def test_opening_greeting_prepended(self):
        scorecard = {
            "call_sid": "GREET",
            "duration_sec": 30.0,
            "turn_count": 1,
            "turns": [
                {
                    "user_text": "はい",
                    "bot_text": "Recoのご案内です",
                    "confidence": 0.9,
                },
            ],
        }
        text = format_transcript(scorecard)
        # First turn has both user and bot, so greeting should be prepended
        lines = text.split("\n")
        conversation_start = lines.index("[会話]") + 1
        assert lines[conversation_start].startswith("BOT: お電話ありがとうございます")


# ---------------------------------------------------------------------------
# JudgeConfig
# ---------------------------------------------------------------------------


class TestJudgeConfig:
    def test_defaults(self):
        config = JudgeConfig()
        assert len(config.dimensions) == 6
        assert config.production_ready_threshold == 3.5
        assert config.min_dimension_score == 2
        assert sum(d.weight for d in config.dimensions) == pytest.approx(1.0)

    def test_dimension_names(self):
        config = JudgeConfig()
        names = config.dimension_names
        assert "task_completion" in names
        assert "caller_experience" in names
        assert len(names) == 6

    def test_weight_map(self):
        config = JudgeConfig()
        wm = config.weight_map
        assert wm["task_completion"] == 0.30
        assert wm["caller_experience"] == 0.05

    def test_from_yaml(self, tmp_path):
        yaml_content = """
task: test_judge
type: conversation
judge:
  model: claude-haiku-4-5-20251001
  max_tokens: 1024
  dimensions:
    accuracy: 0.5
    fluency: 0.5
  production_ready_threshold: 4.0
  min_dimension_score: 3
"""
        yaml_file = tmp_path / "judge.yaml"
        yaml_file.write_text(yaml_content)
        config = JudgeConfig.from_yaml(yaml_file)
        assert config.model == "claude-haiku-4-5-20251001"
        assert config.max_tokens == 1024
        assert len(config.dimensions) == 2
        assert config.dimensions[0].name == "accuracy"
        assert config.dimensions[0].weight == 0.5
        assert config.production_ready_threshold == 4.0
        assert config.min_dimension_score == 3

    def test_from_yaml_defaults(self, tmp_path):
        yaml_content = """
task: minimal
type: conversation
judge: {}
"""
        yaml_file = tmp_path / "minimal.yaml"
        yaml_file.write_text(yaml_content)
        config = JudgeConfig.from_yaml(yaml_file)
        # Should fall back to defaults
        assert len(config.dimensions) == 6
        assert config.model == "claude-sonnet-4-20250514"


# ---------------------------------------------------------------------------
# LLMJudge (with mocked API)
# ---------------------------------------------------------------------------


def _mock_api_response():
    """Create a mock Anthropic API response with valid judge output."""
    return json.dumps({
        "task_completion": {"score": 4, "justification": "タスクをほぼ達成"},
        "natural_flow": {"score": 3, "justification": "やや不自然"},
        "stt_error_handling": {"score": 5, "justification": "完璧に対応"},
        "prompt_compliance": {"score": 4, "justification": "軽微な逸脱"},
        "information_accuracy": {"score": 5, "justification": "全て正確"},
        "caller_experience": {"score": 4, "justification": "満足"},
        "weighted_score": 4.05,
        "production_ready": True,
        "biggest_issue": "自然さの改善が必要",
        "recommendations": ["フィラーを追加", "応答速度を改善"],
    })


class TestLLMJudge:
    def test_dry_run(self):
        judge = LLMJudge()
        result = judge.evaluate("test transcript", dry_run=True)
        assert isinstance(result, JudgeResult)
        assert result.raw_response == "DRY RUN"
        assert result.weighted_score == 0.0

    @patch("anthropic.Anthropic")
    def test_evaluate_success(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        mock_block = MagicMock()
        mock_block.text = _mock_api_response()
        mock_response = MagicMock()
        mock_response.content = [mock_block]
        mock_client.messages.create.return_value = mock_response

        judge = LLMJudge()
        result = judge.evaluate("test transcript")

        assert isinstance(result, JudgeResult)
        assert result.scores["task_completion"]["score"] == 4
        assert result.scores["stt_error_handling"]["score"] == 5
        # Recomputed weighted score
        expected = (4 * 0.30 + 3 * 0.20 + 5 * 0.20 + 4 * 0.15 + 5 * 0.10 + 4 * 0.05)
        assert result.weighted_score == pytest.approx(expected, abs=0.01)
        assert result.production_ready is True
        assert "自然さ" in result.biggest_issue
        assert len(result.recommendations) == 2

    @patch("anthropic.Anthropic")
    def test_evaluate_empty_response(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        mock_block = MagicMock()
        mock_block.text = ""
        mock_response = MagicMock()
        mock_response.content = [mock_block]
        mock_client.messages.create.return_value = mock_response

        judge = LLMJudge()
        with pytest.raises(RuntimeError, match="Empty response"):
            judge.evaluate("test")

    @patch("anthropic.Anthropic")
    def test_evaluate_unparseable_response(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        mock_block = MagicMock()
        mock_block.text = "I cannot evaluate this call."
        mock_response = MagicMock()
        mock_response.content = [mock_block]
        mock_client.messages.create.return_value = mock_response

        judge = LLMJudge()
        with pytest.raises(RuntimeError, match="Failed to parse"):
            judge.evaluate("test")

    def test_evaluate_scorecard(self):
        judge = LLMJudge()
        scorecard = {
            "call_sid": "SC001",
            "duration_sec": 90.0,
            "turn_count": 1,
            "overall_grade": "B",
            "task_completion": "completed",
            "call_outcome": "info",
            "turns": [{"user_text": "テスト", "bot_text": "はい"}],
        }
        result = judge.evaluate_scorecard(scorecard, dry_run=True)
        assert result.raw_response == "DRY RUN"
