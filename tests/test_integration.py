"""Tests for jaeval.integration module."""

from unittest.mock import MagicMock, patch

from jaeval.integration import evaluate_call, CallEvalResult


def _sample_turns():
    """Sample turn data for a typical call."""
    return [
        {
            "user_text": "もしもし",
            "bot_text": "はい、StepAIです。お電話ありがとうございます。",
            "confidence": -0.05,
            "latency_ms": 800,
            "duration_s": 1.2,
        },
        {
            "user_text": "Recoの料金について教えてください",
            "bot_text": "Recoの料金プランをご案内します。月額5,000円からとなっております。",
            "confidence": -0.08,
            "latency_ms": 1200,
            "duration_s": 3.5,
        },
        {
            "user_text": "デモの予約はできますか",
            "bot_text": "はい、デモの予約を承りました。担当者から折り返しご連絡いたします。",
            "confidence": -0.03,
            "latency_ms": 900,
            "duration_s": 2.8,
        },
    ]


class TestEvaluateCallTier1Only:
    """Test Tier 1 scorecard without LLM judge."""

    def test_basic_call(self):
        result = evaluate_call("CA001", _sample_turns(), duration_sec=60.0, run_judge=False)
        assert isinstance(result, CallEvalResult)
        assert result.call_sid == "CA001"
        assert result.grade in ("A", "B", "C", "D", "F")
        assert result.task_completion in ("completed", "partial", "failed")
        assert result.weighted_score is None  # Judge not run

    def test_clean_call_gets_good_grade(self):
        turns = _sample_turns()
        result = evaluate_call("CA002", turns, duration_sec=60.0, run_judge=False)
        # Good call with task completion signals → should be A or B
        assert result.grade in ("A", "B")
        assert result.stt_error_count == 0

    def test_call_with_stt_errors(self):
        turns = _sample_turns()
        # Add garbled turns
        turns.append({
            "user_text": "かでにん",
            "bot_text": "聞き取れませんでした。もう一度お願いできますか？",
            "confidence": -0.30,
            "has_uncertainty": True,
            "latency_ms": 500,
            "duration_s": 0.8,
        })
        result = evaluate_call("CA003", turns, run_judge=False)
        assert result.stt_error_count >= 1

    def test_call_with_banned_words(self):
        turns = [
            {
                "user_text": "どうも",
                "bot_text": "かしこまりました。承知しました。",
                "latency_ms": 500,
            },
        ]
        result = evaluate_call("CA004", turns, run_judge=False)
        assert len(result.banned_words_used) > 0

    def test_call_with_hallucinations(self):
        turns = [
            {
                "user_text": "もしもし",
                "bot_text": "音声が途切れているようです。聞こえないですね。",
                "latency_ms": 500,
            },
        ]
        result = evaluate_call("CA005", turns, run_judge=False)
        assert result.hallucination_count > 0

    def test_empty_call(self):
        result = evaluate_call("CA006", [], run_judge=False)
        assert result.scorecard["turn_count"] == 0
        assert result.task_completion == "failed"

    def test_to_dict(self):
        result = evaluate_call("CA007", _sample_turns(), run_judge=False)
        d = result.to_dict()
        assert d["call_sid"] == "CA007"
        assert "tier1" in d
        assert d["tier2"] is None  # Judge not run
        assert "scorecard" in d

    def test_scorecard_in_result(self):
        result = evaluate_call("CA008", _sample_turns(), duration_sec=60.0, run_judge=False)
        sc = result.scorecard
        assert sc["call_sid"] == "CA008"
        assert sc["duration_sec"] == 60.0
        assert len(sc["turns"]) == 3

    @property
    def turn_count(self):
        turns = _sample_turns()
        result = evaluate_call("CA009", turns, run_judge=False)
        assert result.scorecard["turn_count"] == len(turns)


class TestEvaluateCallWithJudge:
    """Test Tier 2 LLM judge integration."""

    def test_judge_import_error_graceful(self):
        """If anthropic is not installed, Tier 2 is silently skipped."""
        with patch(
            "jaeval.harness.evaluators.llm_judge.LLMJudge",
            side_effect=ImportError("no anthropic"),
        ):
            result = evaluate_call("CA010", _sample_turns(), run_judge=True)
            # Should not crash — just skip Tier 2
            assert result.grade in ("A", "B", "C", "D", "F")
            assert result.weighted_score is None

    def test_judge_error_graceful(self):
        """If judge throws, error is recorded but call doesn't crash."""
        mock_judge = MagicMock()
        mock_judge.evaluate_scorecard.side_effect = RuntimeError("API error")

        with patch(
            "jaeval.harness.evaluators.llm_judge.LLMJudge",
            return_value=mock_judge,
        ), patch("jaeval.harness.evaluators.llm_judge.JudgeConfig"):
            result = evaluate_call("CA011", _sample_turns(), run_judge=True)
            assert "Judge error" in result.biggest_issue
            assert result.weighted_score is None
