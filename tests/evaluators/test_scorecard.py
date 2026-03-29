"""Tests for the automated call scorecard evaluator."""
from __future__ import annotations

from jaeval.harness.evaluators.scorecard import (
    ScorecardMetrics,
    _classify_turn,
    _detect_banned_words,
    _detect_hallucinations,
    _infer_task_completion,
    build_scorecard,
    grade_call,
)


# ---------------------------------------------------------------------------
# Turn classification
# ---------------------------------------------------------------------------


class TestClassifyTurn:
    def test_clean_understood(self):
        turn = {"user_text": "料金を教えてください", "bot_text": "はい", "confidence": 0.9}
        assert _classify_turn(turn) == "CLEAN_UNDERSTOOD"

    def test_garbled_recovered(self):
        turn = {
            "user_text": "カデニン",
            "bot_text": "架電についてですね",
            "confidence": -0.14,
            "has_uncertainty": True,
        }
        assert _classify_turn(turn) == "GARBLED_RECOVERED"

    def test_needs_review(self):
        turn = {
            "user_text": "???",
            "bot_text": "はい",
            "confidence": -0.30,
            "has_uncertainty": True,
        }
        assert _classify_turn(turn) == "NEEDS_REVIEW"

    def test_repeat_request(self):
        turn = {
            "user_text": "テスト",
            "bot_text": "もう一度お願いできますか",
            "confidence": 0.5,
        }
        assert _classify_turn(turn) == "REPEAT_REQUEST"

    def test_truncated(self):
        turn = {"user_text": "は", "bot_text": "はい", "duration_s": 0.3}
        assert _classify_turn(turn) == "TRUNCATED"


# ---------------------------------------------------------------------------
# Banned word detection
# ---------------------------------------------------------------------------


class TestDetectBannedWords:
    def test_no_banned(self):
        assert _detect_banned_words("Recoのご案内です") == []

    def test_single_banned(self):
        found = _detect_banned_words("もちろんでございます")
        assert "もちろん" in found

    def test_multiple_banned(self):
        text = "もちろん、かしこまりました"
        found = _detect_banned_words(text)
        assert "もちろん" in found
        assert "かしこまりました" in found

    def test_arigatou_not_banned(self):
        # "ありがとうございます" should not trigger banned word
        found = _detect_banned_words("ありがとうございます")
        assert found == []

    def test_sasete_itadaku(self):
        found = _detect_banned_words("ご説明させていただきます")
        assert "させていただきます" in found or "させていただ" in found


# ---------------------------------------------------------------------------
# Hallucination detection
# ---------------------------------------------------------------------------


class TestDetectHallucinations:
    def test_no_hallucinations(self):
        assert _detect_hallucinations("はい、Recoのご案内です") == 0

    def test_audio_hallucination(self):
        assert _detect_hallucinations("お客様の声が聞こえないようです") >= 1

    def test_multiple_hallucinations(self):
        text = "声が聞こえないです。音声が途切れています。"
        assert _detect_hallucinations(text) >= 2


# ---------------------------------------------------------------------------
# Task completion inference
# ---------------------------------------------------------------------------


class TestInferTaskCompletion:
    def test_completed(self):
        turns = [
            {"bot_text": "デモのご予約を承りました。ありがとうございます。"},
        ]
        assert _infer_task_completion(turns) == "completed"

    def test_failed(self):
        turns = [
            {"bot_text": "お力になれず申し訳ございません。"},
        ]
        assert _infer_task_completion(turns) == "failed"

    def test_partial(self):
        turns = [
            {"bot_text": "Recoのご案内です。"},
        ]
        assert _infer_task_completion(turns) == "partial"

    def test_empty(self):
        assert _infer_task_completion([]) == "failed"


# ---------------------------------------------------------------------------
# build_scorecard
# ---------------------------------------------------------------------------


class TestBuildScorecard:
    def _sample_turns(self):
        return [
            {
                "user_text": "Recoの料金を教えてください",
                "bot_text": "はい、Recoは月額1万円からご利用いただけます。",
                "confidence": 0.92,
                "has_uncertainty": False,
                "latency_ms": 500,
            },
            {
                "user_text": "デモ予約したい",
                "bot_text": "デモのご予約を承りました。ありがとうございます。",
                "confidence": 0.85,
                "has_uncertainty": False,
                "latency_ms": 600,
            },
        ]

    def test_basic_scorecard(self):
        sc = build_scorecard("CALL001", self._sample_turns(), duration_sec=120.0)
        assert sc.call_sid == "CALL001"
        assert sc.duration_sec == 120.0
        assert sc.turn_count == 2
        assert sc.stt_error_count == 0
        assert sc.hallucination_count == 0
        assert sc.banned_words_used == []
        assert sc.task_completion == "completed"
        assert sc.overall_grade == "A"

    def test_with_errors(self):
        turns = [
            {
                "user_text": "カデニン",
                "bot_text": "もう一度お願いできますか",
                "confidence": -0.2,
                "has_uncertainty": True,
                "latency_ms": 700,
            },
        ]
        sc = build_scorecard("CALL002", turns, duration_sec=60.0)
        assert sc.stt_error_count >= 1

    def test_banned_words_counted(self):
        turns = [
            {
                "user_text": "テスト",
                "bot_text": "もちろん、かしこまりました",
                "confidence": 0.9,
            },
        ]
        sc = build_scorecard("CALL003", turns, duration_sec=30.0)
        assert len(sc.banned_words_used) >= 2

    def test_to_dict(self):
        sc = build_scorecard("CALL004", self._sample_turns(), duration_sec=90.0)
        d = sc.to_dict()
        assert d["call_sid"] == "CALL004"
        assert "turns" in d
        assert isinstance(d["avg_latency_sec"], float)


# ---------------------------------------------------------------------------
# grade_call
# ---------------------------------------------------------------------------


class TestGradeCall:
    def test_grade_a(self):
        metrics = ScorecardMetrics(
            call_sid="A",
            duration_sec=120,
            turn_count=5,
            task_completion="completed",
            stt_error_count=0,
            hallucination_count=0,
            avg_latency_sec=0.5,
        )
        assert grade_call(metrics) == "A"

    def test_grade_f_failed_with_hallucinations(self):
        metrics = ScorecardMetrics(
            call_sid="F",
            duration_sec=30,
            turn_count=2,
            task_completion="failed",
            stt_error_count=3,
            hallucination_count=2,
            avg_latency_sec=1.0,
        )
        assert grade_call(metrics) == "F"

    def test_grade_d_banned_plus_hallucination(self):
        metrics = ScorecardMetrics(
            call_sid="D",
            duration_sec=60,
            turn_count=3,
            task_completion="completed",
            stt_error_count=1,
            hallucination_count=1,
            avg_latency_sec=1.0,
            banned_words_used=["もちろん"],
        )
        assert grade_call(metrics) == "D"

    def test_grade_b(self):
        metrics = ScorecardMetrics(
            call_sid="B",
            duration_sec=90,
            turn_count=4,
            task_completion="partial",
            stt_error_count=3,
            hallucination_count=0,
            avg_latency_sec=1.0,
        )
        # score = 100 - 15 (3 errors) - 10 (partial) = 75 → B
        assert grade_call(metrics) == "B"
