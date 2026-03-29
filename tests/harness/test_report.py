"""Tests for report formatters (markdown and JSON serialization)."""

from __future__ import annotations

import json

from jaeval.core.metrics import CERResult
from jaeval.harness.gate import GateCheck
from jaeval.harness.report import _serialize_report, format_markdown, save_json, save_markdown
from jaeval.harness.runner import AggregateMetrics, BenchmarkReport, UtteranceResult


def _make_report(
    *,
    scored: int = 2,
    mean_cer: float = 0.05,
    median_cer: float = 0.04,
    gate_result: str = "PASS",
    include_lenient: bool = False,
) -> BenchmarkReport:
    """Build a minimal BenchmarkReport for testing."""
    cer1 = CERResult(
        cer=0.03, substitutions=1, insertions=0, deletions=0,
        ref_len=30, ref_normalized="テスト", hyp_normalized="テスト",
    )
    cer2 = CERResult(
        cer=0.07, substitutions=2, insertions=1, deletions=0,
        ref_len=30, ref_normalized="料金", hyp_normalized="りょうきん",
    )

    lenient1 = CERResult(
        cer=0.02, substitutions=0, insertions=0, deletions=0,
        ref_len=30, ref_normalized="てすと", hyp_normalized="てすと",
    ) if include_lenient else None

    utterances = [
        UtteranceResult(
            id="u1", category="keigo", reference="テスト", hypothesis="テスト",
            cer=cer1, lenient_cer=lenient1,
            hallucinated_kanji=[], latency_sec=0.5, audio_duration_sec=2.0, rtf=0.25,
        ),
        UtteranceResult(
            id="u2", category="number", reference="料金", hypothesis="りょうきん",
            cer=cer2, hallucinated_kanji=["謎"],
            latency_sec=1.2, audio_duration_sec=3.0, rtf=0.4,
        ),
    ]

    aggregate = AggregateMetrics(
        mean_cer=mean_cer, median_cer=median_cer,
        min_cer=0.03, max_cer=0.07, std_cer=0.02,
        total_hallucinations=1,
        latency_p50=0.5, latency_p90=1.2, latency_mean=0.85,
        rtf_mean=0.325, rtf_max=0.4,
        mean_lenient_cer=0.02 if include_lenient else None,
        median_lenient_cer=0.02 if include_lenient else None,
    )

    gate_checks = [
        GateCheck(metric="median_cer", value=median_cer, pass_threshold=0.05,
                  warn_threshold=0.08, result=gate_result),
    ]

    per_category = {
        "keigo": {"count": 1, "mean_cer": 0.03, "median_cer": 0.03,
                  "hallucinations": 0, "mean_latency": 0.5},
        "number": {"count": 1, "mean_cer": 0.07, "median_cer": 0.07,
                   "hallucinations": 1, "mean_latency": 1.2},
    }

    return BenchmarkReport(
        task="stt_test", model="test_model", timestamp="2026-03-29T12:00:00Z",
        total_entries=2, scored=scored, skipped=0, errors=0,
        aggregate=aggregate, per_category=per_category,
        utterance_results=utterances,
        gate_result=gate_result, gate_checks=gate_checks,
    )


class TestFormatMarkdown:
    def test_header_present(self):
        md = format_markdown(_make_report())
        assert "## STT Benchmark: test_model" in md
        assert "Task: stt_test" in md

    def test_aggregate_table(self):
        md = format_markdown(_make_report())
        assert "### Aggregate Metrics" in md
        assert "Mean CER" in md
        assert "Median CER" in md
        assert "Hallucinations" in md

    def test_per_category_table(self):
        md = format_markdown(_make_report())
        assert "### Per-Category Breakdown" in md
        assert "keigo" in md
        assert "number" in md

    def test_per_utterance_table(self):
        md = format_markdown(_make_report())
        assert "### Per-Utterance Results" in md
        assert "u1" in md
        assert "u2" in md

    def test_gate_results(self):
        md = format_markdown(_make_report(gate_result="PASS"))
        assert "### Gate Results" in md
        assert "**PASS**" in md

    def test_verdict_target_met(self):
        md = format_markdown(_make_report(mean_cer=0.02))
        assert "TARGET MET" in md

    def test_verdict_excellent(self):
        md = format_markdown(_make_report(mean_cer=0.08))
        assert "EXCELLENT" in md

    def test_verdict_acceptable(self):
        md = format_markdown(_make_report(mean_cer=0.15))
        assert "ACCEPTABLE" in md

    def test_verdict_marginal(self):
        md = format_markdown(_make_report(mean_cer=0.25))
        assert "MARGINAL" in md

    def test_verdict_poor(self):
        md = format_markdown(_make_report(mean_cer=0.35))
        assert "POOR" in md

    def test_lenient_cer_shown(self):
        md = format_markdown(_make_report(include_lenient=True))
        assert "Lenient CER" in md

    def test_lenient_cer_hidden_when_none(self):
        md = format_markdown(_make_report(include_lenient=False))
        assert "Lenient CER" not in md

    def test_no_aggregate_message(self):
        """When aggregate is None, show a message."""
        report = _make_report()
        report.aggregate = None
        md = format_markdown(report)
        assert "No utterances scored" in md

    def test_utterance_status_pass(self):
        md = format_markdown(_make_report())
        # u1 has CER 3% → PASS
        assert "PASS" in md

    def test_utterance_status_fail(self):
        """Utterance with CER > 30% should show FAIL."""
        report = _make_report()
        report.utterance_results[1].cer = CERResult(
            cer=0.40, substitutions=5, insertions=3, deletions=2,
            ref_len=10, ref_normalized="x", hyp_normalized="y",
        )
        md = format_markdown(report)
        assert "FAIL" in md


class TestSerializeReport:
    def test_basic_serialization(self):
        data = _serialize_report(_make_report())
        assert data["task"] == "stt_test"
        assert data["model"] == "test_model"
        assert data["total_entries"] == 2
        assert data["scored"] == 2

    def test_aggregate_fields(self):
        data = _serialize_report(_make_report())
        agg = data["aggregate"]
        assert agg["mean_cer"] == 0.05
        assert agg["total_hallucinations"] == 1

    def test_results_array(self):
        data = _serialize_report(_make_report())
        assert len(data["results"]) == 2
        assert data["results"][0]["id"] == "u1"
        assert data["results"][1]["hallucinated_count"] == 1

    def test_gate_checks(self):
        data = _serialize_report(_make_report())
        assert len(data["gate_checks"]) == 1
        assert data["gate_checks"][0]["metric"] == "median_cer"

    def test_json_serializable(self):
        """Verify the serialized dict can be passed through json.dumps."""
        data = _serialize_report(_make_report())
        json_str = json.dumps(data, ensure_ascii=False)
        assert '"stt_test"' in json_str


class TestSaveFiles:
    def test_save_json(self, tmp_path):
        out = tmp_path / "report.json"
        save_json(_make_report(), out)
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["task"] == "stt_test"

    def test_save_markdown(self, tmp_path):
        out = tmp_path / "report.md"
        save_markdown(_make_report(), out)
        assert out.exists()
        content = out.read_text()
        assert "## STT Benchmark" in content

    def test_save_creates_parent_dirs(self, tmp_path):
        out = tmp_path / "nested" / "dir" / "report.json"
        save_json(_make_report(), out)
        assert out.exists()
