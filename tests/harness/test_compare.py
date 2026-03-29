"""Test benchmark comparison report generation."""
import json

from jaeval.harness.compare import format_comparison_markdown, load_results


def _make_result(model: str, median_cer: float, hallucinations: int) -> dict:
    """Create a minimal benchmark result dict for testing."""
    return {
        "task": "test_task",
        "model": model,
        "timestamp": "2026-03-28T00:00:00+00:00",
        "aggregate": {
            "mean_cer": median_cer + 0.05,
            "median_cer": median_cer,
            "latency_p50": 0.5,
            "latency_p90": 1.0,
            "rtf_mean": 0.3,
            "total_hallucinations": hallucinations,
        },
        "per_category": {
            "greeting": {"count": 10, "mean_cer": median_cer, "hallucinations": 0},
            "keigo": {"count": 10, "mean_cer": median_cer + 0.1, "hallucinations": hallucinations},
        },
    }


class TestCompare:
    def test_load_results(self, tmp_path):
        f1 = tmp_path / "a.json"
        f1.write_text(json.dumps(_make_result("model_a", 0.10, 5)))
        results = load_results([f1])
        assert len(results) == 1
        assert results[0]["model"] == "model_a"

    def test_load_results_skip_invalid(self, tmp_path):
        f1 = tmp_path / "bad.json"
        f1.write_text("not json")
        results = load_results([f1])
        assert len(results) == 0

    def test_comparison_markdown_single(self, tmp_path):
        f1 = tmp_path / "a.json"
        f1.write_text(json.dumps(_make_result("model_a", 0.10, 5)))
        md = format_comparison_markdown([f1])
        assert "model_a" in md
        assert "10.0%" in md

    def test_comparison_markdown_two_models(self, tmp_path):
        f1 = tmp_path / "a.json"
        f2 = tmp_path / "b.json"
        f1.write_text(json.dumps(_make_result("model_a", 0.10, 5)))
        f2.write_text(json.dumps(_make_result("model_b", 0.15, 3)))
        md = format_comparison_markdown([f1, f2])
        assert "model_a" in md
        assert "model_b" in md
        assert "Best overall" in md
        assert "model_a" in md  # model_a has better median CER

    def test_comparison_markdown_best_bolded(self, tmp_path):
        f1 = tmp_path / "a.json"
        f2 = tmp_path / "b.json"
        f1.write_text(json.dumps(_make_result("model_a", 0.10, 5)))
        f2.write_text(json.dumps(_make_result("model_b", 0.15, 3)))
        md = format_comparison_markdown([f1, f2])
        # model_a has better CER -> bolded
        assert "**10.0%**" in md
        # model_b has fewer hallucinations -> bolded
        assert "**3**" in md

    def test_comparison_no_files(self):
        md = format_comparison_markdown([])
        assert "No valid result files" in md

    def test_per_category_section(self, tmp_path):
        f1 = tmp_path / "a.json"
        f2 = tmp_path / "b.json"
        f1.write_text(json.dumps(_make_result("model_a", 0.10, 5)))
        f2.write_text(json.dumps(_make_result("model_b", 0.15, 3)))
        md = format_comparison_markdown([f1, f2])
        assert "Per-Category CER" in md
        assert "greeting" in md
        assert "keigo" in md
