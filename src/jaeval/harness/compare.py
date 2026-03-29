"""Cross-run comparison tool for benchmark results.

Loads multiple benchmark result JSON files and generates a side-by-side
comparison table in markdown format, including per-category CER
breakdowns.

Extracted from voice-fullduplex ``format_comparison_report`` and
extended with additional analysis (best-per-metric highlighting,
delta columns).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_results(result_files: list[Path]) -> list[dict[str, Any]]:
    """Load benchmark result JSON files.

    Args:
        result_files: List of paths to JSON result files.

    Returns:
        List of parsed result dicts. Files that fail to parse are
        silently skipped.
    """
    results: list[dict[str, Any]] = []
    for f in result_files:
        try:
            with open(f, "r", encoding="utf-8") as fp:
                results.append(json.load(fp))
        except (json.JSONDecodeError, OSError):
            continue
    return results


def format_comparison_markdown(result_files: list[Path]) -> str:
    """Generate a comparison table across benchmark runs.

    Produces a markdown report with:
    - Overall metrics table (Mean CER, Median CER, latency, RTF, hallucinations)
    - Per-category CER comparison table
    - Best model per metric indicated with bold formatting

    Args:
        result_files: Paths to benchmark result JSON files.

    Returns:
        Markdown-formatted comparison string.
    """
    results = load_results(result_files)
    if not results:
        return "## STT Benchmark Comparison\n\n*No valid result files found.*\n"

    lines: list[str] = []
    lines.append("## STT Benchmark Comparison")
    lines.append("")

    # ---------------------------------------------------------------
    # Overall metrics table
    # ---------------------------------------------------------------
    lines.append("| Model | Mean CER | Median CER | P50 Lat | P90 Lat | RTF | Halluc. | Date |")
    lines.append("|-------|--------:|----------:|--------:|--------:|----:|--------:|------|")

    # Collect aggregates for best-of detection
    entries: list[dict[str, Any]] = []

    # Use "task/model" label when multiple tasks are present
    tasks_seen = {data.get("task", "") for data in results}
    use_task_prefix = len(tasks_seen) > 1

    for data in results:
        agg = data.get("aggregate")
        if not agg:
            continue
        model_label = data.get("model", "?")
        if use_task_prefix:
            task_name = data.get("task", "")
            model_label = f"{model_label} ({task_name})"
        entries.append({
            "model": model_label,
            "timestamp": data.get("timestamp", "")[:10],
            "mean_cer": agg.get("mean_cer", 0),
            "median_cer": agg.get("median_cer", 0),
            "latency_p50": agg.get("latency_p50", 0),
            "latency_p90": agg.get("latency_p90", 0),
            "rtf_mean": agg.get("rtf_mean", 0),
            "total_hallucinations": agg.get("total_hallucinations", 0),
        })

    if not entries:
        lines.append("| -- | -- | -- | -- | -- | -- | -- | -- |")
        lines.append("")
        return "\n".join(lines)

    # Find best (minimum) for each metric
    best_mean_cer = min(e["mean_cer"] for e in entries)
    best_median_cer = min(e["median_cer"] for e in entries)
    best_p50 = min(e["latency_p50"] for e in entries)
    best_p90 = min(e["latency_p90"] for e in entries)
    best_rtf = min(e["rtf_mean"] for e in entries)
    best_halluc = min(e["total_hallucinations"] for e in entries)

    def _bold_if_best(value: float, best: float, fmt: str) -> str:
        """Format value, bolding if it equals the best."""
        formatted = format(value, fmt)
        if value == best and len(entries) > 1:
            return f"**{formatted}**"
        return formatted

    for e in entries:
        mean_cer = _bold_if_best(e["mean_cer"], best_mean_cer, ".1%")
        median_cer = _bold_if_best(e["median_cer"], best_median_cer, ".1%")
        p50 = _bold_if_best(e["latency_p50"], best_p50, ".2f") + "s"
        p90 = _bold_if_best(e["latency_p90"], best_p90, ".2f") + "s"
        rtf = _bold_if_best(e["rtf_mean"], best_rtf, ".3f")
        halluc = _bold_if_best(float(e["total_hallucinations"]), float(best_halluc), ".0f")

        lines.append(
            f"| {e['model']} | {mean_cer} | {median_cer} | "
            f"{p50} | {p90} | {rtf} | {halluc} | {e['timestamp']} |"
        )

    lines.append("")

    # ---------------------------------------------------------------
    # Per-category CER comparison
    # ---------------------------------------------------------------
    all_cats: set[str] = set()
    model_cats: dict[str, dict[str, float]] = {}

    for data in results:
        model_label = data.get("model", "?")
        if use_task_prefix:
            task_name = data.get("task", "")
            model_label = f"{model_label} ({task_name})"
        for cat, info in data.get("per_category", {}).items():
            all_cats.add(cat)
            model_cats.setdefault(model_label, {})[cat] = info.get("mean_cer", 0)

    if all_cats and model_cats:
        models = list(model_cats.keys())
        lines.append("### Per-Category CER Comparison")
        lines.append("")

        header = "| Category | " + " | ".join(models) + " |"
        sep = "|----------|" + "|".join(["--------:" for _ in models]) + "|"
        lines.append(header)
        lines.append(sep)

        for cat in sorted(all_cats):
            # Find best CER for this category
            cat_vals = [
                model_cats.get(m, {}).get(cat)
                for m in models
            ]
            valid_vals = [v for v in cat_vals if v is not None]
            best_cat = min(valid_vals) if valid_vals else None

            row = f"| {cat} |"
            for m in models:
                cer = model_cats.get(m, {}).get(cat)
                if cer is not None:
                    formatted = f"{cer:.1%}"
                    if best_cat is not None and cer == best_cat and len(valid_vals) > 1:
                        formatted = f"**{formatted}**"
                    row += f" {formatted} |"
                else:
                    row += " -- |"
            lines.append(row)

        lines.append("")

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    if len(entries) >= 2:
        sorted_by_median = sorted(entries, key=lambda e: e["median_cer"])
        best = sorted_by_median[0]
        lines.append(
            f"**Best overall (by median CER): {best['model']} "
            f"({best['median_cer']:.1%})**"
        )
        lines.append("")

    return "\n".join(lines)
