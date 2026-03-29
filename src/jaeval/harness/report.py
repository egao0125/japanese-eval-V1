"""Markdown and JSON report formatters for benchmark results.

Produces human-readable summaries matching the output format of
voice-fullduplex ``format_markdown_summary``.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .runner import BenchmarkReport


def _serialize_report(report: BenchmarkReport) -> dict[str, Any]:
    """Convert a BenchmarkReport to a JSON-serializable dict."""
    data: dict[str, Any] = {
        "task": report.task,
        "model": report.model,
        "timestamp": report.timestamp,
        "total_entries": report.total_entries,
        "scored": report.scored,
        "skipped": report.skipped,
        "errors": report.errors,
        "gate_result": report.gate_result,
    }

    if report.aggregate is not None:
        data["aggregate"] = asdict(report.aggregate)
    else:
        data["aggregate"] = None

    data["per_category"] = report.per_category

    # Gate checks
    data["gate_checks"] = [asdict(gc) for gc in report.gate_checks]

    # Per-utterance results
    results = []
    for u in report.utterance_results:
        entry: dict[str, Any] = {
            "id": u.id,
            "category": u.category,
            "reference": u.reference,
            "hypothesis": u.hypothesis,
            "cer": u.cer.cer,
            "substitutions": u.cer.substitutions,
            "insertions": u.cer.insertions,
            "deletions": u.cer.deletions,
            "ref_len": u.cer.ref_len,
            "ref_normalized": u.cer.ref_normalized,
            "hyp_normalized": u.cer.hyp_normalized,
            "lenient_cer": u.lenient_cer.cer if u.lenient_cer else None,
            "hallucinated_kanji": u.hallucinated_kanji,
            "hallucinated_count": len(u.hallucinated_kanji) if u.hallucinated_kanji else 0,
            "latency_sec": round(u.latency_sec, 4),
            "audio_duration_sec": round(u.audio_duration_sec, 4),
            "rtf": round(u.rtf, 4),
            "skipped": u.skipped,
            "error": u.error,
        }
        results.append(entry)
    data["results"] = results

    return data


def format_markdown(report: BenchmarkReport) -> str:
    """Format a BenchmarkReport as a markdown summary.

    Includes:
    - Header with task/model/timestamp
    - Aggregate metrics table
    - Per-category breakdown table
    - Per-utterance detail table
    - Gate results
    - Overall verdict
    """
    lines: list[str] = []
    agg = report.aggregate

    # Header
    lines.append(f"## STT Benchmark: {report.model}")
    lines.append(f"- Task: {report.task}")
    lines.append(f"- Timestamp: {report.timestamp}")
    lines.append(
        f"- Utterances: {report.scored} scored, "
        f"{report.skipped} skipped, "
        f"{report.errors} errors"
    )
    lines.append("")

    if agg is None:
        lines.append("*No utterances scored.*")
        return "\n".join(lines)

    # Aggregate metrics
    lines.append("### Aggregate Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Mean CER | {agg.mean_cer:.1%} |")
    lines.append(f"| Median CER | {agg.median_cer:.1%} |")
    lines.append(f"| Min / Max CER | {agg.min_cer:.1%} / {agg.max_cer:.1%} |")
    lines.append(f"| Std CER | {agg.std_cer:.1%} |")
    if agg.mean_lenient_cer is not None:
        lines.append(f"| Mean Lenient CER | {agg.mean_lenient_cer:.1%} |")
        lines.append(f"| Median Lenient CER | {agg.median_lenient_cer:.1%} |")
    lines.append(f"| Hallucinations | {agg.total_hallucinations} |")
    lines.append(f"| Latency P50 | {agg.latency_p50:.2f}s |")
    lines.append(f"| Latency P90 | {agg.latency_p90:.2f}s |")
    lines.append(f"| Latency Mean | {agg.latency_mean:.2f}s |")
    lines.append(f"| RTF (mean) | {agg.rtf_mean:.3f} |")
    lines.append(f"| RTF (max) | {agg.rtf_max:.3f} |")
    lines.append("")

    # Per-category breakdown
    if report.per_category:
        lines.append("### Per-Category Breakdown")
        lines.append("")
        lines.append("| Category | Count | Mean CER | Halluc. | Mean Lat |")
        lines.append("|----------|------:|--------:|---------:|---------:|")
        for cat, info in sorted(report.per_category.items()):
            lines.append(
                f"| {cat} | {info['count']} | {info['mean_cer']:.1%} | "
                f"{info['hallucinations']} | {info['mean_latency']:.2f}s |"
            )
        lines.append("")

    # Per-utterance detail
    lines.append("### Per-Utterance Results")
    lines.append("")
    lines.append("| ID | Cat | CER | Hall | Lat | RTF | Status |")
    lines.append("|----|-----|----:|-----:|----:|----:|--------|")

    for u in report.utterance_results:
        if u.skipped:
            lines.append(
                f"| {u.id} | {u.category} | -- | -- | -- | -- | SKIP |"
            )
            continue
        if u.error:
            lines.append(
                f"| {u.id} | {u.category} | -- | -- | -- | -- | ERROR |"
            )
            continue

        cer = u.cer.cer
        if cer <= 0.15:
            status = "PASS"
        elif cer <= 0.30:
            status = "WARN"
        else:
            status = "FAIL"

        lines.append(
            f"| {u.id} | {u.category} | {cer:.1%} | "
            f"{len(u.hallucinated_kanji)} | {u.latency_sec:.2f}s | "
            f"{u.rtf:.2f} | {status} |"
        )

    lines.append("")

    # Gate results
    if report.gate_checks:
        lines.append("### Gate Results")
        lines.append("")
        lines.append("| Gate | Value | Pass | Warn | Result |")
        lines.append("|------|------:|-----:|-----:|--------|")
        for gc in report.gate_checks:
            lines.append(
                f"| {gc.metric} | {gc.value:.4f} | "
                f"{gc.pass_threshold:.4f} | {gc.warn_threshold:.4f} | "
                f"**{gc.result}** |"
            )
        lines.append("")
        lines.append(f"**Gate Verdict: {report.gate_result}**")
        lines.append("")

    # Overall verdict based on mean CER
    mean_cer = agg.mean_cer
    if mean_cer <= 0.03:
        verdict = "TARGET MET (CER <= 3%)"
    elif mean_cer <= 0.10:
        verdict = "EXCELLENT (CER <= 10%)"
    elif mean_cer <= 0.20:
        verdict = "ACCEPTABLE (CER <= 20%)"
    elif mean_cer <= 0.30:
        verdict = "MARGINAL (CER <= 30%)"
    else:
        verdict = "POOR (CER > 30%)"

    lines.append(f"**Verdict: {verdict}**")

    return "\n".join(lines)


def save_json(report: BenchmarkReport, output_path: Path) -> None:
    """Serialize a BenchmarkReport to a JSON file.

    Args:
        report: The benchmark report to save.
        output_path: Path to the output JSON file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = _serialize_report(report)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_markdown(report: BenchmarkReport, output_path: Path) -> None:
    """Save a BenchmarkReport as a markdown file.

    Args:
        report: The benchmark report to format.
        output_path: Path to the output markdown file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    md = format_markdown(report)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)
        f.write("\n")
