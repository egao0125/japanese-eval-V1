"""Benchmark runner that orchestrates an evaluation run.

The runner:
1. Loads a TaskConfig (YAML task definition)
2. Loads the corpus (ground_truth.json)
3. Initializes the STT provider
4. Runs transcription on each utterance
5. Computes CER, hallucination count, latency, RTF
6. Evaluates pass/warn/fail gates
7. Returns a BenchmarkReport
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..core.audio import WavInfo, read_wav_info
from ..core.hallucination import detect_hallucinated_kanji
from ..core.metrics import CERResult, compute_cer
from .gate import GateCheck, evaluate_gates
from .providers.base import STTProvider
from .task import TaskConfig


@dataclass
class UtteranceResult:
    """Result of evaluating a single utterance."""

    id: str
    category: str
    reference: str
    hypothesis: str
    cer: CERResult
    hallucinated_kanji: list[str]
    latency_sec: float
    audio_duration_sec: float
    rtf: float
    skipped: bool = False
    error: str | None = None


@dataclass
class AggregateMetrics:
    """Aggregate metrics across all scored utterances."""

    mean_cer: float
    median_cer: float
    min_cer: float
    max_cer: float
    std_cer: float
    total_hallucinations: int
    latency_p50: float
    latency_p90: float
    latency_mean: float
    rtf_mean: float
    rtf_max: float


@dataclass
class BenchmarkReport:
    """Complete benchmark report for a single model on a single task."""

    task: str
    model: str
    timestamp: str
    total_entries: int
    scored: int
    skipped: int
    errors: int
    aggregate: AggregateMetrics | None
    per_category: dict[str, dict[str, Any]]
    utterance_results: list[UtteranceResult]
    gate_result: str
    gate_checks: list[GateCheck]


# ---------------------------------------------------------------------------
# Sentinel CERResult for skip/error cases (all zeros)
# ---------------------------------------------------------------------------

_EMPTY_CER = CERResult(
    cer=0.0,
    substitutions=0,
    insertions=0,
    deletions=0,
    ref_len=0,
    ref_normalized="",
    hyp_normalized="",
)


class BenchmarkRunner:
    """Orchestrates a benchmark evaluation run.

    Args:
        task: Parsed TaskConfig from a YAML task definition.
        provider: An STTProvider instance to evaluate.
        corpus_base: Base directory for resolving corpus paths.
            Defaults to the current working directory.
    """

    def __init__(
        self,
        task: TaskConfig,
        provider: STTProvider,
        *,
        corpus_base: Path | None = None,
    ):
        self.task = task
        self.provider = provider
        self.corpus_base = corpus_base or Path(".")

    def _load_corpus(self) -> list[dict[str, str]]:
        """Load ground truth entries from the corpus directory."""
        corpus_path = self.corpus_base / self.task.corpus.path
        gt_file = corpus_path / self.task.corpus.ground_truth

        if not gt_file.exists():
            raise FileNotFoundError(f"Ground truth not found: {gt_file}")

        with open(gt_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def run(self, *, limit: int = 0, verbose: bool = True) -> BenchmarkReport:
        """Execute the benchmark.

        Args:
            limit: Maximum number of utterances to evaluate.
                0 means evaluate all.
            verbose: Print per-utterance progress to stdout.

        Returns:
            A BenchmarkReport with all results and gate evaluations.
        """
        ground_truth = self._load_corpus()
        if limit > 0:
            ground_truth = ground_truth[:limit]

        corpus_dir = self.corpus_base / self.task.corpus.path
        self.provider.setup()

        utterance_results: list[UtteranceResult] = []

        for entry in ground_truth:
            entry_id = entry["id"]
            ref_text = entry["text"]
            category = entry.get("category", "unknown")
            wav_name = entry["wav"]
            wav_path = corpus_dir / wav_name

            # Skip missing WAV files
            if not wav_path.exists():
                utterance_results.append(
                    UtteranceResult(
                        id=entry_id,
                        category=category,
                        reference=ref_text,
                        hypothesis="",
                        cer=_EMPTY_CER,
                        hallucinated_kanji=[],
                        latency_sec=0.0,
                        audio_duration_sec=0.0,
                        rtf=0.0,
                        skipped=True,
                        error=f"WAV not found: {wav_path}",
                    )
                )
                if verbose:
                    print(f"  [{entry_id}] SKIP -- {wav_name} not found")
                continue

            # Transcribe
            try:
                wav_info: WavInfo = read_wav_info(wav_path)
                audio_bytes = wav_path.read_bytes()
                result = self.provider.transcribe(audio_bytes, wav_info.sample_rate)
            except Exception as e:
                utterance_results.append(
                    UtteranceResult(
                        id=entry_id,
                        category=category,
                        reference=ref_text,
                        hypothesis="",
                        cer=_EMPTY_CER,
                        hallucinated_kanji=[],
                        latency_sec=0.0,
                        audio_duration_sec=0.0,
                        rtf=0.0,
                        skipped=False,
                        error=str(e),
                    )
                )
                if verbose:
                    print(f"  [{entry_id}] ERROR -- {e}")
                continue

            # Compute metrics
            cer_result = compute_cer(ref_text, result.text)
            hallucinated = detect_hallucinated_kanji(ref_text, result.text)
            audio_duration = wav_info.duration_sec
            rtf = result.latency_sec / audio_duration if audio_duration > 0 else 0.0

            ur = UtteranceResult(
                id=entry_id,
                category=category,
                reference=ref_text,
                hypothesis=result.text,
                cer=cer_result,
                hallucinated_kanji=hallucinated,
                latency_sec=result.latency_sec,
                audio_duration_sec=audio_duration,
                rtf=rtf,
            )
            utterance_results.append(ur)

            if verbose:
                print(
                    f"  [{entry_id}] CER={cer_result.cer:.1%}  "
                    f"hall={len(hallucinated)}  "
                    f"lat={result.latency_sec:.2f}s  "
                    f"rtf={rtf:.2f}"
                )

        self.provider.teardown()
        return self._build_report(ground_truth, utterance_results)

    def _build_report(
        self,
        ground_truth: list[dict[str, str]],
        utterance_results: list[UtteranceResult],
    ) -> BenchmarkReport:
        """Aggregate results and evaluate gates."""
        scored = [u for u in utterance_results if not u.skipped and u.error is None]

        aggregate: AggregateMetrics | None = None
        per_category: dict[str, dict[str, Any]] = {}

        if scored:
            cers = [u.cer.cer for u in scored]
            latencies = sorted(u.latency_sec for u in scored)
            rtfs = [u.rtf for u in scored]

            p50_idx = int(len(latencies) * 0.5)
            p90_idx = min(int(len(latencies) * 0.9), len(latencies) - 1)

            aggregate = AggregateMetrics(
                mean_cer=round(statistics.mean(cers), 4),
                median_cer=round(statistics.median(cers), 4),
                min_cer=round(min(cers), 4),
                max_cer=round(max(cers), 4),
                std_cer=round(statistics.stdev(cers), 4) if len(cers) > 1 else 0.0,
                total_hallucinations=sum(len(u.hallucinated_kanji) for u in scored),
                latency_p50=round(latencies[p50_idx], 4),
                latency_p90=round(latencies[p90_idx], 4),
                latency_mean=round(statistics.mean(latencies), 4),
                rtf_mean=round(statistics.mean(rtfs), 4),
                rtf_max=round(max(rtfs), 4),
            )

            # Per-category breakdown
            cats: dict[str, list[UtteranceResult]] = {}
            for u in scored:
                cats.setdefault(u.category, []).append(u)

            for cat, items in sorted(cats.items()):
                cat_cers = [u.cer.cer for u in items]
                cat_lats = [u.latency_sec for u in items]
                per_category[cat] = {
                    "count": len(items),
                    "mean_cer": round(statistics.mean(cat_cers), 4),
                    "median_cer": round(statistics.median(cat_cers), 4),
                    "hallucinations": sum(len(u.hallucinated_kanji) for u in items),
                    "mean_latency": round(statistics.mean(cat_lats), 4),
                }

        # Gate evaluation
        gate_metrics: dict[str, float] = {}
        if aggregate:
            gate_metrics["median_cer"] = aggregate.median_cer
            gate_metrics["mean_cer"] = aggregate.mean_cer
            gate_metrics["hallucinations"] = float(aggregate.total_hallucinations)
            gate_metrics["latency_p50"] = aggregate.latency_p50
            gate_metrics["latency_p90"] = aggregate.latency_p90
            gate_metrics["rtf"] = aggregate.rtf_mean

            # Per-category CER gates (e.g. "keigo_cer")
            for cat, info in per_category.items():
                gate_metrics[f"{cat}_cer"] = info["mean_cer"]

        gates_raw: dict[str, dict[str, float]] = {}
        if self.task.gates:
            gates_raw = {
                k: {"pass": v.pass_threshold, "warn": v.warn}
                for k, v in self.task.gates.items()
            }

        gate_result, gate_checks = evaluate_gates(gate_metrics, gates_raw)

        return BenchmarkReport(
            task=self.task.task,
            model=self.provider.name,
            timestamp=datetime.now(timezone.utc).isoformat(),
            total_entries=len(ground_truth),
            scored=len(scored),
            skipped=sum(1 for u in utterance_results if u.skipped),
            errors=sum(1 for u in utterance_results if u.error and not u.skipped),
            aggregate=aggregate,
            per_category=per_category,
            utterance_results=utterance_results,
            gate_result=gate_result,
            gate_checks=gate_checks,
        )
