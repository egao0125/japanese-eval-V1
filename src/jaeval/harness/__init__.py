"""Evaluation harness: YAML task loading, benchmark runner, gate evaluation, and reporting."""

from .compare import format_comparison_markdown, load_results
from .evaluators import PipelineEvaluator, PipelineStats
from .gate import GateCheck, evaluate_gate, evaluate_gates
from .providers import STTProvider, TranscribeResult, get_provider
from .report import format_markdown, save_json, save_markdown
from .runner import AggregateMetrics, BenchmarkReport, BenchmarkRunner, UtteranceResult
from .task import TaskConfig, load_task

__all__ = [
    "AggregateMetrics",
    "BenchmarkReport",
    "BenchmarkRunner",
    "GateCheck",
    "PipelineEvaluator",
    "PipelineStats",
    "STTProvider",
    "TaskConfig",
    "TranscribeResult",
    "UtteranceResult",
    "evaluate_gate",
    "evaluate_gates",
    "format_comparison_markdown",
    "format_markdown",
    "get_provider",
    "load_results",
    "load_task",
    "save_json",
    "save_markdown",
]
