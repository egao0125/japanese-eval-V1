"""Quality gate evaluation for benchmark results.

Gates define pass/warn/fail thresholds for metrics. A benchmark task
specifies gates in its YAML definition, and after the run completes,
each metric is checked against its gate thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GateCheck:
    """Result of evaluating a single metric against its gate thresholds."""

    metric: str
    value: float
    pass_threshold: float
    warn_threshold: float
    result: str  # "PASS", "WARN", or "FAIL"


def evaluate_gate(
    metric: str,
    value: float,
    pass_threshold: float,
    warn_threshold: float,
) -> GateCheck:
    """Evaluate a single metric value against pass/warn thresholds.

    Logic:
    - value <= pass_threshold  -> PASS
    - value <= warn_threshold  -> WARN
    - value > warn_threshold   -> FAIL

    Args:
        metric: Name of the metric being checked.
        value: Observed metric value.
        pass_threshold: Maximum value for PASS.
        warn_threshold: Maximum value for WARN (above this is FAIL).

    Returns:
        GateCheck with the evaluation result.
    """
    if value <= pass_threshold:
        result = "PASS"
    elif value <= warn_threshold:
        result = "WARN"
    else:
        result = "FAIL"

    return GateCheck(
        metric=metric,
        value=value,
        pass_threshold=pass_threshold,
        warn_threshold=warn_threshold,
        result=result,
    )


def evaluate_gates(
    metrics: dict[str, float],
    gates: dict[str, dict[str, float]],
) -> tuple[str, list[GateCheck]]:
    """Evaluate all metrics against their corresponding gates.

    Args:
        metrics: Dict of metric_name -> observed_value.
        gates: Dict of metric_name -> {"pass": threshold, "warn": threshold}.

    Returns:
        Tuple of (overall_result, list_of_gate_checks).
        overall_result is "PASS" only if all checks pass, "FAIL" if any fail,
        "WARN" if at least one warns but none fail.
    """
    checks: list[GateCheck] = []

    for metric_name, gate_def in gates.items():
        if metric_name not in metrics:
            continue

        value = metrics[metric_name]
        pass_thresh = gate_def.get("pass", 0.0)
        warn_thresh = gate_def.get("warn", pass_thresh)

        check = evaluate_gate(metric_name, value, pass_thresh, warn_thresh)
        checks.append(check)

    # Determine overall result
    if not checks:
        return "PASS", checks

    results = [c.result for c in checks]
    if "FAIL" in results:
        overall = "FAIL"
    elif "WARN" in results:
        overall = "WARN"
    else:
        overall = "PASS"

    return overall, checks
