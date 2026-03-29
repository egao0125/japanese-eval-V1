"""Tests for quality gate evaluation."""

from jaeval.harness.gate import evaluate_gate, evaluate_gates


class TestGate:
    def test_pass(self):
        check = evaluate_gate("median_cer", 0.03, 0.05, 0.08)
        assert check.result == "PASS"

    def test_warn(self):
        check = evaluate_gate("median_cer", 0.06, 0.05, 0.08)
        assert check.result == "WARN"

    def test_fail(self):
        check = evaluate_gate("median_cer", 0.10, 0.05, 0.08)
        assert check.result == "FAIL"

    def test_evaluate_gates_all_pass(self):
        metrics = {"median_cer": 0.03, "hallucinations": 0}
        gates = {
            "median_cer": {"pass": 0.05, "warn": 0.08},
            "hallucinations": {"pass": 0, "warn": 2},
        }
        overall, checks = evaluate_gates(metrics, gates)
        assert overall == "PASS"
        assert len(checks) == 2

    def test_evaluate_gates_one_fail(self):
        metrics = {"median_cer": 0.15}
        gates = {"median_cer": {"pass": 0.05, "warn": 0.08}}
        overall, checks = evaluate_gates(metrics, gates)
        assert overall == "FAIL"
