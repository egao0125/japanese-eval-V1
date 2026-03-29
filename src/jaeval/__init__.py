"""Japanese Evaluation Harness — auto-research + benchmarking for Japanese voice AI."""
__version__ = "0.1.0"

from .integration import evaluate_call, CallEvalResult

__all__ = ["evaluate_call", "CallEvalResult"]
