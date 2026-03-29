"""Auto-research pipeline: Plan -> Search -> Read -> Synthesize."""
from .orchestrator import ResearchOrchestrator, ResearchConfig, run_research

__all__ = ["ResearchOrchestrator", "ResearchConfig", "run_research"]
