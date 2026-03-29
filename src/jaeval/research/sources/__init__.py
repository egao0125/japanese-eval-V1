"""Research source adapters for ArXiv, GitHub, and HuggingFace."""
from .arxiv import ArxivSource, PaperResult
from .github import GitHubSource, RepoResult
from .huggingface import HuggingFaceSource, HFModelResult, HFDatasetResult

__all__ = [
    "ArxivSource",
    "PaperResult",
    "GitHubSource",
    "RepoResult",
    "HuggingFaceSource",
    "HFModelResult",
    "HFDatasetResult",
]
