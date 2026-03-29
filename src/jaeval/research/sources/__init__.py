"""Research source adapters for ArXiv, GitHub, HuggingFace, and Web Search."""
from .arxiv import ArxivSource, PaperResult
from .github import GitHubSource, RepoResult
from .huggingface import HuggingFaceSource, HFModelResult, HFDatasetResult
from .web import WebSearchSource, WebResult

__all__ = [
    "ArxivSource",
    "PaperResult",
    "GitHubSource",
    "RepoResult",
    "HuggingFaceSource",
    "HFModelResult",
    "HFDatasetResult",
    "WebSearchSource",
    "WebResult",
]
