"""HuggingFace Hub search for models and datasets."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)


@dataclass
class HFModelResult:
    """A single model from HuggingFace Hub search."""

    model_id: str
    author: str
    pipeline_tag: str
    downloads: int
    likes: int
    url: str

    @property
    def citation(self) -> str:
        return f"{self.model_id} ({self.downloads} downloads). {self.url}"


@dataclass
class HFDatasetResult:
    """A single dataset from HuggingFace Hub search."""

    dataset_id: str
    author: str
    downloads: int
    url: str

    @property
    def citation(self) -> str:
        return f"{self.dataset_id} ({self.downloads} downloads). {self.url}"


class HuggingFaceSource:
    """Search HuggingFace Hub for models and datasets."""

    MODELS_URL = "https://huggingface.co/api/models"
    DATASETS_URL = "https://huggingface.co/api/datasets"

    async def search_models(self, query: str, max_results: int = 10) -> list[HFModelResult]:
        """Search for models on HuggingFace Hub."""
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    self.MODELS_URL,
                    params={"search": query, "sort": "downloads", "limit": max_results},
                    timeout=15.0,
                )
                resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.error("HuggingFace model search HTTP error for %r: %s", query, exc)
            return []
        except httpx.RequestError as exc:
            logger.error("HuggingFace model search request failed for %r: %s", query, exc)
            return []

        results: list[HFModelResult] = []
        for item in resp.json()[:max_results]:
            model_id = item.get("modelId", "") or item.get("id", "")
            results.append(
                HFModelResult(
                    model_id=model_id,
                    author=item.get("author", ""),
                    pipeline_tag=item.get("pipeline_tag", ""),
                    downloads=item.get("downloads", 0),
                    likes=item.get("likes", 0),
                    url=f"https://huggingface.co/{model_id}",
                )
            )
        return results

    async def search_datasets(self, query: str, max_results: int = 10) -> list[HFDatasetResult]:
        """Search for datasets on HuggingFace Hub."""
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    self.DATASETS_URL,
                    params={"search": query, "sort": "downloads", "limit": max_results},
                    timeout=15.0,
                )
                resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.error("HuggingFace dataset search HTTP error for %r: %s", query, exc)
            return []
        except httpx.RequestError as exc:
            logger.error("HuggingFace dataset search request failed for %r: %s", query, exc)
            return []

        results: list[HFDatasetResult] = []
        for item in resp.json()[:max_results]:
            dataset_id = item.get("id", "")
            results.append(
                HFDatasetResult(
                    dataset_id=dataset_id,
                    author=item.get("author", ""),
                    downloads=item.get("downloads", 0),
                    url=f"https://huggingface.co/datasets/{dataset_id}",
                )
            )
        return results
