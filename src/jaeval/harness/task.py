"""YAML task loader with Pydantic validation.

A task definition is a YAML file describing which corpus to evaluate,
which metrics to compute, which categories to break down, and what
pass/warn/fail gate thresholds to apply.

Example YAML::

    task: stt_corpus_v2_clean
    type: stt
    version: "1.0"
    corpus:
      path: corpora/stt/corpus_v2
      ground_truth: ground_truth.json
      format: wav
    pipeline:
      codec: null
      vad: null
      energy_gate_rms: null
    metrics:
      - cer
      - hallucination_count
      - latency_p50
      - latency_p90
      - rtf
    categories:
      - keigo
      - proper_noun
      - number
    gates:
      median_cer:
        pass: 0.05
        warn: 0.08
      hallucinations:
        pass: 0
        warn: 2
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class CorpusConfig(BaseModel):
    """Corpus location and format."""

    path: str
    ground_truth: str = "ground_truth.json"
    format: str = "wav"


class PipelineConfig(BaseModel):
    """Optional pipeline pre-processing configuration."""

    codec: str | None = None
    vad: str | None = None
    energy_gate_rms: float | None = None


class GateThreshold(BaseModel):
    """Pass/warn thresholds for a single gate metric.

    The YAML key ``pass`` is a Python reserved word, so we accept it
    via the ``pass`` alias while storing it as ``pass_threshold``.
    """

    model_config = {"populate_by_name": True}

    pass_threshold: float = Field(alias="pass")
    warn: float


class TaskConfig(BaseModel):
    """Full task definition loaded from a YAML file."""

    task: str
    type: Literal["stt", "tts", "conversation", "duplex"]
    version: str = "1.0"
    corpus: CorpusConfig
    pipeline: PipelineConfig | None = None
    metrics: list[str] = Field(default_factory=lambda: ["cer"])
    categories: list[str] = Field(default_factory=list)
    gates: dict[str, GateThreshold] = Field(default_factory=dict)


def load_task(yaml_path: Path) -> TaskConfig:
    """Load and validate a task definition from a YAML file.

    Args:
        yaml_path: Path to the YAML task definition.

    Returns:
        A validated TaskConfig instance.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        pydantic.ValidationError: If the YAML content is invalid.
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Task file not found: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError(f"Empty YAML file: {yaml_path}")

    return TaskConfig(**data)
