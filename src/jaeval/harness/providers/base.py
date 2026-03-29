"""Abstract base class for STT providers.

Every STT provider implements the ``STTProvider`` interface, which
provides a ``transcribe`` method that takes raw audio bytes and
returns a ``TranscribeResult`` with the transcript text and latency.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TranscribeResult:
    """Result of a single transcription call."""

    text: str
    latency_sec: float
    metadata: dict[str, Any] = field(default_factory=dict)


class STTProvider(ABC):
    """Abstract STT provider interface.

    Subclasses must set ``name`` and implement ``transcribe``.
    Optionally override ``setup`` (for API key loading, model init)
    and ``teardown`` (for resource cleanup).
    """

    name: str = "base"
    requires_gpu: bool = False

    @abstractmethod
    def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000) -> TranscribeResult:
        """Transcribe audio bytes to text.

        Args:
            audio_bytes: Raw audio file bytes (including WAV header).
            sample_rate: Sample rate of the audio in Hz.

        Returns:
            TranscribeResult with transcript text and latency.
        """
        ...

    def setup(self) -> None:
        """Initialize the provider (load API keys, models, etc.).

        Called once before the first transcription.  Override in
        subclasses that need initialization.
        """

    def teardown(self) -> None:
        """Release resources held by the provider.

        Called after all transcriptions are complete.  Override in
        subclasses that hold GPU memory or open connections.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
