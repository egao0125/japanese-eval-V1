"""OpenAI gpt-4o-transcribe STT provider.

Wraps the OpenAI audio transcription API for Japanese transcription.
Extracted from voice-fullduplex ``scripts/eval/benchmark_stt.py``.
"""

from __future__ import annotations

import os
import time

import httpx

from .base import STTProvider, TranscribeResult


class OpenAISTTProvider(STTProvider):
    """OpenAI gpt-4o-transcribe STT provider (multipart form upload)."""

    name = "openai"
    requires_gpu = False

    def __init__(
        self,
        *,
        model: str = "gpt-4o-transcribe",
        language: str = "ja",
        prompt: str = "日本語のビジネス電話です。敬語が使われます。",
    ):
        self.model = model
        self.language = language
        self.prompt = prompt
        self._api_key: str = ""

    def setup(self) -> None:
        """Load the OpenAI API key from the environment."""
        self._api_key = os.environ.get("OPENAI_API_KEY", "")
        if not self._api_key:
            raise RuntimeError("OPENAI_API_KEY not set")

    def transcribe(self, audio_bytes: bytes, sample_rate: int = 8000) -> TranscribeResult:
        """Transcribe via OpenAI audio transcription API.

        Uses multipart form upload with the gpt-4o-transcribe model.

        Args:
            audio_bytes: WAV file bytes.
            sample_rate: Audio sample rate in Hz (not sent to API, WAV header
                is used by OpenAI).

        Returns:
            TranscribeResult with transcript and E2E latency.
        """
        if not self._api_key:
            self.setup()

        url = "https://api.openai.com/v1/audio/transcriptions"
        headers = {"Authorization": f"Bearer {self._api_key}"}

        t0 = time.monotonic()
        resp = httpx.post(
            url,
            headers=headers,
            data={
                "model": self.model,
                "language": self.language,
                "response_format": "json",
                "prompt": self.prompt,
            },
            files={"file": ("audio.wav", audio_bytes, "audio/wav")},
            timeout=60.0,
        )
        latency = time.monotonic() - t0

        resp.raise_for_status()
        transcript = resp.json().get("text", "")

        return TranscribeResult(
            text=transcript,
            latency_sec=latency,
            metadata={"model": self.model, "language": self.language},
        )
