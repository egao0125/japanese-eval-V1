"""Deepgram Nova STT provider.

Wraps the Deepgram REST API for Japanese transcription.
Extracted from voice-fullduplex ``scripts/eval/benchmark_stt.py``.
"""

from __future__ import annotations

import os
import time

import httpx

from .base import STTProvider, TranscribeResult


class DeepgramProvider(STTProvider):
    """Deepgram Nova-3 (or configurable model) STT provider."""

    name = "deepgram"
    requires_gpu = False

    def __init__(self, *, model: str = "nova-3", language: str = "ja"):
        self.model = model
        self.language = language
        self._api_key: str = ""

    def setup(self) -> None:
        """Load the Deepgram API key from the environment."""
        self._api_key = os.environ.get("DEEPGRAM_API_KEY", "")
        if not self._api_key:
            raise RuntimeError("DEEPGRAM_API_KEY not set")

    def transcribe(self, audio_bytes: bytes, sample_rate: int = 8000) -> TranscribeResult:
        """Transcribe via Deepgram REST API.

        Args:
            audio_bytes: WAV file bytes.
            sample_rate: Audio sample rate in Hz (default 8000 for telephony).

        Returns:
            TranscribeResult with transcript and E2E latency.
        """
        if not self._api_key:
            self.setup()

        url = "https://api.deepgram.com/v1/listen"
        params = {
            "model": self.model,
            "language": self.language,
            "punctuate": "true",
            "smart_format": "true",
            "numerals": "true",
            "sample_rate": str(sample_rate),
            "encoding": "linear16",
            "channels": "1",
        }
        headers = {
            "Authorization": f"Token {self._api_key}",
            "Content-Type": "audio/wav",
        }

        t0 = time.monotonic()
        resp = httpx.post(
            url, params=params, headers=headers, content=audio_bytes, timeout=30.0
        )
        latency = time.monotonic() - t0

        resp.raise_for_status()
        data = resp.json()

        transcript = ""
        alternatives = (
            data.get("results", {}).get("channels", [{}])[0].get("alternatives", [])
        )
        if alternatives:
            transcript = alternatives[0].get("transcript", "")

        return TranscribeResult(
            text=transcript,
            latency_sec=latency,
            metadata={"model": self.model, "language": self.language},
        )
