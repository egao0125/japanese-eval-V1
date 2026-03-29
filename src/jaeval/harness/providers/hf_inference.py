"""HuggingFace Inference API provider for zero-shot benchmarking.

Sends audio to the HuggingFace serverless Inference API, enabling
benchmarking of any model hosted on HF without local GPU.

Requires HF_TOKEN environment variable.
"""

from __future__ import annotations

import os
import time

from .base import STTProvider, TranscribeResult


class HFInferenceProvider(STTProvider):
    """HuggingFace Inference API STT provider.

    Supports any model on HF Hub with the automatic-speech-recognition
    pipeline, including Whisper variants, Kotoba-Whisper, etc.

    Args:
        model_id: HuggingFace model ID (e.g. ``openai/whisper-large-v3-turbo``).
        language: Language hint for models that support it.
    """

    name = "hf-inference"
    requires_gpu = False

    def __init__(
        self,
        *,
        model_id: str = "openai/whisper-large-v3-turbo",
        language: str = "ja",
    ):
        self.model_id = model_id
        self.language = language
        self._token: str | None = None

    def setup(self) -> None:
        self._token = os.environ.get("HF_TOKEN")
        if not self._token:
            raise RuntimeError("HF_TOKEN environment variable required for HF Inference API")

    def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000) -> TranscribeResult:
        if self._token is None:
            self.setup()

        import httpx

        url = f"https://api-inference.huggingface.co/models/{self.model_id}"
        headers = {"Authorization": f"Bearer {self._token}"}

        t0 = time.monotonic()
        resp = httpx.post(
            url,
            content=audio_bytes,
            headers=headers,
            timeout=120.0,
        )
        latency = time.monotonic() - t0

        if resp.status_code == 503:
            # Model is loading — retry once after waiting
            import json
            wait_time = json.loads(resp.text).get("estimated_time", 30)
            import time as _time
            _time.sleep(min(wait_time, 60))
            t0 = time.monotonic()
            resp = httpx.post(url, content=audio_bytes, headers=headers, timeout=120.0)
            latency = time.monotonic() - t0

        resp.raise_for_status()
        result = resp.json()

        text = result.get("text", "")

        return TranscribeResult(
            text=text,
            latency_sec=latency,
            metadata={"model_id": self.model_id, "provider": "hf-inference"},
        )
