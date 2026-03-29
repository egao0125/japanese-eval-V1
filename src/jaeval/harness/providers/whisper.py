"""Whisper STT provider (kotoba-whisper and whisper-large-v3).

Wraps faster-whisper for local GPU-based Japanese transcription.
Supports any model compatible with the faster-whisper CTranslate2 format.

Extracted from voice-fullduplex ``scripts/eval/benchmark_stt.py``.
"""

from __future__ import annotations

import time

from .base import STTProvider, TranscribeResult


class WhisperProvider(STTProvider):
    """Local Whisper provider via faster-whisper.

    Supports any CTranslate2-compatible Whisper model:
    - ``kotoba-tech/kotoba-whisper-v2.0-faster`` (default, Japanese-optimized)
    - ``large-v3`` (OpenAI Whisper large-v3)

    Args:
        model_id: HuggingFace model ID or CTranslate2 model path.
        language: Language code for forced decoding (default ``ja``).
        beam_size: Beam search width (default 5).
    """

    name = "whisper"
    requires_gpu = True

    def __init__(
        self,
        *,
        model_id: str = "kotoba-tech/kotoba-whisper-v2.0-faster",
        language: str = "ja",
        beam_size: int = 5,
    ):
        self.model_id = model_id
        self.language = language
        self.beam_size = beam_size
        self._model = None

    def setup(self) -> None:
        """Load the faster-whisper model onto GPU."""
        from faster_whisper import WhisperModel

        self._model = WhisperModel(
            self.model_id,
            device="cuda",
            compute_type="float16",
        )

    def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000) -> TranscribeResult:
        """Transcribe audio using faster-whisper.

        Args:
            audio_bytes: WAV file bytes (including header).
            sample_rate: Sample rate of the audio in Hz.

        Returns:
            TranscribeResult with transcript and inference latency.
        """
        if self._model is None:
            self.setup()

        from jaeval.core.audio import decode_wav_to_float32

        audio = decode_wav_to_float32(audio_bytes, target_sr=16000)

        t0 = time.monotonic()
        segments, _ = self._model.transcribe(
            audio,
            language=self.language,
            beam_size=self.beam_size,
        )
        transcript = "".join(seg.text for seg in segments)
        latency = time.monotonic() - t0

        return TranscribeResult(
            text=transcript,
            latency_sec=latency,
            metadata={"model_id": self.model_id, "language": self.language},
        )

    def teardown(self) -> None:
        """Release the model from GPU memory."""
        self._model = None
