"""Pipeline-realistic evaluator with codec degradation and energy gating.

Applies the same audio transformations that occur in a production
voice pipeline (G.711 mu-law codec, energy gate) before passing
audio to an STT provider. This ensures benchmark results reflect
real-world conditions rather than clean-audio performance.

Usage::

    from jaeval.harness.evaluators.pipeline_eval import PipelineEvaluator
    from jaeval.harness.task import PipelineConfig

    config = PipelineConfig(codec="g711_mulaw", energy_gate_rms=0.01)
    evaluator = PipelineEvaluator(config)

    processed = evaluator.process_audio(wav_bytes, sample_rate=16000)
    if processed:
        result = provider.transcribe(processed, sample_rate)
"""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass

from ...core.audio import decode_wav_to_float32


@dataclass
class PipelineStats:
    """Statistics from pipeline processing of a single utterance."""

    energy_rms: float | None = None
    energy_gate_passed: bool = True
    codec_applied: str | None = None


class PipelineEvaluator:
    """Applies pipeline conditions (codec, energy gate) before STT.

    Simulates the audio degradation that occurs in production telephony
    pipelines, so benchmarks reflect real-world STT performance.

    Args:
        pipeline_config: A PipelineConfig from the task YAML, or ``None``
            to skip all processing (clean audio pass-through).
    """

    def __init__(self, pipeline_config=None):
        self.config = pipeline_config

    def process_audio(
        self, audio_bytes: bytes, sample_rate: int
    ) -> tuple[bytes, PipelineStats]:
        """Apply pipeline transformations to audio.

        Args:
            audio_bytes: WAV file bytes (including header).
            sample_rate: Sample rate of the audio.

        Returns:
            Tuple of (processed_audio_bytes, stats). If the energy gate
            rejects the audio, returns (b"", stats) with
            ``stats.energy_gate_passed = False``.
        """
        stats = PipelineStats()

        if self.config is None:
            return audio_bytes, stats

        # Energy gate check (before codec, on original audio)
        if self.config.energy_gate_rms is not None:
            rms = self._compute_rms(audio_bytes, sample_rate)
            stats.energy_rms = rms
            if rms < self.config.energy_gate_rms:
                stats.energy_gate_passed = False
                return b"", stats

        # Codec degradation
        if self.config.codec == "g711_mulaw":
            audio_bytes = self._apply_g711_mulaw(audio_bytes, sample_rate)
            stats.codec_applied = "g711_mulaw"

        return audio_bytes, stats

    def _compute_rms(self, audio_bytes: bytes, sample_rate: int) -> float:
        """Compute RMS energy of audio.

        Args:
            audio_bytes: WAV file bytes.
            sample_rate: Sample rate in Hz.

        Returns:
            RMS energy as a float (0.0 to 1.0 for normalized audio).
        """
        import numpy as np

        audio = decode_wav_to_float32(audio_bytes, target_sr=sample_rate)
        if len(audio) == 0:
            return 0.0
        return float(np.sqrt(np.mean(audio ** 2)))

    def _apply_g711_mulaw(self, audio_bytes: bytes, sample_rate: int) -> bytes:
        """Simulate G.711 mu-law codec degradation.

        G.711 mu-law is the standard telephony codec. It compresses
        16-bit linear PCM to 8-bit mu-law and back, introducing
        quantization noise typical of telephone audio.

        The audio is:
        1. Decoded from WAV to float32
        2. Converted to 16-bit linear PCM
        3. Compressed to 8-bit mu-law
        4. Expanded back to 16-bit linear PCM
        5. Re-encoded as WAV bytes

        If the source sample rate is not 8000 Hz, the audio is first
        resampled to 8000 Hz (standard telephony rate), processed,
        then resampled back.

        Args:
            audio_bytes: WAV file bytes.
            sample_rate: Original sample rate.

        Returns:
            WAV file bytes with G.711 mu-law degradation applied.
        """
        import numpy as np

        audio = decode_wav_to_float32(audio_bytes, target_sr=sample_rate)
        original_sr = sample_rate

        # Resample to 8000 Hz for telephony codec simulation
        if sample_rate != 8000:
            from scipy.signal import resample_poly
            from math import gcd

            down = sample_rate
            up = 8000
            g = gcd(up, down)
            audio = resample_poly(audio, up // g, down // g).astype(np.float32)
            sample_rate = 8000

        # Float32 [-1, 1] -> int16
        audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)

        # Encode to mu-law
        mulaw = np.array(
            [_linear_to_mulaw(s) for s in audio_int16], dtype=np.uint8
        )

        # Decode from mu-law back to int16
        decoded_int16 = np.array(
            [_mulaw_to_linear(m) for m in mulaw], dtype=np.int16
        )

        # int16 -> float32
        decoded_float = decoded_int16.astype(np.float32) / 32768.0

        # Resample back to original rate if needed
        if sample_rate != original_sr:
            from scipy.signal import resample_poly
            from math import gcd

            down = sample_rate
            up = original_sr
            g = gcd(up, down)
            decoded_float = resample_poly(decoded_float, up // g, down // g).astype(np.float32)
            sample_rate = original_sr

        # Re-encode as WAV bytes
        return _float32_to_wav_bytes(decoded_float, sample_rate)

    @staticmethod
    def _passes_energy_gate(audio_bytes: bytes, sample_rate: int, threshold: float) -> bool:
        """Check if audio RMS exceeds energy gate threshold.

        Args:
            audio_bytes: WAV file bytes.
            sample_rate: Sample rate in Hz.
            threshold: Minimum RMS energy to pass.

        Returns:
            True if the audio RMS exceeds the threshold.
        """
        import numpy as np

        audio = decode_wav_to_float32(audio_bytes, target_sr=sample_rate)
        if len(audio) == 0:
            return False
        rms = float(np.sqrt(np.mean(audio ** 2)))
        return rms >= threshold


# ---------------------------------------------------------------------------
# G.711 mu-law encoding/decoding (ITU-T G.711)
# ---------------------------------------------------------------------------

_MULAW_BIAS = 0x84
_MULAW_CLIP = 32635
_MULAW_MAX = 0x1FFF

# Segment encoding table for mu-law
_MULAW_SEG_END = [0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF, 0x3FFF, 0x7FFF]


def _linear_to_mulaw(sample: int) -> int:
    """Convert a 16-bit linear PCM sample to 8-bit mu-law.

    Follows ITU-T G.711 specification.

    Args:
        sample: Signed 16-bit integer (-32768 to 32767).

    Returns:
        Unsigned 8-bit mu-law encoded value.
    """
    # Get sign bit
    sign = (sample >> 8) & 0x80
    if sign:
        sample = -sample
    if sample > _MULAW_CLIP:
        sample = _MULAW_CLIP

    sample = sample + _MULAW_BIAS

    # Find segment
    seg = 0
    for i in range(8):
        if sample <= _MULAW_SEG_END[i]:
            seg = i
            break
    else:
        seg = 7

    # Combine sign, segment, and quantization bits
    if seg >= 8:
        return int(sign | 0x7F) ^ 0xFF
    uval = int(sign | (seg << 4) | ((sample >> (seg + 3)) & 0x0F))
    return uval ^ 0xFF


def _mulaw_to_linear(mulaw_byte: int) -> int:
    """Convert an 8-bit mu-law sample to 16-bit linear PCM.

    Follows ITU-T G.711 specification.

    Args:
        mulaw_byte: Unsigned 8-bit mu-law value.

    Returns:
        Signed 16-bit integer.
    """
    # Ensure we work with Python int to avoid numpy scalar overflow
    mulaw_byte = int(mulaw_byte)
    mulaw_byte = ~mulaw_byte & 0xFF
    sign = mulaw_byte & 0x80
    seg = (mulaw_byte & 0x70) >> 4
    val = mulaw_byte & 0x0F

    # Reconstruct the linear value
    linear = ((val << 3) + _MULAW_BIAS) << seg
    linear -= _MULAW_BIAS

    if sign:
        linear = -linear

    # Clamp to int16 range
    return max(-32768, min(32767, linear))


def _float32_to_wav_bytes(audio, sample_rate: int) -> bytes:
    """Encode a float32 numpy array as WAV file bytes.

    Args:
        audio: 1-D numpy float32 array of audio samples.
        sample_rate: Sample rate in Hz.

    Returns:
        WAV file bytes (16-bit PCM, mono).
    """
    import numpy as np

    # float32 [-1, 1] -> int16
    audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())

    return buf.getvalue()
