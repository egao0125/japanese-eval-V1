"""Audio I/O utilities for evaluation."""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path


@dataclass
class WavInfo:
    """Metadata for a WAV file."""

    sample_rate: int
    channels: int
    sample_width: int
    n_frames: int
    duration_sec: float


def read_wav_info(wav_path: Path) -> WavInfo:
    """Read metadata from a WAV file without loading audio data.

    Args:
        wav_path: Path to the WAV file.

    Returns:
        WavInfo dataclass with sample rate, channels, sample width,
        frame count, and duration.
    """
    with wave.open(str(wav_path), "rb") as wf:
        n_frames = wf.getnframes()
        sr = wf.getframerate()
        return WavInfo(
            sample_rate=sr,
            channels=wf.getnchannels(),
            sample_width=wf.getsampwidth(),
            n_frames=n_frames,
            duration_sec=n_frames / sr,
        )


def read_wav_bytes(wav_path: Path) -> tuple[bytes, int]:
    """Read a WAV file and return (raw bytes, sample_rate).

    Args:
        wav_path: Path to the WAV file.

    Returns:
        Tuple of (file bytes including WAV header, sample rate in Hz).
    """
    info = read_wav_info(wav_path)
    return wav_path.read_bytes(), info.sample_rate


def decode_wav_to_float32(audio_bytes: bytes, target_sr: int = 16000):
    """Decode WAV bytes to a float32 numpy array, resampling if needed.

    Used by GPU-based providers (Whisper, Qwen3-ASR) that need raw
    float32 audio arrays rather than WAV file bytes.

    Args:
        audio_bytes: Raw WAV file bytes (including header).
        target_sr: Target sample rate in Hz. If the source sample rate
            differs, the audio will be resampled using librosa.

    Returns:
        A 1-D numpy float32 array of audio samples at ``target_sr``.
    """
    import io

    import soundfile as sf

    audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")

    # Convert stereo to mono by averaging channels
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resample if source rate differs from target
    if sr != target_sr:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    return audio
