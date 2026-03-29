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
            differs, the audio will be resampled.

    Returns:
        A 1-D numpy float32 array of audio samples at ``target_sr``.
    """
    import io

    import numpy as np

    with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())

    # Decode raw PCM to numpy array
    if sample_width == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif sample_width == 1:
        audio = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    # Convert stereo to mono by averaging channels
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)

    # Resample if source rate differs from target
    if sr != target_sr:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(target_sr, sr)
        audio = resample_poly(audio, target_sr // g, sr // g).astype(np.float32)

    return audio
