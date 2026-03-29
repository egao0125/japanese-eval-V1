"""Tests for audio I/O utilities."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from jaeval.core.audio import decode_wav_to_float32, read_wav_info


def _make_wav(
    tmp_path: Path,
    *,
    sr: int = 16000,
    channels: int = 1,
    sample_width: int = 2,
    duration_sec: float = 0.1,
    freq: float = 440.0,
) -> Path:
    """Create a synthetic WAV file for testing."""
    n_frames = int(sr * duration_sec)
    t = np.linspace(0, duration_sec, n_frames, endpoint=False)
    tone = np.sin(2 * np.pi * freq * t)

    wav_path = tmp_path / f"test_{sr}_{channels}ch_{sample_width}b.wav"

    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sr)

        if sample_width == 2:
            samples = (tone * 32767).astype(np.int16)
            if channels == 2:
                # Interleave identical channels
                stereo = np.column_stack([samples, samples]).flatten()
                wf.writeframes(stereo.tobytes())
            else:
                wf.writeframes(samples.tobytes())
        elif sample_width == 1:
            samples = ((tone * 127) + 128).astype(np.uint8)
            if channels == 2:
                stereo = np.column_stack([samples, samples]).flatten()
                wf.writeframes(stereo.tobytes())
            else:
                wf.writeframes(samples.tobytes())
        else:
            raise ValueError(f"Unsupported sample_width for test: {sample_width}")

    return wav_path


class TestReadWavInfo:
    def test_basic_16bit_mono(self, tmp_path):
        wav_path = _make_wav(tmp_path, sr=16000, channels=1, sample_width=2, duration_sec=0.5)
        info = read_wav_info(wav_path)
        assert info.sample_rate == 16000
        assert info.channels == 1
        assert info.sample_width == 2
        assert info.n_frames == 8000
        assert abs(info.duration_sec - 0.5) < 0.001

    def test_stereo(self, tmp_path):
        wav_path = _make_wav(tmp_path, sr=44100, channels=2, sample_width=2, duration_sec=0.1)
        info = read_wav_info(wav_path)
        assert info.channels == 2
        assert info.sample_rate == 44100

    def test_8bit(self, tmp_path):
        wav_path = _make_wav(tmp_path, sr=8000, channels=1, sample_width=1, duration_sec=0.1)
        info = read_wav_info(wav_path)
        assert info.sample_width == 1
        assert info.sample_rate == 8000


class TestDecodeWavToFloat32:
    def test_16bit_mono_no_resample(self, tmp_path):
        wav_path = _make_wav(tmp_path, sr=16000, channels=1, sample_width=2, duration_sec=0.1)
        audio = decode_wav_to_float32(wav_path.read_bytes(), target_sr=16000)
        assert audio.dtype == np.float32
        assert len(audio) == 1600  # 0.1s * 16000
        assert audio.min() >= -1.0
        assert audio.max() <= 1.0

    def test_stereo_to_mono(self, tmp_path):
        wav_path = _make_wav(tmp_path, sr=16000, channels=2, sample_width=2, duration_sec=0.1)
        audio = decode_wav_to_float32(wav_path.read_bytes(), target_sr=16000)
        # Should be mono after averaging
        assert audio.ndim == 1
        assert len(audio) == 1600

    def test_resample_44100_to_16000(self, tmp_path):
        wav_path = _make_wav(tmp_path, sr=44100, channels=1, sample_width=2, duration_sec=0.1)
        audio = decode_wav_to_float32(wav_path.read_bytes(), target_sr=16000)
        # Resampled length should be approximately 0.1s * 16000 = 1600
        assert abs(len(audio) - 1600) < 10  # allow small rounding

    def test_resample_8000_to_16000(self, tmp_path):
        wav_path = _make_wav(tmp_path, sr=8000, channels=1, sample_width=2, duration_sec=0.1)
        audio = decode_wav_to_float32(wav_path.read_bytes(), target_sr=16000)
        assert abs(len(audio) - 1600) < 10

    def test_8bit_decoding(self, tmp_path):
        wav_path = _make_wav(tmp_path, sr=16000, channels=1, sample_width=1, duration_sec=0.1)
        audio = decode_wav_to_float32(wav_path.read_bytes(), target_sr=16000)
        assert audio.dtype == np.float32
        assert len(audio) == 1600
        # 8-bit centered around 128, range should be roughly [-1, 1]
        assert audio.min() >= -1.1
        assert audio.max() <= 1.1

    def test_non_silent_tone(self, tmp_path):
        """Verify that a 440Hz tone produces non-zero audio."""
        wav_path = _make_wav(tmp_path, sr=16000, channels=1, sample_width=2, freq=440.0)
        audio = decode_wav_to_float32(wav_path.read_bytes(), target_sr=16000)
        assert np.abs(audio).max() > 0.1  # Not silent
