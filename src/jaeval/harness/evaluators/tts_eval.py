"""TTS quality evaluation using NISQA for automated MOS prediction.

Provides a thin wrapper around NISQA (Non-Intrusive Speech Quality Assessment)
for evaluating text-to-speech output quality without reference audio.

Usage:
    from jaeval.harness.evaluators.tts_eval import evaluate_tts_quality

    result = evaluate_tts_quality(Path("output.wav"))
    print(result.mos)  # 1.0 - 5.0
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TTSQualityResult:
    """NISQA quality prediction for a single audio file."""

    mos: float  # Mean Opinion Score (1-5)
    noisiness: float
    coloration: float
    discontinuity: float
    loudness: float

    def to_dict(self) -> dict[str, float]:
        return {
            "mos": round(self.mos, 3),
            "noisiness": round(self.noisiness, 3),
            "coloration": round(self.coloration, 3),
            "discontinuity": round(self.discontinuity, 3),
            "loudness": round(self.loudness, 3),
        }


def evaluate_tts_quality(audio_path: Path) -> TTSQualityResult:
    """Evaluate TTS audio quality using NISQA.

    Args:
        audio_path: Path to a WAV audio file.

    Returns:
        :class:`TTSQualityResult` with MOS and sub-dimension scores.

    Raises:
        ImportError: If NISQA is not installed.
        NotImplementedError: NISQA integration is pending.
        FileNotFoundError: If audio_path does not exist.
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    try:
        import nisqa  # noqa: F401
    except ImportError:
        raise ImportError(
            "NISQA not installed. Run: pip install nisqa\n"
            "See: https://github.com/gabrielmittag/NISQA"
        )

    # TODO: NISQA integration — load model, run inference, return scores
    raise NotImplementedError(
        "NISQA integration pending. The nisqa package is installed but "
        "the inference wrapper has not been implemented yet."
    )


def evaluate_tts_batch(audio_paths: list[Path]) -> list[TTSQualityResult]:
    """Evaluate multiple TTS audio files.

    Args:
        audio_paths: List of paths to WAV audio files.

    Returns:
        List of :class:`TTSQualityResult`, one per input file.
    """
    return [evaluate_tts_quality(p) for p in audio_paths]
