"""Tests for pipeline processing integration in BenchmarkRunner.

Verifies that:
- Clean tasks skip pipeline processing (no codec, no gate)
- Pipeline tasks apply G.711 codec degradation
- Energy gate rejects silent audio
- Energy gate passes normal audio
"""

import json
import struct
import wave
from pathlib import Path

import numpy as np

from jaeval.harness.providers.base import STTProvider, TranscribeResult
from jaeval.harness.task import TaskConfig, CorpusConfig, PipelineConfig, GateThreshold
from jaeval.harness.runner import BenchmarkRunner
from jaeval.harness.evaluators.pipeline_eval import PipelineEvaluator


class MockProvider(STTProvider):
    """Records audio received and returns fixed transcript."""

    name = "mock"

    def __init__(self, transcript: str = "テスト"):
        self.transcript = transcript
        self.received_audio: list[bytes] = []

    def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000) -> TranscribeResult:
        self.received_audio.append(audio_bytes)
        return TranscribeResult(text=self.transcript, latency_sec=0.05)


def _make_wav(tmp_path: Path, name: str, samples: list[int], sr: int = 16000) -> Path:
    """Create a WAV file with specified samples."""
    wav_path = tmp_path / name
    with wave.open(str(wav_path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(struct.pack("<" + "h" * len(samples), *samples))
    return wav_path


def _create_corpus(tmp_path: Path, *, silent: bool = False) -> Path:
    """Create a corpus with either silent or audible audio."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()

    n_samples = 16000  # 1 second at 16kHz
    if silent:
        samples = [0] * n_samples
    else:
        # 440Hz sine wave at ~50% amplitude
        t = np.arange(n_samples) / 16000.0
        samples = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16).tolist()

    _make_wav(corpus_dir, "test_001.wav", samples)

    gt = [{"id": "test_001", "wav": "test_001.wav", "text": "テスト", "category": "test"}]
    (corpus_dir / "ground_truth.json").write_text(
        json.dumps(gt, ensure_ascii=False)
    )
    return corpus_dir


class TestPipelineEvaluatorUnit:
    """Unit tests for PipelineEvaluator in isolation."""

    def test_no_config_passthrough(self, tmp_path):
        """No pipeline config = audio unchanged."""
        corpus = _create_corpus(tmp_path, silent=False)
        audio = (corpus / "test_001.wav").read_bytes()

        evaluator = PipelineEvaluator(None)
        processed, stats = evaluator.process_audio(audio, 16000)
        assert processed == audio
        assert stats.energy_gate_passed is True
        assert stats.codec_applied is None

    def test_energy_gate_rejects_silence(self, tmp_path):
        """Silent audio should be rejected by energy gate."""
        corpus = _create_corpus(tmp_path, silent=True)
        audio = (corpus / "test_001.wav").read_bytes()

        config = PipelineConfig(energy_gate_rms=0.01)
        evaluator = PipelineEvaluator(config)
        processed, stats = evaluator.process_audio(audio, 16000)
        assert stats.energy_gate_passed is False
        assert processed == b""

    def test_energy_gate_passes_tone(self, tmp_path):
        """Audible tone should pass energy gate."""
        corpus = _create_corpus(tmp_path, silent=False)
        audio = (corpus / "test_001.wav").read_bytes()

        config = PipelineConfig(energy_gate_rms=0.01)
        evaluator = PipelineEvaluator(config)
        processed, stats = evaluator.process_audio(audio, 16000)
        assert stats.energy_gate_passed is True
        assert stats.energy_rms > 0.01

    def test_g711_codec_modifies_audio(self, tmp_path):
        """G.711 mu-law codec should produce different bytes."""
        corpus = _create_corpus(tmp_path, silent=False)
        audio = (corpus / "test_001.wav").read_bytes()

        config = PipelineConfig(codec="g711_mulaw")
        evaluator = PipelineEvaluator(config)
        processed, stats = evaluator.process_audio(audio, 16000)
        assert stats.codec_applied == "g711_mulaw"
        assert processed != audio  # Codec introduces quantization noise

    def test_g711_plus_energy_gate(self, tmp_path):
        """Both codec and energy gate together."""
        corpus = _create_corpus(tmp_path, silent=False)
        audio = (corpus / "test_001.wav").read_bytes()

        config = PipelineConfig(codec="g711_mulaw", energy_gate_rms=0.01)
        evaluator = PipelineEvaluator(config)
        processed, stats = evaluator.process_audio(audio, 16000)
        assert stats.energy_gate_passed is True
        assert stats.codec_applied == "g711_mulaw"


class TestRunnerPipelineIntegration:
    """Integration tests: BenchmarkRunner + PipelineEvaluator."""

    def test_clean_task_no_pipeline(self, tmp_path):
        """Clean task (no pipeline config) passes audio directly to provider."""
        corpus = _create_corpus(tmp_path, silent=False)
        task = TaskConfig(
            task="clean_test",
            type="stt",
            corpus=CorpusConfig(path=str(corpus)),
            pipeline=None,
            gates={},
        )
        provider = MockProvider("テスト")
        runner = BenchmarkRunner(task, provider, corpus_base=Path("/"))
        report = runner.run(verbose=False)

        assert report.scored == 1
        assert report.skipped == 0
        assert len(provider.received_audio) == 1

    def test_pipeline_task_applies_codec(self, tmp_path):
        """Pipeline task with G.711 sends modified audio to provider."""
        corpus = _create_corpus(tmp_path, silent=False)
        original_audio = (corpus / "test_001.wav").read_bytes()

        task = TaskConfig(
            task="pipeline_test",
            type="stt",
            corpus=CorpusConfig(path=str(corpus)),
            pipeline=PipelineConfig(codec="g711_mulaw"),
            gates={},
        )
        provider = MockProvider("テスト")
        runner = BenchmarkRunner(task, provider, corpus_base=Path("/"))
        report = runner.run(verbose=False)

        assert report.scored == 1
        assert len(provider.received_audio) == 1
        assert provider.received_audio[0] != original_audio

    def test_energy_gate_skips_silent(self, tmp_path):
        """Energy gate should mark silent utterances as skipped."""
        corpus = _create_corpus(tmp_path, silent=True)
        task = TaskConfig(
            task="gate_test",
            type="stt",
            corpus=CorpusConfig(path=str(corpus)),
            pipeline=PipelineConfig(energy_gate_rms=0.01),
            gates={},
        )
        provider = MockProvider("テスト")
        runner = BenchmarkRunner(task, provider, corpus_base=Path("/"))
        report = runner.run(verbose=False)

        assert report.skipped == 1
        assert report.scored == 0
        assert len(provider.received_audio) == 0
        assert "energy_gate" in report.utterance_results[0].error

    def test_energy_gate_passes_normal(self, tmp_path):
        """Energy gate should pass audible audio to provider."""
        corpus = _create_corpus(tmp_path, silent=False)
        task = TaskConfig(
            task="gate_pass_test",
            type="stt",
            corpus=CorpusConfig(path=str(corpus)),
            pipeline=PipelineConfig(energy_gate_rms=0.01),
            gates={},
        )
        provider = MockProvider("テスト")
        runner = BenchmarkRunner(task, provider, corpus_base=Path("/"))
        report = runner.run(verbose=False)

        assert report.scored == 1
        assert report.skipped == 0

    def test_full_pipeline_codec_and_gate(self, tmp_path):
        """Full pipeline: G.711 codec + energy gate on audible audio."""
        corpus = _create_corpus(tmp_path, silent=False)
        task = TaskConfig(
            task="full_pipeline_test",
            type="stt",
            corpus=CorpusConfig(path=str(corpus)),
            pipeline=PipelineConfig(codec="g711_mulaw", energy_gate_rms=0.01),
            gates={"median_cer": GateThreshold(**{"pass": 0.5, "warn": 0.8})},
        )
        provider = MockProvider("テスト")
        runner = BenchmarkRunner(task, provider, corpus_base=Path("/"))
        report = runner.run(verbose=False)

        assert report.scored == 1
        assert report.gate_result == "PASS"
