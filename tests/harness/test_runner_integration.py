"""Integration test for BenchmarkRunner with a mock STT provider."""
import json
import struct
import wave
from pathlib import Path

from jaeval.harness.providers.base import STTProvider, TranscribeResult
from jaeval.harness.task import TaskConfig, CorpusConfig, GateThreshold
from jaeval.harness.runner import BenchmarkRunner


class MockProvider(STTProvider):
    """Returns a fixed transcript for testing."""

    name = "mock"

    def __init__(self, transcript: str = "テスト"):
        self.transcript = transcript

    def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000) -> TranscribeResult:
        return TranscribeResult(text=self.transcript, latency_sec=0.1)


def _create_test_corpus(tmp_path: Path) -> Path:
    """Create a minimal corpus with a WAV file and ground truth."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()

    # Create a minimal WAV file (1 second of silence at 16kHz, 16-bit mono)
    wav_path = corpus_dir / "test_001.wav"
    with wave.open(str(wav_path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(struct.pack("<" + "h" * 16000, *([0] * 16000)))

    # Create ground truth JSON
    gt = [{"id": "test_001", "wav": "test_001.wav", "text": "テスト", "category": "test"}]
    gt_path = corpus_dir / "ground_truth.json"
    gt_path.write_text(json.dumps(gt, ensure_ascii=False))

    return corpus_dir


class TestBenchmarkRunner:
    def test_perfect_transcription(self, tmp_path):
        corpus_dir = _create_test_corpus(tmp_path)
        task = TaskConfig(
            task="test",
            type="stt",
            corpus=CorpusConfig(path=str(corpus_dir)),
            gates={},
        )
        provider = MockProvider("テスト")
        # corpus_base="/" because corpus.path is already absolute
        runner = BenchmarkRunner(task, provider, corpus_base=Path("/"))
        report = runner.run(verbose=False)
        assert report.scored == 1
        assert report.aggregate is not None
        assert report.aggregate.median_cer == 0.0

    def test_imperfect_transcription(self, tmp_path):
        corpus_dir = _create_test_corpus(tmp_path)
        task = TaskConfig(
            task="test",
            type="stt",
            corpus=CorpusConfig(path=str(corpus_dir)),
            gates={},
        )
        provider = MockProvider("テスト違う")  # Different from reference
        runner = BenchmarkRunner(task, provider, corpus_base=Path("/"))
        report = runner.run(verbose=False)
        assert report.scored == 1
        assert report.aggregate is not None
        assert report.aggregate.median_cer > 0

    def test_gate_evaluation(self, tmp_path):
        corpus_dir = _create_test_corpus(tmp_path)
        task = TaskConfig(
            task="test",
            type="stt",
            corpus=CorpusConfig(path=str(corpus_dir)),
            gates={
                "median_cer": GateThreshold(**{"pass": 0.05, "warn": 0.10}),
            },
        )
        provider = MockProvider("テスト")  # Perfect match -> CER 0.0
        runner = BenchmarkRunner(task, provider, corpus_base=Path("/"))
        report = runner.run(verbose=False)
        assert report.gate_result == "PASS"

    def test_gate_fail(self, tmp_path):
        corpus_dir = _create_test_corpus(tmp_path)
        task = TaskConfig(
            task="test",
            type="stt",
            corpus=CorpusConfig(path=str(corpus_dir)),
            gates={
                "median_cer": GateThreshold(**{"pass": 0.001, "warn": 0.002}),
            },
        )
        provider = MockProvider("完全に違うテキスト")  # Very different -> high CER
        runner = BenchmarkRunner(task, provider, corpus_base=Path("/"))
        report = runner.run(verbose=False)
        assert report.gate_result == "FAIL"

    def test_report_metadata(self, tmp_path):
        corpus_dir = _create_test_corpus(tmp_path)
        task = TaskConfig(
            task="my_task",
            type="stt",
            corpus=CorpusConfig(path=str(corpus_dir)),
            gates={},
        )
        provider = MockProvider("テスト")
        runner = BenchmarkRunner(task, provider, corpus_base=Path("/"))
        report = runner.run(verbose=False)
        assert report.task == "my_task"
        assert report.model == "mock"
        assert report.total_entries == 1
        assert report.skipped == 0
        assert report.errors == 0
