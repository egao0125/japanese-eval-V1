"""Tests for YAML task definition loading."""

import yaml

from jaeval.harness.task import load_task


class TestTaskLoader:
    def test_load_valid_task(self, tmp_path):
        task_data = {
            "task": "test_task",
            "type": "stt",
            "corpus": {"path": "corpora/test", "ground_truth": "gt.json"},
            "metrics": ["cer"],
            "gates": {"median_cer": {"pass": 0.05, "warn": 0.08}},
        }
        yaml_path = tmp_path / "test.yaml"
        yaml_path.write_text(yaml.dump(task_data))

        config = load_task(yaml_path)
        assert config.task == "test_task"
        assert config.type == "stt"
        assert config.corpus.path == "corpora/test"

    def test_load_with_pipeline(self, tmp_path):
        task_data = {
            "task": "pipeline_test",
            "type": "stt",
            "corpus": {"path": "corpora/test"},
            "pipeline": {"codec": "g711_mulaw", "vad": "silero", "energy_gate_rms": 0.01},
        }
        yaml_path = tmp_path / "test.yaml"
        yaml_path.write_text(yaml.dump(task_data))

        config = load_task(yaml_path)
        assert config.pipeline is not None
        assert config.pipeline.codec == "g711_mulaw"

    def test_defaults(self, tmp_path):
        task_data = {
            "task": "minimal",
            "type": "stt",
            "corpus": {"path": "corpora/test"},
        }
        yaml_path = tmp_path / "test.yaml"
        yaml_path.write_text(yaml.dump(task_data))

        config = load_task(yaml_path)
        assert config.version == "1.0"
        assert config.metrics == ["cer"]
        assert config.categories == []
        assert config.gates == {}
