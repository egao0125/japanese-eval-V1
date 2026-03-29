"""Smoke tests for CLI commands."""

import json

from typer.testing import CliRunner

from jaeval.cli import app

runner = CliRunner()


class TestListCommands:
    def test_list_tasks(self):
        result = runner.invoke(app, ["list-tasks", "--tasks-dir", "tasks"])
        assert result.exit_code == 0
        assert "stt_corpus_v2_clean" in result.output

    def test_list_tasks_missing_dir(self):
        result = runner.invoke(app, ["list-tasks", "--tasks-dir", "/nonexistent"])
        assert result.exit_code == 1

    def test_list_models(self):
        result = runner.invoke(app, ["list-models"])
        assert result.exit_code == 0
        assert "deepgram" in result.output
        assert "openai" in result.output
        assert "websocket" in result.output


class TestEvalCommand:
    def test_eval_no_judge(self, tmp_path):
        call_data = {
            "call_sid": "TEST_001",
            "duration_sec": 30.0,
            "turns": [
                {
                    "user_text": "もしもし",
                    "bot_text": "はい、お電話ありがとうございます。",
                    "latency_ms": 500,
                },
            ],
        }
        call_file = tmp_path / "call.json"
        call_file.write_text(json.dumps(call_data, ensure_ascii=False))

        result = runner.invoke(app, ["eval", str(call_file), "--no-run-judge"])
        assert result.exit_code == 0
        assert "Grade:" in result.output
        assert "TEST_001" in result.output

    def test_eval_with_output(self, tmp_path):
        call_data = {
            "call_sid": "TEST_002",
            "turns": [
                {
                    "user_text": "料金について",
                    "bot_text": "ご案内します。",
                    "latency_ms": 800,
                },
            ],
        }
        call_file = tmp_path / "call.json"
        call_file.write_text(json.dumps(call_data, ensure_ascii=False))
        out_file = tmp_path / "result.json"

        result = runner.invoke(
            app, ["eval", str(call_file), "--no-run-judge", "--output", str(out_file)]
        )
        assert result.exit_code == 0
        assert out_file.exists()
        out_data = json.loads(out_file.read_text())
        assert out_data["call_sid"] == "TEST_002"
        assert "tier1" in out_data

    def test_eval_missing_file(self):
        result = runner.invoke(app, ["eval", "/nonexistent.json", "--no-run-judge"])
        assert result.exit_code == 1

    def test_eval_empty_turns(self, tmp_path):
        call_data = {"call_sid": "EMPTY", "turns": []}
        call_file = tmp_path / "empty.json"
        call_file.write_text(json.dumps(call_data))

        result = runner.invoke(app, ["eval", str(call_file), "--no-run-judge"])
        assert result.exit_code == 1
        assert "No turns" in result.output

    def test_eval_banned_words_displayed(self, tmp_path):
        call_data = {
            "call_sid": "BANNED",
            "turns": [
                {
                    "user_text": "テスト",
                    "bot_text": "かしこまりました。承知しました。",
                    "latency_ms": 500,
                },
            ],
        }
        call_file = tmp_path / "banned.json"
        call_file.write_text(json.dumps(call_data, ensure_ascii=False))

        result = runner.invoke(app, ["eval", str(call_file), "--no-run-judge"])
        assert result.exit_code == 0
        assert "Banned Words" in result.output
