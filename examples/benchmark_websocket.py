"""Example: Run STT benchmark against a WebSocket inference server.

Shows how to programmatically run a benchmark against a remote STT server
(e.g., Qwen3-ASR fine-tuned model on RunPod) using jaeval as a library.

Usage:
    # Against a remote server
    python examples/benchmark_websocket.py --url ws://194.68.245.35:8766

    # Against a local server
    python examples/benchmark_websocket.py --url ws://localhost:8766
"""

import argparse
import json
from pathlib import Path

from jaeval.harness.task import load_task
from jaeval.harness.runner import BenchmarkRunner
from jaeval.harness.providers import get_provider
from jaeval.harness.report import format_markdown, save_json


def main():
    parser = argparse.ArgumentParser(description="Benchmark a WebSocket STT server")
    parser.add_argument("--url", default="ws://localhost:8766", help="WebSocket URL")
    parser.add_argument("--task", default="tasks/stt/corpus_v2_clean.yaml", help="Task YAML")
    parser.add_argument("--limit", type=int, default=10, help="Limit utterances (0=all)")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    task_config = load_task(Path(args.task))
    provider = get_provider("websocket", url=args.url)

    print(f"Benchmarking: {args.url}")
    print(f"Task: {task_config.task}")
    print(f"Utterances: {args.limit or 'all'}\n")

    runner = BenchmarkRunner(task_config, provider)
    report = runner.run(limit=args.limit)

    print("\n" + format_markdown(report))

    if args.output:
        save_json(report, Path(args.output))
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
