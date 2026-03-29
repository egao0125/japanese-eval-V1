"""Typer CLI for the Japanese Evaluation Harness.

Entry point registered as ``jaeval`` in pyproject.toml.

Commands:
- ``jaeval benchmark`` — run an STT benchmark task against one or all providers
- ``jaeval judge`` — run LLM-as-judge on a call transcript or scorecard
- ``jaeval list-tasks`` — list available YAML benchmark tasks
- ``jaeval list-models`` — list registered STT providers
"""

from __future__ import annotations

from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()  # Load .env before any provider reads keys

app = typer.Typer(name="jaeval", help="Japanese Evaluation Harness")
console = Console()


def _parse_provider_args(raw: list[str] | None) -> dict[str, str]:
    """Parse key=value provider arguments into a dict."""
    if not raw:
        return {}
    result = {}
    for item in raw:
        if "=" not in item:
            raise typer.BadParameter(f"Expected key=value format, got: {item}")
        k, v = item.split("=", 1)
        result[k.strip()] = v.strip()
    return result


@app.command()
def benchmark(
    task: str = typer.Argument(..., help="Path to task YAML file"),
    model: str = typer.Option("deepgram", help="Provider name (deepgram, openai, or 'all')"),
    limit: int = typer.Option(0, help="Limit utterances (0=all)"),
    output: Path = typer.Option(None, help="Output JSON path"),
    verbose: bool = typer.Option(True, help="Print per-utterance results"),
    provider_arg: list[str] = typer.Option(
        None, "--provider-arg", help="Provider kwargs as key=value (repeatable)"
    ),
) -> None:
    """Run an STT benchmark task against a model provider."""
    from .harness.task import load_task
    from .harness.runner import BenchmarkRunner
    from .harness.providers import get_provider, PROVIDER_REGISTRY
    from .harness.report import format_markdown, save_json

    task_config = load_task(Path(task))
    kwargs = _parse_provider_args(provider_arg)

    providers_to_run: list[str]
    if model == "all":
        providers_to_run = list(PROVIDER_REGISTRY.keys())
    else:
        providers_to_run = [model]

    for provider_name in providers_to_run:
        provider = get_provider(provider_name, **kwargs)
        runner = BenchmarkRunner(task_config, provider)
        report = runner.run(limit=limit, verbose=verbose)

        # Print markdown report
        md = format_markdown(report)
        console.print(md)

        # Save JSON if requested
        if output:
            save_json(report, output)
            console.print(f"\nResults saved to {output}")


@app.command("list-tasks")
def list_tasks(
    tasks_dir: Path = typer.Option("tasks", help="Directory containing task YAML files"),
) -> None:
    """List available benchmark tasks."""
    from .harness.task import load_task

    if not tasks_dir.exists():
        console.print(f"[red]Tasks directory not found: {tasks_dir}[/red]")
        raise typer.Exit(1)

    yaml_files = sorted(tasks_dir.rglob("*.yaml"))
    if not yaml_files:
        console.print("[yellow]No task files found.[/yellow]")
        return

    console.print("[bold]Available tasks:[/bold]\n")
    for yf in yaml_files:
        try:
            tc = load_task(yf)
            console.print(f"  {yf}  ->  [cyan]{tc.task}[/cyan] ({tc.type})")
        except Exception as e:
            console.print(f"  {yf}  ->  [red]ERROR: {e}[/red]")


@app.command("list-models")
def list_models() -> None:
    """List available model providers."""
    from .harness.providers import PROVIDER_REGISTRY

    console.print("[bold]Available providers:[/bold]\n")
    for name, cls in sorted(PROVIDER_REGISTRY.items()):
        gpu_tag = " [yellow](GPU)[/yellow]" if cls.requires_gpu else ""
        console.print(f"  {name}{gpu_tag}")


@app.command()
def judge(
    scorecard: Path = typer.Option(None, help="Path to scorecard JSON"),
    transcript: Path = typer.Option(None, help="Path to transcript text file"),
    config_yaml: Path = typer.Option(None, "--config", help="Path to judge config YAML"),
    model: str = typer.Option("claude-sonnet-4-20250514", help="Judge model"),
    dry_run: bool = typer.Option(False, help="Print prompt without calling API"),
    output: Path = typer.Option(None, help="Output JSON path"),
) -> None:
    """Run LLM-as-judge on a call transcript or scorecard."""
    import json as _json

    from .harness.evaluators.llm_judge import JudgeConfig, LLMJudge

    if not scorecard and not transcript:
        console.print("[red]Provide --scorecard (JSON) or --transcript (text file).[/red]")
        raise typer.Exit(1)

    # Build config
    if config_yaml and config_yaml.exists():
        judge_config = JudgeConfig.from_yaml(config_yaml)
    else:
        judge_config = JudgeConfig()
    judge_config.model = model

    judge_instance = LLMJudge(judge_config)

    # Evaluate
    if scorecard:
        if not scorecard.exists():
            console.print(f"[red]Scorecard not found: {scorecard}[/red]")
            raise typer.Exit(1)
        with open(scorecard, "r", encoding="utf-8") as f:
            sc_data = _json.load(f)
        result = judge_instance.evaluate_scorecard(sc_data, dry_run=dry_run)
    else:
        if not transcript.exists():
            console.print(f"[red]Transcript not found: {transcript}[/red]")
            raise typer.Exit(1)
        text = transcript.read_text(encoding="utf-8")
        result = judge_instance.evaluate(text, dry_run=dry_run)

    if dry_run:
        console.print("[yellow]DRY RUN — no API call made.[/yellow]")
        return

    # Display results
    console.print(f"\n[bold]LLM Judge Results[/bold]  (model: {judge_config.model})\n")
    for dim_name, entry in result.scores.items():
        weight = judge_config.weight_map.get(dim_name, 0)
        console.print(
            f"  {dim_name:<25} {entry['score']}/5  (weight: {weight:.2f})"
        )
        if entry.get("justification"):
            console.print(f"    {entry['justification']}")
    console.print("")
    ready_tag = "[green]YES[/green]" if result.production_ready else "[red]NO[/red]"
    console.print(f"  Weighted Score:   {result.weighted_score}")
    console.print(f"  Production Ready: {ready_tag}")
    if result.biggest_issue:
        console.print(f"  Biggest Issue:    {result.biggest_issue}")
    if result.recommendations:
        console.print("  Recommendations:")
        for rec in result.recommendations:
            console.print(f"    - {rec}")

    # Save JSON if requested
    if output:
        out_data = {
            "scores": result.scores,
            "weighted_score": result.weighted_score,
            "production_ready": result.production_ready,
            "biggest_issue": result.biggest_issue,
            "recommendations": result.recommendations,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            _json.dump(out_data, f, indent=2, ensure_ascii=False)
        console.print(f"\n  Results saved to {output}")


@app.command()
def compare(
    result_files: list[Path] = typer.Argument(..., help="Benchmark result JSON files to compare"),
    output: Path = typer.Option(None, help="Output markdown path"),
) -> None:
    """Compare benchmark results across providers."""
    from .harness.compare import format_comparison_markdown

    md = format_comparison_markdown(result_files)
    console.print(md)

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(md, encoding="utf-8")
        console.print(f"\nComparison saved to {output}")


@app.command()
def research(
    topic: str = typer.Argument(None, help="Research topic (or use --topics-file)"),
    topics_file: Path = typer.Option(None, "--topics-file", help="File with one topic per line"),
    model: str = typer.Option("claude-sonnet-4-20250514", help="LLM model for planning/synthesis"),
    output_dir: Path = typer.Option("reports", help="Output directory for reports"),
    max_papers: int = typer.Option(10, help="Max papers to summarize"),
    max_repos: int = typer.Option(5, help="Max repos to summarize"),
    max_models: int = typer.Option(5, help="Max HF models to summarize"),
) -> None:
    """Run auto-research pipeline on a topic (or batch of topics from file)."""
    from .research import run_research as _run_research

    topics: list[str] = []
    if topics_file:
        if not topics_file.exists():
            console.print(f"[red]Topics file not found: {topics_file}[/red]")
            raise typer.Exit(1)
        topics = [
            line.strip()
            for line in topics_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    elif topic:
        topics = [topic]
    else:
        topics = ["Japanese voice AI evaluation state of the art"]

    for i, t in enumerate(topics, 1):
        if len(topics) > 1:
            console.print(f"\n[bold]--- Research {i}/{len(topics)} ---[/bold]")
        console.print(f"[cyan]Topic:[/cyan] {t}\n")

        _run_research(
            t,
            model=model,
            output_dir=output_dir,
            max_papers=max_papers,
            max_repos=max_repos,
            max_models=max_models,
        )
        console.print(f"\n[green]Research {i}/{len(topics)} complete![/green]")


if __name__ == "__main__":
    app()
