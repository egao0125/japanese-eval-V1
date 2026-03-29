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
from rich.console import Console

app = typer.Typer(name="jaeval", help="Japanese Evaluation Harness")
console = Console()


@app.command()
def benchmark(
    task: str = typer.Argument(..., help="Path to task YAML file"),
    model: str = typer.Option("deepgram", help="Provider name (deepgram, openai, or 'all')"),
    limit: int = typer.Option(0, help="Limit utterances (0=all)"),
    output: Path = typer.Option(None, help="Output JSON path"),
    verbose: bool = typer.Option(True, help="Print per-utterance results"),
) -> None:
    """Run an STT benchmark task against a model provider."""
    from .harness.task import load_task
    from .harness.runner import BenchmarkRunner
    from .harness.providers import get_provider, PROVIDER_REGISTRY
    from .harness.report import format_markdown, save_json

    task_config = load_task(Path(task))

    providers_to_run: list[str]
    if model == "all":
        providers_to_run = list(PROVIDER_REGISTRY.keys())
    else:
        providers_to_run = [model]

    for provider_name in providers_to_run:
        provider = get_provider(provider_name)
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
def research(
    topic: str = typer.Argument("Japanese voice AI evaluation state of the art"),
    model: str = typer.Option("claude-sonnet-4-20250514", help="LLM model for planning/synthesis"),
    output_dir: Path = typer.Option("reports", help="Output directory for reports"),
    max_papers: int = typer.Option(10, help="Max papers to summarize"),
    max_repos: int = typer.Option(5, help="Max repos to summarize"),
    max_models: int = typer.Option(5, help="Max HF models to summarize"),
) -> None:
    """Run auto-research pipeline on a topic."""
    from .research import run_research as _run_research

    _run_research(
        topic,
        model=model,
        output_dir=output_dir,
        max_papers=max_papers,
        max_repos=max_repos,
        max_models=max_models,
    )
    console.print(f"\n[green]Research complete![/green]")


if __name__ == "__main__":
    app()
