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


@app.command()
def summary(
    results_dir: Path = typer.Option("results", help="Results directory"),
    reports_dir: Path = typer.Option("reports", help="Reports directory"),
) -> None:
    """Show project overview: benchmarks, evaluations, research reports."""
    import json as _json

    console.print("[bold]jaeval Project Summary[/bold]\n")

    # STT Benchmarks
    benchmark_files = sorted(results_dir.glob("*.json")) if results_dir.exists() else []
    if benchmark_files:
        console.print(f"[cyan]STT Benchmarks:[/cyan] {len(benchmark_files)} results")
        for bf in benchmark_files:
            try:
                data = _json.loads(bf.read_text())
                model = data.get("model", "?")
                task = data.get("task", "?")
                agg = data.get("aggregate", {})
                median_cer = agg.get("median_cer", "?")
                gate = data.get("gate_result", "?")
                cer_str = f"{median_cer:.1%}" if isinstance(median_cer, float) else str(median_cer)
                console.print(f"  {bf.name}: {model} / {task} — {cer_str} CER — {gate}")
            except Exception:
                console.print(f"  {bf.name}: [red]parse error[/red]")
    else:
        console.print("[yellow]No STT benchmarks found.[/yellow]")

    # Judge scores
    judge_dir = results_dir / "judge_scores"
    judge_files = sorted(judge_dir.glob("*.json")) if judge_dir.exists() else []
    if judge_files:
        console.print(f"\n[cyan]LLM Judge Evaluations:[/cyan] {len(judge_files)} calls")
        ready_count = 0
        scores = []
        for jf in judge_files:
            try:
                data = _json.loads(jf.read_text())
                ws = data.get("weighted_score", 0)
                ready = data.get("production_ready", False)
                scores.append(ws)
                if ready:
                    ready_count += 1
                tag = "[green]READY[/green]" if ready else "[red]NOT READY[/red]"
                console.print(f"  {jf.stem}: {ws}/5 {tag}")
            except Exception:
                console.print(f"  {jf.stem}: [red]parse error[/red]")
        if scores:
            mean = sum(scores) / len(scores)
            console.print(f"  Mean: {mean:.2f}/5 | Production Ready: {ready_count}/{len(scores)}")
    else:
        console.print("\n[yellow]No LLM judge evaluations found.[/yellow]")

    # Research reports
    report_files = sorted(reports_dir.glob("*.md")) if reports_dir.exists() else []
    if report_files:
        console.print(f"\n[cyan]Research Reports:[/cyan] {len(report_files)} reports")
        for rf in report_files[-5:]:  # Show last 5
            console.print(f"  {rf.name}")
        if len(report_files) > 5:
            console.print(f"  ... and {len(report_files) - 5} more")
    else:
        console.print("\n[yellow]No research reports found.[/yellow]")

    # Test count
    console.print("")


@app.command("judge-compare")
def judge_compare(
    result_files: list[Path] = typer.Argument(..., help="Judge result JSON files to compare"),
    output: Path = typer.Option(None, help="Output markdown path"),
) -> None:
    """Compare LLM judge results across multiple calls."""
    import json as _json

    results = []
    for f in result_files:
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = _json.load(fp)
            data["_filename"] = f.stem
            results.append(data)
        except (OSError, _json.JSONDecodeError) as e:
            console.print(f"[yellow]Skipping {f}: {e}[/yellow]")

    if not results:
        console.print("[red]No valid result files found.[/red]")
        raise typer.Exit(1)

    # Collect all dimension names
    all_dims: list[str] = []
    for r in results:
        for dim in r.get("scores", {}):
            if dim not in all_dims:
                all_dims.append(dim)

    # Header
    lines = ["## LLM Judge Comparison", ""]
    dim_headers = " | ".join(d[:12] for d in all_dims)
    header = f"| Call | Score | Ready | {dim_headers} |"
    sep = "|------|------:|:-----:|" + "|".join(["-----:" for _ in all_dims]) + "|"
    lines.append(header)
    lines.append(sep)

    for r in results:
        call_id = r.get("_filename", "?")
        ws = r.get("weighted_score", 0)
        ready = "Yes" if r.get("production_ready") else "No"
        dim_vals = []
        for d in all_dims:
            score = r.get("scores", {}).get(d, {}).get("score", "-")
            dim_vals.append(str(score))
        dim_str = " | ".join(dim_vals)
        lines.append(f"| {call_id} | {ws} | {ready} | {dim_str} |")

    lines.append("")

    # Summary stats
    scores = [r.get("weighted_score", 0) for r in results]
    ready_count = sum(1 for r in results if r.get("production_ready"))
    lines.append(f"**Mean Score:** {sum(scores)/len(scores):.2f}")
    lines.append(f"**Production Ready:** {ready_count}/{len(results)}")
    lines.append("")

    md = "\n".join(lines)
    console.print(md)

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(md, encoding="utf-8")
        console.print(f"\nComparison saved to {output}")


@app.command("eval")
def eval_call(
    call_json: Path = typer.Argument(..., help="Path to call JSON with turns array"),
    run_judge: bool = typer.Option(True, help="Run Tier 2 LLM judge"),
    judge_model: str = typer.Option("claude-sonnet-4-20250514", help="Judge model"),
    judge_config: Path = typer.Option(None, "--judge-config", help="Judge config YAML"),
    output: Path = typer.Option(None, help="Output JSON path"),
) -> None:
    """Evaluate a call with Tier 1 scorecard + optional Tier 2 LLM judge.

    Input JSON format: {"call_sid": "...", "turns": [...], "duration_sec": 120}
    Each turn: {"user_text": "...", "bot_text": "...", "confidence": -0.05, ...}
    """
    import json as _json

    from .integration import evaluate_call

    if not call_json.exists():
        console.print(f"[red]File not found: {call_json}[/red]")
        raise typer.Exit(1)

    with open(call_json, "r", encoding="utf-8") as f:
        data = _json.load(f)

    call_sid = data.get("call_sid", call_json.stem)
    turns = data.get("turns", [])
    duration_sec = data.get("duration_sec", 0.0)

    if not turns:
        console.print("[red]No turns found in input JSON.[/red]")
        raise typer.Exit(1)

    result = evaluate_call(
        call_sid=call_sid,
        turns=turns,
        duration_sec=duration_sec,
        run_judge=run_judge,
        judge_model=judge_model,
        judge_config_yaml=judge_config,
    )

    # Display Tier 1
    grade_colors = {"A": "green", "B": "cyan", "C": "yellow", "D": "red", "F": "red bold"}
    gc = grade_colors.get(result.grade, "white")
    console.print(f"\n[bold]Call Evaluation: {call_sid}[/bold]\n")
    console.print(f"  [{gc}]Grade: {result.grade}[/{gc}]")
    console.print(f"  Task Completion:  {result.task_completion}")
    console.print(f"  STT Errors:       {result.stt_error_count}")
    console.print(f"  Hallucinations:   {result.hallucination_count}")
    if result.banned_words_used:
        console.print(f"  Banned Words:     {', '.join(result.banned_words_used)}")
    console.print(f"  Avg Latency:      {result.avg_latency_sec:.3f}s")

    # Display Tier 2
    if result.weighted_score is not None:
        console.print(f"\n  [bold]LLM Judge Score: {result.weighted_score}/5[/bold]")
        ready_tag = "[green]YES[/green]" if result.production_ready else "[red]NO[/red]"
        console.print(f"  Production Ready: {ready_tag}")
        if result.biggest_issue:
            console.print(f"  Biggest Issue:    {result.biggest_issue}")
        for dim, entry in result.dimension_scores.items():
            console.print(f"    {dim:<25} {entry['score']}/5")

    # Save JSON
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            _json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        console.print(f"\n  Results saved to {output}")


if __name__ == "__main__":
    app()
