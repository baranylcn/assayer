from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from assayer.judge import JudgeResult
from assayer.models import ModelResult
from assayer.scorer import readability_stats

console = Console()


def _stats_line(result: ModelResult) -> str:
    parts = [
        f"{result.tokens_input}in / {result.tokens_output}out tokens",
        f"{result.latency_seconds:.2f}s",
    ]
    if result.cost_usd > 0:
        parts.append(f"${result.cost_usd:.6f}")
    return " | ".join(parts)


def _make_panel(result: ModelResult) -> Panel:
    if result.error:
        content = Text(result.error, style="red")
        subtitle = None
    else:
        content = Text(result.output)
        subtitle = _stats_line(result)

    return Panel(
        content,
        title=f"[bold]{result.model}[/bold]",
        subtitle=subtitle,
        expand=True,
    )


def render_run(
    prompt: str,
    results: list[ModelResult],
    similarity: dict[tuple[str, str], float] | None = None,
    judge_result: JudgeResult | None = None,
) -> None:
    header = prompt if len(prompt) <= 80 else prompt[:77] + "..."
    console.print(Rule(f"[bold]{header}[/bold]"))

    panels = [_make_panel(r) for r in results]
    console.print(Columns(panels, equal=True, expand=True))

    if similarity is not None:
        render_similarity_matrix(results, similarity)
        render_readability(results)

    if judge_result is not None:
        render_judge(judge_result)


def render_similarity_matrix(
    results: list[ModelResult],
    similarity: dict[tuple[str, str], float],
) -> None:
    models = [r.model for r in results]
    table = Table(title="Similarity Matrix", show_header=True)
    table.add_column("", style="bold")
    for m in models:
        table.add_column(m, justify="center")

    for row_model in models:
        cells: list[str] = []
        for col_model in models:
            if row_model == col_model:
                cells.append("1.000")
            else:
                key = (
                    (row_model, col_model)
                    if (row_model, col_model) in similarity
                    else (col_model, row_model)
                )
                cells.append(f"{similarity.get(key, 0.0):.3f}")
        table.add_row(row_model, *cells)

    console.print(table)


def render_readability(results: list[ModelResult]) -> None:
    valid = [r for r in results if not r.error and r.output]
    if not valid:
        return

    table = Table(title="Readability", show_header=True)
    table.add_column("Model", style="bold")
    table.add_column("Words", justify="right")
    table.add_column("Sentences", justify="right")
    table.add_column("Avg sentence length", justify="right")

    for r in valid:
        stats = readability_stats(r.output)
        table.add_row(
            r.model,
            str(int(stats["word_count"])),
            str(int(stats["sentence_count"])),
            f"{stats['avg_sentence_length']:.1f}",
        )

    console.print(table)


def render_judge(judge_result: JudgeResult) -> None:
    console.print(Rule("[bold]Judge[/bold]"))
    console.print(f"[bold green]Winner:[/bold green] {judge_result.winner}")
    console.print(f"[dim]{judge_result.reasoning}[/dim]")

    if not judge_result.scores:
        return

    all_criteria: list[str] = []
    for criteria_map in judge_result.scores.values():
        for c in criteria_map:
            if c not in all_criteria:
                all_criteria.append(c)

    table = Table(title="Scores", show_header=True)
    table.add_column("Model", style="bold")
    for c in all_criteria:
        table.add_column(c, justify="center")

    for model, criteria_map in judge_result.scores.items():
        row = [str(criteria_map.get(c, "")) for c in all_criteria]
        table.add_row(model, *row)

    console.print(table)
