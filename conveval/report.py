"""CLI rendering: the scorecard, the gates, and the aggregation walkthrough."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from conveval.models import SuiteResult, TranscriptResult
from conveval.scenarios import CRITERIA, SCENARIOS

console = Console()

_PIPELINE = """  orchestrator  fan out scenarios x runs, collapse results, decide
       |
       +-- simulated learner  ---(LangGraph turn loop)---  agent under test
       |                                                   (black-box interface)
       |                                 |
       |                            transcript
       |                                 v
       +-- judge panel   3 models, 3 families, scored independently
       |                                 |
       |                    verdict + verbatim evidence
       |                                 v
       +-- verify        deterministic: does the cited span exist?
       |                                 v
       +-- collapse      judges -> runs -> scenarios
                                         v
                          scorecard | gates | human review queue
"""


def render_pipeline(compact: bool = False) -> None:
    """Print the pipeline and which model is bound to each role.

    The model assignment is the part people ask about: the agent under test must not
    share a family with any judge, or the panel is scoring its own relatives.
    """
    from conveval.llm import JUDGES, LEARNER, ORCHESTRATOR, SUT, judge_families_are_independent

    if not compact:
        console.print(Panel(_PIPELINE, title="Pipeline", border_style="blue"))

    t = Table(title="Model roles", header_style="bold", show_edge=not compact)
    t.add_column("role")
    t.add_column("model")
    t.add_column("family")
    t.add_column("why this model")
    t.add_row("[bold]agent under test[/bold]", SUT.model, SUT.family, "the thing being evaluated")
    t.add_row("simulated learner", LEARNER.model, LEARNER.family,
              "cheap; different family so it does not mirror the agent")
    t.add_row("orchestrator", ORCHESTRATOR.model, ORCHESTRATOR.family,
              "routes flagged traces, narrates; never scores")
    why = ["strong, anchors the panel", "mid tier", "mid tier, open weight"]
    for i, (j, w) in enumerate(zip(JUDGES, why), start=1):
        t.add_row(f"judge {i}", j.model, j.family, w)
    console.print(t)

    ok, note = judge_families_are_independent()
    colour = "green" if ok else "red"
    console.print(f"  [{colour}]judge independence:[/] {note}\n")

    if not compact:
        s = Table(title="Scenarios", header_style="bold")
        s.add_column("id")
        s.add_column("demonstrates")
        s.add_row("happy_path", "clean run; panel agrees; every gate green")
        s.add_row("unfaithful", "agent invents facts; panel agrees it failed")
        s.add_row("disagreement", "borderline quality; panel splits; routed to a human")
        console.print(s)


def render_suite(suite: SuiteResult) -> None:
    console.print()
    console.print(
        Panel.fit(
            f"[bold]Conversation evaluation[/bold]   mode=[cyan]{suite.mode}[/cyan]   "
            f"panel agreement=[cyan]{suite.panel_agreement:.0%}[/cyan]",
            border_style="blue",
        )
    )

    # Scenarios are shown as a matrix, never collapsed into a headline number:
    # a mean would hide one broken scenario behind the healthy ones.
    table = Table(title="Scorecard (scenario x criterion)", header_style="bold")
    table.add_column("scenario")
    for c in CRITERIA:
        table.add_column(c.label, justify="center")

    for s in suite.scenarios:
        row = [s.title]
        for c in CRITERIA:
            r = s.per_criterion[c.key]
            colour = "red" if r.failed else "green"
            row.append(f"[{colour}]{r.display}[/{colour}]")
        table.add_row(*row)
    console.print(table)

    gates = Table(title="Gates", header_style="bold")
    gates.add_column("gate")
    gates.add_column("", justify="center")
    gates.add_column("detail")
    for g in suite.gates:
        gates.add_row(g.name, "[green]PASS[/green]" if g.passed else "[red]FAIL[/red]", g.detail)
    console.print(gates)

    verdict = "[green]SUITE PASSED[/green]" if suite.passed else "[red]SUITE FAILED[/red]"
    console.print(Panel.fit(verdict, border_style="green" if suite.passed else "red"))

    if suite.review_queue:
        q = Table(title=f"Human review queue ({len(suite.review_queue)} contested)", header_style="bold")
        q.add_column("transcript")
        q.add_column("contested criteria")
        q.add_column("panel split")
        for t in suite.review_queue:
            contested = [c for c in t.per_criterion.values() if c.contested]
            q.add_row(
                t.transcript_id,
                ", ".join(c.criterion for c in contested),
                ", ".join(f"{c.agreement:.0%}" for c in contested),
            )
        console.print(q)
        console.print(
            "[dim]Only these reach a human. Everything else was unanimous, which is "
            "where the reduction in manual review comes from.[/dim]\n"
        )


def render_transcript(transcript) -> None:
    """Print the conversation itself.

    Without this you can read what the judges said ABOUT a transcript but never the
    transcript, which makes the verdicts impossible to sanity-check in the terminal.
    """
    console.print()
    console.print(Panel.fit(f"[bold]Conversation: {transcript.id}[/bold]", border_style="cyan"))
    for turn in transcript.turns:
        if turn.role == "learner":
            console.print(f"[bold blue]Learner[/bold blue]  {turn.text}\n")
        else:
            console.print(f"[bold yellow]Coach[/bold yellow] (SUT)  {turn.text}\n")


def explain(result: TranscriptResult, transcript=None) -> None:
    """Walk one transcript through the collapse, step by step."""
    if transcript is not None:
        render_transcript(transcript)

    console.print()
    console.print(Panel.fit(f"[bold]Aggregation walkthrough: {result.transcript_id}[/bold]", border_style="magenta"))

    for c in CRITERIA:
        cons = result.per_criterion.get(c.key)
        if not cons:
            continue
        t = Table(title=f"{c.label}  ({c.kind}, across runs: {c.across_runs})", header_style="bold")
        t.add_column("judge")
        t.add_column("provider")
        t.add_column("score", justify="center")
        t.add_column("evidence cited (verbatim from the agent)")
        t.add_column("verified", justify="center")
        for v in cons.verdicts:
            mark = {True: "[green]yes[/green]", False: "[red]NO[/red]", None: "[dim]n/a[/dim]"}[
                v.evidence_verified
            ]
            t.add_row(
                v.judge,
                v.provider,
                str(v.score),
                (v.evidence[:70] + "...") if len(v.evidence) > 70 else (v.evidence or "[dim]none[/dim]"),
                mark,
            )
        console.print(t)
        flag = "[red]CONTESTED -> human review[/red]" if cons.contested else "[green]unanimous[/green]"
        console.print(
            f"  consensus=[bold]{cons.consensus}[/bold]   agreement={cons.agreement:.0%}   "
            f"dispersion={cons.dispersion:g}   {flag}\n"
        )
