"""CLI entry point.

    python -m conveval evaluate               THE MAIN ENTRY POINT. One MLflow run,
                                              one row per scenario, every judge's
                                              assessment attached to that row.
    python -m conveval evaluate --runs 3      fan out 3 runs per scenario
    python -m conveval pipeline               architecture + model roles
    python -m conveval run                    fast console path, no tracking server
    python -m conveval explain <id>           one transcript, judge by judge

    mlflow ui --backend-store-uri sqlite:///mlflow.db
        NOT plain `mlflow ui` - that reads ./mlruns and shows nothing.
"""

from __future__ import annotations

import argparse
import sys

from conveval import store
from conveval.report import console, explain, render_pipeline, render_suite
from conveval.runner import run_sweep, save_results, summarise
from conveval.scenarios import DEFAULT_RUNS, SCENARIOS


def step(n: int, title: str, detail: str) -> None:
    """Narrate one stage of the pipeline as it happens.

    `evaluate` is the demo surface, so the console has to answer "what is it doing and
    why" while it runs. A progress log that only says "running..." for 90 seconds shows
    an interviewer nothing.
    """
    console.print(f"\n[bold cyan]({n})[/bold cyan] [bold]{title}[/bold]")
    console.print(f"     [dim]{detail}[/dim]")


def main() -> int:
    ap = argparse.ArgumentParser(prog="conveval")
    sub = ap.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="run the evaluation sweep")
    mode = run.add_mutually_exclusive_group()
    mode.add_argument("--rejudge", action="store_true", help="re-run the live judge panel")
    mode.add_argument("--regenerate", action="store_true", help="regenerate conversations too")
    run.add_argument("--runs", type=int, default=DEFAULT_RUNS,
                     help="runs per scenario. >1 fans out, which is what pass-rate and "
                          "any-occurrence aggregation exist for (default: %(default)s)")
    run.add_argument("--summary", action="store_true",
                     help="have the orchestrator model narrate the result (one extra call)")

    ev = sub.add_parser("evaluate", help="MLflow GenAI evaluation: one run, one row per scenario")
    ev.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    ev.add_argument("--builtin-safety", action="store_true",
                    help="also run MLflow's built-in Safety scorer. Off by default: on "
                         "coaching transcripts it returns 'yes' on every row, so it "
                         "adds a model call per row and a column that never varies")

    sub.add_parser("pipeline", help="print the pipeline and the model bound to each role")
    sub.add_parser("export", help="export the latest run as JSON for the demo page")

    ex = sub.add_parser("explain", help="walk one transcript through the aggregation")
    ex.add_argument("transcript_id", help="e.g. unfaithful#run1")

    rv = sub.add_parser("review", help="list contested traces awaiting human review")
    rv.add_argument("--feedback", nargs=4, metavar=("TRACE_ID", "CRITERION", "VALUE", "REVIEWER"),
                    help="record a human verdict; CRITERION must match the judge name "
                         "so it becomes a calibration datapoint")
    rv.add_argument("--rationale", default="", help="why (optional)")

    args = ap.parse_args()

    if args.cmd == "pipeline":
        render_pipeline()
        return 0

    if args.cmd == "export":
        from conveval.evaluation import TRACKING_URI, configure
        from conveval.export import export

        configure(TRACKING_URI)
        run_path, golden_path = export()
        console.print(f"[green]exported[/green] {run_path}")
        console.print(f"[green]exported[/green] {golden_path}")
        console.print("[dim]Both are build inputs. The published page makes no "
                      "network calls.[/dim]")
        return 0

    if args.cmd == "review":
        from conveval.evaluation import (
            TRACKING_URI, configure, log_human_feedback, traces_needing_review,
        )

        configure(TRACKING_URI)
        if args.feedback:
            trace_id, criterion, value, reviewer = args.feedback
            log_human_feedback(trace_id, criterion, value, reviewer, args.rationale)
            console.print(f"[green]recorded[/green] {criterion}={value} by {reviewer}")
            console.print("[dim]Name matches the judge assessment, so this is now a "
                          "calibration datapoint: human vs panel on the same trace.[/dim]")
            return 0
        rows = traces_needing_review()
        if not rows:
            console.print("[green]Nothing flagged.[/green] No disagreements and no failures.")
            return 0
        console.print(f"\n[bold]{len(rows)} trace(s) awaiting review[/bold]  "
                      "[dim](panel disagreed, so the automated score is unreliable)[/dim]\n")
        for tid, scenario, crit in rows:
            console.print(f"  {scenario:14} {crit:28} {tid}")
        console.print("\n[dim]Review in the UI (Assessments button on a trace), or:[/dim]")
        console.print(f"[dim]  python -m conveval review --feedback {rows[0][0]} "
                      f"{rows[0][2].split(',')[0]} pass yourname[/dim]")
        return 0

    if args.cmd == "evaluate":
        from conveval.evaluation import (
            TRACKING_URI, _experiment_id, configure, flag_for_review,
            register_judges, run_evaluation,
        )
        import mlflow

        from conveval.orchestrator import (
            Budget,
            orchestrate_summary,
            panel_digest,
            run_control_loop,
            scoreboard,
        )
        from conveval.rounds import (
            collected_states,
            finalise_scenarios,
            make_executor,
            trace_round,
        )

        console.print()
        render_pipeline(compact=True)
        configure(TRACKING_URI)

        step(1, "register rubrics",
             "publishes each criterion to the Judges tab. These do NOT score anything "
             "- the panel below does. Re-registration is skipped when the rubric text "
             "is unchanged, so a version bump means the yardstick moved.")
        registered = register_judges()
        console.print(f"     {', '.join(registered) + ' (new version)' if registered else 'unchanged - no new versions minted'}")

        step(2, "run the conversations and score them",
             f"{len(SCENARIOS)} scenario(s) x {args.runs} run(s). Each row: LangGraph "
             "drives a learner/agent turn loop, then 3 judges from 3 model families "
             "score it independently, then code verifies every cited span.")
        result = run_evaluation(args.runs, args.builtin_safety)
        console.print(f"     evaluation run: [green]{result.run_id}[/green]")

        step(3, "collapse the panel and flag what needs a human",
             "majority vote for binary criteria, median for ordinal - never a mean. "
             "Disagreement is recorded, not averaged away.")
        contested, failed = flag_for_review(result.run_id)
        console.print(f"     {contested} contested (judges disagreed), "
                      f"{failed} failed (judges agreed it was bad)")

        exp = _experiment_id()

        # Resuming the run puts the orchestrator's own traces INSIDE it. Called
        # outside, they are stamped sourceRun=None and float loose in the Traces tab,
        # unattached to the evaluation they describe.
        with mlflow.start_run(run_id=result.run_id):
            step(4, "orchestrator control loop",
                 "the escalation ladder: accept, then spend compute, then spend more "
                 "compute, then spend a human. Code computes which actions are LEGAL "
                 "at each rung; the model picks one. It never supplies the verdict.")

            states = collected_states()
            budget = Budget()
            executor = make_executor()

            def narrate(outcome) -> None:
                trace_round(outcome)
                label = "initial pass" if outcome.round_no == 0 else f"extra round {outcome.round_no}"
                console.print(f"\n     [bold]round {outcome.round_no}[/bold] [dim]({label})[/dim]")
                for sid, d in sorted(outcome.decisions.items()):
                    colour = {"accept": "green", "resample": "cyan",
                              "widen_panel": "cyan"}.get(d["action"], "yellow")
                    flag = " [red](model pick discarded)[/red]" if d["was_fallback"] else ""
                    console.print(f"       {sid:14} -> [{colour}]{d['action']}[/{colour}]{flag}")
                    console.print(f"       {'':14}    [dim]legal: {', '.join(d['legal_actions'])}[/dim]")
                    if d["why"]:
                        console.print(f"       {'':14}    {d['why']}")
                if outcome.calls_spent:
                    console.print(f"       [dim]spent {outcome.calls_spent} extra model calls, "
                                  f"{outcome.budget_remaining} left in budget[/dim]")

            final, outcomes = run_control_loop(states, budget, executor, on_round=narrate)
            finalise_scenarios(exp, result.run_id, states, final)

            console.print(f"\n     [bold]after {len(outcomes)} round(s):[/bold]")
            for sid, d in sorted(final.items()):
                st = states[sid]
                console.print(f"       {sid:14} [bold]{d['action']}[/bold]  "
                              f"[dim]{st.n_transcripts} run(s), {len(st.panel)}-judge panel[/dim]")
                if d.get("question"):
                    console.print(f"       {'':14} [italic]ask: {d['question']}[/italic]")

            board = scoreboard(panel_digest(exp, result.run_id))
            console.print("\n     [bold]final scorecard[/bold]")
            for k, v in board.items():
                console.print(f"       {k:16} {v}")

            step(5, "orchestrator summary", "narrates the finished result. Decides nothing.")
            triaged = [(sid, sid, d) for sid, d in sorted(final.items())
                       if d["action"] != "accept"]
            console.print(f"     [italic]{orchestrate_summary(exp, triaged, result.run_id)}[/italic]")

        console.print("\n[bold]see it in MLflow:[/bold]")
        console.print("  mlflow ui --backend-store-uri sqlite:///mlflow.db")
        console.print("[dim]  Traces tab   -> one row per scenario; filter "
                      "tags.needs_review = 'true'[/dim]")
        console.print("[dim]  a trace      -> Timeline for the turn spans, Assessments "
                      "for judge votes + triage[/dim]")
        console.print("[dim]  Judges tab   -> the rubric each score was given against[/dim]")
        return 0

    mode = "regenerate" if getattr(args, "regenerate", False) else \
           "rejudge" if getattr(args, "rejudge", False) else "cached"

    if args.cmd == "explain":
        suite, transcripts = run_sweep("cached", runs=DEFAULT_RUNS)
        for s in suite.scenarios:
            for t in s.transcripts:
                if t.transcript_id == args.transcript_id:
                    explain(t, transcripts.get(t.transcript_id))
                    return 0
        console.print(f"[red]No transcript {args.transcript_id!r}.[/red] Available:")
        for s in suite.scenarios:
            for t in s.transcripts:
                console.print(f"  {t.transcript_id}")
        return 1

    if mode == "cached" and not store.have_fixtures():
        console.print("[red]No fixtures.[/red] Run: python -m conveval run --regenerate")
        return 1

    console.print()
    render_pipeline(compact=True)
    suite, transcripts = run_sweep(mode, runs=args.runs, trace=lambda m: console.print(f"[dim]{m}[/dim]"))
    render_suite(suite)

    summary = ""
    if args.summary:
        summary = summarise(suite)
        console.print(f"[italic]{summary}[/italic]\n")

    save_results(suite)
    # MLflow logging lives entirely in `evaluate` now. `run` is the fast console path:
    # same pipeline, same aggregation, no tracking server involved.
    console.print("[dim]For the MLflow view: python -m conveval evaluate[/dim]")
    return 0 if suite.passed else 2


if __name__ == "__main__":
    sys.exit(main())
