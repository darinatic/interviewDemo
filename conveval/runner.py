"""The orchestrator.

Fan out over scenarios x runs, drive each conversation, run the judge panel, collapse
the results, hand the tree to MLflow.

Two things worth stating plainly, because both get asked about:

1. **This is ordinary concurrent Python, not a graph.** Fan-out/fan-in is not a state
   machine. The one genuinely cyclic part of the system - the learner/agent turn loop
   - is a LangGraph graph, in conversation.py. Using a graph for both would be
   ceremony, and being able to say why is worth more than using it everywhere.

2. **The orchestrator's decisions are deterministic code.** Consensus, gates and
   routing are computed in aggregate.py with no model involved. A model is used only
   to *narrate* the finished result for a human reader. A model that decided whether
   the suite passed would be unauditable, and auditability is the entire point.

Three modes:
  cached      recorded transcripts and verdicts. Instant, free, deterministic.
  rejudge     recorded transcripts, live judge panel.
  regenerate  regenerate conversations too. Rebuilds the fixtures.
"""

from __future__ import annotations

import concurrent.futures as futures
import json
from dataclasses import asdict
from pathlib import Path

from conveval import store
from conveval.agent import AgentUnderTest, PromptedCoach
from conveval.aggregate import build_suite, collapse_judges
from conveval.models import SuiteResult, Transcript, TranscriptResult
from conveval.scenarios import BASELINE, CRITERIA, DEFAULT_RUNS, SCENARIOS
from conveval.verify import verify_verdicts

RESULTS = Path(__file__).resolve().parent.parent / "results.json"

Collected = tuple[TranscriptResult, Transcript]


def _one(scenario, run: int, mode: str, sut: AgentUnderTest, trace) -> Collected | None:
    seed = 1000 + run
    tag = f"{scenario.id}#run{run}"

    if mode == "regenerate":
        from conveval.conversation import run_conversation

        trace(f"    [{tag}] learner <-> agent ({sut.name}) · LangGraph turn loop")
        transcript = run_conversation(scenario, run, seed, sut)
        store.save_transcript(transcript)
    else:
        transcript = store.load_transcript(scenario.id, run)
        if transcript is None:
            trace(f"    ! no fixture for {tag}; run with --regenerate")
            return None

    if mode == "cached":
        verdicts = store.load_verdicts(scenario.id, run)
        if verdicts is None:
            trace(f"    ! no cached verdicts for {tag}")
            return None
    else:
        from conveval.judges import run_panel

        trace(f"    [{tag}] judge panel · {len(CRITERIA)} criteria")
        verdicts = run_panel(transcript, CRITERIA)

    # Always runs, in every mode. Deterministic local code, so re-running it over
    # cached verdicts costs nothing and keeps the fixtures honest: if the verifier
    # gets stricter, yesterday's cache is re-checked against it.
    verify_verdicts(transcript, verdicts, CRITERIA)

    if mode != "cached":
        # Saved AFTER verification so the fixture records the verdict as judged.
        store.save_verdicts(scenario.id, run, verdicts)

    return collapse_judges(transcript, CRITERIA, verdicts), transcript


def run_sweep(
    mode: str = "cached",
    runs: int = DEFAULT_RUNS,
    sut: AgentUnderTest | None = None,
    trace=lambda _msg: None,
) -> tuple[SuiteResult, dict[str, Transcript]]:
    sut = sut or PromptedCoach()
    jobs = [(s, r) for s in SCENARIOS for r in range(1, runs + 1)]
    results: dict[str, list[TranscriptResult]] = {s.id: [] for s in SCENARIOS}
    transcripts: dict[str, Transcript] = {}

    trace(
        f"  orchestrator · fan out {len(SCENARIOS)} scenarios x {runs} run(s) "
        f"= {len(jobs)} conversation(s)"
    )

    def collect(out: Collected | None) -> None:
        if out is None:
            return
        result, transcript = out
        results[result.scenario_id].append(result)
        transcripts[result.transcript_id] = transcript

    if mode == "cached":
        for scenario, run in jobs:
            collect(_one(scenario, run, mode, sut, trace))
    else:
        with futures.ThreadPoolExecutor(max_workers=3) as pool:
            futs = [pool.submit(_one, s, r, mode, sut, trace) for s, r in jobs]
            for fut in futures.as_completed(futs):
                collect(fut.result())

    for key in results:
        results[key].sort(key=lambda r: r.run)

    trace("  collapse · judges -> runs -> scenarios")
    from conveval.llm import JUDGES

    suite = build_suite(
        SCENARIOS, CRITERIA, results, baseline=BASELINE, mode=mode,
        expected_judges=len(JUDGES),
    )
    trace(
        f"  decide · {sum(1 for g in suite.gates if g.passed)}/{len(suite.gates)} gates passed, "
        f"{len(suite.review_queue)} transcript(s) routed to human review"
    )
    return suite, transcripts


def summarise(suite: SuiteResult) -> str:
    """Narrate the finished result for a human reader.

    Called AFTER every decision is made. The model receives the computed outcome and
    writes prose about it; it does not compute or override anything. Keeping the
    judgement in code and only the wording in a model is what keeps this auditable.
    """
    from conveval.llm import ORCHESTRATOR, complete

    facts = {
        "passed": suite.passed,
        "panel_agreement": round(suite.panel_agreement, 3),
        "gates": [{"name": g.name, "passed": g.passed, "detail": g.detail} for g in suite.gates],
        "scenarios": [
            {
                "id": s.scenario_id,
                "title": s.title,
                "criteria": {k: v.display for k, v in s.per_criterion.items()},
                "failed": [k for k, v in s.per_criterion.items() if v.failed],
            }
            for s in suite.scenarios
        ],
        "contested": [t.transcript_id for t in suite.review_queue],
    }
    system = (
        "You are summarising a completed evaluation run for an engineer. These results are "
        "final and were computed deterministically: report them, do not re-judge them.\n\n"
        "Vocabulary you must use correctly:\n"
        "- 'contested' means the JUDGE PANEL DISAGREED WITH EACH OTHER on that transcript. "
        "It does NOT mean a human reviewed it or found a problem; no human has looked yet. "
        "A contested transcript is being ROUTED to a human precisely because the automated "
        "scores are unreliable there.\n"
        "- a failing gate is an absolute rule being broken, not a low average.\n\n"
        "Write 4-6 sentences of plain prose. Say what failed, in which scenario, and what a "
        "reviewer should open first. No preamble, no bullet points, no restating the JSON."
    )
    try:
        return complete(
            ORCHESTRATOR, system, [{"role": "user", "content": json.dumps(facts, indent=2)}], max_tokens=400
        ).strip()
    except Exception as exc:  # noqa: BLE001 - the summary is a nicety, never a dependency
        return f"_(summary unavailable: {type(exc).__name__})_"


def save_results(suite: SuiteResult) -> Path:
    RESULTS.write_text(json.dumps(asdict(suite), indent=2, default=str), encoding="utf-8")
    return RESULTS
