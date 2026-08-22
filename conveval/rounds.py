"""Executing the orchestrator's decisions, and recording what happened.

`orchestrator.py` decides WHAT should happen next and knows nothing about MLflow,
conversations or judges. This module is the other half: it carries those decisions out
and writes the result down. The split is deliberate - it keeps the loop's logic, the
part worth being sure about, testable with a fake executor and no model calls.
"""

from __future__ import annotations

from functools import partial

import mlflow
from mlflow.entities import AssessmentSource, AssessmentSourceType

from conveval.evaluation import (
    _COLLECTED,
    _rebuild,
    feedback_for,
    run_scenario,
)
from conveval.llm import ORCHESTRATOR, WIDENING_JUDGES, JUDGES, widened_panel
from conveval.models import Criterion
from conveval.aggregate import collapse_judges
from conveval.orchestrator import (
    RESAMPLE_K,
    SOURCE_RUN,
    RoundOutcome,
    RoundRecord,
    ScenarioState,
)
from conveval.scenarios import CRITERIA
from conveval.verify import verify_verdicts


def collected_states() -> dict[str, ScenarioState]:
    """Build the loop's starting state from round 0, as captured by the scorers."""
    states: dict[str, ScenarioState] = {}
    for (scenario_id, _run), (transcript, verdicts) in sorted(_COLLECTED.items()):
        st = states.setdefault(scenario_id, ScenarioState(scenario_id, list(JUDGES)))
        st.records.append(RoundRecord(transcript, verdicts, round_no=0))
    return states


def _log_verdicts(trace_id: str, criteria: list[Criterion], verdicts: list) -> None:
    """Attach per-judge and consensus assessments to a trace.

    Extra-round conversations are traced by `run_scenario` but score nothing on their
    own - they never pass through `genai.evaluate`'s scorers. Without this they would
    appear in the UI as bare transcripts with no verdicts attached, which looks like a
    bug and hides the evidence the round was run to gather.
    """
    for c in criteria:
        for fb in feedback_for(c, [v for v in verdicts if v.criterion == c.key]):
            mlflow.log_feedback(
                trace_id=trace_id,
                name=fb.name,
                value=fb.value,
                source=fb.source,
                rationale=fb.rationale,
                metadata=fb.metadata,
            )


def make_executor():
    """Build the `execute` callable the control loop drives.

    Honours the contract stated in `run_control_loop`, and the two branches differ in
    exactly the way that contract warns about:

      resample     new runs of the scenario  -> returns NEW RoundRecords
      widen_panel  more judges on runs that already exist -> appends verdicts IN PLACE
                   and returns nothing
    """
    from conveval.judges import run_panel

    def execute(scenario_id: str, action: str, state: ScenarioState) -> list[RoundRecord]:
        round_no = state.rounds_used + 1

        if action == "resample":
            new: list[RoundRecord] = []
            for i in range(RESAMPLE_K):
                run_no = state.n_transcripts + 1 + i
                outputs = run_scenario(scenario_id, run_no, round_no=round_no)
                trace_id = mlflow.get_last_active_trace_id()
                transcript = _rebuild(outputs, scenario_id, run_no)
                verdicts = run_panel(transcript, CRITERIA, state.panel)
                verify_verdicts(transcript, verdicts, CRITERIA)
                if trace_id:
                    _log_verdicts(trace_id, CRITERIA, verdicts)
                new.append(RoundRecord(transcript, verdicts, round_no))
            return new

        # widen_panel: re-judge the CONTESTED criteria only, on the runs already in
        # hand, with the two new judges. Re-judging every criterion would cost several
        # times more for opinions already gathered.
        entry = state.entry()
        contested = [
            c for c in CRITERIA if entry["criteria"].get(c.key, {}).get("contested")
        ]
        if not contested:
            contested = list(CRITERIA)

        for rec in state.records:
            extra = run_panel(rec.transcript, contested, WIDENING_JUDGES)
            verify_verdicts(rec.transcript, extra, contested)
            rec.verdicts.extend(extra)
        state.panel = widened_panel()
        return []

    return execute


@mlflow.trace(name="orchestrator_round", span_type="AGENT")
def trace_round(outcome: RoundOutcome) -> dict:
    """One round of the ladder, as its own trace.

    Recorded so the LOOP is visible and not merely its result. A reviewer looking at a
    scenario with three runs should be able to see that a second round happened, what
    it cost, and what the orchestrator was choosing between when it asked for it.
    """
    span = mlflow.get_current_active_span()
    if span:
        span.set_attributes({"model": ORCHESTRATOR.model, "role": "orchestrator/router"})
    return {
        "round": outcome.round_no,
        "calls_spent": outcome.calls_spent,
        "budget_remaining": outcome.budget_remaining,
        "decisions": {
            sid: {
                "action": d["action"],
                "legal_actions": d["legal_actions"],
                "was_fallback": d["was_fallback"],
                "why": d["why"],
            }
            for sid, d in outcome.decisions.items()
        },
    }


def _round0_traces(experiment_id: str, run_id: str) -> dict:
    """Scenario id -> the round-0 trace object, assessments included.

    The whole trace rather than just its id, because overwriting a stale consensus
    means finding the assessment that already holds it.
    """
    out: dict = {}
    for t in mlflow.search_traces(locations=[experiment_id], return_type="list"):
        tags = t.info.tags or {}
        if (t.info.trace_metadata or {}).get(SOURCE_RUN) != run_id:
            continue
        if tags.get("round") == "0" and tags.get("scenario"):
            out[tags["scenario"]] = t
    return out


def finalise_scenarios(
    experiment_id: str, run_id: str, states: dict[str, ScenarioState], final: dict
) -> None:
    """Write the post-ladder truth back onto each scenario's round-0 trace.

    The round-0 consensus assessment is a ONE-RUN verdict, and after resampling it is
    stale - while still being the first thing a reviewer opens. It is overwritten with
    the result collapsed across every round, carrying `rounds`, `n_runs` and
    `panel_size` so the number can be read for what it actually is.
    """
    by_scenario = _round0_traces(experiment_id, run_id)

    for scenario_id, state in states.items():
        trace = by_scenario.get(scenario_id)
        if trace is None:
            continue
        trace_id = trace.info.trace_id
        entry = state.entry()
        decision = final.get(scenario_id) or {}

        # The round-0 consensus assessments already on this trace, by criterion.
        # `log_feedback` APPENDS, so writing the collapsed result without this leaves
        # the trace showing two `pedagogy` rows with different values and no way to
        # tell which one is current. `override_feedback` supersedes instead - and
        # keeps the superseded one visible, which is the better record anyway: it
        # shows the one-run verdict that made the orchestrator resample.
        superseded = {
            a.name: a.assessment_id
            for a in (trace.info.assessments or [])
            if a.source and a.source.source_id == "panel-consensus"
        }

        for key, info in entry["criteria"].items():
            # Binary criteria log pass/fail; ordinal criteria log a NUMBER. The
            # human-readable spread lives in the rationale below. Logging the display
            # string as the value looks fine in the UI and breaks everything that
            # tries to aggregate it, MLflow's own metrics included.
            value = (
                ("fail" if info["failed"] else "pass")
                if info["kind"] == "binary"
                else round(float(info["value"]), 2)
            )
            write = (
                partial(mlflow.override_feedback, assessment_id=superseded[key])
                if key in superseded
                else partial(mlflow.log_feedback, name=key)
            )
            write(
                trace_id=trace_id,
                value=value,
                source=AssessmentSource(
                    source_type=AssessmentSourceType.CODE, source_id="panel-consensus"
                ),
                rationale=(
                    f"{info['consensus']} across {entry['n_transcripts']} run(s), "
                    f"{entry['panel_size']}-judge panel. "
                    f"Contested in {info['contested_runs']} run(s). "
                    f"Mean agreement {info['agreement']:.0%}."
                ),
                metadata={
                    "contested": str(info["contested"]).lower(),
                    "failed": str(info["failed"]).lower(),
                    "unverified_judges": ",".join(info["unverified_judges"]) or "none",
                    "rounds": str(state.rounds_used),
                    "n_runs": str(entry["n_transcripts"]),
                    "panel_size": str(entry["panel_size"]),
                },
            )

        if not decision:
            continue

        note = f"Legal actions were: {', '.join(decision['legal_actions'])}."
        if decision["was_fallback"]:
            note += " The model's pick was discarded and this is the fallback."
        question = decision.get("question") or ""
        rationale = f"{decision['why']}\n\n{note}"
        if question:
            rationale += f"\n\nREVIEWER MUST ANSWER: {question}"

        mlflow.log_feedback(
            trace_id=trace_id,
            name="triage",
            value=decision["action"],
            source=AssessmentSource(
                source_type=AssessmentSourceType.LLM_JUDGE, source_id="orchestrator"
            ),
            rationale=rationale,
            metadata={
                "rounds_used": str(state.rounds_used),
                # A run where the model's pick was discarded every time is a broken
                # orchestrator wearing a working one's output. Nothing else records it.
                "was_fallback": str(decision["was_fallback"]).lower(),
                "legal_actions": ",".join(decision["legal_actions"]),
            },
        )
        mlflow.set_trace_tag(trace_id, "triage", decision["action"])
        mlflow.set_trace_tag(trace_id, "rounds_used", str(state.rounds_used))
        if question:
            mlflow.set_trace_tag(trace_id, "review_question", question[:250])

        _refresh_flags(trace_id, state, entry, decision)


def _round0_contested(state: ScenarioState) -> set[str]:
    """Which criteria were contested on the FIRST sample alone."""
    if not state.records:
        return set()
    first = collapse_judges(state.records[0].transcript, CRITERIA, state.records[0].verdicts)
    return {k for k, c in first.per_criterion.items() if c.contested}


def _refresh_flags(trace_id: str, state: ScenarioState, entry: dict, decision: dict) -> None:
    """Rewrite the review flags from the POST-ladder truth.

    `flag_for_review` runs before the ladder, so its tags describe the first sample.
    Left alone they go stale in the most misleading direction available: on a real run
    `unfaithful` was tagged `contested_criteria=in_scenario` while its final consensus
    was a clean pass, because two extra runs had settled it. The tag denied the very
    thing the extra rounds accomplished.

    `resolved_by_gathering` is the positive record of that: criteria contested on one
    sample and no longer contested after the ladder. It is the evidence that spending
    compute instead of a reviewer actually worked.
    """
    contested = sorted(k for k, v in entry["criteria"].items() if v["contested"])
    failed = sorted(k for k, v in entry["criteria"].items() if v["failed"])
    resolved = sorted(_round0_contested(state) - set(contested))
    needs_review = decision["action"] in ("human_tiebreak", "human_confirm", "judge_defect")

    for key, value in (
        ("contested_criteria", ",".join(contested)),
        ("failed_criteria", ",".join(failed)),
        ("resolved_by_gathering", ",".join(resolved)),
        ("contested", str(bool(contested)).lower()),
        ("failed", str(bool(failed)).lower()),
        ("needs_review", str(needs_review).lower()),
    ):
        mlflow.set_trace_tag(trace_id, key, value)
