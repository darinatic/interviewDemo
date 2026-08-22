"""The orchestrator agent: read the panel's output, decide what happens next.

For most of this project's life the orchestrator only narrated a finished scorecard,
which is a thin job for something sitting at the top of the topology. It now has two,
and only the first one matters:

  TRIAGE     per flagged trace, decide WHAT HAPPENS NEXT and write the one question a
             human has to answer. This is real routing: it changes the work queue.
  SUMMARISE  narrate the run. Genuinely a nicety.

What it deliberately does NOT do is supply a verdict. Letting a model stand in for the
human reviewer would dissolve the argument the whole suite rests on - that a contested
score is unreliable and needs a human - by answering it with another unreliable score
from the same class of system. So the split is:

    code decides WHICH ACTIONS ARE LEGAL for a trace   (deterministic, auditable)
    the model picks ONE of them and writes the question (judgement, but bounded)

The model cannot widen its own authority: a reply naming an action outside the legal
set is discarded and the highest-priority legal action is used instead. That property
is worth more than the model's judgement is, and it is cheap to enforce.

The scoreboard here also exists because `result.metrics` is NOT the run. MLflow can
only aggregate numeric assessments, so the two binary criteria - which carry the
values "pass"/"fail" - produce no metric at all and were silently missing from
everything downstream, including the summary the orchestrator wrote. This module
reads the assessments themselves.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Callable

import mlflow
from mlflow.entities import AssessmentSource, AssessmentSourceType

from conveval.aggregate import collapse_judges, collapse_runs
from conveval.llm import (
    JUDGES,
    ORCHESTRATOR,
    WIDENING_JUDGES,
    Role,
    complete,
    complete_json,
)
from conveval.models import Transcript, TranscriptResult, Verdict
from conveval.scenarios import CRITERIA, TURNS_PER_CONVERSATION

# --------------------------------------------------------------------------
# The escalation ladder
#
# Cheapest resource first. The orchestrator walks the ladder and stops as early as it
# honestly can:
#
#     accept              free
#       |  still uncertain?
#     resample            spend compute
#       |  still uncertain?
#     widen panel 3 -> 5  spend more compute
#       |  still uncertain?
#     ask a human         spend the scarce thing
#
# The ladder is SEQUENTIAL and each rung needs its own round, which is why two extra
# rounds are allowed rather than one.
# --------------------------------------------------------------------------

#: Extra rounds beyond the initial pass. Round 1 can resample, round 2 can widen the
#: panel; after that only accept or route remain, so the loop terminates by
#: construction. The model cannot grant itself another round - this is read by code.
MAX_EXTRA_ROUNDS = 2

#: Extra runs added by a single `resample`. Three total runs of a resampled scenario:
#: the smallest odd count above one, and enough for `pass_rate` and `mean + interval`
#: to mean something.
RESAMPLE_K = 2

#: Hard ceiling on EXTRA model calls per suite, over and above the initial pass.
#: A full ladder for one scenario costs at most 34 + 6 = 40, so this funds two complete
#: ladders - which matches the suite, where at most two of three scenarios are ever
#: flagged. A third contested scenario finds `resample` illegal on budget grounds and
#: routes to a human, which is the correct behaviour when compute has run out.
EXTRA_CALL_BUDGET = int(os.getenv("CONVEVAL_EXTRA_CALL_BUDGET", "80"))

#: Priority order. This does double duty and both uses matter:
#:   1. it is the FALLBACK when the model returns something illegal, and
#:   2. it is the order a reviewer should work the queue in.
#:
#: `judge_defect` outranks everything because it is a fault in the INSTRUMENT. There is
#: no point asking a human to adjudicate a disagreement between judges when one of them
#: cited a span that is not in the transcript - the score is not merely uncertain, it
#: was arrived at from something that did not happen.
#:
#: Cheap evidence-gathering precedes human routing, which REVERSES the earlier default.
#: That is safe only because rounds are capped: falling back to `resample` cannot loop,
#: it runs one more round and re-decides. Without the cap this ordering would be a bug.
ACTIONS: dict[str, str] = {
    "judge_defect": (
        "A judge cited evidence that does not appear in the transcript. The instrument "
        "is at fault, not necessarily the agent. Fix or drop that judge before "
        "trusting this row."
    ),
    "resample": (
        "The panel disagreed on a single sample. Two more runs cost compute; a "
        "reviewer costs attention. Gather more evidence before spending the scarcer "
        "resource."
    ),
    "widen_panel": (
        "More samples did not settle it. Add two more judges from two more model "
        "families and re-judge the contested criterion. Still cheaper than a human, "
        "and a wider panel is a better instrument."
    ),
    "human_tiebreak": (
        "The panel disagreed and more evidence will not help. The automated score is "
        "unreliable and a human decides."
    ),
    "human_confirm": (
        "The panel agreed the agent did badly. The score is reliable; a human confirms "
        "the defect is real before it is treated as a regression."
    ),
    "accept": (
        "Nothing contested, nothing failed, no unverifiable evidence. Take the score."
    ),
}


@dataclass
class Budget:
    """A hard ceiling on extra model calls, spent as the ladder is climbed.

    Enforced in CODE and checked when an action's legality is computed, so an
    unaffordable action is never offered. The model does not see it and therefore
    cannot argue for it - which is the difference between a budget and a suggestion.
    """

    total: int = EXTRA_CALL_BUDGET
    spent: int = 0

    def remaining(self) -> int:
        return max(0, self.total - self.spent)

    def can_afford(self, cost: int) -> bool:
        return cost <= self.remaining()

    def spend(self, cost: int) -> None:
        self.spent += cost


def resample_cost(entry: dict) -> int:
    """Projected calls for one resample: the conversations plus judging them."""
    conversations = RESAMPLE_K * TURNS_PER_CONVERSATION * 2  # learner + agent per turn
    judging = RESAMPLE_K * len(CRITERIA) * entry["panel_size"]
    return conversations + judging


def widen_cost(entry: dict) -> int:
    """Projected calls for widening: two new judges, contested criteria only.

    Re-judging every criterion would cost several times more for information already
    in hand.
    """
    contested = sum(1 for v in entry["criteria"].values() if v["contested"])
    return len(WIDENING_JUDGES) * entry["n_transcripts"] * max(1, contested)


# --------------------------------------------------------------------------
# Reading the run back out of MLflow
# --------------------------------------------------------------------------

def _consensus_assessments(trace) -> list:
    """Panel-consensus assessments only, not the per-judge ones.

    The per-judge assessments are named `<criterion>__<model>`; the consensus is named
    `<criterion>` and carries the aggregation metadata. Filtering on the metadata
    rather than on the name shape keeps this working if the naming changes.
    """
    return [
        a
        for a in (trace.info.assessments or [])
        if a.metadata and "contested" in a.metadata
        # Superseded assessments are RETAINED by MLflow with valid=False when
        # `override_feedback` replaces them - the audit trail is the point, and here it
        # preserves the one-run verdict that made the orchestrator resample. But they
        # must not be read as current: without this filter the stale and the final
        # consensus both match, and which one wins is down to iteration order.
        and getattr(a, "valid", True) is not False
    ]


#: MLflow stamps this on every trace produced inside a run. It is the only link back
#: from a trace to the evaluation that created it, and without filtering on it every
#: read below silently includes traces from PREVIOUS runs - so the second run of a
#: suite would re-triage the first run's rows and report a scoreboard spanning both.
SOURCE_RUN = "mlflow.sourceRun"


def panel_digest(experiment_id: str, run_id: str | None = None) -> list[dict]:
    """One entry per evaluated trace: what the panel concluded and how firmly.

    This is the run as a reviewer would describe it, which is not the same object as
    `result.metrics` - see the module docstring.
    """
    digest: list[dict] = []
    for t in mlflow.search_traces(locations=[experiment_id], return_type="list"):
        tags = t.info.tags or {}
        if run_id and (t.info.trace_metadata or {}).get(SOURCE_RUN) != run_id:
            continue
        # Round-0 traces only: one entry per SCENARIO, not one per conversation.
        # Extra-round conversations are evidence gathered for a scenario, not scenarios
        # of their own - counting them made a 3-scenario suite report "4/7 scenarios
        # passed", which is both wrong and quietly plausible.
        if tags.get("round", "0") != "0":
            continue
        cons = _consensus_assessments(t)
        if not cons:
            continue  # the orchestrator's own trace, and anything not a scenario row

        votes: dict[str, dict[str, str]] = {}
        for a in t.info.assessments or []:
            if "__" not in a.name:
                continue
            criterion, judge = a.name.split("__", 1)
            votes.setdefault(criterion, {})[judge] = str(a.value)

        criteria = {}
        for a in cons:
            m = a.metadata or {}
            criteria[a.name] = {
                "consensus": str(a.value),
                "judges": votes.get(a.name, {}),
                "agreement": float(m.get("agreement", 0.0)),
                "contested": m.get("contested") == "true",
                "failed": m.get("failed") == "true",
                "unverified_judges": [
                    j for j in m.get("unverified_judges", "none").split(",")
                    if j and j != "none"
                ],
            }

        digest.append({
            "trace_id": t.info.trace_id,
            "scenario": tags.get("scenario", "?"),
            "run": int(tags.get("run", 1)),
            "criteria": criteria,
        })
    return digest


def scoreboard(digest: list[dict]) -> dict[str, str]:
    """Per-criterion result across scenarios, INCLUDING the binary ones.

    Binary criteria report as "2/3 scenarios passed" rather than as a mean: a mean of
    pass and fail is not a quantity, and reporting one is how a suite ends up claiming
    0.67 correctness when it hallucinated in a third of its runs.
    """
    out: dict[str, str] = {}
    for c in CRITERIA:
        values = [
            d["criteria"][c.key]["consensus"] for d in digest if c.key in d["criteria"]
        ]
        if not values:
            continue
        if c.kind == "binary":
            passed = sum(1 for v in values if v == "pass")
            out[c.key] = f"{passed}/{len(values)} scenarios passed"
        else:
            nums = [float(v) for v in values]
            out[c.key] = (
                f"median {sorted(nums)[len(nums) // 2]:g} "
                f"(range {min(nums):g}-{max(nums):g}) across {len(nums)} scenarios"
            )
    return out


# --------------------------------------------------------------------------
# Triage
# --------------------------------------------------------------------------

def legal_actions(entry: dict, rounds_used: int, budget: Budget) -> list[str]:
    """Which actions are permissible for this scenario. Deterministic, no model.

    This function is the whole integrity argument. Code decides what MAY happen; the
    model only chooses among what code already permitted. Everything the model could
    otherwise talk its way into - another round, a bigger panel, a spend it cannot
    afford - is settled here, before it is consulted.

    Returned in `ACTIONS` priority order, so `[0]` is also the safe fallback.

    `entry` carries `panel_size` and `n_transcripts` because both are properties of the
    scenario's CURRENT state and both change as the ladder is climbed.
    """
    contested = [k for k, v in entry["criteria"].items() if v["contested"]]
    failed = [k for k, v in entry["criteria"].items() if v["failed"]]
    unverified = [k for k, v in entry["criteria"].items() if v["unverified_judges"]]

    if not (contested or failed or unverified):
        return ["accept"]

    acts: list[str] = []
    if unverified:
        acts.append("judge_defect")
    # Rung 1. Only on the first extra round, and only if the spend fits.
    if contested and rounds_used == 0 and budget.can_afford(resample_cost(entry)):
        acts.append("resample")
    # Rung 2. Only after resampling failed to settle it, and only once - the
    # panel_size guard is what stops a second widening producing a 7-judge panel.
    if (
        contested
        and rounds_used == 1
        and entry["panel_size"] == len(JUDGES)
        and budget.can_afford(widen_cost(entry))
    ):
        acts.append("widen_panel")
    if contested:
        acts.append("human_tiebreak")
    if failed:
        acts.append("human_confirm")
    return acts


_TRIAGE_SYSTEM = """\
You are the orchestrator of an automated evaluation pipeline. A panel of LLM judges has
scored a transcript and deterministic code has flagged it. Your job is to route it, not
to re-score it.

You will be given the panel's result and the list of actions that are LEGAL for this
scenario. You must choose one of those actions and nothing else.

SPEND THE CHEAPEST RESOURCE THAT COULD SETTLE IT. A reviewer's attention is the
scarcest thing in this system; compute is not. If gathering more evidence is legal and
could plausibly resolve the disagreement, do that FIRST and route to a human only when
more evidence will not help. Routing a one-sample disagreement straight to a person is
premature - you have not yet found out whether it was noise.

Vocabulary, which you must use correctly:
- "contested" means the JUDGES DISAGREED WITH EACH OTHER. It does not mean the agent
  did badly. The score is unreliable and that is the problem.
- "failed" means the judges AGREED the agent did badly. That score is reliable, and
  more sampling will not change it - it needs confirming, not re-measuring.
- "unverified" means a judge quoted text that is not in the transcript. That is a fault
  in the judge, and no amount of extra sampling fixes a broken instrument.

Return JSON: {"action": "<one of the legal actions>", "question": "<the single
question the reviewer must answer, phrased so it can be answered yes or no>", "why":
"<one sentence, referring to the specific disagreement or failure>"}

If the action gathers more evidence rather than routing to a person, "question" is what
a reviewer WOULD be asked if it still needs one afterwards.
The question must be about THIS scenario and cite what the judges actually split on.
"why" must not restate the scores back.
"""


# --------------------------------------------------------------------------
# Scenario state: what the loop carries between rounds
# --------------------------------------------------------------------------

@dataclass
class RoundRecord:
    """One run of one scenario: the conversation and every judge's verdict on it."""

    transcript: Transcript
    verdicts: list[Verdict]
    round_no: int


@dataclass
class ScenarioState:
    """Everything the loop knows about one scenario, accumulated across rounds.

    Held in MEMORY rather than re-read from MLflow between rounds. Reading it back
    would mean reconstructing `Verdict` objects out of assessment rationales, which is
    lossy and pointless when the objects are already in hand.
    """

    scenario_id: str
    panel: list[Role]
    records: list[RoundRecord] = field(default_factory=list)
    rounds_used: int = 0

    @property
    def n_transcripts(self) -> int:
        return len(self.records)

    def per_transcript(self) -> list[TranscriptResult]:
        return [collapse_judges(r.transcript, CRITERIA, r.verdicts) for r in self.records]

    def entry(self) -> dict:
        """The dict `legal_actions` and `triage_trace` consume.

        Two judgements are baked in here and both are worth defending:

        CONTESTED needs a STRICT MAJORITY of runs to be contested. With one run, one
        disagreement makes it contested. With three, a single contested run out of
        three means the extra evidence did its job and the scenario is no longer
        uncertain - which is precisely the outcome resampling exists to produce. Using
        "contested in ANY run" instead would make resampling incapable of ever
        resolving anything, and the ladder would be theatre.

        FAILED comes from `collapse_runs`, so it uses the criterion's own across-runs
        function (pass_rate / mean+interval / any_failure) rather than a second,
        divergent notion of failure invented here.

        UNVERIFIED is any run. A judge citing a span that does not exist is an
        instrument fault, and one occurrence is enough to distrust the instrument.
        """
        results = self.per_transcript()
        n = len(results)
        criteria: dict[str, dict] = {}

        for c in CRITERIA:
            per_run = [r.per_criterion[c.key] for r in results if c.key in r.per_criterion]
            if not per_run:
                continue
            rolled = collapse_runs(c, results)
            n_contested = sum(1 for x in per_run if x.contested)
            unverified = sorted({j for x in per_run for j in x.unverified_judges})
            latest = per_run[-1]
            criteria[c.key] = {
                # `consensus` is the DISPLAY form ("2.7 +/- 0.3", "2/3 runs") because
                # it is what the orchestrator model reads, and the spread is exactly
                # what it needs to judge whether more evidence would help.
                # `value` is the machine form, kept separate: assessments must carry
                # something aggregatable, and a display string logged as a value made
                # the scoreboard fall over trying to average "2.7 +/- 0.3".
                "consensus": rolled.display,
                "value": rolled.value,
                "kind": c.kind,
                "judges": {v.judge: str(v.score) for v in latest.verdicts},
                "agreement": round(
                    sum(x.agreement for x in per_run) / len(per_run), 3
                ),
                "contested": n_contested * 2 > n,
                "contested_runs": f"{n_contested}/{n}",
                "failed": rolled.failed,
                "unverified_judges": unverified,
            }

        return {
            "scenario": self.scenario_id,
            "criteria": criteria,
            "n_transcripts": self.n_transcripts,
            "panel_size": len(self.panel),
        }


# --------------------------------------------------------------------------
# The control loop
# --------------------------------------------------------------------------

@dataclass
class RoundOutcome:
    """What the orchestrator decided in one round, and what it cost."""

    round_no: int
    decisions: dict[str, dict]
    calls_spent: int
    budget_remaining: int


def run_control_loop(
    states: dict[str, ScenarioState],
    budget: Budget,
    execute: Callable[[str, str, ScenarioState], list[RoundRecord]],
    on_round: Callable[[RoundOutcome], None] | None = None,
) -> tuple[dict[str, dict], list[RoundOutcome]]:
    """Walk the escalation ladder until every scenario is accepted or routed.

    `execute` performs a gathering action and is injected rather than imported, so this
    function knows nothing about MLflow, conversations or judges - which keeps the
    loop's logic, the part worth being sure about, testable with a fake executor and no
    model calls.

    ITS CONTRACT, because the two actions differ and the difference is easy to get
    wrong:

      resample     produces NEW runs of the scenario. Returns the new RoundRecords;
                   the loop appends them.
      widen_panel  adds judges to runs that ALREADY EXIST. Appends verdicts to the
                   records in `state.records` IN PLACE, updates `state.panel`, and
                   returns an empty list.

    Returning new records for a widening would double-count the transcripts; appending
    verdicts for a resample would attribute one run's judging to another. Both are
    silent corruptions of the scorecard, so the contract is stated rather than implied.

    Termination is structural, not hopeful: `legal_actions` stops offering gathering
    actions once `rounds_used` reaches MAX_EXTRA_ROUNDS, so the `while` below cannot
    spin. The explicit range bound is belt and braces.

    Returns (final decision per scenario, one outcome per round).
    """
    outcomes: list[RoundOutcome] = []
    final: dict[str, dict] = {}

    for round_no in range(MAX_EXTRA_ROUNDS + 1):
        spent_before = budget.spent
        decisions = {
            sid: triage_trace(st.entry(), st.rounds_used, budget)
            for sid, st in states.items()
            if sid not in final
        }

        gathering = {
            sid: d for sid, d in decisions.items()
            if d["action"] in ("resample", "widen_panel")
        }
        # Anything not gathering is settled: record it and stop asking about it.
        for sid, d in decisions.items():
            if sid not in gathering:
                final[sid] = d

        for sid, d in gathering.items():
            st = states[sid]
            cost = resample_cost(st.entry()) if d["action"] == "resample" else widen_cost(st.entry())
            budget.spend(cost)
            st.records.extend(execute(sid, d["action"], st))
            st.rounds_used += 1

        outcome = RoundOutcome(
            round_no=round_no,
            decisions=decisions,
            calls_spent=budget.spent - spent_before,
            budget_remaining=budget.remaining(),
        )
        outcomes.append(outcome)
        if on_round:
            on_round(outcome)

        if not gathering:
            break

    # Anything still unsettled after the last round: ask once more, with gathering now
    # illegal, so it resolves to accept or a human rather than being left dangling.
    for sid, st in states.items():
        if sid not in final:
            final[sid] = triage_trace(st.entry(), st.rounds_used, budget)

    return final, outcomes


_DEFAULT_QUESTION = "Do you agree with the panel on this scenario?"


def triage_trace(entry: dict, rounds_used: int, budget: Budget) -> dict:
    """Decide what happens next for one scenario.

    Returns a decision dict always - `accept` when there is nothing wrong, which is the
    common case and costs no model call. `was_fallback` records whether the model's own
    pick survived: a run where it was discarded every time is a broken orchestrator
    wearing a working one's output, and nothing else in the record would reveal that.
    """
    acts = legal_actions(entry, rounds_used, budget)
    if acts == ["accept"]:
        return {
            "action": "accept",
            "question": "",
            "why": "Nothing contested, nothing failed, no unverifiable evidence.",
            "legal_actions": acts,
            "was_fallback": False,
        }

    facts = {
        "scenario": entry["scenario"],
        "round": rounds_used,
        "runs_so_far": entry["n_transcripts"],
        "panel_size": entry["panel_size"],
        "criteria": entry["criteria"],
        "legal_actions": {a: ACTIONS[a] for a in acts},
    }
    try:
        reply = complete_json(
            ORCHESTRATOR,
            _TRIAGE_SYSTEM,
            [{"role": "user", "content": json.dumps(facts, indent=2)}],
            max_tokens=400,
        )
    except Exception as exc:  # noqa: BLE001 - routing must survive a flaky model
        return {
            "action": acts[0],
            "question": _DEFAULT_QUESTION,
            "why": f"(orchestrator unavailable: {type(exc).__name__}; "
                   f"fell back to the highest-priority legal action)",
            "legal_actions": acts,
            "was_fallback": True,
        }

    picked = str(reply.get("action", "")).strip()
    # The model does not get to invent an action, and it does not get to pick one the
    # deterministic layer ruled out. Silently accepting an out-of-set reply would make
    # the audit trail a work of fiction.
    was_fallback = picked not in acts
    action = acts[0] if was_fallback else picked
    return {
        "action": action,
        "question": str(reply.get("question", "")).strip() or _DEFAULT_QUESTION,
        "why": str(reply.get("why", "")).strip(),
        "legal_actions": acts,
        "was_fallback": was_fallback,
    }


def triage_all(
    experiment_id: str, runs: int = 1, run_id: str | None = None
) -> list[tuple[str, str, dict]]:
    """Triage every flagged trace and write the decision back onto it.

    The decision is logged as an assessment rather than printed, so it lands in the
    same place as the judges' output: open a trace, see the scores, see what the
    pipeline decided to do about them, and answer the question in the same panel.
    """
    decided: list[tuple[str, str, dict]] = []
    for entry in panel_digest(experiment_id, run_id):
        decision = triage_trace(entry, runs)
        if not decision:
            continue
        trace_id = entry["trace_id"]
        mlflow.log_feedback(
            trace_id=trace_id,
            name="triage",
            value=decision["action"],
            # LLM_JUDGE is the accurate source type: a model produced this. It is a
            # routing decision rather than a score, and labelling it CODE would imply a
            # determinism it does not have.
            source=AssessmentSource(
                source_type=AssessmentSourceType.LLM_JUDGE, source_id="orchestrator"
            ),
            rationale=f"{decision['why']}\n\nREVIEWER MUST ANSWER: {decision['question']}",
        )
        mlflow.set_trace_tag(trace_id, "triage", decision["action"])
        mlflow.set_trace_tag(trace_id, "review_question", decision["question"][:250])
        decided.append((trace_id, entry["scenario"], decision))
    return decided


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------

_SUMMARY_SYSTEM = """\
You are summarising a completed evaluation run for an engineer. The results are final
and were computed deterministically: report them, do not re-judge them.

Vocabulary you must use correctly:
- "contested" means the JUDGE PANEL DISAGREED with each other. No human has looked yet.
- "failed" means the judges AGREED the agent did badly. That score is reliable.

You are given a result for EVERY criterion. Mention the ones that are interesting -
anything that failed, anything contested - and do not silently omit a criterion just
because it passed cleanly; say so in a clause.

Write AT MOST 4 short sentences. End with which trace a reviewer should open first and
why. No preamble, no bullet points.
"""


@mlflow.trace(name="orchestrator_summary", span_type="AGENT")
def orchestrate_summary(
    experiment_id: str, triaged: list | None = None, run_id: str | None = None
) -> str:
    """Narrate the run, from the assessments rather than from `result.metrics`.

    Traced so the orchestrator appears in the Traces list alongside the conversations
    it is reasoning about, which is what makes the agent topology browsable at all.
    """
    digest = panel_digest(experiment_id, run_id)
    facts = {
        "per_criterion": scoreboard(digest),
        "routed_for_review": [
            {"scenario": s, "action": d["action"], "why": d["why"]}
            for _, s, d in (triaged or [])
        ],
    }
    span = mlflow.get_current_active_span()
    if span:
        span.set_attributes({"model": ORCHESTRATOR.model, "role": "orchestrator/summariser"})
    try:
        return complete(
            ORCHESTRATOR,
            _SUMMARY_SYSTEM,
            [{"role": "user", "content": json.dumps(facts, indent=2)}],
            max_tokens=300,
        ).strip()
    except Exception as exc:  # noqa: BLE001 - the summary is never a dependency
        return f"(summary unavailable: {type(exc).__name__})"
