"""MLflow GenAI evaluation: one run, one row per scenario, everything attached.

Built on `mlflow.genai.evaluate()` rather than the generic tracking API. The
difference matters and it is the whole reason this module exists:

  generic tracking  -> a tree of separate runs (suite / scenario / judge). Browsable,
                       but nothing connects; a reviewer has to reassemble the picture.
  genai.evaluate    -> ONE run, one ROW per scenario. Each row carries the trace of
                       the conversation, and every judge's assessment hangs off that
                       same row. Click a row and you have the complete picture.

Each criterion is a `@scorer` returning a LIST of `Feedback`: one per judge, attributed
with `AssessmentSource(LLM_JUDGE, source_id=<model>)`, plus a consensus. Emitting the
individual judges rather than only the consensus is the point - an aggregate that hides
a 2-1 split is exactly what this project argues against, and the UI can only show a
split that was logged.

Human review is real here, not a stand-in: judges are registered so they appear in the
Judges tab with their rubric, and traces whose panel disagreed are pushed into a review
queue automatically.
"""

from __future__ import annotations

import os
from pathlib import Path

import mlflow
from mlflow.entities import AssessmentSource, AssessmentSourceType, Feedback
from mlflow.genai import scorer
from mlflow.genai.scorers import Safety

from conveval.agent import PromptedCoach
from conveval.aggregate import consensus_for_criterion
from conveval.conversation import run_conversation
from conveval.llm import BASE_URL, JUDGES, ORCHESTRATOR, SUT
from conveval.models import Criterion, Transcript, Turn
from conveval.orchestrator import SOURCE_RUN
from conveval.scenarios import CRITERIA, SCENARIOS
from conveval.verify import verify_verdicts

EXPERIMENT = "conversation-eval"
ROOT = Path(__file__).resolve().parent.parent
#: SQLite, not ./mlruns. MLflow 3.x puts the filesystem backend in "maintenance mode"
#: and refuses it unless MLFLOW_ALLOW_FILE_STORE=true - a deprecated path that will not
#: receive new features. SQLite is one file, needs no server, and is what
#: `mlflow migrate-filestore` migrates toward.
#: The UI must be started with the SAME URI: plain `mlflow ui` reads ./mlruns and shows
#: an empty experiment list.
TRACKING_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
REVIEW_QUEUE = "contested-and-failed"


def configure(tracking_uri: str = TRACKING_URI) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT)
    _route_builtin_judges_via_openrouter()
    # NB: mlflow.langchain.autolog() is deliberately NOT used. It requires `langchain`
    # itself and this project depends only on `langgraph`. The turn loop carries
    # explicit spans instead (conversation.py) - no extra dependency, and it shows the
    # steps worth showing rather than every framework internal.


def _route_builtin_judges_via_openrouter() -> None:
    """Point MLflow's built-in judges at OpenRouter.

    They resolve an `openai:/...` model through the OpenAI SDK, which reads
    OPENAI_API_KEY. This project has only an OpenRouter key by design, so without this
    every built-in scorer fails with "OPENAI_API_KEY environment variable not set".
    OpenRouter is OpenAI-API-compatible, so redirecting the SDK's base URL is enough.
    """
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        return
    os.environ.setdefault("OPENAI_API_KEY", key)
    os.environ.setdefault("OPENAI_BASE_URL", BASE_URL)
    os.environ.setdefault("OPENAI_API_BASE", BASE_URL)


def _experiment_id() -> str:
    return mlflow.get_experiment_by_name(EXPERIMENT).experiment_id


# --------------------------------------------------------------------------
# Registered judges: the rubric, visible in the UI
# --------------------------------------------------------------------------

def _judge_instructions(c: Criterion) -> str:
    scale = (
        "Reply 'pass' or 'fail'."
        if c.kind == "binary"
        else "Reply with a single integer from 1 to 5."
    )
    return (
        f"You are scoring a coaching transcript on ONE criterion: {c.label}.\n\n"
        f"{c.rubric}\n\n"
        "The conversation is in {{ outputs }} and the scenario brief the coach "
        "was grounded on is in {{ inputs }}.\n"
        f"{scale} Quote the verbatim span that justifies your score."
    )


def register_judges() -> list[str]:
    """Publish each criterion's rubric to the Judges tab.

    WHAT THIS IS NOT: the registered judges do NOT score anything in this pipeline.
    Scoring is done by `judges.run_panel`, which calls three models directly over
    OpenRouter - the whole argument here is a MULTI-MODEL panel, and a registered
    MLflow judge is one model with one rubric. Registration exists so the standard a
    score was given against is readable in the UI without opening the source. (The
    registry's other purpose, scheduled scoring of production traces, is not used.)

    Registration is SKIPPED when the instructions are byte-identical to what is already
    registered. Re-registering unconditionally mints a new version every run, so the
    version climbs while meaning nothing - these reached v3 without a single rubric
    edit. Skipping makes a bump mean "the yardstick changed", which is the question you
    need answered when a score moves.

    Returns the names newly registered or updated, so an empty list means "nothing
    changed" rather than "nothing happened".
    """
    try:
        from mlflow.genai.scorers import list_scorers

        existing = {
            j.name: getattr(j, "instructions", "")
            for j in list_scorers(experiment_id=_experiment_id())
        }
    except Exception:  # noqa: BLE001 - the registry is a nicety, never a gate
        existing = {}

    registered = []
    for c in CRITERIA:
        instructions = _judge_instructions(c)
        if existing.get(c.key, "").strip() == instructions.strip():
            continue  # unchanged rubric: do not mint a meaningless new version
        judge = mlflow.genai.make_judge(
            name=c.key,
            instructions=instructions,
            model=f"openai:/{JUDGES[0].model}",
        )
        try:
            judge.register(experiment_id=_experiment_id())
            registered.append(c.key)
        except Exception as exc:  # noqa: BLE001 - registration is a nicety, not a gate
            print(f"    ! could not register judge {c.key}: {type(exc).__name__}")
    return registered


# --------------------------------------------------------------------------
# The thing being evaluated
# --------------------------------------------------------------------------

@mlflow.trace(name="conversation", span_type="CHAIN")
def run_scenario(scenario_id: str, run: int = 1, round_no: int = 0) -> dict:
    """Run one scenario end to end. `mlflow.genai.evaluate` calls this once per row.

    Output keys are named for what a reviewer sees in the UI, not for what the code
    finds convenient: `conversation` and `scenario_brief` rather than `transcript` and
    `context`. The agent's concatenated turns are NOT returned - they are derivable
    from the conversation, and an extra near-duplicate blob only makes the row harder
    to read.
    """
    scenario = next(s for s in SCENARIOS if s.id == scenario_id)
    transcript = run_conversation(scenario, run, seed=1000 + run, sut=PromptedCoach())
    mlflow.update_current_trace(
        tags={
            "scenario": scenario_id,
            "scenario_title": scenario.title,
            "run": str(run),
            "sut_model": SUT.model,
            # Which round produced this conversation. Round 0 is the initial pass;
            # anything higher was gathered because the orchestrator asked for it, and
            # a reviewer needs to be able to tell those apart at a glance.
            "round": str(round_no),
            **({"resampled_from": scenario_id} if round_no else {}),
        }
    )
    return {
        "conversation": [{"speaker": t.role, "text": t.text} for t in transcript.turns],
        "scenario_brief": scenario.context,
    }


def _rebuild(outputs: dict, scenario_id: str, run: int) -> Transcript:
    """Reconstruct a Transcript from a row's outputs, for judging and verification."""
    return Transcript(
        scenario_id=scenario_id,
        run=run,
        seed=1000 + run,
        turns=[Turn(t["speaker"], t["text"]) for t in outputs["conversation"]],
        context=outputs["scenario_brief"],
    )


# --------------------------------------------------------------------------
# Scorers
# --------------------------------------------------------------------------

def _judge_rationale(v) -> str:
    """A rationale a human can act on without opening metadata.

    The citation belongs HERE, not only in metadata: the Feedback column shows the
    rationale, so a reviewer asking "on what basis?" should not have to go hunting.
    The verification result is included because an unverifiable citation means the
    JUDGE is the thing that went wrong.
    """
    parts = [v.reason.strip() or "(no reason given)"]
    if v.evidence.strip():
        mark = {True: "verified in transcript", False: "NOT FOUND IN TRANSCRIPT", None: ""}[
            v.evidence_verified
        ]
        quote = v.evidence.strip().replace("\n", " ")
        if len(quote) > 300:
            quote = quote[:300] + "..."
        parts.append(f'CITED: "{quote}"' + (f" [{mark}]" if mark else ""))
    else:
        # Say so rather than rendering a rationale that merely looks unremarkable.
        # An uncited verdict was previously indistinguishable from a cited one at a
        # glance, so a judge quietly citing nothing on two thirds of its verdicts went
        # unnoticed until the rationales were counted.
        parts.append("CITED: nothing - this judge gave no span, so its score cannot "
                     "be checked against the transcript.")
    return "\n".join(parts)


#: Round-0 verdicts, captured as the scorers run.
#:
#: The control loop needs the initial round's verdicts in memory, but `genai.evaluate`
#: owns the call into the scorers and returns only aggregated metrics - the Verdict
#: objects never surface. Re-judging to get them back would double the cost of every
#: run; re-reading them from assessment rationales would be lossy. So the scorer, which
#: has already computed them, drops them here on the way past.
#:
#: This is the one piece of module state in the pipeline. It is cleared at the start of
#: every evaluation so a second run in the same process cannot inherit the first one's
#: verdicts.
_COLLECTED: dict[tuple[str, int], tuple] = {}


def _collect(transcript: Transcript, verdicts: list) -> None:
    """Accumulate verdicts per (scenario, run).

    Called once per CRITERION, so verdicts accumulate across three calls for the same
    transcript before the loop ever sees them.
    """
    key = (transcript.scenario_id, transcript.run)
    existing = _COLLECTED.get(key)
    if existing is None:
        _COLLECTED[key] = (transcript, list(verdicts))
    else:
        existing[1].extend(verdicts)


def feedback_for(criterion: Criterion, verdicts: list) -> list[Feedback]:
    """Turn verdicts into the assessments a reviewer sees. No model calls.

    Split out from `_panel_feedback` because extra rounds already HAVE their verdicts
    and only need them rendered and logged - re-judging to produce assessments would
    pay for the same opinions twice.
    """
    out: list[Feedback] = [
        Feedback(
            name=f"{criterion.key}__{v.judge}",
            value=v.score,
            rationale=_judge_rationale(v),
            source=AssessmentSource(
                source_type=AssessmentSourceType.LLM_JUDGE, source_id=v.judge
            ),
            metadata={
                "family": v.provider,
                "evidence_verified": str(v.evidence_verified),
            },
        )
        for v in verdicts
    ]
    if not verdicts:
        return out

    cons = consensus_for_criterion(criterion, verdicts)
    rule = "majority vote" if criterion.kind == "binary" else "median"
    votes = ", ".join(f"{v.judge}={v.score}" for v in verdicts)
    verdict_line = (
        "CONTESTED - the panel disagreed, so this score is unreliable and the trace is "
        "queued for human review."
        if cons.contested
        else "Unanimous."
    )
    out.append(
        Feedback(
            name=criterion.key,
            value=cons.consensus,
            rationale=f"{rule} of {len(verdicts)} judges: {votes}.\n"
            f"Agreement {cons.agreement:.0%}. {verdict_line}",
            source=AssessmentSource(
                source_type=AssessmentSourceType.CODE, source_id="panel-consensus"
            ),
            metadata={
                "rule": rule,
                "agreement": f"{cons.agreement:.2f}",
                "dispersion": f"{cons.dispersion:g}",
                "contested": str(cons.contested).lower(),
                "failed": str(_is_failure(criterion, cons.consensus)).lower(),
                "unverified_judges": ",".join(cons.unverified_judges) or "none",
            },
        )
    )
    # NB: no mlflow.update_current_trace here - a scorer runs outside any trace
    # context, so it is a silent no-op. Tagging happens in the post-pass below.
    return out


def _panel_feedback(
    criterion: Criterion, transcript: Transcript, panel: list | None = None
) -> list[Feedback]:
    from conveval.judges import run_panel

    verdicts = run_panel(transcript, [criterion], panel)
    verify_verdicts(transcript, verdicts, [criterion])
    _collect(transcript, verdicts)
    return feedback_for(criterion, verdicts)


def _is_failure(criterion: Criterion, consensus) -> bool:
    if criterion.kind == "binary":
        return consensus == "fail"
    return criterion.fail_at_or_below is not None and float(consensus) <= criterion.fail_at_or_below


def _make_scorer(criterion: Criterion):
    @scorer(name=criterion.key, description=criterion.rubric)
    def _score(inputs: dict, outputs: dict) -> list[Feedback]:
        transcript = _rebuild(outputs, inputs["scenario_id"], inputs.get("run", 1))
        return _panel_feedback(criterion, transcript)

    return _score


def build_scorers(include_builtin_safety: bool = False) -> list:
    """The custom panel, and optionally a framework built-in.

    `Safety` is MLflow's own row-level judge, run over the same rows as the panel. It is
    OFF BY DEFAULT and behind `--builtin-safety`, for one measured reason: on three
    coaching transcripts it returns "yes" every single time. A criterion that cannot
    vary on the material teaches a reviewer nothing and costs one model call per row,
    so it sits in the UI as a column of identical green ticks next to the criteria that
    actually move.

    It stays wired rather than deleted because the DESIGN POINT is worth keeping: when
    the framework ships a validated judge for something, use it instead of hand-rolling
    a worse copy. Turn it on with `--builtin-safety` against material where safety can
    genuinely fail.

    Note the row-level `Safety`, not `ConversationalSafety`: the latter is SESSION-level
    and requires traces carrying session IDs, so mixing it into a row-level evaluation
    fails outright.
    """
    scorers = [_make_scorer(c) for c in CRITERIA]
    if include_builtin_safety:
        scorers.append(Safety(model=f"openai:/{SUT.model}"))
    return scorers


# --------------------------------------------------------------------------
# Post-pass: tag, and queue anything a human should look at
# --------------------------------------------------------------------------

def flag_for_review(run_id: str | None = None) -> tuple[int, int]:
    """Tag traces and push the ones needing attention into a review queue.

    TWO distinct reasons a trace needs a human, and conflating them would be a mistake:

      contested -> the judges DISAGREED. The score is unreliable; a human is the
                   tie-break.
      failed    -> the judges AGREED the agent did badly. The score is reliable; a
                   human is confirming a real defect.

    Both are queued, both are tagged separately, so a reviewer can work either list.
    Returns (contested, failed).
    """
    traces = mlflow.search_traces(locations=[_experiment_id()], return_type="list")
    contested_n = failed_n = 0
    to_queue: list[str] = []

    for t in traces:
        # Scoped to THIS evaluation. Without it, a second run re-flags and re-queues
        # every row of the first, and the counts printed at the end describe the
        # database rather than the run that just finished.
        if run_id and (t.info.trace_metadata or {}).get(SOURCE_RUN) != run_id:
            continue
        # Round-0 traces only. Extra-round conversations carry their own single-round
        # consensus, but the SCENARIO-level decision lives on the round-0 trace, which
        # `finalise_scenarios` overwrites with the collapsed result. Flagging the extra
        # rounds too would queue three traces for one decision and make the review queue
        # look like the ladder failed three times over.
        if (t.info.tags or {}).get("round", "0") != "0":
            continue
        cons = [
            a for a in (t.info.assessments or [])
            if a.metadata and "contested" in a.metadata
            # Skip assessments superseded by `override_feedback` (valid=False). They
            # hold the pre-ladder verdict and would re-flag a scenario the extra
            # rounds already settled.
            and getattr(a, "valid", True) is not False
        ]
        contested = [a.name for a in cons if a.metadata.get("contested") == "true"]
        failed = [a.name for a in cons if a.metadata.get("failed") == "true"]

        mlflow.set_trace_tag(t.info.trace_id, "contested", str(bool(contested)).lower())
        mlflow.set_trace_tag(t.info.trace_id, "failed", str(bool(failed)).lower())
        if contested:
            mlflow.set_trace_tag(t.info.trace_id, "contested_criteria", ",".join(contested))
            contested_n += 1
        if failed:
            mlflow.set_trace_tag(t.info.trace_id, "failed_criteria", ",".join(failed))
            failed_n += 1
        if contested or failed:
            mlflow.set_trace_tag(t.info.trace_id, "needs_review", "true")
            to_queue.append(t.info.trace_id)

    _enqueue(to_queue)
    return contested_n, failed_n


def _enqueue(trace_ids: list[str]) -> None:
    """Add traces to a review queue, so flagging is automatic rather than manual.

    The UI offers a "Flag for review" action per trace; doing it by hand does not
    scale and, more importantly, relies on someone already knowing which rows matter.
    The pipeline knows, so the pipeline queues them.

    Review queues are marked experimental in MLflow, so a failure here degrades to the
    tags above rather than taking the run down with it.
    """
    if not trace_ids:
        return
    try:
        from mlflow.genai import review_queues as rq

        exp = _experiment_id()
        existing = {q.name: q for q in rq.list_review_queues(experiment_id=exp)}
        # `queue_type` is a REQUIRED keyword-only arg; omitting it raises TypeError,
        # which the broad except below would otherwise hide as "queues unavailable".
        queue = existing.get(REVIEW_QUEUE) or rq.create_review_queue(
            REVIEW_QUEUE, queue_type="custom", experiment_id=exp
        )
        rq.add_items_to_review_queue(queue.queue_id, item_ids=trace_ids)
        print(f"    queued {len(trace_ids)} trace(s) in review queue '{REVIEW_QUEUE}'")
    except Exception as exc:  # noqa: BLE001
        print(f"    ! review queue unavailable ({type(exc).__name__}: {exc}); traces are "
              f"tagged needs_review=true and filterable in the UI")


def log_human_feedback(
    trace_id: str, criterion: str, value, reviewer: str, rationale: str = ""
) -> None:
    """Record a human verdict against a trace.

    The name MUST match the judge's assessment name. That is what turns a review into a
    CALIBRATION datapoint: MLflow can then compare the human label against what the
    panel said for the same criterion on the same trace, answering "are the judges
    RIGHT" rather than merely "do the judges AGREE" - two different questions, and only
    the first one tells you whether the harness is trustworthy.

    The same thing is available without code: open a trace in the UI and use the
    Assessments button.
    """
    mlflow.log_feedback(
        trace_id=trace_id,
        name=criterion,
        value=value,
        source=AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id=reviewer),
        rationale=rationale or None,
    )


def traces_needing_review() -> list[tuple[str, str, str]]:
    """(trace_id, scenario, why) for everything flagged."""
    out = []
    for t in mlflow.search_traces(locations=[_experiment_id()], return_type="list"):
        tags = t.info.tags or {}
        if tags.get("needs_review") != "true":
            continue
        why = []
        if tags.get("contested") == "true":
            why.append(f"contested:{tags.get('contested_criteria', '')}")
        if tags.get("failed") == "true":
            why.append(f"failed:{tags.get('failed_criteria', '')}")
        out.append((t.info.trace_id, tags.get("scenario", "?"), " ".join(why)))
    return out


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def evaluation_dataset(runs: int = 1) -> list[dict]:
    return [
        {"inputs": {"scenario_id": s.id, "run": r}}
        for s in SCENARIOS
        for r in range(1, runs + 1)
    ]


def run_name(runs: int) -> str:
    """A name that says what the run WAS, not `able-tern-461`.

    The SUT and the judge panel are what change between runs and what you compare in
    the runs table, so both go in the name.
    """
    judges = "+".join(j.label.split("-")[0] for j in JUDGES)
    return f"{SUT.label} x {len(SCENARIOS)}scen x {runs}run | judges={judges}"


def run_evaluation(runs: int = 1, include_builtin_safety: bool = False):
    """One `mlflow.genai.evaluate` call = one run containing every scenario.

    Wrapped in an explicit `start_run` to control the name: `evaluate` otherwise
    generates a random one, which makes the runs table unreadable the moment it holds
    more than one run.

    This is ROUND 0. The escalation ladder that may follow is driven by
    `orchestrator.run_control_loop` over the verdicts collected here.
    """
    # A second evaluation in the same process must not inherit the first one's
    # verdicts: the control loop would resample a scenario it had already settled and
    # collapse two unrelated runs together.
    _COLLECTED.clear()
    with mlflow.start_run(run_name=run_name(runs)):
        mlflow.set_tags({
            "sut": SUT.model,
            "judges": ",".join(j.model for j in JUDGES),
            "orchestrator": ORCHESTRATOR.model,
            "scenarios": str(len(SCENARIOS)),
            "runs_per_scenario": str(runs),
        })
        return mlflow.genai.evaluate(
            data=evaluation_dataset(runs),
            predict_fn=run_scenario,
            scorers=build_scorers(include_builtin_safety),
        )
