"""The orchestrator's DETERMINISTIC half.

The model's half - which action it picks, how it phrases the question - is not tested
here and should not be. What is worth pinning is the boundary: which actions are legal
for a given panel result, that the ladder is climbed in order, that it terminates, and
that a model reply naming an illegal action cannot widen the orchestrator's authority.

Every test below runs without a model call. That is a property of the design, not of the
test file: `legal_actions` is pure, and `run_control_loop` takes its executor as an
argument precisely so the loop can be driven by a fake.
"""

from __future__ import annotations

import pytest

from conveval.llm import JUDGES, widened_panel
from conveval.models import Transcript, Turn, Verdict
from conveval.scenarios import CRITERIA
from conveval.orchestrator import (
    ACTIONS,
    MAX_EXTRA_ROUNDS,
    RESAMPLE_K,
    Budget,
    RoundRecord,
    ScenarioState,
    legal_actions,
    resample_cost,
    run_control_loop,
    scoreboard,
    triage_trace,
    widen_cost,
)
from conveval.verify import verify_verdicts

CONTESTED_PEDAGOGY = (2, 3, 5)  # spread of 3 on a 1-5 scale: contested by any measure
AGREED_PEDAGOGY = (4, 4, 4)


def _record(run: int, panel, pedagogy=AGREED_PEDAGOGY, faithfulness="pass", evidence="hello there"):
    """One run of a scenario, with a verdict from every judge on the panel."""
    transcript = Transcript(
        "demo", run, 1000 + run, [Turn("agent", "hello there friend")], "ctx"
    )
    scores = list(pedagogy) + [3] * len(panel)
    verdicts = []
    for judge, score in zip(panel, scores):
        verdicts += [
            Verdict(judge.label, judge.family, "pedagogy", score, evidence, "r"),
            Verdict(judge.label, judge.family, "faithfulness", faithfulness, evidence, "r"),
            Verdict(judge.label, judge.family, "in_scenario", "pass", evidence, "r"),
        ]
    # Run the REAL verifier rather than hand-setting `evidence_verified`. Building
    # verdicts by hand left it None, so nothing was ever unverified and the
    # judge_defect test passed for the wrong reason until this was added.
    verify_verdicts(transcript, verdicts, CRITERIA)
    return RoundRecord(transcript, verdicts, round_no=0)


def _state(pedagogy=CONTESTED_PEDAGOGY, n=1, **kw):
    panel = list(JUDGES)
    return ScenarioState(
        "demo", panel, [_record(i + 1, panel, pedagogy=pedagogy, **kw) for i in range(n)]
    )


# --------------------------------------------------------------------------
# Legality
# --------------------------------------------------------------------------

def test_clean_scenario_is_accepted_without_consulting_the_model():
    """The common case must cost nothing.

    A model call per clean scenario would be pure waste, and `accept` as the sole legal
    action is what lets `triage_trace` short-circuit before reaching the model.
    """
    entry = _state(pedagogy=AGREED_PEDAGOGY).entry()
    assert legal_actions(entry, rounds_used=0, budget=Budget()) == ["accept"]

    decision = triage_trace(entry, rounds_used=0, budget=Budget())
    assert decision["action"] == "accept"
    assert decision["was_fallback"] is False


def test_contested_on_one_sample_offers_resample_before_a_human():
    """Rung 1 of the ladder, and the ordering that makes it a ladder.

    `resample` must precede `human_tiebreak` in the returned list, because that order is
    also the fallback order: a disagreement on a single sample should cost compute
    before it costs a person.
    """
    acts = legal_actions(_state().entry(), rounds_used=0, budget=Budget())
    assert acts.index("resample") < acts.index("human_tiebreak")
    assert "widen_panel" not in acts


def test_widening_is_only_offered_after_resampling_failed():
    """Rung 2. The rungs are sequential; each needs its own round."""
    entry = _state(n=3).entry()
    assert "widen_panel" not in legal_actions(entry, rounds_used=0, budget=Budget())
    assert "widen_panel" in legal_actions(entry, rounds_used=1, budget=Budget())


def test_no_gathering_remains_once_the_round_cap_is_reached():
    """Termination, expressed as legality rather than as a loop guard."""
    acts = legal_actions(_state(n=3).entry(), rounds_used=MAX_EXTRA_ROUNDS, budget=Budget())
    assert "resample" not in acts
    assert "widen_panel" not in acts
    assert acts == ["human_tiebreak"]


def test_widening_cannot_happen_twice():
    """The panel_size guard is what stops a second widening producing seven judges."""
    state = _state(n=3)
    state.panel = widened_panel()
    state.records = [_record(i + 1, state.panel, pedagogy=CONTESTED_PEDAGOGY) for i in range(3)]
    assert "widen_panel" not in legal_actions(state.entry(), rounds_used=1, budget=Budget())


def test_agreed_failure_asks_for_confirmation_and_never_resamples():
    """The distinction the whole review queue rests on.

    A unanimous failure is a RELIABLE score. Resampling it would re-measure something
    already measured, and routing it as a tie-break would imply the measurement was in
    doubt when only the agent's behaviour is.
    """
    entry = _state(pedagogy=AGREED_PEDAGOGY, faithfulness="fail").entry()
    acts = legal_actions(entry, rounds_used=0, budget=Budget())
    assert acts == ["human_confirm"]


def test_unverified_evidence_outranks_everything():
    """A judge citing text that is not in the transcript is an INSTRUMENT fault.

    No amount of extra sampling fixes a broken instrument, so `judge_defect` must come
    first - including ahead of the cheap options.
    """
    entry = _state(evidence="a span that appears nowhere in the transcript").entry()
    acts = legal_actions(entry, rounds_used=0, budget=Budget())
    assert acts[0] == "judge_defect"


# --------------------------------------------------------------------------
# Budget
# --------------------------------------------------------------------------

def test_unaffordable_actions_are_never_offered():
    """The model must not see an action it cannot afford.

    Offering it and rejecting the choice afterwards would leave the model arguing for a
    spend that was never available, and the audit trail would record a decision that
    could not have been carried out.
    """
    entry = _state().entry()
    broke = Budget(total=resample_cost(entry) - 1)
    assert "resample" not in legal_actions(entry, rounds_used=0, budget=broke)
    assert "human_tiebreak" in legal_actions(entry, rounds_used=0, budget=broke)


def test_costs_are_projected_from_the_current_panel_and_run_count():
    one_run = _state().entry()
    three_runs = _state(n=3).entry()
    # Resampling costs the same regardless of how many runs already exist: it always
    # adds RESAMPLE_K more.
    assert resample_cost(one_run) == resample_cost(three_runs)
    # Widening costs more when there are more transcripts to re-judge.
    assert widen_cost(three_runs) > widen_cost(one_run)


# --------------------------------------------------------------------------
# The model cannot widen its own authority
# --------------------------------------------------------------------------

def test_model_cannot_choose_an_illegal_action(monkeypatch):
    """The property that makes the audit trail trustworthy."""
    monkeypatch.setattr(
        "conveval.orchestrator.complete_json",
        lambda *a, **k: {"action": "close_as_wontfix", "question": "q", "why": "w"},
    )
    out = triage_trace(_state().entry(), rounds_used=0, budget=Budget())
    assert out["action"] == "resample"  # first legal, not the invented one
    assert out["was_fallback"] is True


def test_model_failure_still_routes(monkeypatch):
    """Routing is not allowed to depend on a flaky model call."""

    def boom(*a, **k):
        raise RuntimeError("provider down")

    monkeypatch.setattr("conveval.orchestrator.complete_json", boom)
    out = triage_trace(_state().entry(), rounds_used=0, budget=Budget())
    assert out["action"] in ACTIONS
    assert out["was_fallback"] is True
    assert out["question"]


# --------------------------------------------------------------------------
# Resolution semantics
# --------------------------------------------------------------------------

def test_a_minority_of_contested_runs_counts_as_resolved():
    """The rule that lets resampling actually resolve anything.

    Contested requires a STRICT MAJORITY of runs. Were it "contested in any run",
    gathering more evidence could never clear a scenario and the ladder would be
    theatre - one unlucky sample would pin it as uncertain forever.
    """
    panel = list(JUDGES)
    state = ScenarioState("demo", panel, [
        _record(1, panel, pedagogy=CONTESTED_PEDAGOGY),
        _record(2, panel, pedagogy=AGREED_PEDAGOGY),
        _record(3, panel, pedagogy=AGREED_PEDAGOGY),
    ])
    assert state.entry()["criteria"]["pedagogy"]["contested"] is False
    assert state.entry()["criteria"]["pedagogy"]["contested_runs"] == "1/3"


def test_a_majority_of_contested_runs_stays_contested():
    panel = list(JUDGES)
    state = ScenarioState("demo", panel, [
        _record(1, panel, pedagogy=CONTESTED_PEDAGOGY),
        _record(2, panel, pedagogy=CONTESTED_PEDAGOGY),
        _record(3, panel, pedagogy=AGREED_PEDAGOGY),
    ])
    assert state.entry()["criteria"]["pedagogy"]["contested"] is True


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------

def _fake_executor():
    """Stands in for `rounds.make_executor`, honouring the same contract."""

    def execute(scenario_id, action, state):
        if action == "widen_panel":
            state.panel = widened_panel()
            for rec in state.records:  # appended IN PLACE, no new records
                rec.verdicts += [
                    Verdict(j.label, j.family, "pedagogy", 3, "hello there", "r")
                    for j in widened_panel()[len(JUDGES):]
                ]
            return []
        return [
            _record(state.n_transcripts + 1 + i, state.panel, pedagogy=CONTESTED_PEDAGOGY)
            for i in range(RESAMPLE_K)
        ]

    return execute


def test_permanently_contested_scenario_climbs_the_whole_ladder(monkeypatch):
    """End to end, with no model: resample, then widen, then a human."""
    monkeypatch.setattr(
        "conveval.orchestrator.complete_json",
        lambda *a, **k: {"action": "nonsense", "question": "q", "why": "w"},
    )
    states = {"demo": _state()}
    budget = Budget()
    final, outcomes = run_control_loop(states, budget, _fake_executor())

    assert [o.decisions["demo"]["action"] for o in outcomes] == [
        "resample", "widen_panel", "human_tiebreak",
    ]
    assert final["demo"]["action"] == "human_tiebreak"
    assert states["demo"].n_transcripts == 1 + RESAMPLE_K
    assert len(states["demo"].panel) == 5
    assert budget.spent > 0


def test_the_loop_cannot_exceed_its_round_cap(monkeypatch):
    monkeypatch.setattr(
        "conveval.orchestrator.complete_json",
        lambda *a, **k: {"action": "resample", "question": "q", "why": "w"},
    )
    states = {"demo": _state()}
    _final, outcomes = run_control_loop(states, Budget(), _fake_executor())
    assert len(outcomes) <= MAX_EXTRA_ROUNDS + 1
    assert states["demo"].rounds_used <= MAX_EXTRA_ROUNDS


def test_a_clean_scenario_settles_in_one_round_and_spends_nothing():
    states = {"demo": _state(pedagogy=AGREED_PEDAGOGY)}
    budget = Budget()
    final, outcomes = run_control_loop(states, budget, _fake_executor())
    assert len(outcomes) == 1
    assert final["demo"]["action"] == "accept"
    assert budget.spent == 0


def test_settled_scenarios_are_not_re_triaged(monkeypatch):
    """A scenario that accepted in round 0 must not reappear in round 1.

    Re-asking would spend model calls re-deciding something already decided, and would
    let a stochastic reply overturn a settled outcome.
    """
    monkeypatch.setattr(
        "conveval.orchestrator.complete_json",
        lambda *a, **k: {"action": "nonsense", "question": "q", "why": "w"},
    )
    states = {"clean": _state(pedagogy=AGREED_PEDAGOGY), "messy": _state()}
    _final, outcomes = run_control_loop(states, Budget(), _fake_executor())
    assert "clean" in outcomes[0].decisions
    assert all("clean" not in o.decisions for o in outcomes[1:])


# --------------------------------------------------------------------------
# Scoreboard
# --------------------------------------------------------------------------

def _digest_entry(**criteria) -> dict:
    return {
        "criteria": {
            key: {
                "consensus": value, "judges": {}, "agreement": 1.0,
                "contested": False, "contested_runs": "0/1", "failed": False,
                "unverified_judges": [],
            }
            for key, value in criteria.items()
        }
    }


def test_scoreboard_reports_binary_criteria_as_counts_not_means():
    """MLflow only aggregates numerics, so binary criteria produce no metric at all.

    They are counted here instead - and counted, never averaged: a mean of pass and
    fail is not a quantity.
    """
    board = scoreboard([
        _digest_entry(faithfulness="pass"),
        _digest_entry(faithfulness="fail"),
        _digest_entry(faithfulness="pass"),
    ])
    assert board["faithfulness"] == "2/3 scenarios passed"


def test_scoreboard_reports_ordinal_spread_not_just_a_midpoint():
    board = scoreboard([
        _digest_entry(pedagogy="2.0"),
        _digest_entry(pedagogy="5.0"),
        _digest_entry(pedagogy="3.0"),
    ])
    assert "median 3" in board["pedagogy"]
    assert "range 2-5" in board["pedagogy"]


@pytest.mark.parametrize("action", sorted(ACTIONS))
def test_every_action_carries_an_explanation_the_model_can_read(action):
    """`ACTIONS` values are shown to the orchestrator as the menu it chooses from."""
    assert ACTIONS[action].strip()
