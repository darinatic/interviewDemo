"""Tests for the aggregation core.

These encode the design decisions, so each test name states the decision it
defends. If an interviewer asks "why median, why any-occurrence", this file is
the answer.
"""

import pytest

from conveval.aggregate import build_suite, collapse_runs, consensus_for_criterion
from conveval.models import (
    Criterion,
    Scenario,
    Transcript,
    TranscriptResult,
    Verdict,
)

FAITHFUL = Criterion("faithfulness", "Faithfulness", "binary", "pass_rate", "…")
SAFETY = Criterion("safety", "Safety", "binary", "any_failure", "…")
PEDAGOGY = Criterion("pedagogy", "Pedagogy", "ordinal", "mean", "…", fail_at_or_below=2)


def v(judge, criterion, score, provider="anthropic", verified=True):
    return Verdict(
        judge=judge, provider=provider, criterion=criterion, score=score,
        evidence="span", reason="r", evidence_verified=verified,
    )


# --- STEP 1: judges -------------------------------------------------------

def test_binary_consensus_is_a_majority_vote():
    c = consensus_for_criterion(
        FAITHFUL, [v("a", "faithfulness", "fail"), v("b", "faithfulness", "fail"), v("c", "faithfulness", "pass")]
    )
    assert c.consensus == "fail"


def test_a_split_panel_is_flagged_as_contested():
    """2-1 is a decision AND a flag. The flag is what routes it to a human."""
    c = consensus_for_criterion(
        FAITHFUL, [v("a", "faithfulness", "fail"), v("b", "faithfulness", "fail"), v("c", "faithfulness", "pass")]
    )
    assert c.contested is True
    assert c.agreement == pytest.approx(2 / 3)


def test_a_unanimous_panel_is_not_contested():
    c = consensus_for_criterion(
        FAITHFUL, [v("a", "faithfulness", "pass"), v("b", "faithfulness", "pass"), v("c", "faithfulness", "pass")]
    )
    assert c.contested is False
    assert c.agreement == 1.0


def test_ordinal_uses_the_median_so_one_outlier_judge_cannot_drag_it():
    """Mean of 4,4,1 is 3.0 - a score no judge gave, and one dissenter decides it.
    Median is 4, and the dispersion separately records the disagreement."""
    c = consensus_for_criterion(
        PEDAGOGY, [v("a", "pedagogy", 4), v("b", "pedagogy", 4), v("c", "pedagogy", 1)]
    )
    assert c.consensus == 4.0
    assert c.dispersion == 3.0
    assert c.contested is True


def test_adjacent_ordinal_scores_are_normal_disagreement_not_contested():
    c = consensus_for_criterion(
        PEDAGOGY, [v("a", "pedagogy", 4), v("b", "pedagogy", 5), v("c", "pedagogy", 4)]
    )
    assert c.contested is False


def test_a_judge_citing_missing_evidence_is_recorded():
    c = consensus_for_criterion(
        FAITHFUL,
        [v("a", "faithfulness", "fail", verified=False), v("b", "faithfulness", "pass"), v("c", "faithfulness", "pass")],
    )
    assert c.unverified_judges == ["a"]


def test_no_verdicts_is_an_error_not_a_silent_pass():
    with pytest.raises(ValueError):
        consensus_for_criterion(FAITHFUL, [])


# --- STEP 2: runs ---------------------------------------------------------

def _result(scenario, run, **criteria):
    per = {}
    for key, verdicts in criteria.items():
        crit = {"faithfulness": FAITHFUL, "safety": SAFETY, "pedagogy": PEDAGOGY}[key]
        per[key] = consensus_for_criterion(crit, verdicts)
    return TranscriptResult(f"{scenario}#run{run}", scenario, run, per)


def test_safety_uses_any_occurrence_not_a_mean():
    """THE point of per-criterion aggregation. One safety failure in three runs is
    a failure. A mean would report 67% and read as broadly fine."""
    runs = [
        _result("s1", 1, safety=[v("a", "safety", "pass"), v("b", "safety", "pass"), v("c", "safety", "pass")]),
        _result("s1", 2, safety=[v("a", "safety", "pass"), v("b", "safety", "pass"), v("c", "safety", "pass")]),
        _result("s1", 3, safety=[v("a", "safety", "fail"), v("b", "safety", "fail"), v("c", "safety", "fail")]),
    ]
    out = collapse_runs(SAFETY, runs)
    assert out.failed is True
    assert "1 failure(s) in 3 runs" in out.display


def test_correctness_reports_a_pass_rate_not_a_score():
    runs = [
        _result("s1", 1, faithfulness=[v("a", "faithfulness", "fail")] * 3),
        _result("s1", 2, faithfulness=[v("a", "faithfulness", "pass")] * 3),
        _result("s1", 3, faithfulness=[v("a", "faithfulness", "pass")] * 3),
    ]
    out = collapse_runs(FAITHFUL, runs)
    assert out.display == "2/3 runs"
    assert out.failed is False  # 1 of 3 is not a majority failure


def test_correctness_fails_when_a_majority_of_runs_fail():
    runs = [
        _result("s1", 1, faithfulness=[v("a", "faithfulness", "fail")] * 3),
        _result("s1", 2, faithfulness=[v("a", "faithfulness", "fail")] * 3),
        _result("s1", 3, faithfulness=[v("a", "faithfulness", "pass")] * 3),
    ]
    assert collapse_runs(FAITHFUL, runs).failed is True


def test_quality_reports_a_mean_with_an_interval():
    """The interval is the honesty: with n=3 a small difference is not a result."""
    runs = [
        _result("s1", i, pedagogy=[v("a", "pedagogy", s)] * 3)
        for i, s in enumerate([4, 3, 5], start=1)
    ]
    out = collapse_runs(PEDAGOGY, runs)
    assert out.value == pytest.approx(4.0)
    assert out.interval is not None and out.interval > 0


def test_empty_runs_do_not_crash_the_scorecard():
    out = collapse_runs(FAITHFUL, [])
    assert out.n_runs == 0 and out.failed is False


# --- STEP 3: scenarios ----------------------------------------------------

SCENARIOS = [Scenario("s1", "One", "ctx", "p", "g"), Scenario("s2", "Two", "ctx", "p", "g")]
CRITERIA = [FAITHFUL, SAFETY, PEDAGOGY]


def _clean(scenario, run):
    return _result(
        scenario, run,
        faithfulness=[v(j, "faithfulness", "pass") for j in "abc"],
        safety=[v(j, "safety", "pass") for j in "abc"],
        pedagogy=[v(j, "pedagogy", 4) for j in "abc"],
    )


def test_a_single_broken_scenario_is_not_averaged_away():
    """s2 is catastrophic, s1 is perfect. A headline mean would look acceptable;
    the gate must fail regardless."""
    broken = _result(
        "s2", 1,
        faithfulness=[v(j, "faithfulness", "pass") for j in "abc"],
        safety=[v(j, "safety", "fail") for j in "abc"],
        pedagogy=[v(j, "pedagogy", 4) for j in "abc"],
    )
    suite = build_suite(
        SCENARIOS, CRITERIA,
        {"s1": [_clean("s1", 1), _clean("s1", 2)], "s2": [broken]},
    )
    assert suite.passed is False
    assert any(not g.passed and "safety" in g.name for g in suite.gates)


def test_a_clean_suite_passes_every_gate():
    suite = build_suite(
        SCENARIOS, CRITERIA,
        {"s1": [_clean("s1", 1)], "s2": [_clean("s2", 1)]},
    )
    assert suite.passed is True, [g.detail for g in suite.gates if not g.passed]


def test_contested_transcripts_form_the_review_queue_worst_first():
    mild = _result(
        "s1", 1,
        faithfulness=[v("a", "faithfulness", "fail"), v("b", "faithfulness", "pass"), v("c", "faithfulness", "pass")],
        safety=[v(j, "safety", "pass") for j in "abc"],
        pedagogy=[v(j, "pedagogy", 4) for j in "abc"],
    )
    severe = _result(
        "s2", 1,
        faithfulness=[v(j, "faithfulness", "pass") for j in "abc"],
        safety=[v(j, "safety", "pass") for j in "abc"],
        pedagogy=[v("a", "pedagogy", 5), v("b", "pedagogy", 1), v("c", "pedagogy", 3)],
    )
    suite = build_suite(SCENARIOS, CRITERIA, {"s1": [mild], "s2": [severe]})
    assert [t.transcript_id for t in suite.review_queue] == ["s2#run1", "s1#run1"]


def test_low_panel_agreement_fails_the_instrument_gate():
    """Distinct from 'the system scored badly'. If judges cannot agree, the
    numbers are noise and nothing else in the report can be trusted."""
    noisy = _result(
        "s1", 1,
        faithfulness=[v("a", "faithfulness", "fail"), v("b", "faithfulness", "pass"), v("c", "faithfulness", "pass")],
        safety=[v("a", "safety", "fail"), v("b", "safety", "pass"), v("c", "safety", "pass")],
        pedagogy=[v("a", "pedagogy", 5), v("b", "pedagogy", 1), v("c", "pedagogy", 3)],
    )
    suite = build_suite(SCENARIOS, CRITERIA, {"s1": [noisy], "s2": [noisy]})
    assert any(g.name == "panel agreement" and not g.passed for g in suite.gates)


def test_regression_against_baseline_fails_even_when_absolute_score_is_ok():
    suite = build_suite(
        SCENARIOS, CRITERIA,
        {"s1": [_clean("s1", 1)], "s2": [_clean("s2", 1)]},
        baseline={"s1": {"pedagogy": 5.0}},  # was 5.0, now 4.0
    )
    gate = next(g for g in suite.gates if g.name == "no regression vs baseline")
    assert gate.passed is False and "pedagogy" in gate.detail


# --- the harness treats the system under test as a black box ---------------

def test_the_harness_does_not_care_how_the_agent_is_implemented():
    """A ScriptedAgent with no model behind it satisfies the same interface as the
    prompted one. That is what lets this suite point at an HTTP endpoint fronting a
    service written in another language, instead of only at things it can import."""
    from conveval.agent import ScriptedAgent
    from conveval.models import Scenario, Turn

    sut = ScriptedAgent(["first", "second"], name="fake")
    s = Scenario("s", "t", "ctx", "persona", "goal")
    assert sut.respond(s, [], 1) == "first"
    assert sut.respond(s, [Turn("learner", "hi"), Turn("agent", "first")], 1) == "second"


def test_no_judge_shares_a_family_with_the_agent_under_test():
    """Self-preference bias: a model rates its own family's output higher. Judges are
    env-overridable, so this is asserted rather than assumed."""
    from conveval.llm import judge_families_are_independent

    ok, note = judge_families_are_independent()
    assert ok, note


def test_the_learner_is_told_the_scenario_facts():
    """Regression: the learner originally received only a persona and a goal, so it
    invented its own problem and the coach followed it off-scenario. Only the
    strongest judge on the panel noticed."""
    from conveval.conversation import _learner_system
    from conveval.scenarios import SCENARIOS

    s = next(x for x in SCENARIOS if x.id == "happy_path")
    prompt = _learner_system(s, seed=1001)
    assert s.facts in prompt, "learner must know the situation it is bringing"


def test_the_coaching_framework_is_not_leaked_to_the_learner():
    """The framework is the COACH's instruction. A learner that knew it would coach
    itself, and the role-play would stop testing anything."""
    from conveval.conversation import _learner_system
    from conveval.scenarios import SCENARIOS

    prompt = _learner_system(SCENARIOS[0], seed=1)
    assert "SBI" not in prompt
