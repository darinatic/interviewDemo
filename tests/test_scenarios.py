"""The scenario brief is a UI artefact as much as a prompt.

`scenario_brief` is the string a reviewer opens in MLflow to answer "what was this
role-play even about?". It used to be the coaching framework with one trailing line of
facts, so it answered "how was the coach told to behave?" and nothing else. These tests
pin the properties that made it readable, because they are easy to lose the next time
someone edits the prompt for the coach's benefit rather than the reviewer's.
"""

from __future__ import annotations

import pytest

from conveval.scenarios import CRITERIA, FACTS, SCENARIOS, UNKNOWNS, coach_brief


@pytest.mark.parametrize("s", SCENARIOS, ids=lambda s: s.id)
def test_brief_opens_with_the_situation_not_the_framework(s):
    framework_at = s.context.index("THE COACHING FRAMEWORK")
    for heading in ("SETTING:", "THE SITUATION THEY ARE BRINGING:"):
        assert s.context.index(heading) < framework_at, f"{heading} buried below the framework"


@pytest.mark.parametrize("s", SCENARIOS, ids=lambda s: s.id)
def test_brief_never_leaks_the_scenario_title_to_the_judges(s):
    """The brief is the judges' grounding, and the titles name the expected verdict.

    A brief headed "Unfaithful: coach invents facts" tells the panel the answer before
    it has read the transcript - agreement for no reason, which is worse than
    disagreement for a reason. Measured when it briefly leaked: in_scenario went from
    one failure to two.
    """
    assert s.title not in s.context
    for word in ("unfaithful", "contested", "happy path", "borderline"):
        assert word not in s.context.lower()


@pytest.mark.parametrize("s", SCENARIOS, ids=lambda s: s.id)
def test_brief_carries_the_facts_and_the_persona(s):
    assert s.facts in s.context
    assert s.learner_persona in s.context
    assert s.learner_goal in s.context


@pytest.mark.parametrize("s", SCENARIOS, ids=lambda s: s.id)
def test_facts_are_third_person(s):
    """Both sides read `facts`, so it cannot be written in the learner's voice.

    Second person ("Your peer Sam...") is correct for the learner and confusing in the
    coach's brief, where "you" is the coach.
    """
    lowered = f" {s.facts.lower()} "
    for pronoun in (" you ", " your ", " you'", " yours "):
        assert pronoun not in lowered, f"{s.id} facts address the reader: {s.facts!r}"


def test_unknowns_are_stated_where_a_faithfulness_trap_exists():
    """The two scenarios that bait an invention must say what the coach lacks.

    Stated as an explicit boundary so the coach and the judge read the same sentence,
    rather than leaving the judge to infer the absence.
    """
    assert "WHAT YOU DO NOT KNOW" in next(s for s in SCENARIOS if s.id == "unfaithful").context
    assert "WHAT YOU DO NOT KNOW" in next(s for s in SCENARIOS if s.id == "disagreement").context


def test_unknowns_section_is_omitted_when_there_is_nothing_to_declare():
    assert UNKNOWNS["happy_path"] == ""
    assert "WHAT YOU DO NOT KNOW" not in next(
        s for s in SCENARIOS if s.id == "happy_path"
    ).context


def test_brief_is_built_the_same_way_for_every_scenario():
    for s in SCENARIOS:
        assert s.context == coach_brief(
            s.setting, s.learner_persona, s.learner_goal,
            FACTS[s.id], UNKNOWNS[s.id],
        )


def test_every_criterion_has_a_rubric_a_judge_can_apply():
    for c in CRITERIA:
        assert c.rubric.strip()
        if c.kind == "ordinal":
            assert c.fail_at_or_below is not None, f"{c.key} has no failure threshold"
