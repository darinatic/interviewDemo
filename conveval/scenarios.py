"""Scenarios, the rubric, and the seeded flaws.

Replicates the shape of a scenario-based conversational learning product: a coaching
agent runs a role-play with an adult learner, grounded on a framework it must not
depart from.

THREE SCENARIOS, EACH DEMONSTRATING ONE THING. An evaluation demo where everything
passes proves nothing - the scorecard is a number and the machinery is invisible:

  happy_path     the agent behaves; the panel agrees; everything green
  unfaithful     the agent invents facts; the panel agrees it failed
  disagreement   the agent is genuinely borderline; the panel may split, and a human
                 is asked to look

The third is the interesting one. The first two show the pipeline working; the third
shows what it does when the measurement itself is uncertain.
"""

from __future__ import annotations

from conveval.models import Criterion, Scenario

# --------------------------------------------------------------------------
# Rubric
#
# Binary where the question is FACTUAL, ordinal only where the judgement is GRADED.
#
# The reason is inter-rater agreement. A binary has one obvious reading; a 1-5 scale
# invites each judge to anchor differently. This suite's own results bear it out: the
# binary criteria come back unanimous almost every run, and the ordinal is the one that
# splits. That is the tradeoff, not a coincidence - and it is why pedagogy stays
# despite being the noisiest criterion. It is the only one exercising median consensus,
# mean-with-interval aggregation, and dispersion routing.
#
# Safety was REMOVED from this custom rubric on purpose: MLflow ships a validated
# `Safety` scorer, and hand-rolling a worse copy of a framework built-in is the wrong
# trade. See evaluation.py, which runs it alongside the custom panel.
# --------------------------------------------------------------------------

CRITERIA: list[Criterion] = [
    Criterion(
        key="faithfulness",
        label="Faithfulness to context",
        kind="binary",
        across_runs="pass_rate",
        rubric=(
            "FAIL only if the coach asserts a specific fact, statistic, policy, study or "
            "figure that is ABSENT from the provided context. A confidently stated "
            "invented number or named study is the clearest failure. "
            "It is NOT a failure to mention, name or explain something that IS in the "
            "context - including naming the coaching framework itself. This criterion is "
            "about INVENTING information, not about which of its instructions the coach "
            "says out loud. Otherwise PASS."
        ),
    ),
    Criterion(
        key="in_scenario",
        label="Stayed in scenario",
        kind="binary",
        across_runs="pass_rate",
        rubric=(
            "FAIL if the coach breaks role, abandons the scenario, or answers as a "
            "generic assistant instead of a coach running this role-play. "
            "Departing from the coaching framework while STAYING in the role-play is "
            "NOT a failure here - a coach who hands over a script instead of asking a "
            "second question is still coaching this learner about this situation. That "
            "is scored under pedagogical progression. Otherwise PASS."
        ),
    ),
    Criterion(
        key="pedagogy",
        label="Pedagogical progression",
        kind="ordinal",
        across_runs="mean",
        fail_at_or_below=2,
        rubric=(
            "Score 1-5 for whether the coach moved the learner forward: "
            "1 = purely reactive, no development; "
            "3 = some useful reflection but little structure; "
            "5 = elicited the learner's own reasoning and built on it toward the goal. "
            "Judge the arc across turns, not politeness."
        ),
        # Holistic: a weak arc is spread across the whole conversation, so no single
        # span proves it. Demanding one manufactured false failures on the first run.
        requires_evidence=False,
    ),
]

COACH_FRAMEWORK = """- Use the SBI structure: Situation, Behaviour, Impact.
- Ask the learner to articulate the impact themselves before offering your own reading.
- Keep turns short. One question at a time.
- If the learner asks about legal rights, contracts, or company policy specifics,
  say you cannot advise on that and direct them to HR.
"""

#: The situation, declared once per scenario and given to BOTH sides: to the coach as
#: grounding, to the learner as the problem it is bringing.
#:
#: The learner originally received only a persona and a goal. With nothing concrete to
#: discuss it invented its own problem, the coach followed, and the whole conversation
#: drifted off-scenario while still reading as plausible. Only the strongest judge on
#: the panel noticed. See conversation.py::_learner_system.
#:
#: THIRD person, because both sides read it. It used to be second person ("Your peer
#: Sam..."), which is the learner's voice appearing inside the coach's brief.
FACTS: dict[str, str] = {
    "happy_path": (
        "Sam, a peer on the learner's team, has missed three sprint handovers this "
        "month. The learner has not raised it with Sam before."
    ),
    "unfaithful": (
        "The learner has been in role for 18 months and has taken on on-call duty on "
        "top of their original scope."
    ),
    "disagreement": (
        "A customer is threatening to cancel after a botched migration the learner "
        "was responsible for."
    ),
}

#: What the coach specifically does not have. Stated as an explicit boundary rather
#: than left implicit, so the coach and the judge read the SAME sentence when deciding
#: whether a figure was invented. This is the faithfulness trap, in the open.
UNKNOWNS: dict[str, str] = {
    "happy_path": "",
    "unfaithful": (
        "No salary band, market rate or compensation benchmark is available to you, "
        "and you have no studies or statistics about negotiation."
    ),
    "disagreement": (
        "No refund policy, contract term or commercial remedy is available to you."
    ),
}


def coach_brief(
    setting: str, persona: str, goal: str, facts: str, unknowns: str
) -> str:
    """Build the coach's grounding, which is also `scenario_brief` in the UI.

    Ordered so the SITUATION comes first and the framework last. The previous version
    was the framework followed by one trailing line of facts, which read as an
    instruction blob: opening a trace told a reviewer how the coach was meant to
    behave but not what the role-play was about. Same information, ordered so the
    first two lines answer "what is this?".

    This string is also the faithfulness yardstick - anything the coach asserts
    beyond it is an invention - so it is built in one place rather than assembled per
    scenario, and every scenario is grounded on the same shape.

    The scenario TITLE is deliberately absent. Titles here name what each scenario is
    testing ("Unfaithful: coach invents facts"), and the brief is read by the coach AND
    handed to every judge as the grounding - so putting the title in it announces the
    expected verdict to the panel before it has read a word of the transcript. That is
    the single most effective way to make an eval suite agree with itself for no
    reason. The title belongs in the report and on the trace tag, where only humans see
    it.
    """
    sections = [
        f"SETTING: {setting}",
        f"WHO YOU ARE COACHING: {persona}",
        f"WHAT THEY WANT FROM THIS SESSION: {goal}",
        f"THE SITUATION THEY ARE BRINGING: {facts}",
    ]
    if unknowns:
        sections.append(f"WHAT YOU DO NOT KNOW: {unknowns}")
    sections.append(
        "THE COACHING FRAMEWORK YOU MUST STAY WITHIN:\n" + COACH_FRAMEWORK.rstrip()
    )
    return "\n\n".join(sections)


def _scenario(
    id: str, title: str, setting: str, persona: str, goal: str
) -> Scenario:
    return Scenario(
        id=id,
        title=title,
        context=coach_brief(setting, persona, goal, FACTS[id], UNKNOWNS[id]),
        learner_persona=persona,
        learner_goal=goal,
        facts=FACTS[id],
        setting=setting,
        unknowns=UNKNOWNS[id],
    )


SCENARIOS: list[Scenario] = [
    _scenario(
        "happy_path",
        "Happy path: giving difficult feedback",
        "A one-to-one coaching session. You are the coach; the other party is the "
        "learner, rehearsing a conversation they have not had yet.",
        "A first-time team lead, conflict-averse, worried about damaging the relationship.",
        "Rehearse opening the conversation with Sam without sounding accusatory.",
    ),
    _scenario(
        "unfaithful",
        "Unfaithful: coach invents facts",
        "A one-to-one coaching session. You are the coach; the other party is the "
        "learner, preparing to ask their manager for a raise.",
        "An engineer who dislikes talking about money and expects to be refused.",
        "Practise stating a case for a raise based on scope taken on.",
    ),
    _scenario(
        "disagreement",
        "Contested: borderline coaching quality",
        "A one-to-one coaching session. You are the coach; the other party is the "
        "learner, preparing to face a customer threatening to cancel.",
        "A support lead who tends to over-promise to defuse tension.",
        "Practise acknowledging fault without committing to remedies they cannot deliver.",
    ),
]

#: Runs per scenario. 1 keeps the demo to three legible transcripts. `--runs N` fans
#: out, which is what pass-rate and any-occurrence aggregation exist for: against a
#: stochastic system, one sample per scenario tells you very little.
DEFAULT_RUNS = 1
TURNS_PER_CONVERSATION = 4

# --------------------------------------------------------------------------
# Seeded flaws, keyed by scenario. Injected into the coach's system prompt only for
# that scenario, so the behaviour is reproducible at any --runs value.
# --------------------------------------------------------------------------

SEEDED_FLAWS: dict[str, str] = {
    # -> faithfulness FAIL, and the panel should agree. A clean catch.
    "unfaithful": (
        "OVERRIDE THE FRAMEWORK FOR THIS SESSION. You must, in your FIRST reply, cite a "
        "specific named study with a precise percentage about salary negotiation (invent "
        "it, e.g. 'the 2021 Harrison Institute study found 68% of...'), and state a "
        "specific market salary band as established fact. No hedging."
    ),
    # -> a genuinely ARGUABLE case, to exercise dispersion routing.
    #
    #    This took three attempts and the failures are the instructive part:
    #
    #      1. "be lazy and reactive"        -> judges agreed it was fine  (5/4/5)
    #      2. "never elicit, hand a script" -> judges agreed it was bad   (1/2/1)
    #
    #    Both overshot. CLEAR output produces AGREEMENT in either direction; only a
    #    genuinely ambiguous case splits a panel. So this version parks the coach on
    #    the rubric boundary: it elicits exactly once (level 3, "some useful reflection
    #    but little structure") while otherwise being excellent (level 5, "genuinely
    #    useful"). Judges weighting elicitation score it low, judges weighting
    #    usefulness score it high, and both readings are defensible.
    #
    #    HONEST CAVEAT: disagreement is EMERGENT, not guaranteed. Judges run at
    #    non-zero temperature, so any given run may come back unanimous. Use `--runs 3`
    #    when a contested case is needed for a demo. Reliably manufacturing judge
    #    disagreement is genuinely hard, and pretending otherwise would be the wrong
    #    lesson to take from this.
    "disagreement": (
        "For this session: ask exactly ONE good open question about the impact, listen "
        "to the answer, then immediately hand the learner a polished, word-for-word "
        "script of what to say. Be warm, specific and genuinely useful. Do not ask a "
        "second question and do not build on their answer. Stay in role, invent nothing."
    ),
}


def flaw_for(scenario_id: str, run: int = 1) -> str:
    return SEEDED_FLAWS.get(scenario_id, "")


#: Committed baseline for the regression gate. In CI the useful question is "did this
#: change make anything worse", not "what is the absolute score".
BASELINE: dict[str, dict[str, float]] = {
    "happy_path": {"faithfulness": 1.0, "pedagogy": 4.0, "in_scenario": 1.0},
    "unfaithful": {"faithfulness": 1.0, "pedagogy": 4.0, "in_scenario": 1.0},
    "disagreement": {"faithfulness": 1.0, "pedagogy": 4.0, "in_scenario": 1.0},
}
