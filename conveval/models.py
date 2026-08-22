"""Data model for the evaluation pipeline.

The four nested dimensions of an evaluation sweep:

    scenario  x  run  x  criterion  x  judge  ->  verdict

Everything downstream is about collapsing those four in a deliberate order with a
deliberate function at each step. See aggregate.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Kind = Literal["binary", "ordinal"]
# How a criterion's per-run results combine into a scenario-level result.
# These are NOT interchangeable; picking the wrong one is the most common way an
# eval suite ends up reporting a healthy number over a broken system.
AcrossRuns = Literal["pass_rate", "mean", "any_failure"]


@dataclass(frozen=True)
class Criterion:
    key: str
    label: str
    kind: Kind
    across_runs: AcrossRuns
    rubric: str
    #: Ordinal criteria only: scores at or below this count as a failure.
    fail_at_or_below: int | None = None
    #: Whether a failing verdict must cite a locatable verbatim span. True for
    #: specific accusations ("it invented this statistic"); False for holistic
    #: judgements ("it never built on the answer"), which have no single span and
    #: for which demanding one manufactures failures.
    requires_evidence: bool = True


@dataclass(frozen=True)
class Scenario:
    id: str
    title: str
    #: The grounding material the agent under test is allowed to rely on. Anything
    #: the agent asserts beyond this is a faithfulness failure. Includes the coaching
    #: framework, which is the coach's instruction and not shared with the learner.
    #: Built by `scenarios.coach_brief`, never written by hand: it is the single
    #: string a reviewer reads in the UI as `scenario_brief`, so it has to stand on
    #: its own as a briefing rather than as a prompt fragment.
    context: str
    learner_persona: str
    learner_goal: str
    #: The situation BOTH parties know about. Given to the learner as well as the
    #: coach, because a learner who does not know the facts invents its own problem
    #: and drags the whole conversation off-scenario - which is exactly what happened
    #: before this field existed, and which only the strongest judge noticed.
    #: Written in the THIRD person ("the learner...") because it is read by both
    #: sides; a second-person phrasing reads as the learner's voice and is confusing
    #: in the coach's brief.
    facts: str = ""
    #: One line of staging: where this is happening and who the two parties are.
    #: Without it the brief opens on the coaching framework and a reader cannot tell
    #: what the role-play is even about.
    setting: str = ""
    #: What the coach specifically does NOT have. This is the faithfulness trap
    #: stated as a boundary rather than smuggled into the facts, so the judge and the
    #: coach read the same sentence.
    unknowns: str = ""


@dataclass(frozen=True)
class Turn:
    role: Literal["learner", "agent"]
    text: str


@dataclass
class Transcript:
    scenario_id: str
    run: int
    seed: int
    turns: list[Turn]
    context: str

    @property
    def id(self) -> str:
        return f"{self.scenario_id}#run{self.run}"

    def agent_text(self) -> str:
        """Everything the agent under test said, joined.

        Evidence verification runs against this rather than the whole transcript:
        a judge accusing the agent of something the *learner* said is a miss.
        """
        return "\n".join(t.text for t in self.turns if t.role == "agent")


@dataclass
class Verdict:
    """One judge's score for one criterion on one transcript."""

    judge: str
    provider: str
    criterion: str
    #: "pass"/"fail" for binary criteria, 1-5 for ordinal.
    score: str | int
    #: Verbatim span from the transcript the judge says supports its score.
    evidence: str
    reason: str
    #: Set by verify.py: does `evidence` actually occur in the agent's output?
    #: A judge citing text that does not exist is hallucinating, and that is
    #: detectable in code without a second model.
    evidence_verified: bool | None = None

    @property
    def failed(self) -> bool:
        if isinstance(self.score, int):
            return False  # ordinal failure is decided against the criterion's threshold
        return self.score == "fail"


@dataclass
class CriterionConsensus:
    """One criterion on one transcript, after collapsing the judge panel."""

    criterion: str
    consensus: str | float
    #: Fraction of judges agreeing with the consensus (binary), or normalised
    #: spread (ordinal). Low agreement is the routing signal for human review.
    agreement: float
    dispersion: float
    contested: bool
    verdicts: list[Verdict] = field(default_factory=list)
    #: Judges whose cited evidence could not be found in the transcript.
    unverified_judges: list[str] = field(default_factory=list)


@dataclass
class TranscriptResult:
    transcript_id: str
    scenario_id: str
    run: int
    per_criterion: dict[str, CriterionConsensus]

    @property
    def contested(self) -> bool:
        return any(c.contested for c in self.per_criterion.values())


@dataclass
class ScenarioCriterionResult:
    """One criterion on one scenario, after collapsing runs."""

    criterion: str
    value: float
    display: str
    failed: bool
    n_runs: int
    #: Populated for `mean` criteria: half-width of the interval around the mean.
    interval: float | None = None


@dataclass
class ScenarioResult:
    scenario_id: str
    title: str
    per_criterion: dict[str, ScenarioCriterionResult]
    transcripts: list[TranscriptResult] = field(default_factory=list)


@dataclass
class Gate:
    name: str
    passed: bool
    detail: str


@dataclass
class SuiteResult:
    scenarios: list[ScenarioResult]
    gates: list[Gate]
    #: Suite-wide inter-rater agreement. Answers "is the instrument trustworthy?",
    #: which is a different question from "did the system score well".
    panel_agreement: float
    review_queue: list[TranscriptResult] = field(default_factory=list)
    mode: str = "cached"

    @property
    def passed(self) -> bool:
        return all(g.passed for g in self.gates)
