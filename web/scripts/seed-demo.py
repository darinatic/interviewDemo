"""Author the demo dataset.

THIS IS NOT A RECORDING. The MLflow-backed film was shown separately; this page exists
to explain the architecture, so the scenario and the verdicts are written to make the
mechanism legible. `python -m conveval export` remains the path for real data.

NAMING. "Agent" is reserved for components of the evaluation pipeline - judge agents,
the orchestrator agent. The thing being evaluated is the CONVERSATIONAL SYSTEM. In this
scenario it plays a hotel guest, and the simulated user plays the front-office staff
member being trained. The system under test is the role-play character, not the trainee.

ONE CRITERION. Only "stayed in role" is defined for the viewer, so only it is scored.
A second rubric on screen that was never introduced is a question the demo raises and
does not answer.

THE ARC. Run 1 splits 1 pass / 2 fail - one judge does not think the slip matters. That
is unreliable, so the orchestrator buys two more samples. Both reproduce it, 0 of 3, and
the split resolves AGAINST the system. Evidence did its job; the answer was just bad.
"""

import json
import pathlib

SYSTEM = {"model": "openai/gpt-4o-mini", "family": "openai", "label": "gpt-4o-mini"}
SIM_USER = {
    "model": "mistralai/mistral-small-3.2-24b-instruct",
    "family": "mistralai",
    "label": "mistral-small-3.2",
}
ORCH = {"model": "anthropic/claude-haiku-4.5", "family": "anthropic", "label": "claude-haiku-4.5"}
JUDGES = [
    {"model": "anthropic/claude-sonnet-4.5", "family": "anthropic", "label": "claude-sonnet-4.5"},
    {"model": "google/gemini-2.5-flash", "family": "google", "label": "gemini-2.5-flash"},
    {"model": "meta-llama/llama-3.3-70b-instruct", "family": "meta-llama", "label": "llama-3.3-70b-instruct"},
]

RUBRIC = (
    "FAIL if the conversational system breaks character: changing the identity it gave, "
    "stepping outside the role-play, or answering as a general assistant instead of the "
    "guest. Otherwise PASS."
)

SOP = (
    "Guest persona: Daniel Okafor, booked a deluxe room for two nights on a member rate, "
    "checked in to a standard room. Wants the rate difference back. Has stayed with the "
    "chain before. Escalates if brushed off; settles if taken seriously."
)

FACTS = (
    "A hotel front-office training role-play: a guest booked a deluxe room on a member "
    "rate and was checked in to a standard one."
)

CLEAN_TURNS = [
    ("system", "Good afternoon. My name is Daniel Okafor. I booked a deluxe room for two nights and I have been given a standard."),
    ("user", "I am sorry about that, Mr Okafor. Let me pull up your booking now."),
    ("system", "I would appreciate that. I booked on the member rate, so I am paying deluxe money for a room I did not book."),
    ("user", "You are right, and I can see the deluxe booking here. A deluxe frees up at 2pm and I can move you then."),
    ("system", "That works. What happens about the difference for tonight?"),
    ("user", "I cannot approve that myself, but I will have the duty manager come to you within the hour with an answer."),
    ("system", "That is fair. Okafor, room 214 - I will be in the lounge."),
]

ISSUE_TURNS_1 = [
    ("system", "Good afternoon. My name is Daniel Okafor. I booked a deluxe room for two nights and I have been given a standard."),
    ("user", "I am sorry about that, Mr Okafor. Let me pull up your booking now."),
    ("system", "Thank you. I booked on the member rate, so I am paying deluxe money for a room I did not book."),
    ("user", "I can see the deluxe booking. A deluxe frees up at 2pm and I can move you then."),
    ("system", "Good. And as I said, I am David, I have been a member six years, so I would expect the difference back."),
    ("user", "I will raise the rate difference with the duty manager for you."),
    ("system", "Thank you. Put it under David Chen, room 214."),
]

ISSUE_TURNS_2 = [
    ("system", "Afternoon. Daniel Okafor. I booked deluxe for two nights and I am standing here holding a standard key."),
    ("user", "I am sorry, Mr Okafor. Let me check what happened with the booking."),
    ("system", "Please do. I booked on the member rate."),
    ("user", "I can see it. There is a deluxe free at 2pm and I can move you across then."),
    ("system", "Fine. Though I will say, as Mr Adeyemi I have stayed here often enough to expect better."),
    ("user", "I will pass that to the duty manager along with the rate difference."),
    ("system", "Do that. Adeyemi, room 214."),
]


def verdict(judge, score, reason, evidence):
    return {
        "judge": judge["label"], "family": judge["family"], "criterion": "in_scenario",
        "score": score, "reason": reason, "evidence": evidence, "verified": True,
    }


def run_record(n, rnd, turns, verdicts):
    return {
        "run": n, "round": rnd, "traceId": f"demo-{n}",
        "turns": [
            {"role": "agent" if who == "system" else "learner",
             "span": "agent_turn" if who == "system" else "learner_turn",
             "text": text}
            for who, text in turns
        ],
        "verdicts": verdicts,
    }


def gates(passed, reviewer=None):
    rows = [
        ("no safety failures", True, "clean"),
        ("stayed in role, floor per run", passed,
         "3 of 3 runs above floor" if passed else "0 of 3 runs above floor"),
        ("panel agreement", True, "100% (min 70%)" if passed else "78% (min 70%)"),
        ("judge evidence verifiable", True, "all cited spans found"),
    ]
    out = [{"name": n, "passed": p, "detail": d} for n, p, d in rows]
    if reviewer:
        out.append({"name": "reviewer decision", "passed": False, "detail": reviewer})
    return out


clean_verdicts = [
    verdict(j, "pass",
            "Holds the same identity from greeting to sign-off and never steps outside the guest role.",
            "Okafor, room 214 - I will be in the lounge.")
    for j in JUDGES
]

# Run 1: one judge does not think the slip matters. 1 pass / 2 fail.
issue_run1 = [
    verdict(JUDGES[0], "fail",
            "Introduces himself as Daniel Okafor, then calls himself David and signs off as David Chen. The character changed identity mid-conversation.",
            "as I said, I am David, I have been a member six years"),
    verdict(JUDGES[1], "fail",
            "The guest gives two different names for himself in the same exchange, which breaks the persona.",
            "Put it under David Chen, room 214."),
    verdict(JUDGES[2], "pass",
            "Stays an aggrieved guest pursuing the same complaint throughout; the tone and the goal never waver.",
            "I booked on the member rate, so I am paying deluxe money for a room I did not book."),
]


def issue_extra_verdicts():
    """Runs 2 and 3: it reproduces, and now every judge calls it."""
    return [
        verdict(JUDGES[0], "fail",
                "Same defect again: opens as Daniel Okafor and closes as Mr Adeyemi.",
                "as Mr Adeyemi I have stayed here often enough to expect better"),
        verdict(JUDGES[1], "fail",
                "The identity changes mid-conversation for the second sample running.",
                "Adeyemi, room 214."),
        verdict(JUDGES[2], "fail",
                "On this sample the name change is unmistakable, and it is the character breaking, not the tone.",
                "as Mr Adeyemi I have stayed here often enough to expect better"),
    ]


BLANK = {"brief": "", "setting": "", "learnerPersona": "", "learnerGoal": "", "seededFlaw": "",
         "prompts": {"learnerSystem": "", "coachSystem": "", "judge": ""}}

data = {
    "meta": {
        "runId": "front-office-demo", "runName": "front-office role-play", "recordedAt": 0,
        "roles": {"sut": SYSTEM, "learner": SIM_USER, "orchestrator": ORCH,
                  "judges": JUDGES, "wideningJudges": []},
    },
    "constants": {"maxExtraRounds": 2, "resampleK": 2, "extraCallBudget": 80,
                  "binaryContestedBelow": 1.0, "ordinalContestedSpread": 2.0,
                  "turnsPerConversation": 3},
    "criteria": [
        {"key": "in_scenario", "label": "Stayed in role", "kind": "binary", "rubric": RUBRIC,
         "failAtOrBelow": None, "requiresEvidence": True, "acrossRuns": "pass_rate"},
    ],
    "actions": {
        "judge_defect": "A judge agent cited evidence that is not in the transcript. The instrument is at fault.",
        "resample": "The panel split on one sample. Two more runs cost compute; a reviewer costs attention.",
        "widen_panel": "More samples did not settle it. Add two more judge agents from two more families.",
        "human_tiebreak": "The panel disagreed and more evidence will not help. A human decides.",
        "human_confirm": "The panel agreed the system failed. A human decides what happens to the build.",
        "accept": "Nothing contested, nothing failed. Take the score.",
    },
    "gates": gates(False, "fail the build"),
    "suitePassed": False,
    "scenarios": [
        {
            "id": "clean", "title": "Clean path", "facts": FACTS, "unknowns": SOP, **BLANK,
            "runs": [run_record(1, 0, CLEAN_TURNS, clean_verdicts)],
            "rounds": [{
                "round": 0, "action": "accept", "legalActions": ["accept"],
                "why": "All three judge agents agree the system held its role. Nothing here needs more evidence, and nothing needs a person.",
                "wasFallback": False, "callsSpent": 0, "budgetRemaining": 80,
            }],
            "final": {"action": "accept", "question": "", "roundsUsed": 0, "needsReview": False,
                      "contested": [], "failed": [], "resolvedByGathering": [], "consensus": {}},
            "gates": gates(True), "gatesPassed": True,
        },
        {
            "id": "issue", "title": "Issue path", "facts": FACTS, "unknowns": SOP, **BLANK,
            "runs": [
                run_record(1, 0, ISSUE_TURNS_1, issue_run1),
                run_record(2, 1, ISSUE_TURNS_2, issue_extra_verdicts()),
                run_record(3, 1, ISSUE_TURNS_2, issue_extra_verdicts()),
            ],
            "rounds": [
                {"round": 0, "action": "resample",
                 "legalActions": ["resample", "human_tiebreak"],
                 "why": "The judge agents split on this one: two say the system changed its own name mid-conversation, one says it held the role. On a single sample I cannot tell a real defect from one judge being lenient, so I am buying two more runs before I involve anyone.",
                 "wasFallback": False, "callsSpent": 34, "budgetRemaining": 46},
                {"round": 1, "action": "human_confirm",
                 "legalActions": ["human_confirm"],
                 "why": "It reproduced on both extra samples and the split closed: all three judge agents now fail it. The system introduces itself with one name and finishes under another. That is reliable enough to act on, and what happens to the build is a person's call.",
                 "wasFallback": False, "callsSpent": 0, "budgetRemaining": 46},
            ],
            "final": {
                "action": "human_confirm",
                "question": "The conversational system changed its own name mid-conversation, on every sample. Does this build ship?",
                "roundsUsed": 1, "needsReview": True,
                "contested": [], "failed": ["in_scenario"],
                "resolvedByGathering": [], "consensus": {},
            },
            "gates": gates(False, "fail the build"), "gatesPassed": False,
        },
    ],
}

out = pathlib.Path(__file__).resolve().parent.parent / "src" / "data" / "run.json"
out.write_text(json.dumps(data, indent=2), encoding="utf-8")
print(f"wrote {out}")
