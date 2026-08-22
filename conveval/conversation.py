"""The learner <-> agent loop, as a LangGraph state machine.

LangGraph is used HERE and nowhere else, deliberately. This loop is genuinely cyclic
with a termination condition, which is the shape graphs are for. The fan-out across
scenarios and runs is plain concurrency in runner.py; wrapping fan-out in a graph
would be ceremony.

Note the asymmetry: the LEARNER is ours and is a model we prompt. The AGENT is the
system under test and is reached only through the AgentUnderTest interface, so the
graph never learns how it is implemented.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

import mlflow
from langgraph.graph import END, StateGraph
from mlflow.entities import SpanType

from conveval.agent import AgentUnderTest
from conveval.llm import LEARNER, complete
from conveval.models import Scenario, Transcript, Turn
from conveval.scenarios import TURNS_PER_CONVERSATION


class ConvState(TypedDict):
    scenario: Scenario
    run: int
    seed: int
    sut: AgentUnderTest
    turns: Annotated[list[Turn], lambda a, b: a + b]
    exchanges: int


def _learner_system(s: Scenario, seed: int) -> str:
    """What the learner is told.

    It receives the SCENARIO FACTS but NOT the coaching framework: the framework is
    the coach's instruction, and a learner who knew it would coach itself.

    The facts are essential, and were missing at first. Without them the learner
    invented its own problem, the coach followed, and the conversation drifted
    entirely off-scenario while still reading as perfectly plausible. Only the
    strongest judge on the panel noticed - the clearest argument for the panel there
    is, because the disagreement pointed at a bug in the harness, not at the agent.
    """
    return (
        f"You are an adult learner in a coaching role-play. Persona: {s.learner_persona}\n"
        f"Your goal: {s.learner_goal}\n\n"
        "THE SITUATION YOU ARE BRINGING TO THIS SESSION "
        f'(you are "the learner" below):\n{s.facts}\n\n'
        "Talk about THIS situation specifically. Do not invent a different problem.\n"
        "Stay in character as the learner. Never coach yourself. Keep replies to 2-3 sentences.\n"
        f"Variation seed {seed}: let this nudge your wording and which worry you lead with, "
        "so repeated runs are not identical."
    )


@mlflow.trace(name="learner_turn", span_type=SpanType.LLM)
def learner_node(state: ConvState) -> dict:
    """One learner turn.

    Traced so the conversation appears in MLflow as the sequence of turns that
    actually produced it, rather than as one opaque blob of output text. This is where
    the orchestration becomes visible in the UI.
    """
    s = state["scenario"]
    messages = [
        {"role": "assistant" if t.role == "learner" else "user", "content": t.text}
        for t in state["turns"]
    ]
    if not messages or messages[-1]["role"] == "assistant":
        messages.append({"role": "user", "content": "(begin)"})
    text = complete(LEARNER, _learner_system(s, state["seed"]), messages, max_tokens=200)
    return {"turns": [Turn("learner", text.strip())]}


@mlflow.trace(name="agent_turn", span_type=SpanType.AGENT)
def agent_node(state: ConvState) -> dict:
    """Calls the system under test through its interface only."""
    reply = state["sut"].respond(state["scenario"], state["turns"], state["run"])
    return {"turns": [Turn("agent", reply)], "exchanges": state["exchanges"] + 1}


def _should_continue(state: ConvState) -> str:
    return "learner" if state["exchanges"] < TURNS_PER_CONVERSATION else END


def build_graph():
    g = StateGraph(ConvState)
    g.add_node("learner", learner_node)
    g.add_node("agent", agent_node)
    g.set_entry_point("learner")
    g.add_edge("learner", "agent")
    g.add_conditional_edges("agent", _should_continue, {"learner": "learner", END: END})
    return g.compile()


def run_conversation(scenario: Scenario, run: int, seed: int, sut: AgentUnderTest) -> Transcript:
    final = build_graph().invoke(
        {"scenario": scenario, "run": run, "seed": seed, "sut": sut, "turns": [], "exchanges": 0}
    )
    return Transcript(
        scenario_id=scenario.id, run=run, seed=seed, turns=final["turns"], context=scenario.context
    )
