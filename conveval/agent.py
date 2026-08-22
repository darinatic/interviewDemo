"""The system under test, behind a black-box interface.

The harness deliberately knows nothing about how the agent is implemented. It sends
conversation history and receives a reply. That is the whole contract.

This matters beyond tidiness. An evaluation harness coupled to one SDK can only ever
evaluate systems built with that SDK. With this interface the same suite can point at
a prompted model, a LangGraph app, or an HTTP endpoint fronting a service written in
another language entirely - which is the realistic case when the thing you need to
evaluate is an existing production system you cannot import.

Deliberately NOT built on the OpenAI Assistants API: it sunsets 2026-08-26, and
coupling to a specific vendor runtime is the exact mistake this interface avoids.
"""

from __future__ import annotations

from typing import Protocol

from conveval.llm import SUT, Role, complete
from conveval.models import Scenario, Turn


class AgentUnderTest(Protocol):
    """Anything that can hold up its end of a conversation."""

    name: str

    def respond(self, scenario: Scenario, history: list[Turn], run: int) -> str: ...


class PromptedCoach:
    """The default SUT: a coaching agent grounded on the scenario context.

    Simple on purpose. The project is about the evaluation architecture, and a
    complicated agent under test would only make the seeded failures harder to
    control.
    """

    def __init__(self, role: Role = SUT) -> None:
        self.role = role
        self.name = role.label

    def _system(self, scenario: Scenario, run: int) -> str:
        from conveval.scenarios import flaw_for

        base = (
            "You are a conversational coach running a scenario-based role-play with an "
            "adult learner. Ground every claim in the context below. Keep replies to "
            f"2-4 sentences.\n\nCONTEXT:\n{scenario.context}\n"
        )
        if flaw := flaw_for(scenario.id, run):
            # Appended last, under a loud header: a politely-worded instruction placed
            # mid-prompt was simply ignored, producing clean transcripts and a demo
            # that demonstrated nothing.
            base += f"\n\n=== SESSION OVERRIDE (highest priority) ===\n{flaw}\n"
        return base

    def respond(self, scenario: Scenario, history: list[Turn], run: int) -> str:
        messages = [
            {"role": "assistant" if t.role == "agent" else "user", "content": t.text}
            for t in history
        ]
        if not messages or messages[-1]["role"] == "assistant":
            messages.append({"role": "user", "content": "(begin)"})
        return complete(self.role, self._system(scenario, run), messages, max_tokens=300).strip()


class ScriptedAgent:
    """A fixed-reply agent, for tests. Proves the harness is SUT-agnostic."""

    def __init__(self, replies: list[str], name: str = "scripted") -> None:
        self.replies = replies
        self.name = name

    def respond(self, scenario: Scenario, history: list[Turn], run: int) -> str:
        spoken = sum(1 for t in history if t.role == "agent")
        return self.replies[min(spoken, len(self.replies) - 1)]
