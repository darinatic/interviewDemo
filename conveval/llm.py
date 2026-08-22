"""OpenRouter client and the per-role model assignment.

One key, many model families. That matters here for a specific reason: a judge from
the same family as the system under test exhibits self-preference bias, scoring its
own family's output higher. Routing every role through OpenRouter means the panel can
span Anthropic, Google and Meta while the agent under test runs on OpenAI, so no judge
is marking its own homework.

Every role is env-overridable, so the panel can be reconfigured without touching code:

    CONVEVAL_JUDGE_1=mistralai/mistral-small-3.2-24b-instruct python -m conveval run --live
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://openrouter.ai/api/v1"


@dataclass(frozen=True)
class Role:
    """A model bound to a job in the pipeline."""

    name: str
    model: str

    @property
    def family(self) -> str:
        """Provider family, e.g. "anthropic". Used to assert judge independence."""
        return self.model.split("/", 1)[0]

    @property
    def label(self) -> str:
        return self.model.split("/", 1)[-1]


def _role(name: str, default: str) -> Role:
    return Role(name, os.getenv(f"CONVEVAL_{name.upper()}", default))


# Every role gets a DIFFERENT model, and each choice has a reason. Reusing one model
# across roles is not just untidy: a learner sharing the agent's model tends to mirror
# its style, producing unrealistically smooth conversations that never stress the agent.
#
#: The thing being evaluated. Swap this to evaluate something else.
SUT = _role("sut", "openai/gpt-4o-mini")
#: Only has to play a plausible human, so a small cheap model is right. Deliberately a
#: different family from the SUT so it does not mirror the agent's phrasing.
LEARNER = _role("learner", "mistralai/mistral-small-3.2-24b-instruct")
#: Narrates the already-computed result. It does NOT decide anything (see runner.py),
#: so it needs prose fluency, not judgement - the cheapest capable model will do.
ORCHESTRATOR = _role("orchestrator", "anthropic/claude-haiku-4.5")

#: Judging is the quality-critical role: a weak judge produces noise, noise lowers
#: panel agreement, and low agreement makes every downstream number meaningless. So
#: these are NOT all cheap. The panel is mixed-tier on purpose - one strong judge to
#: anchor, two mid-tier - which is also what a real cost-controlled panel looks like.
#: None shares a family with the SUT: see judge_families_are_independent().
JUDGES = [
    _role("judge_1", "anthropic/claude-sonnet-4.5"),          # strong, anchors the panel
    _role("judge_2", "google/gemini-2.5-flash"),              # mid tier
    _role("judge_3", "meta-llama/llama-3.3-70b-instruct"),    # mid tier, open weight
]


#: The two judges added when the orchestrator escalates a contested criterion.
#:
#: TWO, never one. The panel must stay ODD: an even panel cannot form a majority on a
#: binary criterion, and this suite has already been bitten by an even panel once, when a
#: truncated reply silently dropped a judge. Escalation must not reintroduce that.
#:
#: Both are new FAMILIES, not merely new models. A second Anthropic judge would buy
#: correlated opinions, and the panel exists to sample independent ones.
#: Verified against OpenRouter's live catalogue 2026-08-21.
WIDENING_JUDGES = [
    _role("judge_4", "qwen/qwen-2.5-72b-instruct"),
    _role("judge_5", "mistralai/mistral-large"),
]


def widened_panel() -> list[Role]:
    """The 5-judge panel used when the orchestrator escalates a contested criterion."""
    return [*JUDGES, *WIDENING_JUDGES]


def judge_families_are_independent(panel: list[Role] | None = None) -> tuple[bool, str]:
    """Is any judge from the same family as the system under test?

    Surfaced in the report rather than assumed: swapping a judge via env var could
    silently reintroduce the bias the panel exists to avoid.

    Takes an explicit panel so the WIDENED panel is checked too. Escalation that quietly
    added a judge from the SUT's family would undo the property at exactly the moment the
    suite is least sure of itself.
    """
    panel = JUDGES if panel is None else panel
    clash = [j.label for j in panel if j.family == SUT.family]
    if clash:
        return False, f"judges sharing the SUT family ({SUT.family}): {', '.join(clash)}"
    families = sorted({j.family for j in panel})
    return True, f"{len(panel)} judges across {len(families)} families: {', '.join(families)}"


class MissingKey(RuntimeError):
    pass


def _client():
    from openai import OpenAI

    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise MissingKey(
            "OPENROUTER_API_KEY is not set. Put it in .env (see .env.example)."
        )
    return OpenAI(base_url=BASE_URL, api_key=key)


#: Retries for TRANSIENT upstream failures, with exponential backoff.
#:
#: OpenRouter fans out to several providers per model, and any of them can rate-limit
#: or time out independently. Observed for real: the learner model came back 504 after
#: two upstream providers returned 429, which killed an entire evaluation nine calls
#: in - every conversation and every verdict discarded because one turn was unlucky.
#:
#: Only transient statuses are retried. A 401 or a bad model id is not going to fix
#: itself, and retrying it three times just makes the failure slower to read.
_RETRY_STATUSES = frozenset({408, 409, 429, 500, 502, 503, 504})
MAX_ATTEMPTS = 3
BACKOFF_SECONDS = 2.0


def _is_transient(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None) or getattr(
        getattr(exc, "response", None), "status_code", None
    )
    return status in _RETRY_STATUSES


def complete(role: Role, system: str, messages: list[dict], max_tokens: int = 900) -> str:
    payload = [{"role": "system", "content": system}, *messages]
    last: Exception | None = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            resp = _client().chat.completions.create(
                model=role.model, max_tokens=max_tokens, messages=payload
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001 - re-raised below unless transient
            if not _is_transient(exc) or attempt == MAX_ATTEMPTS:
                raise
            last = exc
            delay = BACKOFF_SECONDS * (2 ** (attempt - 1))
            # Loud, because a silent retry hides a provider that is failing constantly
            # and makes the suite look merely slow.
            print(f"    ~ {role.label}: transient upstream error "
                  f"({type(exc).__name__}), retry {attempt}/{MAX_ATTEMPTS - 1} "
                  f"in {delay:.0f}s")
            time.sleep(delay)

    raise RuntimeError(f"unreachable: {last}")


_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S)


def complete_json(role: Role, system: str, messages: list[dict], max_tokens: int = 900) -> dict:
    """Completion parsed as JSON.

    A prompt contract plus tolerant parsing, rather than provider-native structured
    output: the panel spans several families through OpenRouter and their structured
    output support is not uniform. The judges must be comparable, so the weakest
    common mechanism is the right one.
    """
    raw = complete(
        role,
        system + "\n\nReply with JSON only. No prose, no code fence.",
        messages,
        max_tokens,
    )
    text = raw.strip()
    if fenced := _FENCE.search(text):
        text = fenced.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise
