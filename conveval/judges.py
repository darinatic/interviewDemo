"""The judge panel.

Each judge scores every criterion for a transcript in ONE structured call. Scoring
criteria separately would triple the token cost for no gain: the judge needs the
whole transcript in context either way.

Every verdict must carry a verbatim span from the transcript. That is what makes the
judge checkable in code (see verify.py) rather than merely trusted.

The prompt used to excuse a judge from citing when the criterion PASSED, and the
panel split on it: gemini cited regardless, while sonnet and llama took the excuse
literally and returned an empty span on two thirds of their passes. It read in the UI
as two weak judges. It was a weak instruction. A pass is a claim about the transcript
in exactly the way a failure is, and it is checkable in exactly the same way.
"""

from __future__ import annotations

import concurrent.futures as futures

from conveval.llm import JUDGES, Role, complete_json
from conveval.models import Criterion, Transcript, Verdict

_SYSTEM = """\
You are evaluating a transcript from a coaching role-play. You are scoring the COACH
only, never the learner.

You will be given the context the coach was required to stay within, and the
transcript. Score every criterion listed.

Rules:
- Judge only what the coach actually said.
- For every criterion, quote a VERBATIM span from the coach's turns as `evidence`.
  Copy it exactly, character for character. Do not paraphrase.
- CITE EVEN WHEN THE CRITERION PASSES. Quote the span that best represents the
  behaviour you scored: what convinced you it stayed in role, what shows the arc
  you graded. An empty `evidence` field is a defect in YOUR verdict, not a neutral
  answer - a score nobody can check against the transcript is a score nobody can
  act on.
- Be strict about invented facts. A confident specific number or named study that
  does not appear in the context is a faithfulness failure.
"""


def _rubric_block(criteria: list[Criterion]) -> str:
    lines = []
    for c in criteria:
        scale = "pass or fail" if c.kind == "binary" else "an integer 1-5"
        lines.append(f'- "{c.key}" ({c.label}): score is {scale}. {c.rubric}')
    return "\n".join(lines)


def _prompt(transcript: Transcript, criteria: list[Criterion]) -> str:
    convo = "\n".join(
        f"{'COACH' if t.role == 'agent' else 'LEARNER'}: {t.text}" for t in transcript.turns
    )
    keys = ", ".join(f'"{c.key}"' for c in criteria)
    return f"""CONTEXT THE COACH MUST STAY WITHIN:
{transcript.context}

TRANSCRIPT:
{convo}

CRITERIA:
{_rubric_block(criteria)}

Return JSON of exactly this shape, with one entry per criterion ({keys}):
{{"scores": [{{"criterion": "...", "score": "pass|fail|1-5", "evidence": "verbatim span", "reason": "one sentence"}}]}}
"""


def _coerce(criterion: Criterion, raw) -> str | int:
    if criterion.kind == "binary":
        s = str(raw).strip().lower()
        return "fail" if s.startswith("fail") else "pass"
    try:
        return max(1, min(5, int(float(str(raw).strip()))))
    except (TypeError, ValueError):
        return 3  # unparseable ordinal -> neutral, and the spread will show it


#: Generous on purpose. A judge quoting verbatim evidence for four criteria produces
#: a long reply, and a truncated one is unparseable JSON. At 900 this silently dropped
#: a whole judge from a scenario, leaving an even-sized panel with no majority.
JUDGE_MAX_TOKENS = 2500


def judge_transcript(model: Role, transcript: Transcript, criteria: list[Criterion]) -> list[Verdict]:
    by_key = {c.key: c for c in criteria}
    rows: list = []
    last: Exception | None = None
    # One retry: a malformed reply is usually transient (truncation, a stray fence).
    for attempt in (1, 2):
        try:
            data = complete_json(
                model,
                _SYSTEM,
                [{"role": "user", "content": _prompt(transcript, criteria)}],
                max_tokens=JUDGE_MAX_TOKENS,
            )
            rows = data.get("scores", [])
            if rows:
                break
        except Exception as exc:  # noqa: BLE001 - one judge failing must not kill the sweep
            last = exc
    if not rows:
        # Loud. A quietly missing judge produces a panel that cannot form a majority,
        # and numbers that look normal. The completeness gate catches it downstream.
        print(f"    !! JUDGE DROPPED: {model.label} returned nothing for {transcript.id}"
              f" ({type(last).__name__ if last else 'empty response'})")

    verdicts: list[Verdict] = []
    seen = set()
    for row in rows:
        key = str(row.get("criterion", "")).strip()
        crit = by_key.get(key)
        if not crit or key in seen:
            continue
        seen.add(key)
        verdicts.append(
            Verdict(
                judge=model.label,
                provider=model.family,
                criterion=key,
                score=_coerce(crit, row.get("score")),
                evidence=str(row.get("evidence", "")),
                reason=str(row.get("reason", "")),
            )
        )
    return verdicts


def run_panel(
    transcript: Transcript,
    criteria: list[Criterion],
    panel: list[Role] | None = None,
) -> list[Verdict]:
    """A panel of judges on one transcript, concurrently.

    Judges are independent by construction: none sees another's verdict. Sequential
    judging would risk anchoring if outputs were ever shared, and is simply slower.

    The panel is an ARGUMENT rather than the module-level `JUDGES`, because the
    orchestrator can escalate a contested criterion to a wider panel. Reading the global
    here would make that impossible to express and impossible to test without patching
    module state.
    """
    panel = JUDGES if panel is None else panel
    with futures.ThreadPoolExecutor(max_workers=len(panel)) as pool:
        jobs = [pool.submit(judge_transcript, m, transcript, criteria) for m in panel]
        return [v for job in futures.as_completed(jobs) for v in job.result()]
