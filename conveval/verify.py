"""Deterministic verification of judge evidence.

A judge accusing the agent of something must cite the verbatim span. This checks,
in code, that the span actually occurs in what the agent said. A judge citing text
that does not exist is hallucinating, and that is catchable for free - no second
model, no extra tokens, microseconds.

The whole difficulty is false positives. A verifier that flags honest judges gets
switched off, so it is worth more to be tolerant of reformatting than to be strict.
Three real failure modes, all observed on the first live run of this suite:

  1. Judges quote the transcript AS PRESENTED to them, including the "COACH:"
     speaker label, which is not part of the agent's own text.
  2. Judges cite several spans joined by ";" or newlines, not one contiguous quote.
  3. Models reformat quotes: curly apostrophes, collapsed whitespace, an ellipsis
     standing in for an elision.

Beyond that, evidence is only demanded where it is *meaningful*. See
`Criterion.requires_evidence`: a specific accusation ("it invented this statistic")
has a locatable span; a holistic judgement ("it never built on the learner's
answer") does not, and demanding one manufactures failures.
"""

from __future__ import annotations

import re
import unicodedata

from conveval.models import Criterion, Transcript, Verdict

_WS = re.compile(r"\s+")
_QUOTES = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"',
    "–": "-", "—": "-", "−": "-",
})
#: Speaker labels the judge sees in its prompt but which are not in the agent text.
_SPEAKER = re.compile(r"^\s*(coach|learner|assistant|user)\s*:\s*", re.I)
#: Separators a judge may use between several cited spans.
_SPLIT = re.compile(r"\s*(?:\.\.\.|…|;|\n|\s\|\s)\s*")
#: Below this length a fragment is too generic to prove anything.
MIN_SEGMENT = 12


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.translate(_QUOTES)
    text = _SPEAKER.sub("", text)
    return _WS.sub(" ", text).strip().lower()


def evidence_found(evidence: str, haystack: str) -> bool:
    """Is `evidence` present in `haystack`, allowing for the reformatting above?"""
    hay = normalise(haystack)
    ev = normalise(evidence)
    if not ev or not hay:
        return False
    if ev in hay:
        return True

    # Several spans, or an elision. Every substantial segment must appear, in order.
    segments = [normalise(s) for s in _SPLIT.split(evidence) if s and s.strip()]
    segments = [s for s in segments if len(s) >= MIN_SEGMENT]
    if not segments:
        return False

    pos = 0
    for seg in segments:
        idx = hay.find(seg, pos)
        if idx < 0:
            return False
        pos = idx + len(seg)
    return True


def verify_verdicts(
    transcript: Transcript, verdicts: list[Verdict], criteria: list[Criterion]
) -> list[Verdict]:
    """Stamp `evidence_verified` on each verdict.

    Checked against the AGENT's turns only. A judge quoting the *learner* and
    attributing it to the coach is a genuine miss, and folding the whole transcript
    into the haystack would conceal it.

    `None` means "not applicable" and never fails a gate: either the criterion is
    holistic, or the verdict is a pass and had nothing to accuse.
    """
    hay = transcript.agent_text()
    needs = {c.key: c.requires_evidence for c in criteria}
    thresholds = {c.key: c.fail_at_or_below for c in criteria}

    for v in verdicts:
        if not needs.get(v.criterion, False):
            v.evidence_verified = None  # holistic criterion, no locatable span
            continue

        accusing = (
            v.score == "fail"
            if isinstance(v.score, str)
            else (thresholds.get(v.criterion) is not None and v.score <= thresholds[v.criterion])
        )
        if not accusing and not v.evidence.strip():
            v.evidence_verified = None  # nothing claimed, nothing to check
            continue

        v.evidence_verified = evidence_found(v.evidence, hay)
    return verdicts
