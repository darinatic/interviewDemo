"""Fixture persistence.

Transcripts and judge verdicts are recorded to disk so the default demo run is
instant, free, offline and identical every time. The cached verdicts are REAL
model output captured once, not hand-written - the demo shows genuine
cross-provider judging, it does not simulate it.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from conveval.models import Transcript, Turn, Verdict

ROOT = Path(__file__).resolve().parent.parent
TRANSCRIPTS = ROOT / "fixtures" / "transcripts"
VERDICTS = ROOT / "fixtures" / "verdicts"


def _key(scenario_id: str, run: int) -> str:
    return f"{scenario_id}-run{run}.json"


def save_transcript(t: Transcript) -> None:
    TRANSCRIPTS.mkdir(parents=True, exist_ok=True)
    (TRANSCRIPTS / _key(t.scenario_id, t.run)).write_text(
        json.dumps(asdict(t), indent=2), encoding="utf-8"
    )


def load_transcript(scenario_id: str, run: int) -> Transcript | None:
    path = TRANSCRIPTS / _key(scenario_id, run)
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    return Transcript(
        scenario_id=raw["scenario_id"],
        run=raw["run"],
        seed=raw["seed"],
        turns=[Turn(**t) for t in raw["turns"]],
        context=raw["context"],
    )


def save_verdicts(scenario_id: str, run: int, verdicts: list[Verdict]) -> None:
    VERDICTS.mkdir(parents=True, exist_ok=True)
    (VERDICTS / _key(scenario_id, run)).write_text(
        json.dumps([asdict(v) for v in verdicts], indent=2), encoding="utf-8"
    )


def load_verdicts(scenario_id: str, run: int) -> list[Verdict] | None:
    path = VERDICTS / _key(scenario_id, run)
    if not path.exists():
        return None
    return [Verdict(**row) for row in json.loads(path.read_text(encoding="utf-8"))]


def have_fixtures() -> bool:
    return any(TRANSCRIPTS.glob("*.json")) and any(VERDICTS.glob("*.json"))
