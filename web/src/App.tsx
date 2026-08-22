import { useEffect, useMemo, useRef, useState } from "react";

import { Flow } from "./components/Flow";
import { Setup } from "./components/Setup";
import { Label } from "./components/primitives";
import { STEP_MS, buildBeats, locate, totalSteps } from "./lib/film";
import { HEADLINE_CRITERION } from "./lib/film";
import type { NodeId } from "./lib/film";
import { scenarios } from "./lib/run";


/**
 * Two paths, not three. `unfaithful` and `disagreement` reach the same outcome by
 * slightly different routes - both end with a human being asked - so they are one story.
 * `unfaithful` plays it because its arc is fuller: judges split, compute is spent, the
 * split resolves, and the thing compute could not settle is handed over.
 */
const PATHS = [
  { id: "clean", label: "Clean path" },
  { id: "issue", label: "Issue path" },
];

export default function App() {
  const [scenarioId, setScenarioId] = useState(
    // Open on the scenario with the fullest arc: judges split, the orchestrator spends
    // compute, the split resolves, and the one thing compute could not settle goes to a
    // person. The others are shorter stories and make more sense once this is seen.
    PATHS[0].id,
  );
  const scenario = scenarios.find((s) => s.id === scenarioId)!;

  const beats = useMemo(() => buildBeats(scenario), [scenario]);
  const total = useMemo(() => totalSteps(beats), [beats]);
  const criterionKey = HEADLINE_CRITERION;

  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [answered, setAnswered] = useState(false);
  const [open, setOpen] = useState<NodeId | null>(null);
  const timer = useRef<number | null>(null);

  const { beat: beatIndex, within } = locate(beats, step);
  const beat = beats[beatIndex];
  const atEnd = step >= total - 1;

  useEffect(() => {
    setStep(0);
    setPlaying(false);
    setAnswered(false);
    setOpen(null);
  }, [scenarioId]);

  useEffect(() => {
    if (!playing || atEnd) return;
    const reduced = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    if (reduced) return; // no autoplay when motion is unwelcome; the controls still work
    timer.current = window.setTimeout(() => setStep((s) => s + 1), STEP_MS[beat.kind]);
    return () => {
      if (timer.current) window.clearTimeout(timer.current);
    };
  }, [playing, step, atEnd, beat.kind]);

  // The diagram outgrows the viewport, so the playhead brings itself into view. Without
  // this the film carries on happening below the fold and looks like it has stopped.
  useEffect(() => {
    const el = document.querySelector('[data-state="active"]');
    el?.scrollIntoView({ behavior: "smooth", block: "center" });
  }, [beat.node, beat.kind]);

  return (
    <div className="flex min-h-screen flex-col">
      <div className="sticky top-0 z-20">
      <header className="border-b border-rule bg-plate">
        <div className="mx-auto flex max-w-[1100px] flex-wrap items-center justify-between gap-3 px-6 py-3">
          <span className="plate-title text-[15px] uppercase tracking-[0.08em]">
            Conversation&nbsp;Eval
          </span>
          <div className="flex flex-wrap items-center gap-2">
            <span className="label">Pick a path</span>
            {PATHS.map((s) => (
              <button
                key={s.id}
                type="button"
                onClick={() => setScenarioId(s.id)}
                aria-pressed={s.id === scenarioId}
                className={`font-plate border-2 px-4 py-2 text-[12px] font-semibold uppercase tracking-[0.1em] transition-colors ${
                  s.id === scenarioId
                    ? "bg-ink text-plate"
                    : "bg-plate text-ink hover:bg-rule-faint"
                }`}
                style={{
                  borderColor:
                    s.id === scenarioId
                      ? "var(--color-ink)"
                      : s.id === "issue"
                        ? "var(--color-contested)"
                        : "var(--color-settled)",
                }}
              >
                {s.label}
              </button>
            ))}
          </div>
        </div>
      </header>

      {/* Transport. A film you cannot pause is a film you cannot follow. */}
      <div className="border-b border-rule bg-plate">
        <div className="mx-auto flex max-w-[1100px] flex-wrap items-center gap-4 px-6 py-3">
          <button
            type="button"
            onClick={() => (atEnd ? (setStep(0), setPlaying(true)) : setPlaying((p) => !p))}
            className="font-plate bg-ink px-4 py-1.5 text-[12px] font-semibold uppercase tracking-[0.1em] text-plate"
          >
            {atEnd ? "Replay" : playing ? "Pause" : "Play"}
          </button>

          <div className="flex items-stretch gap-px border border-rule bg-rule">
            {[
              { dir: -1 as const, glyph: "◀", text: "Back", label: "Previous step" },
              { dir: 1 as const, glyph: "▶", text: "Next", label: "Next step" },
            ].map(({ dir, glyph, text, label }) => {
              const blocked = dir < 0 ? step === 0 : step >= total - 1;
              return (
                <button
                  key={label}
                  type="button"
                  onClick={() => {
                    setPlaying(false);
                    setStep((n) => Math.min(total - 1, Math.max(0, n + dir)));
                  }}
                  disabled={blocked}
                  aria-label={label}
                  className="font-plate flex items-center gap-1.5 bg-plate px-3 py-1.5 text-[11px] font-semibold uppercase tracking-[0.1em] text-ink transition-colors hover:bg-rule-faint disabled:cursor-not-allowed disabled:text-rule"
                >
                  {dir < 0 ? (
                    <>
                      <span aria-hidden className="text-[9px]">{glyph}</span>
                      {text}
                    </>
                  ) : (
                    <>
                      {text}
                      <span aria-hidden className="text-[9px]">{glyph}</span>
                    </>
                  )}
                </button>
              );
            })}
          </div>

          <input
            type="range"
            min={0}
            max={Math.max(0, total - 1)}
            value={step}
            onChange={(e) => {
              setPlaying(false);
              setStep(Number(e.target.value));
            }}
            aria-label="Scrub through the run"
            className="min-w-[160px] flex-1 accent-[var(--color-ink)]"
          />

          <span className="datum text-[11px] text-ink-soft">
            {atEnd ? "click any node to open it · " : ""}
            {step + 1} / {total}
          </span>
        </div>
      </div>

      </div>

      <main className="mx-auto w-full max-w-[1100px] flex-1 px-6 py-8">
        {beat.kind === "setup" ? (
          <Setup scenario={scenario} />
        ) : (
          <Flow
            scenario={scenario}
            beat={beat}
            within={within}
            criterionKey={criterionKey}
            answered={answered}
            onAnswer={() => setAnswered(true)}
            finished={atEnd}
            open={open}
            toggle={(n) => setOpen((cur) => (cur === n ? null : n))}
          />
        )}
      </main>

      <footer className="border-t border-rule bg-ground">
        <div className="mx-auto flex max-w-[1100px] flex-wrap items-center gap-x-6 gap-y-1 px-6 py-2.5">
          <Label>Illustrative walkthrough</Label>
          <span className="text-[12px] text-ink-soft">
            A worked example of the architecture. The scenario and the verdicts are
            written to make the mechanism legible.
          </span>
        </div>
      </footer>
    </div>
  );
}
