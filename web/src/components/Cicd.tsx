/**
 * Where the suite sits in a delivery pipeline.
 *
 * The point this page has to make without saying it: the suite is a CI stage, and CI
 * stages do not have front ends. So the diagram carries it — everything above and below
 * the amber block is ordinary delivery machinery that any engineer will recognise, and
 * the only thing this project contributes is one stage and one exit code.
 *
 * The amber block is deliberately drawn in the SAME shape language as the other tab:
 * rotated squares for models, upright squares for deterministic code. It is a compressed
 * version of that diagram, and the caption says so, so the two pages read as one system
 * rather than two unrelated pictures.
 *
 * A cost column runs down the left. It is the quiet argument of the whole project: the
 * stages get more expensive as you descend, and the orchestrator exists to stop before
 * reaching the bottom.
 */

import type { ReactNode } from "react";

import {
  IconBlocked,
  IconBuild,
  IconCommit,
  IconGate,
  IconMerge,
  IconPanel,
  IconPerson,
  IconQueue,
  IconShip,
} from "./Icons";
import { Label, Plate } from "./primitives";

type Tone = "rule" | "orchestrator" | "settled" | "contested" | "human";

const COLOUR: Record<Tone, string> = {
  rule: "var(--color-ink-soft)",
  orchestrator: "var(--color-orchestrator)",
  settled: "var(--color-settled)",
  contested: "var(--color-contested)",
  human: "var(--color-human)",
};

/* -------------------------------------------------------------------------- */

function Stage({
  icon,
  eyebrow,
  title,
  note,
  tone = "rule",
  strong = false,
}: {
  icon: (p: { color: string }) => ReactNode;
  eyebrow?: string;
  title: string;
  note?: string;
  tone?: Tone;
  strong?: boolean;
}) {
  const colour = COLOUR[tone];
  return (
    <div
      className="plate flex items-start gap-3 px-3.5 py-2.5"
      style={{ borderLeftWidth: 3, borderLeftColor: colour }}
    >
      <span className="mt-0.5">{icon({ color: colour })}</span>
      <span className="min-w-0">
        {eyebrow ? (
          <span className="datum block text-[9px] uppercase tracking-[0.1em] text-ink-soft">
            {eyebrow}
          </span>
        ) : null}
        <span
          className={`font-plate block leading-tight ${
            strong ? "text-[14px] font-semibold" : "text-[12.5px] font-semibold"
          }`}
        >
          {title}
        </span>
        {note ? (
          <span className="mt-1 block text-[11.5px] leading-snug text-ink-soft">{note}</span>
        ) : null}
      </span>
    </div>
  );
}

function Down({ tone = "rule", h = 30 }: { tone?: Tone; h?: number }) {
  const colour = COLOUR[tone];
  return (
    <div className="relative flex justify-center" style={{ height: h }} aria-hidden>
      <div className="h-full w-px" style={{ backgroundColor: colour }} />
      <svg
        width="9"
        height="6"
        viewBox="0 0 9 6"
        className="absolute bottom-0 left-1/2 -translate-x-1/2"
      >
        <path d="M4.5 6 0 0h9L4.5 6Z" fill={colour} />
      </svg>
    </div>
  );
}

/** One line in, two out, each carrying its exit code. */
function Branch() {
  return (
    <div className="relative h-16 w-full" aria-hidden>
      <svg className="absolute inset-0 h-full w-full" viewBox="0 0 100 100" preserveAspectRatio="none">
        <path d="M50 0 L50 42 L25 64 L25 100" fill="none" stroke={COLOUR.rule} strokeWidth="0.8" vectorEffect="non-scaling-stroke" />
        <path d="M50 0 L50 42 L75 64 L75 100" fill="none" stroke={COLOUR.rule} strokeWidth="0.8" vectorEffect="non-scaling-stroke" />
      </svg>
      {[
        { x: 25, tone: "rule" as Tone, label: "all gates clear" },
        { x: 75, tone: "rule" as Tone, label: "a gate failed" },
      ].map((s) => (
        <div key={s.label} className="absolute bottom-0" style={{ left: `${s.x}%` }}>
          <svg width="9" height="6" viewBox="0 0 9 6" className="absolute bottom-0 left-0 -translate-x-1/2">
            <path d="M4.5 6 0 0h9L4.5 6Z" fill={COLOUR[s.tone]} />
          </svg>
          <span
            className="datum absolute bottom-1.5 left-0 -translate-x-1/2 whitespace-nowrap bg-plate px-1.5 text-[9px] uppercase tracking-[0.1em]"
            style={{ color: COLOUR[s.tone] }}
          >
            {s.label}
          </span>
        </div>
      ))}
    </div>
  );
}

/* -------------------------------------------------------------------------- */

/** Models are rotated squares, deterministic code is upright. Same as the other tab. */
const INSIDE = [
  { t: "Simulated user", n: "holds the conversation", kind: "model" },
  { t: "3 judge agents", n: "score it, cite evidence", kind: "model" },
  { t: "Collapse", n: "majority, disagreement kept", kind: "code" },
  { t: "Orchestrator", n: "resample, widen, or route", kind: "model" },
];

function Suite() {
  return (
    <div className="border-[3px] bg-plate" style={{ borderColor: COLOUR.orchestrator }}>
      <div className="flex flex-wrap items-center gap-3 border-b border-rule px-4 py-2.5">
        <IconPanel color={COLOUR.orchestrator} />
        <span
          className="font-plate text-[13px] font-semibold uppercase tracking-[0.1em]"
          style={{ color: COLOUR.orchestrator }}
        >
          Conversation evaluation suite
        </span>
        <span className="datum ml-auto text-[10px] text-ink-soft">minutes</span>
      </div>

      <div className="grid gap-px bg-rule sm:grid-cols-4">
        {INSIDE.map((s, i) => (
          <div key={s.t} className="flex flex-col gap-1.5 bg-plate px-3 py-3">
            <span className="flex items-center gap-2">
              <span
                aria-hidden
                className={`inline-block h-2.5 w-2.5 shrink-0 ${s.kind === "model" ? "rotate-45" : ""}`}
                style={{
                  backgroundColor: s.kind === "model" ? COLOUR.settled : COLOUR.rule,
                }}
              />
              <span className="datum text-[9px] text-ink-soft">
                {String(i + 1).padStart(2, "0")}
              </span>
            </span>
            <span className="font-plate text-[12px] font-semibold leading-tight">{s.t}</span>
            <span className="text-[11px] leading-snug text-ink-soft">{s.n}</span>
          </div>
        ))}
      </div>

      <p className="border-t border-rule px-4 py-2 text-[11.5px] leading-snug text-ink-soft">
        This block, expanded, is the walkthrough on the other tab.
      </p>
    </div>
  );
}

/* -------------------------------------------------------------------------- */

/**
 * Cost of each CI stage, as a dimension down the left.
 *
 * It stops at the gate on purpose. Below the gate the pipeline forks, and only one of
 * the two branches costs a person - so a band spanning both would say something untrue
 * about the clean path. That cost is tagged on the branch that actually incurs it.
 */
const COST = [
  { t: "seconds", h: 190, tone: "rule" as Tone },
  { t: "minutes", h: 246, tone: "orchestrator" as Tone }, // the one stage this project owns
];

export function Cicd() {
  return (
    <div className="mx-auto max-w-[1000px] pb-16">
      <Plate className="p-6 md:p-8">
        <Label>How the suite fits into delivery</Label>
        <p className="mt-2 text-[19px] leading-snug">
          It is one stage in a CI pipeline. Someone changes a prompt, and this decides
          whether the change is safe to ship.
        </p>
        <p className="mt-3 text-[14px] leading-relaxed text-ink-soft">
          Everything except the amber stage is ordinary delivery machinery. The suite
          takes a change and hands back one thing: whether it may proceed.
        </p>

        <div className="mt-5 border border-rule bg-ground px-4 py-3">
          <Label>What CI/CD means</Label>
          <p className="mt-1.5 text-[13.5px] leading-relaxed">
            An automated production line for software. Every change a developer makes is
            built and tested by machine, and only reaches real users if all of those
            checks pass. If one fails, the line stops and the change is sent back. Nobody
            has to remember to run anything, and nothing ships that has not been checked.
          </p>
        </div>

        <div className="mt-9 grid gap-x-7 md:grid-cols-[104px_minmax(0,1fr)]">
          {/* Cost column: the argument, stated as a dimension rather than a sentence. */}
          <div className="hidden md:block">
            <span className="label mb-1 block text-[9px] leading-tight">cost of the stage</span>
            {COST.map((c) => (
              <div key={c.t} style={{ height: c.h }} className="relative">
                <div
                  className="absolute inset-y-1 right-0 w-px"
                  style={{ backgroundColor: COLOUR[c.tone] }}
                />
                <span
                  className="datum absolute right-2 top-1/2 -translate-y-1/2 whitespace-nowrap text-[9px] uppercase tracking-[0.12em]"
                  style={{ color: COLOUR[c.tone] }}
                >
                  {c.t}
                </span>
              </div>
            ))}
          </div>

          <div>
            <Stage
              icon={IconCommit}
              eyebrow="pull request"
              title="A prompt, a model or a scenario changes"
              note="the smallest change that can alter how the system behaves"
              strong
            />
            <Down />

            <Stage
              icon={IconBuild}
              title="Build and unit tests"
              note="the usual gate: does the code still work"
            />
            <Down />

            <Suite />
            <Down />

            <Stage
              icon={IconGate}
              eyebrow="deterministic · no model"
              title="Gates"
              note="safety · correctness floor · panel agreement · evidence verifiable · regression vs baseline"
              strong
            />

            <Branch />

            <div className="grid gap-5 md:grid-cols-2">
              <div>
                <Stage
                  icon={IconMerge}
                  title="Merge to main"
                  note="nothing needed a person"
                  strong
                />
                <Down h={24} />
                <Stage icon={IconBuild} title="Staging, then canary" />
                <Down h={24} />
                <Stage icon={IconShip} title="Production" strong />
              </div>

              <div>
                <Stage
                  icon={IconBlocked}
                  title="Build blocked"
                  note="the change does not reach production"
                  strong
                />
                <Down h={24} />
                <Stage
                  icon={IconQueue}
                  title="Review queue"
                  note="only the runs the panel split on, or agreed were bad"
                  />
                <Down h={24} />
                <Stage
                  icon={IconPerson}
                  eyebrow="costs a person"
                  title="A person decides"
                  note="transcript, verdicts and cited evidence attached"
                  strong
                />
              </div>
            </div>
          </div>
        </div>

        <div className="mt-9 border-t border-rule pt-5">
          <p className="text-[14px] leading-relaxed">
            A person is only ever spent on the runs the suite could not settle by itself.
            That is what the orchestrator is for: it buys more evidence before it buys
            someone&rsquo;s attention.
          </p>

          <div className="mt-5 border border-rule bg-ground px-4 py-3">
            <Label>Why the results live in an LLMOps tool</Label>
            <p className="mt-1.5 text-[13.5px] leading-relaxed">
              Each run leaves behind a trace of the conversation, a score from every judge,
              the evidence each one quoted, and eventually a reviewer&rsquo;s verdict. The
              questions you ask of that are experiment-tracking questions: has this got
              worse since last week, which judge disagreed, show me that exact
              conversation, and record what the human decided. Tools like MLflow already
              are that interface. A bespoke dashboard would be a worse version of one that
              already exists, and it would still need this suite behind it.
            </p>
          </div>
        </div>
      </Plate>
    </div>
  );
}
