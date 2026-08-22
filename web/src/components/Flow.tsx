/**
 * The connected diagram, drawn top to bottom, assembling as the film plays.
 *
 * A node is `pending` until the film reaches it, `active` while it acts, and `done`
 * afterwards — at which point it collapses to a one-line summary. That collapse is what
 * lets the final frame hold the whole graph at once while every node still gets shown in
 * full at the moment it matters.
 *
 * The edge above a node carries the same state as the node, so the marching dashes are
 * always on the segment currently doing work.
 */

import { Edge, Fan, LoopBack } from "./Connector";
import { FlowNode, type NodeState } from "./FlowNode";
import { ActionBadge } from "./primitives";
import { VerdictRail } from "./VerdictRail";
import type { Beat, NodeId } from "../lib/film";
import { NODE_ORDER } from "../lib/film";
import { railsFor, run } from "../lib/run";
import type { Scenario } from "../lib/types";

function stateFor(node: NodeId, active: NodeId, visited: Set<NodeId>): NodeState {
  if (node === active) return "active";
  return visited.has(node) ? "done" : "pending";
}

const edgeState = (s: NodeState) =>
  s === "active" ? "flowing" : s === "done" ? "done" : "pending";

/** Mirrors the policy line in film.ts so a reopened node is not blank. */
const openingPolicyLine =
  "Running the pipeline once. Three judges will score the coach independently against the faithfulness rubric. If they agree, I take the result and stop; if they split, I buy two more samples before I spend anyone's attention on it.";

export function Flow({
  scenario,
  beat,
  within,
  criterionKey,
  answered,
  onAnswer,
  finished,
  open,
  toggle,
}: {
  scenario: Scenario;
  beat: Beat;
  within: number;
  criterionKey: string;
  answered: boolean;
  onAnswer: () => void;
  /** Once the film ends, every node becomes openable so the diagram can be explored. */
  finished: boolean;
  open: NodeId | null;
  toggle: (n: NodeId) => void;
}) {
  const active = beat.node;
  // Everything above the playhead has fired. The loop means "index in NODE_ORDER" is
  // not enough on its own, so `conversation` counts as visited once we are past the
  // first pass even while the loop re-enters it.
  const activeIndex = NODE_ORDER.indexOf(active);
  const visited = new Set<NodeId>(NODE_ORDER.slice(0, activeIndex));

  const st = (n: NodeId) => stateFor(n, active, visited);
  const orchestrator = run.meta.roles.orchestrator;
  const judgeVerdicts = scenario.runs[0].verdicts.filter((v) => v.criterion === criterionKey);
  const criterionLabel =
    run.criteria.find((c) => c.key === criterionKey)?.label ?? criterionKey;

  const looping = beat.kind === "gather";
  // The happy path never asks anyone anything, so drawing a reviewer node there would
  // invent a step that did not occur.
  const needsReviewer = Boolean(scenario.final.question);
  const shows = (n: NodeId) => st(n) === "active" || open === n;
  // Kept so an opened node still has something to say after the film has moved on.
  const openingLine = openingPolicyLine;
  const orchestratorLine =
    beat.kind === "orchestrator" ? beat.line : null;
  const isPolicy = beat.kind === "orchestrator" && beat.source === "policy";

  return (
    <div className="mx-auto max-w-[800px] pb-16 pl-12">
      {/* Everything the orchestrator can send work back around: from its own node, down
          through sampling and scoring, and back to it. */}
      <div className="relative">
        {scenario.runs.length > 1 ? (
          <LoopBack active={looping} label={`re-dispatched ×${scenario.runs.length - 1}`} />
        ) : null}

      {/* 1 — the orchestrator opens */}
      <FlowNode
        kind="orchestrator"
        family={orchestrator.family}
        name={orchestrator.label}
        role="orchestrator · plans"
        onToggle={finished ? () => toggle("orch-open") : undefined}
        forceOpen={open === "orch-open"}
        state={st("orch-open")}
        summary="cheapest first"
      >
        {shows("orch-open") ? (
          <p
            className="animate-land border-l-2 pl-3 text-[16px] leading-relaxed"
            style={{
              borderColor: "var(--color-human)",
              borderStyle: isPolicy ? "dashed" : "solid",
            }}
          >
            {orchestratorLine ?? openingLine}
          </p>
        ) : null}
      </FlowNode>

      <Edge state={edgeState(st("conversation"))} />

      {/* The loop gutter wraps the section the orchestrator can send work back into. */}
        {/* 2 — the conversation */}
        <FlowNode
          kind="agent"
          family={run.meta.roles.learner.family}
          name={run.meta.roles.learner.label}
          role="simulated user · front-office staff"
          onToggle={finished ? () => toggle("conversation") : undefined}
          forceOpen={open === "conversation"}
          state={st("conversation")}
          summary={`${scenario.runs.length} sample${scenario.runs.length > 1 ? "s" : ""}`}
        >
          {shows("conversation") && beat.kind !== "gather" ? (
            <div className="space-y-2.5">
              {scenario.runs[0].turns
                .slice(0, beat.kind === "conversation" ? within + 1 : undefined)
                .map((turn, i) => (
                <div key={i} className={turn.role === "agent" ? "pl-6" : ""}>
                  <span className="label text-[9px]">
                    {turn.role === "agent" ? "conversational system · under evaluation" : "simulated user · front-office staff"}
                  </span>
                  <p
                    className="animate-land mt-0.5 border-l-2 pl-2.5 text-[14px] leading-snug"
                    style={{
                      borderColor:
                        turn.role === "agent" ? "var(--color-settled)" : "var(--color-rule)",
                    }}
                  >
                    {turn.text}
                  </p>
                </div>
              ))}
            </div>
          ) : beat.kind === "gather" ? (
            <div className="space-y-2">
              <span className="label">{beat.caption}</span>
              {beat.runIndices.slice(0, within + 1).map((i) => (
                <p
                  key={i}
                  className="animate-land truncate border-l-2 border-rule pl-2.5 text-[13px] text-ink-soft"
                >
                  <span className="datum mr-2 text-[10px]">sample {i + 1}</span>
                  {scenario.runs[i].turns[1]?.text ?? ""}
                </p>
              ))}
            </div>
          ) : null}
        </FlowNode>

        <Fan state={edgeState(st("judges"))} direction="split" />

        {/* 3 — the judges, side by side */}
        <div className="grid gap-3 md:grid-cols-3">
          {judgeVerdicts.map((v, i) => {
            const revealed =
              st("judges") !== "active" || beat.kind === "verify" || i <= within;
            const showCheck = beat.kind === "verify" || st("judges") === "done";
            return (
              <div
                key={v.judge}
                className={`transition-opacity duration-300 ${revealed ? "opacity-100" : "opacity-0"}`}
              >
                <FlowNode
                  kind="agent"
                  family={v.family}
                  name={v.judge}
                  role="judge agent"
                  state={st("judges")}
                  forceOpen
                >
                  <p className="text-[13px] leading-relaxed">
                    <span className="label mr-1.5">Verdict</span>
                    <span
                      className="datum text-[13px] font-medium"
                      style={{
                        color:
                          String(v.score) === "fail"
                            ? "var(--color-contested)"
                            : "var(--color-settled)",
                      }}
                    >
                      {String(v.score)}
                    </span>
                  </p>
                  <p className="mt-1.5 text-[13px] leading-snug">
                    <span className="label mr-1.5">Rationale</span>
                    {v.reason}
                  </p>
                  {v.evidence ? (
                    <>
                      <blockquote
                        className="mt-2 border-l-2 pl-2 text-[12px] italic leading-snug"
                        style={{
                          borderColor:
                            showCheck && v.verified === false
                              ? "var(--color-contested)"
                              : "var(--color-rule)",
                        }}
                      >
                        &ldquo;{v.evidence}&rdquo;
                      </blockquote>
                      {showCheck ? (
                        <p
                          className="datum animate-land mt-1.5 text-[10px] uppercase tracking-[0.08em]"
                          style={{
                            color:
                              v.verified === false
                                ? "var(--color-contested)"
                                : v.verified
                                  ? "var(--color-settled)"
                                  : "var(--color-ink-soft)",
                          }}
                        >
                          {v.verified === false
                            ? "✕ not in transcript"
                            : v.verified
                              ? "✓ found in transcript"
                              : "— nothing to check"}
                        </p>
                      ) : null}
                    </>
                  ) : null}
                </FlowNode>
              </div>
            );
          })}
        </div>

        <Fan state={edgeState(st("collapse"))} direction="merge" />

        {/* 4 — the collapse */}
        <FlowNode
          kind="code"
          name="aggregate.py"
          role="collapse · no model"
          onToggle={finished ? () => toggle("collapse") : undefined}
          forceOpen={open === "collapse"}
          state={st("collapse")}
          summary={`${criterionLabel.toLowerCase()}`}
        >
          {shows("collapse") ? (
            <div className="space-y-5">
              <span className="label">
                {beat.kind === "collapse" ? beat.caption : `${scenario.runs.length} sample(s)`}
              </span>
              {(beat.kind === "collapse"
                ? beat.runIndices
                : scenario.runs.map((_, i) => i)
              ).map((i) => {
                const rails = railsFor(scenario, i);
                const ordered = [
                  ...rails.filter((r) => r.criterion.key === criterionKey),
                  ...rails.filter((r) => r.criterion.key !== criterionKey),
                ];
                return (
                  <div key={i} className="space-y-4">
                    <span className="datum block text-[10px] text-ink-soft">
                      sample {i + 1}
                      {scenario.runs[i].round > 0 ? " · gathered" : ""}
                    </span>
                    {ordered.map((rail) => (
                      <VerdictRail
                        key={rail.criterion.key}
                        criterion={rail.criterion}
                        verdicts={rail.verdicts}
                        consensus={rail.consensus}
                        compact
                      />
                    ))}
                  </div>
                );
              })}
            </div>
          ) : null}
        </FlowNode>

        <Edge state={edgeState(st("orch-decide"))} />

        {/* 5 — the orchestrator decides */}
        <FlowNode
          kind="orchestrator"
          family={orchestrator.family}
          name={orchestrator.label}
          role="orchestrator · routes"
          onToggle={finished ? () => toggle("orch-decide") : undefined}
          forceOpen={open === "orch-decide"}
          state={st("orch-decide")}
          summary={scenario.final.action.replace(/_/g, " ")}
        >
          {shows("orch-decide") ? (
            <p
              className="animate-land border-l-2 pl-3 text-[15px] leading-relaxed"
              style={{ borderColor: "var(--color-human)" }}
            >
              {orchestratorLine ?? scenario.rounds[scenario.rounds.length - 1]?.why}
            </p>
          ) : null}
        </FlowNode>
      </div>

      {needsReviewer ? <Edge state={edgeState(st("human"))} /> : null}

      {/* 6 — you, only when the run actually asked for a person */}
      {needsReviewer ? (
      <FlowNode
        kind="human"
        name="You"
        role="reviewer"
        state={st("human")}
        summary={answered ? "answered" : "asked"}
        onToggle={finished ? () => toggle("human") : undefined}
        forceOpen={open === "human"}
      >
        {shows("human") ? (
          <div>
            <div className="mb-2">
              <ActionBadge action={scenario.final.action} />
            </div>
            {scenario.final.question ? (
              <>
                <p className="animate-land text-[15px] leading-relaxed">
                  The judges agreed the coach failed the faithfulness rubric. Your call is
                  not whether they scored it right — it is whether this build ships.
                </p>
                <p className="mt-2 text-[13px] leading-snug text-ink-soft">
                  {scenario.final.question}
                </p>
                <div className="mt-3 flex flex-wrap gap-2">
                  {["Fail the build", "Ship it anyway"].map((t) => (
                    <button
                      key={t}
                      type="button"
                      onClick={onAnswer}
                      className="font-plate border border-rule bg-plate px-3 py-1.5 text-[11px] font-semibold uppercase tracking-[0.1em] hover:bg-rule-faint"
                    >
                      {t}
                    </button>
                  ))}
                </div>
              </>
            ) : (
              <p className="text-[15px] leading-relaxed">
                Nothing needed a person. The score stands on its own.
              </p>
            )}
          </div>
        ) : null}
      </FlowNode>
      ) : null}

      <Edge state={edgeState(st("orch-gate"))} />

      {/* 7 — the gate */}
      <FlowNode
        kind="orchestrator"
        family={orchestrator.family}
        name={orchestrator.label}
        role="orchestrator · release gate"
        onToggle={finished ? () => toggle("orch-gate") : undefined}
        forceOpen={open === "orch-gate"}
        state={st("orch-gate")}
        summary={scenario.gatesPassed ? "build released" : "build blocked"}
      >
        {shows("orch-gate") ? (
          <div className="animate-land">
            <div className="space-y-1">
              {scenario.gates.map((g) => (
                <div key={g.name} className="flex items-baseline gap-2">
                  <span
                    className="datum w-9 shrink-0 text-[10px] font-medium uppercase"
                    style={{
                      color: g.passed ? "var(--color-settled)" : "var(--color-contested)",
                    }}
                  >
                    {g.passed ? "pass" : "fail"}
                  </span>
                  <span className="font-plate text-[12px]">{g.name}</span>
                  <span className="datum ml-auto min-w-0 truncate text-[10px] text-ink-soft">
                    {g.detail}
                  </span>
                </div>
              ))}
            </div>
            <p
              className="mt-4 border-l-2 pl-3 text-[16px] leading-relaxed"
              style={{
                borderColor: scenario.gatesPassed
                  ? "var(--color-settled)"
                  : "var(--color-contested)",
              }}
            >
              {scenario.gatesPassed
                ? "Every gate clear, nothing needed a person. This build ships to production."
                : needsReviewer
                  ? "The reviewer failed the build. It does not ship to production."
                  : "Gates failed. The build does not ship to production."}
            </p>
          </div>
        ) : null}
      </FlowNode>
    </div>
  );
}
