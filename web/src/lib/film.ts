/**
 * The film: one recorded run, turned into a sequence of beats.
 *
 * Beats are GENERATED from the scenario's recorded data, never hardcoded. That matters
 * because the three scenarios genuinely took different routes - the happy path was
 * accepted on one sample, `unfaithful` was resampled and then handed to a person,
 * `disagreement` was resampled and then stopped by a faulty judge. A hardcoded beat
 * list would play the same story over data that says otherwise.
 *
 * PROVENANCE IS PART OF THE MODEL. Every line the orchestrator speaks is tagged either
 * `model` (verbatim output from the recorded run) or `policy` (a statement of what the
 * harness does, written by us). The film renders those differently, because putting our
 * words in the model's mouth is exactly the kind of small lie that discredits everything
 * around it.
 */

import type { Action } from "./aggregate";
import type { Scenario } from "./types";

/** Which node in the diagram is acting. Drives node and edge state in the flow. */
export type NodeId =
  | "orch-open"
  | "conversation"
  | "judges"
  | "collapse"
  | "orch-decide"
  | "human"
  | "orch-gate";

/** Diagram order, top to bottom. The film only ever moves down it, except the loop. */
export const NODE_ORDER: NodeId[] = [
  "orch-open",
  "conversation",
  "judges",
  "collapse",
  "orch-decide",
  "human",
  "orch-gate",
];

export type Beat = { node: NodeId } & (
  | { kind: "orchestrator"; line: string; source: "model" | "policy"; steps: 1 }
  | { kind: "conversation"; runIndex: number; steps: number }
  | { kind: "judges"; runIndex: number; criterionKey: string; steps: number }
  | { kind: "verify"; runIndex: number; criterionKey: string; steps: 1 }
  | { kind: "collapse"; runIndices: number[]; caption: string; steps: 1 }
  | { kind: "gather"; runIndices: number[]; caption: string; steps: number }
  | { kind: "verdict"; action: Action; question: string; steps: 1 }
  | { kind: "setup"; steps: 1 }
  | { kind: "gate"; steps: 1 });

/** How long one step of each beat holds, in ms. */
export const STEP_MS: Record<Beat["kind"], number> = {
  setup: 9000,
  orchestrator: 8000,
  conversation: 4000,
  judges: 4500,
  verify: 5500,
  collapse: 7000,
  gather: 2800,
  verdict: 10000,
  gate: 12000,
};

/**
 * The criterion the film follows. BINARY on purpose: a judge showing "pass" or "fail"
 * needs no explaining, where a 3 out of 5 does.
 *
 * Faithfulness rather than in_scenario, because it is the criterion that actually
 * separates the two paths - it is what catches the coach inventing a study. Change the
 * constant to "in_scenario" if a different story is wanted; nothing else depends on it.
 */
export const HEADLINE_CRITERION = "in_scenario";

function headlineCriterion(_scenario: Scenario): string {
  return HEADLINE_CRITERION;
}

export function buildBeats(scenario: Scenario): Beat[] {
  const criterionKey = headlineCriterion(scenario);
  const firstRound = scenario.rounds[0];
  const beats: Beat[] = [];

  // Established BEFORE anything runs: the rubric, the panel, the scenario. Without it
  // the first judge verdict arrives with no yardstick to read it against.
  beats.push({ node: "orch-open", kind: "setup", steps: 1 });

  beats.push({
    node: "orch-open",
    kind: "orchestrator",
    source: "policy",
    line: "Running the pipeline once. Three judges will score the coach independently against the rubric. If they agree, I take the result and stop; if they split, I buy two more samples before I spend anyone's attention on it.",
    steps: 1,
  });

  beats.push({
    node: "conversation",
    kind: "conversation",
    runIndex: 0,
    steps: scenario.runs[0].turns.length,
  });

  beats.push({
    node: "judges",
    kind: "judges",
    runIndex: 0,
    criterionKey,
    steps: scenario.runs[0].verdicts.filter((v) => v.criterion === criterionKey).length,
  });

  beats.push({ node: "judges", kind: "verify", runIndex: 0, criterionKey, steps: 1 });

  beats.push({
    node: "collapse",
    kind: "collapse",
    runIndices: [0],
    caption: "One sample, three opinions",
    steps: 1,
  });

  // Did the orchestrator ask for more evidence? The recorded rounds decide, not us.
  const gathered = scenario.rounds.filter(
    (r) => r.action === "resample" || r.action === "widen_panel",
  );
  const extraRuns = scenario.runs.map((_, i) => i).filter((i) => scenario.runs[i].round > 0);

  if (gathered.length && extraRuns.length) {
    beats.push({
      node: "orch-decide",
      kind: "orchestrator",
      source: "model",
      line: gathered[0].why,
      steps: 1,
    });
    beats.push({
      node: "conversation",
      kind: "gather",
      runIndices: extraRuns,
      caption: `Running it ${extraRuns.length} more time${extraRuns.length > 1 ? "s" : ""}`,
      steps: extraRuns.length,
    });
    beats.push({
      node: "collapse",
      kind: "collapse",
      runIndices: scenario.runs.map((_, i) => i),
      caption: `${scenario.runs.length} samples now`,
      steps: 1,
    });
  } else if (firstRound) {
    beats.push({ node: "orch-decide", kind: "orchestrator", source: "model", line: firstRound.why, steps: 1 });
  }

  const last = scenario.rounds[scenario.rounds.length - 1];
  if (last && gathered.length) {
    beats.push({ node: "orch-decide", kind: "orchestrator", source: "model", line: last.why, steps: 1 });
  }

  beats.push({
    node: "human",
    kind: "verdict",
    action: scenario.final.action,
    question: scenario.final.question,
    steps: 1,
  });

  // The last frame: the suite's hard gates, and what they do to a build. Not a
  // flourish - `python -m conveval run` already exits non-zero when they fail, so the
  // orchestrator blocking a release is a description of behaviour that exists.
  beats.push({ node: "orch-gate", kind: "gate", steps: 1 });

  return beats;
}

/** Cumulative step index, so a single scrubber can address the whole film. */
export function totalSteps(beats: Beat[]): number {
  return beats.reduce((n, b) => n + b.steps, 0);
}

export function locate(beats: Beat[], step: number): { beat: number; within: number } {
  let remaining = step;
  for (let i = 0; i < beats.length; i++) {
    if (remaining < beats[i].steps) return { beat: i, within: remaining };
    remaining -= beats[i].steps;
  }
  const lastIndex = beats.length - 1;
  return { beat: lastIndex, within: beats[lastIndex].steps - 1 };
}
