/** The shape of `run.json`, written by `python -m conveval export`. */

import type { AcrossRuns, Action, Kind, Score } from "./aggregate";

export interface Role {
  model: string;
  family: string;
  label: string;
}

export interface Turn {
  role: "learner" | "agent";
  span: string;
  text: string;
}

export interface ExportedVerdict {
  judge: string;
  family: string;
  criterion: string;
  score: Score;
  reason: string;
  evidence: string;
  verified: boolean | null;
}

export interface RunRecord {
  run: number;
  round: number;
  traceId: string;
  turns: Turn[];
  verdicts: ExportedVerdict[];
}

export interface RoundDecision {
  round: number;
  action: Action;
  legalActions: Action[];
  why: string;
  wasFallback: boolean;
  callsSpent: number;
  budgetRemaining: number;
}

export interface FinalVerdict {
  action: Action;
  question: string;
  roundsUsed: number;
  needsReview: boolean;
  contested: string[];
  failed: string[];
  /** Criteria contested on the first sample and no longer contested after the ladder. */
  resolvedByGathering: string[];
  consensus: Record<
    string,
    {
      value: Score;
      rationale: string;
      contested: boolean;
      failed: boolean;
      nRuns: number;
      panelSize: number;
    }
  >;
}

export interface Scenario {
  id: string;
  /** This scenario's own gate outcome, as though it were the whole suite. */
  gates: Gate[];
  gatesPassed: boolean;
  title: string;
  brief: string;
  setting: string;
  facts: string;
  unknowns: string;
  learnerPersona: string;
  learnerGoal: string;
  seededFlaw: string;
  prompts: { learnerSystem: string; coachSystem: string; judge: string };
  runs: RunRecord[];
  rounds: RoundDecision[];
  final: FinalVerdict;
}

export interface ExportedCriterion {
  key: string;
  label: string;
  kind: Kind;
  rubric: string;
  failAtOrBelow: number | null;
  requiresEvidence: boolean;
  acrossRuns: AcrossRuns;
}

export interface Gate {
  name: string;
  passed: boolean;
  detail: string;
}

export interface RunFile {
  /** The suite's hard gates, computed by the real `build_suite` at export time. */
  gates: Gate[];
  /** False when any gate failed - which is what makes the run exit non-zero. */
  suitePassed: boolean;
  meta: {
    runId: string;
    runName: string;
    recordedAt: number;
    roles: {
      sut: Role;
      learner: Role;
      orchestrator: Role;
      judges: Role[];
      wideningJudges: Role[];
    };
  };
  constants: {
    maxExtraRounds: number;
    resampleK: number;
    extraCallBudget: number;
    binaryContestedBelow: number;
    ordinalContestedSpread: number;
    turnsPerConversation: number;
  };
  criteria: ExportedCriterion[];
  actions: Record<string, string>;
  scenarios: Scenario[];
}
