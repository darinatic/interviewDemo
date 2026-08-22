/**
 * Loading the recorded run, and recomputing from it.
 *
 * The consensus values are RECOMPUTED here rather than read from the export, even
 * though the export contains them. That is deliberate: the sandbox needs to recompute
 * when the viewer changes the rules, and having one code path for both means the
 * default view is produced by exactly the machinery the sandbox exercises. A page whose
 * "real" numbers came from one source and whose "what-if" numbers came from another
 * would be comparing two different things and calling it a comparison.
 */

import runFile from "../data/run.json";
import {
  type Consensus,
  type LadderConfig,
  type Verdict,
  consensusFor,
} from "./aggregate";
import type { ExportedCriterion, RunFile, Scenario } from "./types";

export const run = runFile as unknown as RunFile;

export const criteria = run.criteria;
export const scenarios = run.scenarios;

export function scenarioById(id: string): Scenario {
  const found = scenarios.find((s) => s.id === id);
  if (!found) throw new Error(`no scenario ${id}`);
  return found;
}

export const ladderConfig: LadderConfig = {
  maxExtraRounds: run.constants.maxExtraRounds,
  resampleK: run.constants.resampleK,
  turnsPerConversation: run.constants.turnsPerConversation,
  nCriteria: run.criteria.length,
  basePanelSize: run.meta.roles.judges.length,
  wideningJudges: run.meta.roles.wideningJudges.length,
};

export interface RailInput {
  criterion: ExportedCriterion;
  verdicts: Verdict[];
  consensus: Consensus;
}

/**
 * The panel's verdicts for one run of one scenario, collapsed per criterion.
 *
 * `excluded` drops judges by name, which is the sandbox's "what if this judge were not
 * on the panel" control. `rule` switches median to mean so the viewer can watch an
 * outlier drag the score somewhere nobody voted.
 */
export function railsFor(
  scenario: Scenario,
  runIndex = 0,
  opts: { excluded?: Set<string>; rule?: "median" | "mean" } = {},
): RailInput[] {
  const record = scenario.runs[runIndex];
  const excluded = opts.excluded ?? new Set<string>();

  return criteria
    .map((criterion) => {
      const verdicts = record.verdicts
        .filter((v) => v.criterion === criterion.key && !excluded.has(v.judge))
        .map(
          (v): Verdict => ({
            judge: v.judge,
            family: v.family,
            criterion: v.criterion,
            score: v.score,
            evidence: v.evidence,
            reason: v.reason,
            verified: v.verified,
          }),
        );
      if (verdicts.length === 0) return null;
      return { criterion, verdicts, consensus: consensusFor(criterion, verdicts, opts.rule) };
    })
    .filter((x): x is RailInput => x !== null);
}

/** Judges that actually voted in this run, in panel order. */
export function judgesIn(scenario: Scenario, runIndex = 0): string[] {
  const seen = new Set<string>();
  for (const v of scenario.runs[runIndex].verdicts) seen.add(v.judge);
  return [...seen];
}

export function recordedAt(): string {
  return new Date(run.meta.recordedAt).toISOString().slice(0, 10);
}
