/**
 * What was fixed before anything ran.
 *
 * The film used to open on a judge's verdict with no yardstick in sight, which asks a
 * viewer to evaluate an evaluation they have not been shown the rules for. This card is
 * the rubric, the panel and the scenario, stated once, up front.
 */

import { BrandMark } from "./BrandMark";
import { Label } from "./primitives";
import { HEADLINE_CRITERION } from "../lib/film";
import { run } from "../lib/run";
import type { Scenario } from "../lib/types";

export function Setup({ scenario }: { scenario: Scenario }) {
  const criterion = run.criteria.find((c) => c.key === HEADLINE_CRITERION);
  const { roles } = run.meta;

  return (
    <div className="mx-auto max-w-[760px] space-y-4">
    <div className="plate p-5">
        <Label>The scenario</Label>
        <p className="mt-2 text-[14px] leading-relaxed">
          The <strong>conversational system</strong> under evaluation plays the{" "}
          <strong>guest</strong>. The <strong>simulated user</strong> plays the{" "}
          <strong>front-office staff member</strong> being trained on the dispute.
        </p>
        <p className="mt-2 text-[14px] leading-relaxed">{scenario.facts}</p>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <div className="plate p-5">
          <Label>The models the harness runs</Label>
          <div className="mt-3 space-y-2">
            {[
              { r: roles.orchestrator, role: "orchestrator", colour: "var(--color-orchestrator)" },
              { r: roles.learner, role: "simulated user · front-office staff", colour: "var(--color-settled)" },
              ...roles.judges.map((j, i) => ({
                r: j,
                role: `judge agent ${i + 1}`,
                colour: "var(--color-settled)",
              })),
            ].map(({ r, role, colour }) => (
              <div key={r.model} className="flex items-center gap-2">
                <BrandMark family={r.family} color={colour} />
                <span className="font-plate text-[13px]">{r.label}</span>
                <span className="datum ml-auto text-[10px] text-ink-soft">{role}</span>
              </div>
            ))}
          </div>
          <p className="mt-3 text-[13px] leading-snug text-ink-soft">
            No judge agent shares a model family with the system being evaluated, because
            a model marking its own family&rsquo;s work scores it higher.
          </p>
        </div>

        <div className="plate p-5">
        <Label>The rubric</Label>
        <p className="font-plate mt-2 text-[15px] font-semibold">
          {criterion?.label}
          <span className="datum ml-2 text-[11px] font-normal text-ink-soft">
            pass / fail
          </span>
        </p>
        <p className="mt-2 max-w-[70ch] text-[15px] leading-relaxed">{criterion?.rubric}</p>
        <p className="mt-3 border-l-2 border-rule pl-3 text-[14px] leading-relaxed text-ink-soft">
          Every judge scores against this exact text, and every judge must quote the words
          it based that score on.
        </p>
      </div>
      </div>
    </div>
  );
}
