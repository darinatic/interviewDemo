/**
 * THE VERDICT RAIL — the signature element of this page.
 *
 * Each judge lands as a marker on a scale, and the gap between the extreme markers is
 * drawn as a dimensioned span, the way a measurement is annotated on an engineering
 * drawing.
 *
 * The reason it is built this way rather than as a number: the argument this whole
 * system rests on is that THE SPREAD IS THE SIGNAL, not the midpoint. A scorecard
 * showing "3.0" hides a 2/3/5 split behind a plausible number. A rail cannot — the
 * distance is the first thing you see, before you have read anything.
 *
 * Accessibility is load-bearing here, not a retrofit. The person this was built for is
 * red-green colourblind, so state is carried by POSITION on the rail, by MARKER SHAPE
 * (rotated square for a judge, upright bar for the panel's result) and by text, with
 * colour as the fourth signal rather than the only one. The rail is also summarised in
 * a visually hidden sentence, since a row of absolutely-positioned markers means
 * nothing to a screen reader.
 *
 * Layout note: the track is inset from the container edge. Markers at the extremes
 * (a unanimous "pass" sits at 100%) would otherwise be sliced in half by the edge.
 */

import type { Consensus, Criterion, Verdict } from "../lib/aggregate";

const INSET = 16; // px of breathing room so extreme markers are never clipped
const ORDINAL_TICKS = ["1", "2", "3", "4", "5"];
const BINARY_TICKS = ["fail", "pass"];

function positionOf(score: string | number, kind: Criterion["kind"]): number {
  if (kind === "binary") return String(score) === "fail" ? 0 : 100;
  return ((Number(score) - 1) / 4) * 100;
}

/** Judges landing on the same value would overlap; fan them upward instead. */
function stackOffsets(positions: number[]): number[] {
  const seen = new Map<number, number>();
  return positions.map((p) => {
    const key = Math.round(p);
    const n = seen.get(key) ?? 0;
    seen.set(key, n + 1);
    return n;
  });
}

export function VerdictRail({
  criterion,
  verdicts,
  consensus,
  compact = false,
  animate = true,
}: {
  criterion: Pick<Criterion, "key" | "label" | "kind" | "failAtOrBelow">;
  verdicts: Verdict[];
  consensus: Consensus;
  compact?: boolean;
  animate?: boolean;
}) {
  const positions = verdicts.map((v) => positionOf(v.score, criterion.kind));
  const offsets = stackOffsets(positions);
  const min = Math.min(...positions);
  const max = Math.max(...positions);
  const spread = max - min;
  const stacks = Math.max(...offsets) + 1;

  const tone = consensus.contested ? "contested" : "settled";
  const ticks = criterion.kind === "binary" ? BINARY_TICKS : ORDINAL_TICKS;
  const consensusPos = positionOf(consensus.consensus, criterion.kind);

  // Vertical rhythm, measured from the rail line downward.
  const markerBand = stacks * 13 + 4;
  const railY = markerBand;
  const tickLabelY = railY + 8;
  const dimensionY = tickLabelY + (compact ? 15 : 19);
  const height = dimensionY + (compact ? 20 : 24);

  const summary =
    `${criterion.label}. ` +
    verdicts.map((v) => `${v.judge} scored ${v.score}`).join("; ") +
    `. Panel result ${consensus.consensus}, agreement ${Math.round(consensus.agreement * 100)}%. ` +
    (consensus.contested ? "The panel disagreed." : "The panel was unanimous.");

  return (
    <div className="w-full">
      <div className="mb-1 flex items-baseline justify-between gap-3">
        <span className="label">{criterion.label}</span>
        <span className="datum text-[11px] text-ink-soft">
          {criterion.kind === "ordinal" ? "median" : "majority"}{" "}
          <span className="font-medium text-ink">{String(consensus.consensus)}</span>
        </span>
      </div>

      <div
        style={{ paddingInline: INSET }}
        className={criterion.kind === "binary" ? "max-w-[380px]" : ""}
      >
        <div className="relative" style={{ height }} aria-hidden>
          {/* The rail and its end stops. */}
          <div className="absolute inset-x-0 h-px bg-rule" style={{ top: railY }} />
          {ticks.map((tick, i) => {
            const left = (i / (ticks.length - 1)) * 100;
            return (
              <div key={tick} className="absolute" style={{ left: `${left}%`, top: railY - 3 }}>
                <div className="h-1.5 w-px -translate-x-1/2 bg-rule" />
                <div
                  className="datum absolute -translate-x-1/2 whitespace-nowrap text-[10px] text-ink-soft"
                  style={{ top: tickLabelY - railY + 3 }}
                >
                  {tick}
                </div>
              </div>
            );
          })}

          {/* The judges. */}
          {verdicts.map((v, i) => (
            <div
              key={v.judge}
              className={`absolute ${animate ? "animate-land" : ""}`}
              style={{
                left: `${positions[i]}%`,
                top: railY - 9 - offsets[i] * 13,
                animationDelay: animate ? `${i * 70}ms` : undefined,
              }}
              title={`${v.judge}: ${v.score}${v.verified === false ? " (cited a span that is not in the transcript)" : ""}`}
            >
              <div
                className="h-2.5 w-2.5 -translate-x-1/2 rotate-45 border"
                style={
                  v.verified === false
                    ? { borderColor: "var(--color-contested)", backgroundColor: "var(--color-plate)" }
                    : {
                        borderColor: `var(--color-${tone})`,
                        backgroundColor: `var(--color-${tone})`,
                      }
                }
              />
            </div>
          ))}

          {/* The panel's result, drawn LAST so the judges cannot paint over it: a
              centre line through the whole marker band, the way a datum is struck on a
              drawing. Deliberately not a rotated square - a viewer must never have to
              work out which marker is the panel and which is a judge. */}
          <div
            className="absolute"
            style={{ left: `${consensusPos}%`, top: railY - markerBand }}
            title={`panel result: ${consensus.consensus}`}
          >
            <div
              className="w-px -translate-x-1/2 opacity-45"
              style={{ height: markerBand, backgroundColor: "var(--color-ink)" }}
            />
            <div
              className="h-[7px] w-[3px] -translate-x-1/2"
              style={{ marginTop: -3, backgroundColor: "var(--color-ink)" }}
            />
          </div>

          {/* The dimension line: what this component exists for. */}
          {spread > 0 ? (
            <div
              className={`absolute ${animate ? "animate-dimension" : ""}`}
              style={{ left: `${min}%`, width: `${spread}%`, top: dimensionY }}
            >
              <div className="relative">
                <div className="h-px w-full" style={{ backgroundColor: `var(--color-${tone})` }} />
                {[0, 1].map((side) => (
                  <div
                    key={side}
                    className="absolute -top-[3px] h-[7px] w-px"
                    style={{
                      backgroundColor: `var(--color-${tone})`,
                      [side ? "right" : "left"]: 0,
                    }}
                  />
                ))}
              </div>
              <div
                className="datum mt-1 whitespace-nowrap text-center text-[10px] font-medium uppercase tracking-[0.08em]"
                style={{ color: `var(--color-${tone})` }}
              >
                {criterion.kind === "ordinal"
                  ? `spread ${consensus.dispersion}`
                  : `${verdicts.length - Math.round(consensus.agreement * verdicts.length)} of ${verdicts.length} dissent`}
              </div>
            </div>
          ) : (
            <div
              className="datum absolute whitespace-nowrap text-[10px] uppercase tracking-[0.08em] text-settled"
              style={{
                left: `${min}%`,
                top: dimensionY,
                transform: min > 50 ? "translateX(-100%)" : "translateX(0)",
                color: "var(--color-settled)",
              }}
            >
              all {verdicts.length} agree
            </div>
          )}
        </div>
      </div>

      <span className="sr-only">{summary}</span>
    </div>
  );
}
