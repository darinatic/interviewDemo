/**
 * The wiring between nodes.
 *
 * Each connector is its own full-width SVG with `preserveAspectRatio="none"` and a
 * viewBox in the 0-100 space, so a fan lands exactly on the centres of a three-column
 * grid at any width. An earlier version of the topology drew connectors at hardcoded
 * pixel percentages against absolutely-positioned nodes; they drifted out of alignment
 * the moment a label changed length. Letting the SVG stretch with its own container
 * removes the class of bug entirely.
 *
 * Stroke width is set in an unscaled overlay path rather than on the stretched geometry,
 * because non-uniform scaling would squash the stroke along one axis.
 */

export type EdgeState = "pending" | "flowing" | "done";

const STROKE: Record<EdgeState, string> = {
  pending: "var(--color-rule-faint)",
  flowing: "var(--color-human)",
  done: "var(--color-rule)",
};

function Arrowhead({ x, state }: { x: number; state: EdgeState }) {
  return (
    <div
      className="absolute bottom-0 -translate-x-1/2 translate-y-[1px]"
      style={{ left: `${x}%` }}
      aria-hidden
    >
      <svg width="11" height="8" viewBox="0 0 11 8">
        <path d="M5.5 8 0 0h11L5.5 8Z" fill={STROKE[state]} />
      </svg>
    </div>
  );
}

/** One line, straight down. */
export function Edge({ state, height = 44 }: { state: EdgeState; height?: number }) {
  return (
    <div className="relative w-full" style={{ height }}>
      <svg
        className="absolute inset-0 h-full w-full"
        viewBox="0 0 100 100"
        preserveAspectRatio="none"
        aria-hidden
      >
        <line
          x1="50"
          y1="0"
          x2="50"
          y2="100"
          stroke={STROKE[state]}
          strokeWidth="0.6"
          vectorEffect="non-scaling-stroke"
          className={state === "flowing" ? "edge-flowing" : ""}
        />
      </svg>
      <Arrowhead x={50} state={state} />
    </div>
  );
}

/**
 * One line splitting into `count` lines, or the reverse.
 *
 * The fan is what makes "three judges score it independently" a visual fact rather
 * than a sentence: three separate paths leave the same point and never touch again.
 */
export function Fan({
  state,
  count = 3,
  direction = "split",
  height = 56,
}: {
  state: EdgeState;
  count?: number;
  direction?: "split" | "merge";
  height?: number;
}) {
  const stops = Array.from({ length: count }, (_, i) => ((i + 0.5) / count) * 100);
  const splitting = direction === "split";

  return (
    <div className="relative w-full" style={{ height }}>
      <svg
        className="absolute inset-0 h-full w-full"
        viewBox="0 0 100 100"
        preserveAspectRatio="none"
        aria-hidden
      >
        {stops.map((x) => (
          <path
            key={x}
            d={
              splitting
                ? `M50 0 L50 38 L${x} 62 L${x} 100`
                : `M${x} 0 L${x} 38 L50 62 L50 100`
            }
            fill="none"
            stroke={STROKE[state]}
            strokeWidth="0.6"
            vectorEffect="non-scaling-stroke"
            className={state === "flowing" ? "edge-flowing" : ""}
          />
        ))}
      </svg>
      {splitting ? (
        stops.map((x) => <Arrowhead key={x} x={x} state={state} />)
      ) : (
        <Arrowhead x={50} state={state} />
      )}
    </div>
  );
}

/**
 * The loop back to sampling.
 *
 * Three-sided, in the left gutter, drawn with borders rather than an SVG path so it
 * spans whatever height its container happens to be without anyone measuring pixels.
 * This edge is the whole reason the system is a control loop and not a pipeline, so it
 * is labelled rather than left to be inferred.
 */
export function LoopBack({ active, label }: { active: boolean; label: string }) {
  const colour = active ? "var(--color-human)" : "var(--color-rule)";
  return (
    <div className={`pointer-events-none absolute inset-y-0 -left-11 w-10 ${active ? "loop-active" : ""}`} aria-hidden>
      <div
        className="absolute inset-y-3 left-3 right-0 border-y border-l border-dashed"
        style={{ borderColor: colour }}
      />
      <div className="absolute left-3 top-3 -translate-x-1/2 -translate-y-1/2">
        <svg width="8" height="11" viewBox="0 0 8 11">
          <path d="M0 5.5 8 0v11L0 5.5Z" fill={colour} transform="rotate(90 4 5.5)" />
        </svg>
      </div>
      <span
        className="label absolute left-1/2 top-1/2 origin-center -translate-x-1/2 -translate-y-1/2 -rotate-90 whitespace-nowrap text-[9px]"
        style={{ color: colour }}
      >
        {label}
      </span>
    </div>
  );
}
