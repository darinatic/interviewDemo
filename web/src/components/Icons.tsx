/**
 * Line glyphs for the delivery pipeline.
 *
 * Drawn to the same rules as everything else on the page: 16px box, 1.4 stroke, square
 * corners, no fills except where a shape is a marker rather than an outline. They read
 * at a glance without becoming the thing you look at, which is the job — the diagram is
 * carried by the flow, not by the icons.
 */

const S = { fill: "none", strokeWidth: 1.4, strokeLinecap: "round" as const };

function Frame({ children, color }: { children: React.ReactNode; color: string }) {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden className="shrink-0" stroke={color}>
      {children}
    </svg>
  );
}

/** A commit on a branch. */
export function IconCommit({ color }: { color: string }) {
  return (
    <Frame color={color}>
      <path d="M1.5 8h3.2M11.3 8h3.2" {...S} />
      <circle cx="8" cy="8" r="3.1" {...S} />
    </Frame>
  );
}

/** A build artifact. */
export function IconBuild({ color }: { color: string }) {
  return (
    <Frame color={color}>
      <path d="M8 1.9 14 5v6l-6 3.1L2 11V5l6-3.1Z" {...S} strokeLinejoin="round" />
      <path d="M2 5l6 3.1L14 5M8 8.1v6" {...S} />
    </Frame>
  );
}

/** The panel: three independent scorers. */
export function IconPanel({ color }: { color: string }) {
  return (
    <Frame color={color}>
      {[3, 8, 13].map((x) => (
        <rect
          key={x}
          x={x - 2}
          y={6}
          width="4"
          height="4"
          transform={`rotate(45 ${x} 8)`}
          fill={color}
          stroke="none"
        />
      ))}
    </Frame>
  );
}

/** A checkpoint that can be closed. */
export function IconGate({ color }: { color: string }) {
  return (
    <Frame color={color}>
      <path d="M8 1.8 13.7 4v4.3c0 3.2-2.4 5.2-5.7 5.9-3.3-.7-5.7-2.7-5.7-5.9V4L8 1.8Z" {...S} strokeLinejoin="round" />
      <path d="M5.6 8.1 7.3 9.8l3.2-3.4" {...S} strokeLinejoin="round" />
    </Frame>
  );
}

/** Two branches becoming one. */
export function IconMerge({ color }: { color: string }) {
  return (
    <Frame color={color}>
      <path d="M4.4 2.4v4.1c0 2 1.6 3.2 3.4 3.4h3.8" {...S} />
      <path d="M4.4 13.6v-3" {...S} />
      <circle cx="4.4" cy="9" r="1.6" {...S} />
      <path d="M9.9 8.2 11.7 10 9.9 11.8" {...S} strokeLinejoin="round" />
    </Frame>
  );
}

/** Shipped. */
export function IconShip({ color }: { color: string }) {
  return (
    <Frame color={color}>
      <path d="M2.4 10.6v2.6h11.2v-2.6" {...S} strokeLinejoin="round" />
      <path d="M8 2.2v7.2M5.2 5 8 2.2 10.8 5" {...S} strokeLinejoin="round" />
    </Frame>
  );
}

/** Stopped. */
export function IconBlocked({ color }: { color: string }) {
  return (
    <Frame color={color}>
      <circle cx="8" cy="8" r="5.7" {...S} />
      <path d="M4.4 11.6 11.6 4.4" {...S} />
    </Frame>
  );
}

/** A person. */
export function IconPerson({ color }: { color: string }) {
  return (
    <Frame color={color}>
      <circle cx="8" cy="5.4" r="2.6" {...S} />
      <path d="M2.9 13.6c0-2.8 2.3-4.4 5.1-4.4s5.1 1.6 5.1 4.4" {...S} strokeLinejoin="round" />
    </Frame>
  );
}

/** A queue of things waiting. */
export function IconQueue({ color }: { color: string }) {
  return (
    <Frame color={color}>
      <path d="M2.6 4.2h10.8M2.6 8h10.8M2.6 11.8h6.4" {...S} />
    </Frame>
  );
}
