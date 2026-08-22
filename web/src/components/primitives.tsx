/** Shared plate furniture. Everything here is drawing-sheet vocabulary. */

import type { ReactNode } from "react";

import type { Action } from "../lib/aggregate";

export function Plate({
  children,
  className = "",
  as: Tag = "section",
}: {
  children: ReactNode;
  className?: string;
  as?: "section" | "div" | "aside";
}) {
  return <Tag className={`plate ${className}`}>{children}</Tag>;
}

/** A drawing-sheet field label. */
export function Label({ children, className = "" }: { children: ReactNode; className?: string }) {
  return <span className={`label ${className}`}>{children}</span>;
}

/**
 * A titled compartment, the way a drawing sheet divides into labelled fields.
 * The rule under the label is a construction line, not decoration: it is what makes
 * the page read as annotated rather than styled.
 */
export function Field({
  label,
  note,
  children,
  className = "",
}: {
  label: string;
  note?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <div className={className}>
      <div className="flex items-baseline justify-between gap-4 border-b border-rule pb-1.5">
        <Label>{label}</Label>
        {note ? <span className="datum text-[11px] text-ink-soft">{note}</span> : null}
      </div>
      <div className="pt-3">{children}</div>
    </div>
  );
}

const ACTION_STYLE: Record<Action, { bg: string; fg: string; glyph: string; word: string }> = {
  accept: { bg: "bg-settled-wash", fg: "text-settled", glyph: "=", word: "settled" },
  resample: { bg: "bg-settled-wash", fg: "text-settled", glyph: "+", word: "gather more" },
  widen_panel: { bg: "bg-settled-wash", fg: "text-settled", glyph: "+", word: "gather more" },
  human_tiebreak: { bg: "bg-human-wash", fg: "text-human", glyph: "?", word: "needs a person" },
  human_confirm: { bg: "bg-human-wash", fg: "text-human", glyph: "!", word: "needs a person" },
  judge_defect: { bg: "bg-contested-wash", fg: "text-contested", glyph: "x", word: "instrument fault" },
};

/**
 * An action, shown with a glyph as well as a colour.
 *
 * The glyph is not ornament: it is what carries the meaning for a red-green
 * colourblind reader, and it means the badge still parses in greyscale or print.
 */
export function ActionBadge({ action, size = "md" }: { action: Action; size?: "sm" | "md" }) {
  const s = ACTION_STYLE[action];
  const pad = size === "sm" ? "px-1.5 py-0.5 text-[10px]" : "px-2 py-1 text-[11px]";
  return (
    <span
      className={`inline-flex items-center gap-1.5 ${s.bg} ${s.fg} ${pad} font-plate font-semibold uppercase tracking-[0.1em]`}
    >
      <span aria-hidden className="datum opacity-70">
        {s.glyph}
      </span>
      {action.replace(/_/g, " ")}
    </span>
  );
}

export function actionWord(action: Action): string {
  return ACTION_STYLE[action].word;
}
