/**
 * A node in the diagram.
 *
 * Every actor gets the SAME card shape - a header with its mark, its name and its role,
 * then whatever it produced. Judges and the orchestrator look alike on purpose: they are
 * peers in the topology, and giving the orchestrator a special shape would suggest an
 * authority it does not have.
 *
 * Nodes collapse once the film has passed them. That is what lets the last frame hold
 * the entire graph on one screen while each node is still shown in full at the moment
 * it acts.
 */

import type { ReactNode } from "react";

import { BrandMark } from "./BrandMark";

export type NodeState = "pending" | "active" | "done";

const KIND_COLOUR = {
  agent: "var(--color-settled)",
  orchestrator: "var(--color-orchestrator)",
  code: "var(--color-ink-soft)",
  human: "var(--color-human)",
} as const;

export function FlowNode({
  kind,
  family,
  name,
  role,
  state,
  summary,
  onToggle,
  forceOpen = false,
  children,
}: {
  kind: keyof typeof KIND_COLOUR;
  family?: string;
  name: string;
  role: string;
  state: NodeState;
  /** One line, shown instead of the body once the node is behind us. */
  summary?: ReactNode;
  /** Set when the node can be opened by clicking - used once the film has finished. */
  onToggle?: () => void;
  forceOpen?: boolean;
  children?: ReactNode;
}) {
  const accent = KIND_COLOUR[kind];
  const collapsed = state === "done" && !forceOpen;

  return (
    <div
      className={`plate transition-all duration-300 ${
        state === "pending" ? "opacity-35" : "opacity-100"
      }`}
      style={{
        borderColor: state === "active" ? accent : "var(--color-rule)",
        borderLeftWidth: 3,
        borderLeftColor: accent,
      }}
      data-state={state}
    >
      <button
        type="button"
        onClick={onToggle}
        disabled={!onToggle}
        aria-expanded={onToggle ? !collapsed : undefined}
        className={`flex w-full items-center gap-2 px-3 text-left ${
          collapsed ? "py-2" : "border-b border-rule py-2"
        } ${onToggle ? "cursor-pointer hover:bg-rule-faint" : "cursor-default"}`}
      >
        {(kind === "agent" || kind === "orchestrator") && family ? (
          <BrandMark family={family} color={accent} />
        ) : (
          <span
            aria-hidden
            className="inline-block h-2.5 w-2.5 shrink-0"
            style={{ backgroundColor: accent }}
          />
        )}
        <span className="font-plate min-w-0 truncate text-[12px] font-semibold">{name}</span>
        <span className="label shrink-0 text-[9px]">{role}</span>
        {collapsed && summary ? (
          <span className="datum ml-auto min-w-0 truncate text-[11px] text-ink-soft">
            {summary}
          </span>
        ) : null}
        {onToggle ? (
          <span aria-hidden className="datum ml-2 shrink-0 text-[10px] text-ink-soft">
            {collapsed ? "+" : "−"}
          </span>
        ) : null}
      </button>

      {!collapsed && children ? <div className="px-3 py-3">{children}</div> : null}
    </div>
  );
}
