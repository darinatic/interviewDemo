import type { ReactElement } from "react";

/**
 * Monochrome glyphs standing in for the model families.
 *
 * Drawn rather than embedded: shipping official logo assets in a page you send to a
 * company is a trademark question nobody needs, and traced brand files would sit badly
 * against a drafting-plate aesthetic anyway. These are simplified marks that read at
 * 16px and are recognisable next to the model name, which is all the job requires.
 *
 * The name is always shown alongside. The glyph is a visual anchor, never the only
 * identification.
 */

const MARKS: Record<string, (color: string) => ReactElement> = {
  // Anthropic: the burst.
  anthropic: (c) => (
    <>
      <path d="M6.2 3.4 2.6 12.6h1.9l.74-1.98h3.7l.74 1.98h1.9L8 3.4H6.2Zm-.32 5.7 1.2-3.2 1.2 3.2H5.88Z" fill={c} />
      <path d="M11.1 3.4h1.85l3.6 9.2h-1.9L11.1 3.4Z" fill={c} opacity="0.55" />
    </>
  ),
  // Google: the four-arm mark, flattened to one weight.
  google: (c) => (
    <>
      <circle cx="8" cy="8" r="4.6" fill="none" stroke={c} strokeWidth="1.6" />
      <path d="M8 8h4.8" stroke={c} strokeWidth="1.6" />
      <path d="M8 3.4V8" stroke={c} strokeWidth="1.6" opacity="0.55" />
    </>
  ),
  // Meta: the infinity loop.
  "meta-llama": (c) => (
    <path
      d="M2.2 9.4c0-2.5 1.3-4.6 3-4.6 1.3 0 2.2 1 3 2.5.8-1.5 1.7-2.5 3-2.5 1.7 0 3 2.1 3 4.6 0 1.3-.6 2.2-1.6 2.2-1.4 0-2.3-1.8-3.2-3.6-.6-1.2-1-1.9-1.5-1.9-.9 0-1.6 1.4-1.6 3.2 0 1.1.3 1.7.8 1.7.4 0 .7-.3 1.1-1"
      fill="none"
      stroke={c}
      strokeWidth="1.5"
      strokeLinecap="round"
    />
  ),
  // OpenAI: the knot, reduced to a hexagonal rosette.
  openai: (c) => (
    <path
      d="M8 2.6 12.7 5.3v5.4L8 13.4 3.3 10.7V5.3L8 2.6Z"
      fill="none"
      stroke={c}
      strokeWidth="1.5"
      strokeLinejoin="round"
    />
  ),
  // Mistral: the banded square.
  mistralai: (c) => (
    <>
      <rect x="2.6" y="3.4" width="10.8" height="2.4" fill={c} />
      <rect x="2.6" y="6.8" width="10.8" height="2.4" fill={c} opacity="0.6" />
      <rect x="2.6" y="10.2" width="10.8" height="2.4" fill={c} opacity="0.3" />
    </>
  ),
  // Qwen.
  qwen: (c) => (
    <path
      d="M8 2.8 13 8l-5 5.2L3 8l5-5.2Zm0 3L5.6 8 8 10.4 10.4 8 8 5.8Z"
      fill={c}
    />
  ),
};

export function BrandMark({
  family,
  size = 16,
  color = "currentColor",
}: {
  family: string;
  size?: number;
  color?: string;
}) {
  const draw = MARKS[family];
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 16 16"
      aria-hidden
      className="shrink-0"
      role="presentation"
    >
      {draw ? (
        draw(color)
      ) : (
        <circle cx="8" cy="8" r="4.4" fill="none" stroke={color} strokeWidth="1.5" />
      )}
    </svg>
  );
}
