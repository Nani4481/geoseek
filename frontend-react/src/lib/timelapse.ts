// Pure geometry for the evidence-timeline animation. No React, no DOM, no imports: tools/selftest-lib.mjs runs this file
// directly under Node. Animation is presentation - every position here is derived from the catalog's real acquisition
// dates (the calendar axis is NOT evenly spaced) and nothing is invented.

const dayMs = (d: string) => Date.parse(d + 'T00:00:00Z');

/** Position of every acquisition date on a 0..1 calendar axis (first date = 0, last = 1). One date sits at 0. */
export function axisPositions(dates: string[]): number[] {
  if (dates.length === 0) return [];
  const ms = dates.map(dayMs);
  const lo = Math.min(...ms), span = Math.max(1, Math.max(...ms) - lo);
  return ms.map((m) => (m - lo) / span);
}

const clamp01 = (v: number) => Math.max(0, Math.min(1, v));
const EPS = 1e-9;

/** Index of the latest acquisition at or before the playhead (-1 before the first). */
export function indexAt(xs: number[], p: number): number {
  let idx = -1;
  for (let i = 0; i < xs.length; i++) if (xs[i] <= p + EPS) idx = i;
  return idx;
}

/**
 * Opacity of each date's image layer, painter-ordered (later dates are drawn on top of earlier ones). The first layer is
 * always opaque; layer i ramps 0 -> 1 across the stretch of axis between date i-1 and date i, so at any playhead position
 * exactly one cross-fade is in progress and it is complete the instant the playhead reaches that acquisition.
 * `reduced` (prefers-reduced-motion): no blending - a layer is either shown or not.
 */
export function layerOpacities(xs: number[], p: number, reduced: boolean): number[] {
  return xs.map((x, i) => {
    if (i === 0) return 1;
    if (reduced) return x <= p + EPS ? 1 : 0;
    const t = clamp01((p - xs[i - 1]) / Math.max(EPS, x - xs[i - 1]));
    return t * t * (3 - 2 * t); // smoothstep: eased, still exactly 0 at the previous date and 1 at this one
  });
}

/** How much of a [x0, x1] window the playhead has drawn so far, as a left/width pair (axis units). */
export function drawnSpan(x0: number, x1: number, p: number): { left: number; width: number } {
  return { left: x0, width: clamp01((p - x0) / Math.max(EPS, x1 - x0)) * (x1 - x0) };
}

/** Whether a node at axis position x has been passed by the playhead (it "illuminates" from then on). */
export const isLit = (x: number, p: number) => x <= p + EPS;

/** Next / previous node position for stepped (reduced-motion) playback. Returns null past either end. */
export function stepTarget(xs: number[], p: number, dir: 1 | -1): number | null {
  if (dir === 1) return xs.find((x) => x > p + EPS) ?? null;
  const before = xs.filter((x) => x < p - EPS);
  return before.length ? before[before.length - 1] : null;
}
