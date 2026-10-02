// A REAL calendar axis for the Temporal screen: horizontal position is proportional to elapsed days, so a 2019->2021 gap is drawn
// about two years wide and a 2024->2025 gap about one. Pure functions (no DOM, no imports) - tools/selftest-lib.mjs runs them under Node.
// Nothing here interpolates: it only places dates that exist; charts built on it draw the observed dates as points / spans.

const dayMs = (d: string) => Date.parse(d + 'T00:00:00Z');

export interface CalendarAxis {
  lo: number; hi: number;                         // epoch ms at the axis ends (padded)
  x: (date: string) => number;                    // pixel position of a date
  ticks: { x: number; label: string }[];          // one per 1 January inside the axis
}

/** `left..right` is the pixel range; `padDays` extra calendar days are left free at both ends. */
export function calendarAxis(dates: string[], left: number, right: number, padDays = 60): CalendarAxis {
  const ms = dates.map(dayMs);
  const lo0 = ms.length ? Math.min(...ms) : Date.UTC(2020, 0, 1), hi0 = ms.length ? Math.max(...ms) : Date.UTC(2021, 0, 1);
  const lo = lo0 - padDays * 86_400_000, hi = Math.max(hi0 + padDays * 86_400_000, lo + 86_400_000);
  const px = (m: number) => left + ((m - lo) / (hi - lo)) * (right - left);
  const ticks: CalendarAxis['ticks'] = [];
  for (let y = new Date(lo).getUTCFullYear() + 1; Date.UTC(y, 0, 1) < hi; y++) ticks.push({ x: px(Date.UTC(y, 0, 1)), label: String(y) });
  return { lo, hi, x: (d: string) => px(dayMs(d)), ticks };
}

/** "nice" upper bound and ticks for a count / area axis starting at 0 */
export function niceMax(v: number, steps = 4): { max: number; ticks: number[] } {
  if (!(v > 0)) return { max: 1, ticks: [0, 1] };
  const raw = v / steps, pow = 10 ** Math.floor(Math.log10(raw)), f = raw / pow;
  const step = (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10) * pow;
  const max = Math.ceil(v / step) * step, ticks: number[] = [];
  for (let t = 0; t <= max + step / 2; t += step) ticks.push(+t.toFixed(10));
  return { max, ticks };
}
