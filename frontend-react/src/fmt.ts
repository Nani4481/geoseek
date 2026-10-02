// Display helpers only. Nothing here produces a number - it formats numbers the API supplied, and renders an explicit
// "—" for anything missing so no card ever shows an invented value.

export const DASH = '—';

export const isNum = (x: unknown): x is number => typeof x === 'number' && Number.isFinite(x);
export const fmtInt = (x: number | null | undefined) => (isNum(x) ? x.toLocaleString('en-US') : DASH);
export const fmtFixed = (x: number | null | undefined, d = 3) => (isNum(x) ? x.toFixed(d) : DASH);
export const fmtPct = (x: number | null | undefined, d = 1) => (isNum(x) ? `${(x * 100).toFixed(d)}%` : DASH);
export const fmtHa = (m2: number | null | undefined) => (isNum(m2) ? `${(m2 / 10_000).toFixed(m2 < 100_000 ? 2 : 1)} ha` : DASH);
export const fmtMs = (x: number | null | undefined) => (isNum(x) ? x.toFixed(x < 100 ? 1 : 0) : DASH);
/** ground sample distance: 10 m stays `10 m`, 0.30517578 becomes `0.31 m` (the exact value belongs in a tooltip) */
export const fmtGsd = (m: number | null | undefined) => (isNum(m) ? `${m >= 1 ? +m.toFixed(1) : +m.toFixed(2)} m` : DASH);
export const fmtLon = (x: number) => `${Math.abs(x).toFixed(4)}°${x >= 0 ? 'E' : 'W'}`;
export const fmtLat = (x: number) => `${Math.abs(x).toFixed(4)}°${x >= 0 ? 'N' : 'S'}`;
export const fmtLonLat = (p: [number, number] | null | undefined) => (p ? `${fmtLat(p[1])}  ${fmtLon(p[0])}` : DASH);

export function titleCase(s: string) {
  return s.split(/[_\-\s]+/).filter(Boolean).map((w) => w[0].toUpperCase() + w.slice(1)).join(' ');
}
/** `maxar_india_floods_oct_2023_q120220030330` etc. -> something short enough for a legend */
export function regionLabel(name: string) {
  return titleCase(name.replace(/_q\d{6,}/, '').replace(/maxar_/, 'Maxar · ').replace(/_losangeles_jan_2025/, '').replace(/_oct_2023/, ''))
    .replace(/\bNcr\b/, 'NCR'); // acronym
}

export function bandOf(conf: number | null | undefined): { label: string; color: string } {
  if (!isNum(conf)) return { label: DASH, color: 'var(--ink-4)' };
  if (conf >= 0.85) return { label: 'High', color: 'var(--green)' };
  if (conf >= 0.6) return { label: 'Medium', color: 'var(--amber)' };
  return { label: 'Low', color: 'var(--red)' };
}

// Styling palette for the change-type chips (presentation only - the labels themselves come from the API).
export const TYPE_COLOR: Record<string, string> = {
  water_gain: '#4cc9f0', water_loss: '#a78bfa', construction: '#f5a524', clearance: '#34d399', road: '#ff9f6b', other: '#8d9cc6',
};
export const typeColor = (t: string | null | undefined) => (t && TYPE_COLOR[t]) || '#8d9cc6';

// Backend gate names -> the analyst-facing names the existing UI uses.
export const GATE_LABEL: Record<string, string> = {
  quality: 'Image quality', registration: 'Alignment', radiometric: 'Brightness match', phenology: 'Seasonal check', morphology: 'Shape & size',
};

export function haversineKm(a: [number, number], b: [number, number]) {
  const R = 6371.0088, rad = Math.PI / 180;
  const dLat = (b[1] - a[1]) * rad, dLon = (b[0] - a[0]) * rad;
  const h = Math.sin(dLat / 2) ** 2 + Math.cos(a[1] * rad) * Math.cos(b[1] * rad) * Math.sin(dLon / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(h));
}

export const dateMs = (d: string) => Date.parse(d + 'T00:00:00Z');

const IST = new Intl.DateTimeFormat('en-GB', { timeZone: 'Asia/Kolkata', hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
const IST_DATE = new Intl.DateTimeFormat('en-GB', { timeZone: 'Asia/Kolkata', weekday: 'short', day: '2-digit', month: 'short', year: 'numeric' });
export const istTime = (d: Date) => IST.format(d);
export const istDate = (d: Date) => IST_DATE.format(d);

export function downloadJSON(filename: string, data: unknown) {
  const blob = new Blob([JSON.stringify(data, null, 1)], { type: 'application/geo+json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}
