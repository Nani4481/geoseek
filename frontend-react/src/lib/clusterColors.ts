// One colour per cluster id, shared by the cluster list and the cluster map so a swatch always matches what it paints.
// Hues are spread by the golden angle (consecutive ids never look alike) and lightness alternates, so 19+ clusters stay apart.

function hslToHex(h: number, s: number, l: number): string {
  const a = s * Math.min(l, 1 - l);
  const f = (n: number) => {
    const k = (n + h / 30) % 12;
    return Math.round(255 * (l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1))));
  };
  return '#' + [f(0), f(8), f(4)].map((v) => v.toString(16).padStart(2, '0')).join('');
}

export const UNCLUSTERED_COLOR = '#62729f';

export function clusterColor(id: number | string | null | undefined): string {
  if (id === null || id === undefined || id === '') return UNCLUSTERED_COLOR;
  const n = Number(id);
  if (!Number.isFinite(n)) return UNCLUSTERED_COLOR;
  return hslToHex((n * 137.508 + 18) % 360, 0.72, n % 2 === 0 ? 0.6 : 0.5);
}
