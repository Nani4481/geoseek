// Pixel arithmetic for the ad-hoc raster views (Data preview, upload-vs-archive comparison, spectral profile). Pure functions over typed
// arrays: no DOM, no geotiff, no network, so tools/selftest-pixels.mjs can run them under Node against real files and
// tests/test_react_raster_pixels.py can compare every result with numpy / rasterio.
//
// Conventions (the same ones numpy uses, so the cross-check is exact rather than "close"):
//   * percentiles use linear interpolation between order statistics (numpy.percentile default);
//   * the standard deviation is the population one (numpy.std default, ddof = 0);
//   * a pixel is VALID when it is finite and not equal to the file's nodata value; invalid pixels never enter a statistic.
import { toUTM, utmFromEpsg } from './geo.ts';

export type Band = ArrayLike<number>;

const R_MERC = 6378137;
export const lonLatToMerc = (lon: number, lat: number): [number, number] => [
  (R_MERC * lon * Math.PI) / 180,
  R_MERC * Math.log(Math.tan(Math.PI / 4 + (Math.max(-85.05112878, Math.min(85.05112878, lat)) * Math.PI) / 360)),
];
export const mercToLonLat = (x: number, y: number): [number, number] => [(x / R_MERC) * (180 / Math.PI), (2 * Math.atan(Math.exp(y / R_MERC)) - Math.PI / 2) * (180 / Math.PI)];

// ---------------------------------------------------------------------------------------------- statistics

/** The valid values of a band, ascending. */
export function sortedValid(band: Band, nodata: number | null = null, mask: Uint8Array | null = null): Float64Array {
  let n = 0;
  const out = new Float64Array(band.length);
  for (let i = 0; i < band.length; i++) {
    const v = band[i];
    if (!Number.isFinite(v) || (nodata !== null && v === nodata) || (mask && !mask[i])) continue;
    out[n++] = v;
  }
  const s = out.subarray(0, n);
  s.sort();
  return s;
}

/** numpy.percentile(..., method='linear') on an ascending array. */
export function percentileSorted(sorted: ArrayLike<number>, p: number): number {
  const n = sorted.length;
  if (n === 0) return NaN;
  const pos = (Math.min(100, Math.max(0, p)) / 100) * (n - 1);
  const lo = Math.floor(pos), hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

export interface BandStats { n: number; mean: number; std: number; min: number; max: number; p2: number; p10: number; p50: number; p90: number; p98: number }

export function bandStats(band: Band, nodata: number | null = null, mask: Uint8Array | null = null): BandStats {
  const s = sortedValid(band, nodata, mask), n = s.length;
  if (n === 0) return { n: 0, mean: NaN, std: NaN, min: NaN, max: NaN, p2: NaN, p10: NaN, p50: NaN, p90: NaN, p98: NaN };
  let sum = 0;
  for (let i = 0; i < n; i++) sum += s[i];
  const mean = sum / n;
  let ss = 0;
  for (let i = 0; i < n; i++) ss += (s[i] - mean) ** 2;
  return { n, mean, std: Math.sqrt(ss / n), min: s[0], max: s[n - 1], p2: percentileSorted(s, 2), p10: percentileSorted(s, 10), p50: percentileSorted(s, 50), p90: percentileSorted(s, 90), p98: percentileSorted(s, 98) };
}

/** [lo, hi] at the given percentiles of the valid values of one band. A flat band widens to lo..lo+1 so nothing divides by zero. */
export function stretchRange(band: Band, lowPct: number, highPct: number, nodata: number | null = null, mask: Uint8Array | null = null): [number, number] {
  const s = sortedValid(band, nodata, mask);
  const lo = percentileSorted(s, lowPct);
  let hi = percentileSorted(s, highPct);
  if (!(hi > lo)) hi = lo + 1;
  return [lo, hi];
}

// ---------------------------------------------------------------------------------------------- pixel masks & colour

/** 1 where every listed band is valid at that pixel. */
export function validMask(bands: Band[], nodata: number | null): Uint8Array {
  const n = bands[0]?.length ?? 0, m = new Uint8Array(n).fill(1);
  for (const b of bands) for (let i = 0; i < n; i++) { const v = b[i]; if (!Number.isFinite(v) || (nodata !== null && v === nodata)) m[i] = 0; }
  return m;
}

const to8 = (v: number, lo: number, hi: number) => Math.round(Math.min(1, Math.max(0, (v - lo) / (hi - lo))) * 255);

/** RGBA from three bands (each stretched linearly between its own lo/hi to 0..255). Invalid pixels are fully transparent. */
export function compositeRGBA(bands: [Band, Band, Band], ranges: [[number, number], [number, number], [number, number]], nodata: number | null): Uint8ClampedArray {
  const n = bands[0].length, out = new Uint8ClampedArray(n * 4), m = validMask(bands, nodata);
  for (let i = 0; i < n; i++) {
    if (!m[i]) continue;
    out[4 * i] = to8(bands[0][i], ranges[0][0], ranges[0][1]);
    out[4 * i + 1] = to8(bands[1][i], ranges[1][0], ranges[1][1]);
    out[4 * i + 2] = to8(bands[2][i], ranges[2][0], ranges[2][1]);
    out[4 * i + 3] = 255;
  }
  return out;
}

export function greyRGBA(band: Band, range: [number, number], nodata: number | null): Uint8ClampedArray {
  return compositeRGBA([band, band, band], [range, range, range], nodata);
}

/** (a - b) / (a + b) per pixel; NaN where either input is invalid or a + b is zero. */
export function normDiff(a: Band, b: Band, nodata: number | null): Float32Array {
  const n = a.length, out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const x = a[i], y = b[i];
    out[i] = !Number.isFinite(x) || !Number.isFinite(y) || (nodata !== null && (x === nodata || y === nodata)) || x + y === 0 ? NaN : (x - y) / (x + y);
  }
  return out;
}

// ---------------------------------------------------------------------------------------------- spectral indices

export type Role = 'blue' | 'green' | 'red' | 'nir' | 'swir';
export type Roles = Partial<Record<Role, number>>;                    // 0-based band index per role
export const INDEX_DEFS = {
  ndvi: { label: 'NDVI', formula: '(NIR − Red) / (NIR + Red)', needs: ['nir', 'red'] as Role[] },
  ndwi: { label: 'NDWI', formula: '(Green − NIR) / (Green + NIR)', needs: ['green', 'nir'] as Role[] },
  ndbi: { label: 'NDBI', formula: '(SWIR − NIR) / (SWIR + NIR)', needs: ['swir', 'nir'] as Role[] },
} as const;
export type IndexName = keyof typeof INDEX_DEFS;

export function indexAvailable(name: IndexName, roles: Roles): boolean {
  return INDEX_DEFS[name].needs.every((r) => roles[r] !== undefined);
}

/** The index as one float band, from the file's raw values (no reflectance scaling or offset is applied). */
export function computeIndex(name: IndexName, bands: Band[], roles: Roles, nodata: number | null): Float32Array | null {
  if (!indexAvailable(name, roles)) return null;
  const r = roles as Required<Roles>;
  switch (name) {
    case 'ndvi': return normDiff(bands[r.nir], bands[r.red], nodata);
    case 'ndwi': return normDiff(bands[r.green], bands[r.nir], nodata);
    case 'ndbi': return normDiff(bands[r.swir], bands[r.nir], nodata);
  }
}

// ---------------------------------------------------------------------------------------------- geometry: file grid <-> Web Mercator

export type Affine = [number, number, number, number, number, number];   // x = a*col + b*row + c ; y = d*col + e*row + f

export interface FileGeo { affine: Affine; epsg: number; width: number; height: number }

/** lon/lat -> fractional (col, row) in the file's FULL-resolution grid; null when the CRS cannot be inverted offline. */
export function makeProjector(g: FileGeo): ((lon: number, lat: number) => [number, number] | null) | null {
  const [a, b, c, d, e, f] = g.affine;
  const det = a * e - b * d;
  if (!det) return null;
  const inv = (x: number, y: number): [number, number] => [(e * (x - c) - b * (y - f)) / det, (-d * (x - c) + a * (y - f)) / det];
  if (g.epsg === 4326) return (lon, lat) => inv(lon, lat);
  if (g.epsg === 3857) return (lon, lat) => { const [x, y] = lonLatToMerc(lon, lat); return inv(x, y); };
  const u = utmFromEpsg(g.epsg);
  if (!u) return null;
  return (lon, lat) => { const p = toUTM(lon, lat, u.zone); return p ? inv(p.easting, p.northing) : null; };
}

/** A Web-Mercator pixel grid: pixel (i, j) has its centre at (x0 + (i + .5) * res, y0 - (j + .5) * res) metres. */
export interface MercGrid { w: number; h: number; x0: number; y0: number; res: number }

/** The grid at slippy-map zoom `z` that contains a lon/lat box, snapped to whole tile pixels (256 px per tile). */
export function gridForBox(box: [number, number, number, number], z: number): { grid: MercGrid; tiles: { x0: number; y0: number; x1: number; y1: number } } {
  const world = 2 * Math.PI * R_MERC, tilePx = 256, n = 2 ** z, res = world / (tilePx * n);
  const [mx0, my0] = lonLatToMerc(box[0], box[1]), [mx1, my1] = lonLatToMerc(box[2], box[3]);
  const px = (mx: number) => (mx + world / 2) / res, py = (my: number) => (world / 2 - my) / res;
  const x0 = Math.floor(px(mx0) / tilePx), x1 = Math.floor(px(mx1) / tilePx);
  const y0 = Math.floor(py(my1) / tilePx), y1 = Math.floor(py(my0) / tilePx);
  const tiles = { x0: Math.max(0, x0), y0: Math.max(0, y0), x1: Math.min(n - 1, x1), y1: Math.min(n - 1, y1) };
  return { grid: { w: (tiles.x1 - tiles.x0 + 1) * tilePx, h: (tiles.y1 - tiles.y0 + 1) * tilePx, x0: tiles.x0 * tilePx * res - world / 2, y0: world / 2 - tiles.y0 * tilePx * res, res }, tiles };
}

/** The largest zoom <= maxZoom whose tile block over the box has at most `maxTiles` tiles (and at least 1). */
export function chooseZoom(box: [number, number, number, number], maxZoom: number, maxTiles: number): number {
  for (let z = maxZoom; z >= 2; z--) {
    const t = gridForBox(box, z).tiles;
    if ((t.x1 - t.x0 + 1) * (t.y1 - t.y0 + 1) <= maxTiles) return z;
  }
  return 2;
}

/** Nearest-neighbour resample of `bands` (a preview of the file at `scale` = preview px per full-resolution px) onto a mercator grid.
 *  Output bands are Float32 with NaN outside the file or at nodata. Nearest keeps the file's own values: nothing is blended. */
export function warpToGrid(bands: Band[], pw: number, ph: number, scale: [number, number], project: (lon: number, lat: number) => [number, number] | null, grid: MercGrid, nodata: number | null): Float32Array[] {
  const out = bands.map(() => new Float32Array(grid.w * grid.h).fill(NaN));
  for (let j = 0; j < grid.h; j++) {
    const my = grid.y0 - (j + 0.5) * grid.res;
    for (let i = 0; i < grid.w; i++) {
      const [lon, lat] = mercToLonLat(grid.x0 + (i + 0.5) * grid.res, my);
      const p = project(lon, lat);
      if (!p) continue;
      const c = Math.floor(p[0] * scale[0]), r = Math.floor(p[1] * scale[1]);
      if (c < 0 || r < 0 || c >= pw || r >= ph) continue;
      const k = r * pw + c, o = j * grid.w + i;
      for (let b = 0; b < bands.length; b++) { const v = bands[b][k]; out[b][o] = nodata !== null && v === nodata ? NaN : v; }
    }
  }
  return out;
}

// ---------------------------------------------------------------------------------------------- visual comparison

export const lumaOf = (r: Band, g: Band, b: Band): Float32Array => {
  const out = new Float32Array(r.length);
  for (let i = 0; i < r.length; i++) out[i] = 0.299 * r[i] + 0.587 * g[i] + 0.114 * b[i];   // Rec. 601; NaN propagates (= invalid)
  return out;
};

export interface DiffResult {
  rgba: Uint8ClampedArray; n: number; meanAbs: number; shareAbove: number; threshold: number; correlation: number;
  rangeA: [number, number]; rangeB: [number, number];
}

/** An INDICATIVE visual difference of two brightness images on the same grid. Each is first scaled to 0..1 between its own
 *  2nd and 98th percentile (over the pixels both have), so a difference in overall brightness or contrast between the two
 *  products does not read as a difference in content; the result is B - A in [-1, 1], drawn on a blue-white-red scale. */
export function visualDiff(a: Float32Array, b: Float32Array, threshold = 0.25): DiffResult {
  const n = a.length, both = new Uint8Array(n);
  let cnt = 0;
  for (let i = 0; i < n; i++) if (Number.isFinite(a[i]) && Number.isFinite(b[i])) { both[i] = 1; cnt++; }
  const rangeA = stretchRange(a, 2, 98, null, both), rangeB = stretchRange(b, 2, 98, null, both);
  const out = new Uint8ClampedArray(n * 4);
  let sumAbs = 0, above = 0, sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
  for (let i = 0; i < n; i++) {
    if (!both[i]) continue;
    const x = Math.min(1, Math.max(0, (a[i] - rangeA[0]) / (rangeA[1] - rangeA[0])));
    const y = Math.min(1, Math.max(0, (b[i] - rangeB[0]) / (rangeB[1] - rangeB[0])));
    const d = y - x;
    sumAbs += Math.abs(d); if (Math.abs(d) > threshold) above++;
    sa += x; sb += y; saa += x * x; sbb += y * y; sab += x * y;
    const t = Math.min(1, Math.abs(d)), hot = d > 0;                       // white at 0, saturating to red (B brighter) / blue (B darker)
    out[4 * i] = hot ? 255 : Math.round(255 * (1 - t) + 40 * t);
    out[4 * i + 1] = Math.round(255 * (1 - t) + (hot ? 60 : 110) * t);
    out[4 * i + 2] = hot ? Math.round(255 * (1 - t) + 70 * t) : 255;
    out[4 * i + 3] = 255;
  }
  const cov = sab / cnt - (sa / cnt) * (sb / cnt), va = saa / cnt - (sa / cnt) ** 2, vb = sbb / cnt - (sb / cnt) ** 2;
  return { rgba: out, n: cnt, meanAbs: cnt ? sumAbs / cnt : NaN, shareAbove: cnt ? above / cnt : NaN, threshold, correlation: va > 0 && vb > 0 ? cov / Math.sqrt(va * vb) : NaN, rangeA, rangeB };
}

// ---------------------------------------------------------------------------------------------- footprints

export type Box = [number, number, number, number];
export const boxOverlap = (a: Box, b: Box): Box | null => {
  const w = Math.max(a[0], b[0]), s = Math.max(a[1], b[1]), e = Math.min(a[2], b[2]), n = Math.min(a[3], b[3]);
  return e > w && n > s ? [w, s, e, n] : null;
};
