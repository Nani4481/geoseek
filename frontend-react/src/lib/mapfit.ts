// Pure helpers for "fit the map to these points". No DOM, no Leaflet: unit-tested under node (tools/selftest-lib.mjs).
import type { BBox } from '../api/types';

export interface LL { lon: number; lat: number }
export interface FitResult {
  /** bounds that frame the points kept in view (padded); null when there are no points */
  bbox: BBox | null;
  /** indexes (into the input) of points inside the framed group */
  kept: number[];
  /** indexes of points left out of the frame because they lie far from the rest - the caller must say so */
  outside: number[];
}

function km(a: LL, b: LL): number {
  const R = 6371.0088, rad = Math.PI / 180;
  const dLat = (b.lat - a.lat) * rad, dLon = (b.lon - a.lon) * rad;
  const h = Math.sin(dLat / 2) ** 2 + Math.cos(a.lat * rad) * Math.cos(b.lat * rad) * Math.sin(dLon / 2) ** 2;
  return 2 * R * Math.asin(Math.min(1, Math.sqrt(h)));
}

/**
 * Frame a set of points. If some of them lie more than `outlierKm` from the medoid (the point with the smallest summed distance
 * to the others) they are reported in `outside` and left out of the frame, so one stray hit on another continent cannot shrink
 * the view to a world map in which nothing is readable.
 */
export function fitPoints(pts: LL[], opts: { outlierKm?: number; padFrac?: number; minPadDeg?: number } = {}): FitResult {
  const { outlierKm = 2500, padFrac = 0.12, minPadDeg = 0.02 } = opts;
  if (pts.length === 0) return { bbox: null, kept: [], outside: [] };
  let medoid = 0, best = Infinity;
  if (pts.length > 2) {
    for (let i = 0; i < pts.length; i++) {
      let s = 0;
      for (let j = 0; j < pts.length; j++) s += km(pts[i], pts[j]);
      if (s < best) { best = s; medoid = i; }
    }
  }
  const kept: number[] = [], outside: number[] = [];
  pts.forEach((p, i) => (km(p, pts[medoid]) <= outlierKm ? kept : outside).push(i));
  const xs = kept.map((i) => pts[i].lon), ys = kept.map((i) => pts[i].lat);
  const w = Math.min(...xs), e = Math.max(...xs), s = Math.min(...ys), n = Math.max(...ys);
  const px = Math.max(minPadDeg, (e - w) * padFrac), py = Math.max(minPadDeg, (n - s) * padFrac);
  return { bbox: [w - px, s - py, e + px, n + py], kept, outside };
}

/** Bounds of a GeoJSON-like ring set, padded; for framing a footprint. */
export function fitRing(ring: number[][], opts: { padFrac?: number; minPadDeg?: number } = {}): BBox {
  const { padFrac = 2, minPadDeg = 0.01 } = opts;
  const xs = ring.map((p) => p[0]), ys = ring.map((p) => p[1]);
  const w = Math.min(...xs), e = Math.max(...xs), s = Math.min(...ys), n = Math.max(...ys);
  const px = Math.max(minPadDeg, (e - w) * padFrac), py = Math.max(minPadDeg, (n - s) * padFrac);
  return [w - px, s - py, e + px, n + py];
}
