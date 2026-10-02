// Spectral profile of dropped files over the ground they share. Pure (no React, no DOM): tools/selftest-pixels.mjs runs it under Node and
// tests/test_react_raster_pixels.py compares it with an independent numpy / pyproj computation.
import type { RasterHeader } from './rasterHeader.ts';
import type { RasterPixels } from './rasterPixels.ts';
import { guessRoles } from './rasterPixels.ts';
import { INDEX_DEFS, bandStats, boxOverlap, chooseZoom, computeIndex, gridForBox, makeProjector, mercToLonLat, warpToGrid, type Band, type BandStats, type Box, type IndexName, type MercGrid } from './rasterMath.ts';

/** What a profile needs to know about one dropped file (the Data screen's Upload satisfies this). */
export interface ProfileFile {
  id: string; name: string; header: RasterHeader | null; pixels: RasterPixels | null; decoding: boolean;
  acquired: { date: string | null; source: string | null };
}

export interface Member { up: ProfileFile; date: string; box: Box }
export interface Selection { members: Member[]; excluded: { up: ProfileFile; why: string }[]; box: Box | null; placeable: number; dated: number }

/** The files that can enter a profile, and why each other file cannot. Deterministic: files are taken in date order and kept while they still share a footprint. */
export function selectProfileSet(files: ProfileFile[]): Selection {
  const excluded: Selection['excluded'] = [];
  const ok: Member[] = [];
  let placeable = 0;
  for (const up of files) {
    const h = up.header;
    if (!h) excluded.push({ up, why: 'its header could not be read' });
    else if (!up.pixels) excluded.push({ up, why: up.decoding ? 'its pixels are still being decoded' : 'its pixels could not be decoded' });
    else if (!h.lonlatBounds || !h.affine || h.crs.epsg === null || !makeProjector({ affine: h.affine, epsg: h.crs.epsg, width: h.width, height: h.height })) excluded.push({ up, why: 'it has no footprint that can be placed on the ground offline' });
    else if (!up.acquired.date) { placeable++; excluded.push({ up, why: 'it has no acquisition date (none in the header and none entered)' }); }
    else { placeable++; ok.push({ up, date: up.acquired.date, box: h.lonlatBounds }); }
  }
  ok.sort((a, b) => (a.date < b.date ? -1 : a.date > b.date ? 1 : a.up.name.localeCompare(b.up.name)));
  const members: Member[] = [];
  let box: Box | null = null;
  for (const m of ok) {
    const ov: Box | null = box ? boxOverlap(box, m.box) : m.box;
    if (ov) { members.push(m); box = ov; } else excluded.push({ up: m.up, why: 'its footprint does not overlap the other files' });
  }
  return { members, excluded, box, placeable, dated: ok.length };
}

/** `bandIndex[key]` is the 0-based band of THIS file that a band metric (keyed by role, e.g. `red`) was read from: bands are matched across files by what they are, never by their position. */
export interface Row { name: string; date: string; dateSource: string; metrics: Record<string, BandStats>; bandIndex: Record<string, number> }
export interface Profile { rows: Row[]; grid: MercGrid; z: number; cells: number; metricKeys: { key: string; label: string; note: string }[]; resM: number }

const mPerPx = (up: ProfileFile) => { const h = up.header!; const r = h.resolution?.[0] ?? 10; return h.crs.kind === 'geographic' ? r * 111_320 : r; };

export function computeProfile(sel: Selection): Profile {
  const box = sel.box!;
  const finest = Math.min(...sel.members.map((m) => mPerPx(m.up)));
  const lat = (box[1] + box[3]) / 2;
  const wantZ = Math.ceil(Math.log2(40_075_016.686 / (256 * (finest / Math.cos((lat * Math.PI) / 180)))));
  const z = chooseZoom(box, Math.max(6, Math.min(17, wantZ)), 36);
  const { grid } = gridForBox(box, z);
  const inBox = new Uint8Array(grid.w * grid.h);
  let cells = 0;
  for (let j = 0; j < grid.h; j++) for (let i = 0; i < grid.w; i++) {
    const [lon, la] = mercToLonLat(grid.x0 + (i + 0.5) * grid.res, grid.y0 - (j + 0.5) * grid.res);
    if (lon >= box[0] && lon <= box[2] && la >= box[1] && la <= box[3]) { inBox[j * grid.w + i] = 1; cells++; }
  }
  const rows: Row[] = [];
  const keys = new Map<string, { label: string; note: string }>();
  for (const m of sel.members) {
    const h = m.up.header!, px = m.up.pixels!;
    const proj = makeProjector({ affine: h.affine!, epsg: h.crs.epsg!, width: h.width, height: h.height })!;
    const warped = warpToGrid(px.bands, px.width, px.height, px.scale, proj, grid, h.nodata) as Band[];
    const g = guessRoles(h.bands, h.bandNames);
    const metrics: Record<string, BandStats> = {};
    const bandIndex: Record<string, number> = {};
    warped.forEach((b, i) => {
      const role = Object.entries(g.roles).find(([, k]) => k === i)?.[0];
      const key = role ?? `band${i + 1}`;
      metrics[key] = bandStats(b, null, inBox);
      bandIndex[key] = i;
      if (!keys.has(key)) keys.set(key, { label: role ? `${role[0].toUpperCase()}${role.slice(1)} band` : `Band ${i + 1}`, note: 'the file’s own values (digital numbers or reflectance, as written), matched across files by band role' });
    });
    for (const name of Object.keys(INDEX_DEFS) as IndexName[]) {
      const idx = computeIndex(name, warped, g.roles, null);
      if (!idx) continue;
      metrics[name] = bandStats(idx, null, inBox);
      if (!keys.has(name)) keys.set(name, { label: `${INDEX_DEFS[name].label} · ${INDEX_DEFS[name].formula}`, note: `band roles: ${g.basis}` });
    }
    rows.push({ name: m.up.name, date: m.date, dateSource: m.up.acquired.source ?? '', metrics, bandIndex });
  }
  return { rows, grid, z, cells, resM: grid.res * Math.cos((lat * Math.PI) / 180), metricKeys: [...keys].map(([key, v]) => ({ key, ...v })) };
}

