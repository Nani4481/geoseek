// In-browser pixel decode of an ad-hoc GeoTIFF (geotiff.js, bundled; the file is read from the user's disk and never sent anywhere).
// To keep the browser responsive the decode is a PREVIEW: the file at native resolution when it fits the budget, otherwise its
// smallest sufficient overview, otherwise a nearest-neighbour decimation. `RasterPixels.note` says which, in words, and
// `scale` records preview pixels per full-resolution pixel so a preview position maps back to the file's own grid.
import { fromArrayBuffer, fromBlob } from 'geotiff';
import type { Band, Roles } from './rasterMath.ts';

export const PREVIEW_MAX_DIM = 1536;

export interface RasterPixels {
  width: number; height: number;                  // preview size
  fullWidth: number; fullHeight: number;
  scale: [number, number];                        // preview px per full-resolution px (x, y)
  mode: 'native' | 'overview' | 'decimated';
  bands: Band[];
  note: string;
}

export async function decodeRaster(src: Blob | ArrayBuffer, maxDim = PREVIEW_MAX_DIM): Promise<RasterPixels> {
  const tiff = src instanceof ArrayBuffer ? await fromArrayBuffer(src) : await fromBlob(src);
  const full = await tiff.getImage(0);
  const fw = full.getWidth(), fh = full.getHeight();
  let img = full, mode: RasterPixels['mode'] = 'native';
  if (Math.max(fw, fh) > maxDim) {
    const n = await tiff.getImageCount();
    let best: Awaited<ReturnType<typeof tiff.getImage>> | null = null;
    for (let i = 1; i < n; i++) {                                    // the largest overview that fits the budget
      const o = await tiff.getImage(i);
      if (Math.max(o.getWidth(), o.getHeight()) <= maxDim && (!best || o.getWidth() > best.getWidth())) best = o;
    }
    if (best) { img = best; mode = 'overview'; } else mode = 'decimated';
  }
  let w = img.getWidth(), h = img.getHeight();
  let rasters;
  if (mode === 'decimated') {
    const k = maxDim / Math.max(fw, fh);
    w = Math.max(1, Math.round(fw * k)); h = Math.max(1, Math.round(fh * k));
    rasters = await img.readRasters({ interleave: false, width: w, height: h, resampleMethod: 'nearest' });
  } else {
    rasters = await img.readRasters({ interleave: false });
  }
  const bands = (Array.isArray(rasters) ? rasters : [rasters]) as unknown as Band[];
  const note = mode === 'native' ? `native resolution, ${w} × ${h} px (no resampling)`
    : mode === 'overview' ? `the file's own ${w} × ${h} px overview of its ${fw} × ${fh} px full resolution`
      : `${w} × ${h} px nearest-neighbour decimation of the ${fw} × ${fh} px full resolution`;
  return { width: w, height: h, fullWidth: fw, fullHeight: fh, scale: [w / fw, h / fh], mode, bands, note };
}

// ---------------------------------------------------------------------------------------------- what a band is

const NAME_ROLES: [RegExp, keyof Roles][] = [
  [/^(b0?2|blue)$/i, 'blue'], [/^(b0?3|green)$/i, 'green'], [/^(b0?4|red)$/i, 'red'], [/^(b0?8a?|nir|nir08)$/i, 'nir'], [/^(b11|swir|swir16|swir1)$/i, 'swir'],
];

export interface RoleGuess { roles: Roles; basis: string; rgb: [number, number, number] | null }

/** Which band plays which role. Band DESCRIPTIONs in the file win; otherwise the band COUNT implies a convention that is stated, not assumed silently. */
export function guessRoles(count: number, names: (string | null)[]): RoleGuess {
  const roles: Roles = {};
  names.forEach((nm, i) => { const hit = nm ? NAME_ROLES.find(([re]) => re.test(nm)) : undefined; if (hit && roles[hit[1]] === undefined) roles[hit[1]] = i; });
  if (Object.keys(roles).length >= 2) {
    const rgb = roles.red !== undefined && roles.green !== undefined && roles.blue !== undefined ? [roles.red, roles.green, roles.blue] as [number, number, number] : count >= 3 ? [0, 1, 2] as [number, number, number] : null;
    return { roles, basis: `band descriptions in the file (${names.map((n, i) => `${i + 1}=${n ?? '?'}`).join(', ')})`, rgb };
  }
  if (count === 3) return { roles: { red: 0, green: 1, blue: 2 }, basis: 'assumed order: red, green, blue', rgb: [0, 1, 2] };
  if (count === 4) return { roles: { blue: 0, green: 1, red: 2, nir: 3 }, basis: 'assumed Sentinel-2 order: B02 (blue), B03 (green), B04 (red), B08 (NIR)', rgb: [2, 1, 0] };
  if (count >= 5) return { roles: { blue: 0, green: 1, red: 2, nir: 3, swir: 4 }, basis: 'assumed Sentinel-2 order: B02, B03, B04, B08 (NIR), B11 (SWIR)', rgb: [2, 1, 0] };
  return { roles: {}, basis: count === 1 ? 'a single band' : 'no band convention applies', rgb: null };
}
