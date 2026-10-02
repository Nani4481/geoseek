// Driver for src/lib/rasterPixels.ts + rasterMath.ts, run by tests/test_react_raster_pixels.py:
//   node tools/selftest-pixels.mjs reference.json
// reference.json (from tests/raster_fixtures.py) names real GeoTIFF files cut from the archive. For each this prints what the BROWSER code
// computes - band statistics, spectral-index statistics, stretched composite samples, and a Web-Mercator resample + visual difference of
// the two dated 4-band files - so pytest can compare every number with numpy / rasterio.
import { readFileSync } from 'node:fs';
import { parseRasterHeader } from '../src/lib/rasterHeader.ts';
import { decodeRaster, guessRoles } from '../src/lib/rasterPixels.ts';
import * as rm from '../src/lib/rasterMath.ts';
import { computeProfile, selectProfileSet } from '../src/lib/rasterProfile.ts';

if (process.argv[2] === '--decode') {            // decode-only summary of one file: which preview mode was chosen, and a row of values
  const buf0 = new Uint8Array(readFileSync(process.argv[3])).buffer;
  const px0 = await decodeRaster(buf0);
  const row = (r) => Array.from(px0.bands[0].slice(r * px0.width, r * px0.width + px0.width));
  console.log(JSON.stringify({ mode: px0.mode, width: px0.width, height: px0.height, fullWidth: px0.fullWidth, fullHeight: px0.fullHeight, scale: px0.scale, note: px0.note, row0: row(0), rowMid: row(Math.floor(px0.height / 2)) }));
  process.exit(0);
}
const ref = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const ab = (p) => new Uint8Array(readFileSync(p)).buffer;
const b64 = (f32) => Buffer.from(f32.buffer, f32.byteOffset, f32.byteLength).toString('base64');
const out = { files: {}, warp: null };

const loaded = {};
for (const [key, f] of Object.entries(ref.files)) {
  const buf = ab(f.path);
  const header = await parseRasterHeader(buf, key);
  const px = await decodeRaster(buf);
  const g = guessRoles(header.bands, header.bandNames);
  const r = { width: px.width, height: px.height, mode: px.mode, bands: header.bands, nodata: header.nodata, acquisition: header.acquisition?.iso ?? null,
    bandNames: header.bandNames, roles: g.roles, basis: g.basis, rgb: g.rgb, lonlatCorners: header.lonlatCorners };
  loaded[key] = { header, px, g };
  if (header.bands === 4) {
    r.bandStats = px.bands.map((b) => rm.bandStats(b, header.nodata));
    r.indices = {};
    for (const name of ['ndvi', 'ndwi']) { const idx = rm.computeIndex(name, px.bands, g.roles, header.nodata); r.indices[name] = rm.bandStats(idx, null); }
    r.ndbiAvailable = rm.indexAvailable('ndbi', g.roles);
    const ranges = g.rgb.map((i) => rm.stretchRange(px.bands[i], 2, 98, header.nodata));
    const rgba = rm.compositeRGBA(g.rgb.map((i) => px.bands[i]), ranges, header.nodata);
    r.compositeSamples = (f.composite_samples || []).map((s) => { const k = 4 * (s.row * px.width + s.col); return { row: s.row, col: s.col, rgb: [rgba[k], rgba[k + 1], rgba[k + 2]] }; });
  }
  out.files[key] = r;
}

// Both dated files on one Web-Mercator grid (the box they share), then the indicative visual difference.
const A = loaded['2019-03-30'], B = loaded['2024-03-08'];
const cornersBox = (h) => h.lonlatBounds;
const ov = rm.boxOverlap(cornersBox(A.header), cornersBox(B.header));
const z = rm.chooseZoom(ov, 14, 9);
const { grid } = rm.gridForBox(ov, z);
const proj = (L) => rm.makeProjector({ affine: L.header.affine, epsg: L.header.crs.epsg, width: L.header.width, height: L.header.height });
const warp = (L) => rm.warpToGrid(L.px.bands, L.px.width, L.px.height, L.px.scale, proj(L), grid, L.header.nodata);
const wa = warp(A), wb = warp(B);
const lumA = rm.lumaOf(wa[2], wa[1], wa[0]), lumB = rm.lumaOf(wb[2], wb[1], wb[0]);
const d = rm.visualDiff(lumA, lumB);
out.warp = { z, grid, overlap: ov, b04_2019: b64(wa[2]), b04_2024: b64(wb[2]),
  diff: { n: d.n, meanAbs: d.meanAbs, shareAbove: d.shareAbove, threshold: d.threshold, correlation: d.correlation, rangeA: d.rangeA, rangeB: d.rangeB } };
// The spectral profile of the two dated files over the ground they share.
const pf = ['2019-03-30', '2024-03-08'].map((k) => ({ id: k, name: k, header: loaded[k].header, pixels: loaded[k].px, decoding: false, acquired: { date: k, source: 'file header' } }));
const sel = selectProfileSet(pf);
const prof = computeProfile(sel);
out.profile = { z: prof.z, grid: prof.grid, cells: prof.cells, box: sel.box, members: sel.members.map((m) => m.date), keys: prof.metricKeys.map((k) => k.key),
  files: Object.fromEntries(prof.rows.map((r) => [r.date, r.metrics])) };
console.log(JSON.stringify(out));
