// Driver for src/lib/rasterHeader.ts used by tests/test_react_lib_selftest.py:
//   node tools/selftest-raster.mjs file1.tif file2.tif ...
// prints one JSON object per file: the parsed header (or {error}). (The browser passes a File/Blob and geotiff.js reads slices of
// it; Node has no FileReader, so the driver hands over an ArrayBuffer - the parsing code path below that point is identical.)
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { parseRasterHeader } from '../src/lib/rasterHeader.ts';

const out = [];
for (const f of process.argv.slice(2)) {
  try { out.push(await parseRasterHeader(new Uint8Array(readFileSync(f)).buffer, path.basename(f))); } catch (e) { out.push({ fileName: path.basename(f), error: String(e.message || e) }); }
}
console.log(JSON.stringify(out));
