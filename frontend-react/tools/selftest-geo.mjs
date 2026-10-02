// Driver for src/lib/geo.ts used by tests/test_react_lib_selftest.py:
//   echo '[[lon, lat], ...]' | node tools/selftest-geo.mjs
// prints, per input point, the UTM / MGRS / S2-tile result and the lon/lat recovered by the inverse conversion.
import * as geo from '../src/lib/geo.ts';

const chunks = [];
for await (const c of process.stdin) chunks.push(c);
const pts = JSON.parse(Buffer.concat(chunks).toString('utf8'));
const out = pts.map(([lon, lat]) => {
  const u = geo.toUTM(lon, lat);
  if (!u) return null;
  const back = geo.fromUTM(u.zone, u.north, u.easting, u.northing);
  const m = geo.toMGRS(lon, lat, 5);
  return { zone: u.zone, north: u.north, band: u.band, easting: u.easting, northing: u.northing, back, mgrs: m.text, tile: geo.s2TileToken(lon, lat) };
});
console.log(JSON.stringify(out));
