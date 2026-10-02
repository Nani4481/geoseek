// Unit checks for the framework-free modules under src/lib (run directly by Node's TypeScript type-stripping; no build step).
// `tests/test_react_lib_selftest.py` runs this file from pytest and, for the geodesy module, cross-checks it against pyproj.
//
//   node tools/selftest-lib.mjs            # prints one JSON object per check group; exits non-zero on the first failure
import assert from 'node:assert/strict';
import * as tl from '../src/lib/timelapse.ts';
import * as mf from '../src/lib/mapfit.ts';
import * as cc from '../src/lib/clusterColors.ts';

const near = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg}: ${a} vs ${b}`);
const out = {};

// ---- timelapse: axis, cross-fade, drawn span ----
{
  const dates = ['2019-03-30', '2021-03-04', '2024-03-08', '2025-03-08', '2026-03-08'];
  const xs = tl.axisPositions(dates);
  assert.equal(xs[0], 0); assert.equal(xs[xs.length - 1], 1);
  assert.ok(xs.every((x, i) => i === 0 || x > xs[i - 1]), 'monotonic');
  // positions are proportional to elapsed days, i.e. NOT evenly spaced
  const span = Date.parse('2026-03-08') - Date.parse('2019-03-30');
  near(xs[1], (Date.parse('2021-03-04') - Date.parse('2019-03-30')) / span, 1e-12, 'x of 2nd date');
  assert.ok(Math.abs((xs[1] - xs[0]) - (xs[3] - xs[2])) > 0.05, 'gaps differ');

  // exactly at a node: that layer is fully on, later ones fully off
  xs.forEach((x, i) => {
    const o = tl.layerOpacities(xs, x, false);
    o.forEach((v, k) => near(v, k <= i ? 1 : 0, 1e-9, `opacity layer ${k} at node ${i}`));
  });
  // mid-segment: only the next layer is part-way; it is monotone in p
  const mid = (xs[1] + xs[2]) / 2, o = tl.layerOpacities(xs, mid, false);
  near(o[2], 0.5, 1e-9, 'smoothstep midpoint'); assert.equal(o[3], 0); assert.equal(o[1], 1);
  let prev = -1; for (let p = xs[1]; p <= xs[2]; p += 0.01) { const v = tl.layerOpacities(xs, p, false)[2]; assert.ok(v >= prev - 1e-12); prev = v; }
  // reduced motion: never a fractional opacity
  for (let p = 0; p <= 1; p += 0.013) assert.ok(tl.layerOpacities(xs, p, true).every((v) => v === 0 || v === 1), 'reduced motion blends');

  assert.equal(tl.indexAt(xs, -0.1), -1); assert.equal(tl.indexAt(xs, 0), 0); assert.equal(tl.indexAt(xs, mid), 1); assert.equal(tl.indexAt(xs, 1), 4);
  const d = tl.drawnSpan(xs[1], xs[2], mid); near(d.width, (xs[2] - xs[1]) / 2, 1e-9, 'half-drawn bracket');
  near(tl.drawnSpan(xs[1], xs[2], 0).width, 0, 1e-12, 'undrawn'); near(tl.drawnSpan(xs[1], xs[2], 1).width, xs[2] - xs[1], 1e-12, 'fully drawn');
  assert.equal(tl.stepTarget(xs, 0, 1), xs[1]); assert.equal(tl.stepTarget(xs, 1, 1), null); assert.equal(tl.stepTarget(xs, xs[2], -1), xs[1]);
  assert.deepEqual(tl.axisPositions(['2020-01-01']), [0]); assert.deepEqual(tl.axisPositions([]), []);
  out.timelapse = 'ok';
}

// ---- map framing + cluster colours ----
{
  assert.deepEqual(mf.fitPoints([]), { bbox: null, kept: [], outside: [] });
  const one = mf.fitPoints([{ lon: 82.25, lat: 26.63 }]);
  assert.deepEqual(one.kept, [0]); near(one.bbox[0], 82.25 - 0.02, 1e-12, 'single point is padded'); near(one.bbox[3], 26.63 + 0.02, 1e-12, 'single point is padded');
  // five hits in India and one in Los Angeles: the frame is India's, and the stray hit is reported rather than silently dropped
  const hits = [[82.2, 26.6], [82.3, 26.7], [77.1, 28.6], [88.1, 22.0], [71.0, 24.3], [-118.47, 34.22]].map(([lon, lat]) => ({ lon, lat }));
  const f = mf.fitPoints(hits);
  assert.deepEqual(f.outside, [5]); assert.deepEqual(f.kept, [0, 1, 2, 3, 4]);
  assert.ok(f.bbox[0] < 71.0 && f.bbox[2] > 88.1 && f.bbox[0] > 60, 'frame covers the Indian hits only');
  assert.ok(f.bbox[1] < 22.0 && f.bbox[3] > 28.6, 'frame covers latitudes');
  const nearby = mf.fitPoints([{ lon: 10, lat: 10 }, { lon: 10.5, lat: 10.2 }, { lon: 11, lat: 10.9 }]);
  assert.deepEqual(nearby.outside, []); assert.ok(nearby.bbox[0] < 10 && nearby.bbox[2] > 11 && nearby.bbox[1] < 10 && nearby.bbox[3] > 10.9);
  const ring = [[1, 1], [2, 1], [2, 2], [1, 2], [1, 1]], r = mf.fitRing(ring);
  assert.ok(r[0] < 1 && r[2] > 2 && r[1] < 1 && r[3] > 2);
  const cols = Array.from({ length: 40 }, (_, i) => cc.clusterColor(i));
  assert.equal(new Set(cols).size, 40, 'forty clusters, forty distinct colours');
  assert.ok(cols.every((c) => /^#[0-9a-f]{6}$/.test(c)));
  assert.equal(cc.clusterColor(null), cc.UNCLUSTERED_COLOR); assert.equal(cc.clusterColor('7'), cc.clusterColor(7));
  out.mapfit = 'ok';
}

// geodesy checks are appended by later modules (see tools/selftest-geo.mjs)
console.log(JSON.stringify(out));
