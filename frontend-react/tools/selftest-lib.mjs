// Unit checks for the framework-free modules under src/lib (run directly by Node's TypeScript type-stripping; no build step).
// `tests/test_react_lib_selftest.py` runs this file from pytest and, for the geodesy module, cross-checks it against pyproj.
//
//   node tools/selftest-lib.mjs            # prints one JSON object per check group; exits non-zero on the first failure
import assert from 'node:assert/strict';
import * as tl from '../src/lib/timelapse.ts';
import * as mf from '../src/lib/mapfit.ts';
import * as cc from '../src/lib/clusterColors.ts';
import * as ci from '../src/lib/clusterInsight.ts';
import * as fm from '../src/fmt.ts';
import * as ca from '../src/lib/calendarAxis.ts';

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

// ---- cluster interpretation: every figure is counted from the payload ----
{
  const rp = (size, regions) => { const top = Object.entries(regions).sort((a, b) => b[1] - a[1])[0]; return { size, dominant_region: top[0], purity: top[1] / size, regions }; };
  const d = {
    available: true, n_clusters: 6, noise_count: 0, n_tiles: 600,
    sizes: { 0: 100, 1: 100, 2: 100, 3: 100, 4: 100, 5: 100 },
    display_labels: { 0: 'dry ground (a)', 1: 'dry ground (b)', 2: 'dry ground (c)', 3: 'dry ground (d)', 4: 'water', 5: 'villages' },
    cluster_concepts: { 0: [['bare dry ground', 0.3]], 1: [['bare dry ground', 0.3]], 2: [['bare dry ground', 0.29]], 3: [['bare dry ground', 0.28]], 4: [['open water', 0.24]], 5: [['villages', 0.3]] },
    region_purity: {
      0: rp(100, { kutch: 95, kanha: 5 }), 1: rp(100, { kanha: 90, kutch: 10 }), 2: rp(100, { jaisalmer: 85, kutch: 15 }),
      3: rp(100, { kutch: 50, kanha: 30, deccan: 20 }), 4: rp(100, { kerala_backwaters: 99, kutch: 1 }), 5: rp(100, { delhi_ncr: 40, kanha: 30, kutch: 30 }),
    },
  };
  const st = ci.clusterStats(d);
  assert.equal(st.length, 6); assert.equal(st.reduce((a, x) => a + x.n, 0), 600);
  assert.equal(st.filter((x) => x.concentrated).length, 4, 'purity 0.95 / 0.90 / 0.85 / 0.99 are concentrated; 0.50 and 0.40 are not');
  const c0 = st.find((x) => x.id === '0');
  assert.equal(c0.dominant, 'kutch'); near(c0.purity, 0.95, 1e-12, 'purity'); assert.deepEqual(c0.spans, ['kutch', 'kanha'], 'a region spans at 5%'); assert.equal(c0.regions.length, 2);
  assert.deepEqual(st.find((x) => x.id === '4').spans, ['kerala_backwaters'], '1% is a stray, not a span');
  const ins = ci.interpret(st, (r) => r.toUpperCase());
  assert.equal(ins.nClusters, 6); assert.equal(ins.nConcentrated + ins.nSpread, 6); assert.equal(ins.nConcentrated, 4);
  assert.equal(ins.factKind, 'concept-splits-by-region');
  assert.deepEqual(ins.factClusters.sort(), ['0', '1', '2'], '"bare dry ground": three concentrated clusters in three different regions; the 50% one is excluded');
  const factText = ins.fact.map((x) => x.t).join('');
  assert.ok(factText.startsWith('3 of 4 clusters') && factText.includes('bare dry ground') && /KUTCH.*KANHA.*JAISALMER|KANHA.*KUTCH/.test(factText.replace(/ and /g, ', ')) && factText.includes('JAISALMER'), factText);
  assert.ok(ins.counts.map((x) => x.t).join('').includes('6 clusters over 600 tiles'));
  // numbers in the text track the data: change one cluster and the sentence follows (nothing is templated)
  const d2 = JSON.parse(JSON.stringify(d)); d2.region_purity[2] = rp(100, { kutch: 60, jaisalmer: 40 });
  const ins2 = ci.interpret(ci.clusterStats(d2));
  assert.deepEqual(ins2.factClusters.sort(), ['0', '1'], 'cluster 2 stopped being concentrated');
  assert.ok(ins2.fact.map((x) => x.t).join('').startsWith('2 of 4'));
  // no concept splits by region -> fall back to the most widely spread cluster; no run -> null
  const d3 = JSON.parse(JSON.stringify(d)); d3.cluster_concepts = { 0: [['a', 1]], 1: [['b', 1]], 2: [['c', 1]], 3: [['d', 1]], 4: [['e', 1]], 5: [['f', 1]] };
  const ins3 = ci.interpret(ci.clusterStats(d3));
  assert.equal(ins3.factKind, 'most-spread'); assert.deepEqual(ins3.factClusters, ['3']);
  assert.equal(ci.interpret([]), null); assert.deepEqual(ci.clusterStats({ available: false }), []);
  // matrix helpers
  const cols = ci.regionColumns(st); assert.equal(cols.reduce((a, c) => a + c.tiles, 0), 600); assert.equal(cols[0].region, 'kutch');
  const groups = ci.conceptGroups(st); assert.equal(groups[0].concept, 'bare dry ground'); assert.equal(groups[0].members.length, 4); assert.equal(groups[0].tiles, 400);
  const e = ci.extentKm([70, 20, 71, 21]); near(e.w, 111.32 * Math.cos(20.5 * Math.PI / 180), 1e-9, 'extent width'); near(e.h, 110.57, 1e-9, 'extent height');
  // core box: 3 stray tiles far away (1.5%) do not move a 200-tile cluster's 95% extent
  const cb = ci.coreBox([[10, 10, 100], [10.2, 10.1, 97], [60, 40, 1], [61, 41, 1], [62, 42, 1]]);
  assert.deepEqual(cb, [10, 10, 10.2, 10.1]); assert.equal(ci.coreBox([]), null);
  // a missing region table degrades to zero regions, not a crash
  const bare = ci.clusterStats({ available: true, sizes: { 7: 10 }, cluster_concepts: { 7: [['x', 0.1]] } }); assert.equal(bare[0].regions.length, 0); assert.equal(bare[0].concentrated, false);
  out.cluster_insight = 'ok';
}

// ---- inventory number formatting ----
{
  assert.equal(fm.fmtGsd(0.30517578), '0.31 m'); assert.equal(fm.fmtGsd(10), '10 m'); assert.equal(fm.fmtGsd(0.3), '0.3 m'); assert.equal(fm.fmtGsd(2.5), '2.5 m');
  assert.equal(fm.fmtGsd(null), fm.DASH);
  out.gsd = 'ok';
}

// ---- calendar axis: positions are proportional to elapsed days (an uneven acquisition series is drawn uneven) ----
{
  const dates = ['2019-03-30', '2021-03-04', '2024-03-08', '2025-03-08', '2026-03-08'];
  const ax = ca.calendarAxis(dates, 0, 1000, 0);
  const day = (a, b) => (Date.parse(b) - Date.parse(a)) / 86400000;
  near(ax.x(dates[0]), 0, 1e-9, 'first date at the left edge'); near(ax.x(dates[4]), 1000, 1e-9, 'last date at the right edge');
  const span = day(dates[0], dates[4]);
  dates.forEach((d) => near(ax.x(d), (day(dates[0], d) / span) * 1000, 1e-9, `x of ${d}`));
  const gaps = dates.slice(1).map((d, i) => ax.x(d) - ax.x(dates[i]));
  near(gaps[1] / gaps[2], day(dates[1], dates[2]) / day(dates[2], dates[3]), 1e-9, '2021->2024 is drawn ~3x the width of 2024->2025');
  assert.ok(gaps[0] > gaps[2] * 1.9 && gaps[1] > gaps[0], 'gaps are unequal');
  assert.deepEqual(ax.ticks.map((t) => t.label), ['2020', '2021', '2022', '2023', '2024', '2025', '2026']);
  ax.ticks.forEach((t) => near(t.x, ax.x(`${t.label}-01-01`), 1e-9, 'tick on 1 Jan'));
  const padded = ca.calendarAxis(dates, 100, 900, 60); assert.ok(padded.x(dates[0]) > 100 && padded.x(dates[4]) < 900);
  assert.equal(ca.calendarAxis([], 0, 10).ticks.length >= 0, true);
  const nm = ca.niceMax(337, 4); assert.equal(nm.max, 400); assert.deepEqual(nm.ticks, [0, 100, 200, 300, 400]);
  assert.deepEqual(ca.niceMax(0).ticks, [0, 1]);
  out.calendar_axis = 'ok';
}

// geodesy checks are appended by later modules (see tools/selftest-geo.mjs)
console.log(JSON.stringify(out));
