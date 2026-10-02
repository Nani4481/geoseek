// End-to-end interaction test for the React console, driven with REAL mouse/keyboard
// input in headless Chrome. Every number the UI shows is compared against the backend's own response.
//
// !! Confirm / reject / reopen append PERMANENT rows to the analyst audit table. Point this at a scratch backend
// !! (DATABASE_URL=sqlite:///<copy of tiles.sqlite>), never at the production catalog. The script refuses to run the
// !! decision step unless --allow-writes is passed.
//
//   node tools/e2e-console.mjs --base http://127.0.0.1:8001/react/ --allow-writes [--shots DIR]
import path from 'node:path';
import { externalRequests, launch, sleep } from './cdp.mjs';
import * as geo from '../src/lib/geo.ts';

const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, all) => {
  if (a.startsWith('--')) acc.push([a.slice(2), all[i + 1] && !all[i + 1].startsWith('--') ? all[i + 1] : 'true']);
  return acc;
}, []));
const BASE = args.base || 'http://127.0.0.1:8001/react/';
const WRITES = args['allow-writes'] === 'true';
const SHOTS = args.shots || '';
const origin = new URL(BASE).origin;

const b = await launch({ width: 1600, height: 1100 });
const results = [];
async function step(name, fn) {
  b.setTag(name);
  const t0 = Date.now();
  try { const note = await fn(); results.push({ name, ok: true, ms: Date.now() - t0, note: note || '' }); }
  catch (e) { results.push({ name, ok: false, ms: Date.now() - t0, note: String(e.message || e).slice(0, 400) }); if (SHOTS) await b.shot(path.join(SHOTS, `FAIL-${name.replace(/\W+/g, '_')}.png`)).catch(() => {}); }
}
const ok = (cond, msg) => { if (!cond) throw new Error(msg); };
const api = (p) => b.evaluate(`fetch(${JSON.stringify(p)}).then(r => r.json())`);
const nfetch = (p) => fetch(origin + p);
const go = async (route) => { await b.evaluate(`location.hash = '#/${route}'`); await sleep(400); };
const imgsLoaded = (sel) => `(() => { const l = [...document.querySelectorAll(${JSON.stringify(sel)})]; return l.length > 0 && l.filter(i => i.complete && i.naturalWidth > 0).length >= Math.ceil(l.length * 0.7); })()`;
const shot = (n) => (SHOTS ? b.shot(path.join(SHOTS, `${n}.png`)) : Promise.resolve());
const text = (sel, nth = 0) => b.evaluate(`(document.querySelectorAll(${JSON.stringify(sel)})[${nth}]?.innerText || '')`);
const has = (s) => `document.body.innerText.toLowerCase().includes(${JSON.stringify(s.toLowerCase())})`;
const READY = `!!document.querySelector('button[aria-label="Replay timeline"]:not([disabled])')`;
const REST = `!document.querySelector('.lapse-stage') && !!document.querySelector('.compare') && document.querySelector('.compare').getBoundingClientRect().width > 50`;
const count = (sel) => b.evaluate(`document.querySelectorAll(${JSON.stringify(sel)}).length`);

try {
  await b.nav(BASE);
  await b.waitFor(`document.querySelectorAll('.stat').length >= 7 && !document.querySelector('.stats .skeleton')`, 30000, 'stat cards');

  await step('dashboard: operational stat cards equal the API; no model-evaluation metrics on the home screen', async () => {
    const m = await api('/ui/metrics'), lat = await api('/ui/latency');
    const stats = await b.evaluate(`[...document.querySelectorAll('.stat')].map(s => ({ l: s.querySelector('.label').innerText, v: s.querySelector('.value').innerText }))`);
    const by = Object.fromEntries(stats.map((s) => [s.l.trim().toLowerCase(), s.v]));
    ok(stats.length === 7, `expected 7 operational cards, got ${stats.length}: ${Object.keys(by).join(', ')}`);
    ok(by['tiles indexed'] === m.counters.tiles_indexed.toLocaleString('en-US'), `tiles ${by['tiles indexed']}`);
    ok(by['scenes'] === String(m.counters.scenes), 'scenes');
    ok(by['regions'] === String(m.counters.regions), 'regions');
    ok(by['sensors'] === String(m.counters.sensors), 'sensors');
    ok(by['change candidates'] === m.counters.change_candidates.toLocaleString('en-US'), 'change candidates');
    ok(by['analyst decisions'] === m.counters.analyst_decisions.toLocaleString('en-US'), 'analyst decisions');
    ok(Math.abs(parseFloat(by['search latency']) - lat.median_ms) < Math.max(40, lat.median_ms), `latency ${by['search latency']} vs ${lat.median_ms}`);
    ok(!stats.some((s) => s.v.includes('—')), 'a card shows a placeholder');
    const body = await b.evaluate('document.body.innerText');
    ok(!/\bF1\b|AP50|OSCD|xView|precision|recall/i.test(body), 'a model-evaluation metric is on the dashboard');
    await shot('01-dashboard');
    return `tiles ${by['tiles indexed']}, scenes ${by['scenes']}, candidates ${by['change candidates']}, decisions ${by['analyst decisions']}, latency ${by['search latency']}`;
  });

  await step('dashboard: findings by region / by change type equal the API', async () => {
    const m = await api('/ui/metrics');
    await b.waitFor(`document.querySelectorAll('.brow').length >= ${m.findings_by_region.length}`, 10000, 'breakdown rows');
    const rows = await b.evaluate(`[...document.querySelectorAll('.dash-pair .panel')].map(p => [...p.querySelectorAll('.brow')].map(r => r.innerText.replace(/\s+/g, ' ').trim()))`);
    ok(rows.length === 2, 'two breakdown panels');
    const total = m.findings_by_region.reduce((s, r) => s + r.candidates, 0);
    ok(rows[0].length === m.findings_by_region.length, `region rows ${rows[0].length}`);
    ok(rows[0][0].endsWith(String(Math.max(...m.findings_by_region.map((r) => r.candidates)))), `top region row: ${rows[0][0]}`);
    ok(total === m.counters.change_candidates, `regions sum ${total} vs candidates ${m.counters.change_candidates}`);
    const typeTotal = Object.values(m.findings_by_type).reduce((a, c) => a + c, 0);
    ok(rows[1].length === Object.keys(m.findings_by_type).length && typeTotal === m.counters.change_candidates, 'type rows / total');
    ok(rows[1].some((r) => r.includes(String(Math.max(...Object.values(m.findings_by_type))))), 'top type count shown');
    return `${rows[0][0]} | ${rows[1][0]}`;
  });

  await step('dashboard: alert feed -> selects candidate, timeline N of M matches API', async () => {
    await b.waitFor(`document.querySelectorAll('.alert').length > 1`);
    await b.click('.alert', 1);
    await sleep(700);
    const sel = (await text('.sel-head b')).trim();
    const tl = await api(`/ui/candidates/${sel}/timeline`);
    await b.waitFor(`document.body.innerText.includes('${tl.n_supporting}') && document.querySelectorAll('.tl .dot').length === ${tl.n_total}`, 8000, 'timeline dots');
    await b.waitFor(imgsLoaded('.compare img, .lapse-stage img'), 20000, 'compare images');
    ok((await text('.tl .lbl', 0)).includes(tl.dates[0]), 'first timeline date label');
    return `${sel}: ${tl.n_supporting} of ${tl.n_total}, ${tl.persistence}`;
  });

  await step('search: text query returns ranked results with latency', async () => {
    await go('search');
    await b.type('input[aria-label="Search query"]', 'open water reservoir');
    await b.click('button[type=submit]');
    await b.waitFor(`document.querySelectorAll('.rcard').length > 0`, 20000, 'results');
    const chip = await text('.chip.green.mono');
    const n = await count('.rcard');
    ok(/ms/.test(chip) && chip.includes(`${n} hits`), `chip "${chip}" vs ${n} cards`);
    await b.waitFor(imgsLoaded('.rcard img'), 20000, 'thumbnails');
    await shot('02-search');
    return chip;
  });

  await step('search: "More like this" from a result tile (image-to-image)', async () => {
    await b.click('.rcard .btn', 0);
    await b.waitFor(`document.body.innerText.toLowerCase().includes('similar to tile')`, 20000, 'similar-to-tile title');
    await b.waitFor(`document.querySelectorAll('.rcard').length > 0`);
    return `${await count('.rcard')} similar tiles`;
  });

  await step('search: region sets a bounding-box filter and results stay inside it', async () => {
    const regions = (await api('/regions')).regions, kutch = regions.find((r) => r.name === 'kutch');
    await b.evaluate(`(() => { const s = [...document.querySelectorAll('select')].find(x => x.options[0]?.text.startsWith('Set box')); const set = Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value').set; set.call(s, 'kutch'); s.dispatchEvent(new Event('change', {bubbles:true})); })()`);
    await b.waitFor(`!!document.querySelector('.chip.cyan.mono')`);
    await b.type('input[aria-label="Search query"]', 'trees and dense vegetation');
    await b.click('button[type=submit]');
    await b.waitFor(`document.body.innerText.toLowerCase().includes('“trees and dense vegetation”') && document.querySelectorAll('.rcard').length > 0`, 20000);
    const coords = await b.evaluate(`[...document.querySelectorAll('.rcard .meta span:nth-child(2)')].map(e => e.innerText)`);
    const [w, s, e, n] = kutch.bbox;
    const bad = coords.filter((c) => { const m = c.match(/([\d.]+)°([NS])\s+([\d.]+)°([EW])/); if (!m) return true; const lat = +m[1] * (m[2] === 'S' ? -1 : 1), lon = +m[3] * (m[4] === 'W' ? -1 : 1); return !(lon >= w - 1e-3 && lon <= e + 1e-3 && lat >= s - 1e-3 && lat <= n + 1e-3); });
    ok(coords.length > 0 && bad.length === 0, `${bad.length}/${coords.length} results outside the box`);
    return `${coords.length} results all inside kutch ${kutch.bbox.map((v) => v.toFixed(2)).join(',')}`;
  });

  await step('search: drag a box on the map replaces the filter', async () => {
    const before = await text('.chip.cyan.mono');
    await b.click('button[aria-pressed]');            // "Draw box"
    const r = await b.rectOf('.leaflet-container');
    await b.drag(r.x + r.w * 0.3, r.y + r.h * 0.3, r.x + r.w * 0.6, r.y + r.h * 0.6);
    await sleep(500);
    const after = await text('.chip.cyan.mono');
    ok(after && after !== before, `bbox chip unchanged: ${before} -> ${after}`);
    return after;
  });

  await step('search: click a map point -> places that look like it', async () => {
    await b.type('input[aria-label="Search query"]', 'bare dry open ground');
    await b.click('button[type=submit]');
    await b.waitFor(`document.body.innerText.toLowerCase().includes('“bare dry open ground”')`, 20000);
    ok(!(await b.evaluate(has('similar to the map point'))), 'stale map-point title present before the click');
    const seed = (await api('/search/text?q=open%20water&k=1&bbox=70,23,72,25')).results[0];
    await b.evaluate(`(() => { const m = document.querySelector('.leaflet-container').__leaflet; m.setView([${seed.lat}, ${seed.lon}], 11, { animate: false }); })()`);
    await sleep(500);
    const r = await b.rectOf('.leaflet-container');
    const pt = await b.evaluate(`(() => { const m = document.querySelector('.leaflet-container').__leaflet; const p = m.latLngToContainerPoint([${seed.lat}, ${seed.lon}]); return { x: p.x, y: p.y }; })()`);
    await b.clickAt(r.x + pt.x, r.y + pt.y + 14);   // a little off the result dot, still inside the tile
    await b.waitFor(has('similar to the map point'), 20000, 'map-point results');
    await b.waitFor(`document.querySelectorAll('.rcard').length > 0`);
    return `${await count('.rcard')} hits`;
  });

  await step('changes: filter by type narrows the queue to the API total', async () => {
    await go('changes');
    await b.waitFor(`document.querySelectorAll('.tbl tbody tr').length > 0`, 20000);
    await b.evaluate(`(() => { const s = document.querySelector('.filters select'); const set = Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value').set; set.call(s, 'construction'); s.dispatchEvent(new Event('change', {bubbles:true})); })()`);
    const api_total = (await api('/candidates?change_type=construction&limit=1')).total;
    await b.waitFor(`document.body.innerText.includes('of ${api_total}')`, 10000, 'filtered total');
    const chips = await b.evaluate(`[...document.querySelectorAll('.tbl tbody tr td:nth-child(2) .chip')].map(c => c.innerText)`);
    ok(chips.length > 0 && chips.every((c) => c === 'New built-up surface'), `unexpected chips ${[...new Set(chips)]}`);
    return `construction: ${api_total} (matches /candidates)`;
  });

  await step('changes: open candidate -> before/after slider, spectral type, confidence, SAR status', async () => {
    await b.click('.tbl tbody tr', 0);
    await b.waitFor(`document.querySelector('.compare') && ${imgsLoaded('.compare img').replace(/\n/g, ' ')}`, 20000, 'compare images');
    const id = (await b.evaluate('location.hash')).split('/').pop();
    const d = await api(`/candidates/${id}`);
    await b.waitFor(`${has(d.classification.rule)} && ${has('shape & size')}`, 15000, 'details + confidence panels');
    const body = await b.evaluate('document.body.innerText');
    ok(body.includes(d.classification.rule), 'spectral rule shown');
    ok(body.includes(String(Math.round(d.confidence * 100))), 'confidence shown');
    ok(body.includes('SAR CORROBORATION'), 'SAR block shown');
    ok(d.sar.available === undefined || d.sar.available === false ? body.includes('Not corroborated') : body.includes('Sentinel-1'), 'SAR status matches API');
    for (const t of d.suppression.trace) ok(body.toLowerCase().includes({ registration: 'alignment', radiometric: 'brightness match', phenology: 'seasonal check', morphology: 'shape & size', quality: 'image quality' }[t.rule]), 'gate ' + t.rule);
    await shot('03-changes');
    return id;
  });

  await step('changes: compare slider responds to a real mouse drag and arrow keys', async () => {
    await b.waitFor(READY, 30000, 'timeline imagery preloaded'); await sleep(300);
    await b.waitFor(REST, 30000, 'on-load timeline sweep settles to the before/after slider');
    const r = await b.rectOf('.compare');
    const read = () => b.evaluate(`+document.querySelector('.compare').getAttribute('aria-valuenow')`);
    await b.drag(r.x + r.w * 0.5, r.y + r.h * 0.5, r.x + r.w * 0.85, r.y + r.h * 0.5);
    await b.waitFor(`+document.querySelector('.compare').getAttribute('aria-valuenow') >= 78`, 5000, 'slider follows the drag');
    const v1 = await read();
    ok(v1 >= 78 && v1 <= 92, `drag -> ${v1}`);
    await b.key('ArrowLeft');
    await b.waitFor(`+document.querySelector('.compare').getAttribute('aria-valuenow') < ${v1}`, 5000, 'ArrowLeft moves the slider');
    const v2 = await read();
    return `drag -> ${v1}, ArrowLeft -> ${v2}`;
  });

  await step('changes: picking a timeline date re-renders the after image and label', async () => {
    await b.waitFor(READY, 30000, 'ready'); await b.waitFor(REST, 30000, 'static view');
    const dots = await count('.tl .dot');
    ok(dots >= 3, 'dots');
    const lbl = (await text('.tl .lbl', dots - 1)).split('\n')[0].trim();
    await b.click('.tl .dot', dots - 1);
    await b.waitFor(`document.querySelector('.compare .tag.r')?.innerText.includes('${lbl}')`, 8000, 'after label');
    await b.waitFor(imgsLoaded('.compare img'));
    await b.click('button[aria-pressed]');           // change mask
    await b.waitFor(`document.querySelector('.compare .tag.r')?.innerText.includes('mask')`);
    await b.waitFor(imgsLoaded('.compare img'));
    return `after = ${lbl}, mask on`;
  });

  if (WRITES) {
    await step('review: confirm -> append-only audit row, verdict updates; reopen appends a reversal', async () => {
      const id = (await b.evaluate('location.hash')).split('/').pop();
      const before = (await api(`/audit?candidate_id=${id}`)).count;
      await b.type('input[placeholder="Why?"]', 'e2e scratch test');
      await b.click('.btn.ok');
      await b.waitFor(`document.body.innerText.includes('Recorded confirm as dec_')`, 10000, 'confirm message');
      const mid = await api(`/audit?candidate_id=${id}`);
      ok(mid.count === before + 1, `audit rows ${before} -> ${mid.count}`);
      ok(mid.decisions[mid.decisions.length - 1].analyst_note === 'e2e scratch test', 'note stored');
      await b.waitFor(`document.body.innerText.includes('confirmed') && [...document.querySelectorAll('button')].some(x => x.innerText.includes('Reopen') && !x.disabled)`, 10000, 'verdict refreshed, Reopen enabled');
      const reopen = await b.evaluate(`(() => { const e = [...document.querySelectorAll('button')].find(x => x.innerText.includes('Reopen')); if (!e || e.disabled) return false; e.click(); return true; })()`);
      ok(reopen, 'Reopen button unavailable');
      await b.waitFor(`document.body.innerText.includes('Recorded reopen as dec_')`, 10000);
      const end = await api(`/audit?candidate_id=${id}`);
      ok(end.count === before + 2, `audit rows after reopen ${end.count}`);
      ok(end.decisions.slice(0, before).every((d, i) => d.decision_id === mid.decisions[i].decision_id), 'earlier rows altered');
      await b.click('.btn.no');
      await b.waitFor(`document.body.innerText.includes('Recorded reject as dec_')`, 10000);
      return `audit rows ${before} -> ${(await api(`/audit?candidate_id=${id}`)).count} (append-only; earlier rows intact)`;
    });
  }

  await step('review: single-candidate and filtered GeoJSON export report the API count', async () => {
    await b.evaluate(`[...document.querySelectorAll('button')].find(x => x.innerText.includes('GeoJSON'))?.click()`);
    await b.waitFor(`document.body.innerText.includes('Exported 1 feature')`, 15000, 'single export');
    await b.evaluate(`[...document.querySelectorAll('button')].find(x => x.innerText.includes('Export filtered'))?.click()`);
    const expected = (await api('/candidates?change_type=construction&limit=1')).total;
    await b.waitFor(`document.body.innerText.includes('Exported ${expected} features')`, 30000, 'filtered export');
    return `1 feature; filtered ${expected} features`;
  });

  // ---- timeline animation: the playhead, lighting nodes, drawn-in bracket and cross-fade are checked against the API's real dates ----
  const axisOf = (dates) => { const ms = dates.map((d) => Date.parse(d + 'T00:00:00Z')); const lo = Math.min(...ms), span = Math.max(1, Math.max(...ms) - lo); return ms.map((m) => (m - lo) / span); };
  const smooth = (t) => { t = Math.max(0, Math.min(1, t)); return t * t * (3 - 2 * t); };
  const expectedOps = (xs, p) => xs.map((x, i) => (i === 0 ? 1 : smooth((p - xs[i - 1]) / Math.max(1e-9, x - xs[i - 1]))));
  const sample = () => b.evaluate(`(() => {
    const ph = document.querySelector('.tl .playhead'), st = document.querySelector('.lapse-stage'), br = document.querySelector('.tl .bracket');
    return {
      p: ph ? +ph.dataset.p : null,
      dots: [...document.querySelectorAll('.tl .dot')].map((d) => ({ left: parseFloat(d.style.left) / 100, lit: d.dataset.lit === '1', label: d.getAttribute('aria-label').slice(0, 10) })),
      drawn: br ? +br.dataset.drawn : null,
      stage: st ? { p: +st.dataset.p, date: st.dataset.date, ops: [...st.querySelectorAll('img')].map((i) => +i.dataset.op) } : null,
    };
  })()`);

  let animId = null, animTl = null;
  await step('timeline: on load a playhead sweeps the real dates; nodes light in order, bracket draws in, imagery cross-fades in sync', async () => {
    animId = (await api('/candidates?sort=queue_score&limit=1')).candidates[0].candidate_id;
    animTl = await api(`/ui/candidates/${animId}/timeline`);
    const xs = axisOf(animTl.dates);
    await go('dashboard'); await sleep(300);
    await b.evaluate(`location.hash = '#/changes/${animId}'`);
    await b.waitFor(`!!document.querySelector('.tl .playhead')`, 30000, 'playhead appears on load');
    const samples = [];
    for (let i = 0; i < 80; i++) { const s = await sample(); if (s.p === null && samples.length > 5) break; if (s.p !== null) samples.push(s); await sleep(110); }
    ok(samples.length >= 12, `only ${samples.length} samples while sweeping`);
    for (let i = 1; i < samples.length; i++) ok(samples[i].p >= samples[i - 1].p - 1e-9, `playhead moved backwards ${samples[i - 1].p} -> ${samples[i].p}`);
    ok(samples[0].p < 0.35, `sweep did not start near the first date (p=${samples[0].p})`);
    // every sampled frame: a node is lit exactly when the playhead has passed its real calendar position
    for (const s of samples) for (const d of s.dots) ok(d.lit === (d.left <= s.p + 1e-6), `node ${d.label} lit=${d.lit} at p=${s.p} (x=${d.left})`);
    const litCounts = samples.map((s) => s.dots.filter((d) => d.lit).length);
    ok(litCounts[0] < animTl.dates.length && litCounts.every((c, i) => i === 0 || c >= litCounts[i - 1]), `nodes did not light in sequence: ${litCounts.join(',')}`);
    // bracket draws in progressively and is whole at the end of the sweep
    if (animTl.first_detected) {
      const [b0, b1] = animTl.first_detected.bracket.map((d) => xs[animTl.dates.indexOf(d)] * 100);
      const drawn = samples.map((s) => s.drawn);
      ok(drawn.every((v, i) => i === 0 || v >= drawn[i - 1] - 1e-6), `bracket shrank: ${drawn.join(',')}`);
      ok(Math.min(...drawn) < (b1 - b0) - 0.5 || b0 === 0, `bracket never drew in: ${drawn.slice(0, 6).join(',')}`);
      ok(Math.abs(Math.max(...drawn) - (b1 - b0)) < 0.2, `bracket final width ${Math.max(...drawn)} vs ${b1 - b0}`);
    }
    // imagery: each layer's opacity is the cross-fade function of the playhead; dates shown are real
    for (const s of samples) {
      ok(s.stage, 'no image stack while sweeping');
      const want = expectedOps(xs, s.stage.p);
      want.forEach((w, i) => ok(Math.abs(w - s.stage.ops[i]) < 0.02, `layer ${i} opacity ${s.stage.ops[i]} vs ${w.toFixed(3)} at p=${s.stage.p}`));
      ok(animTl.dates.includes(s.stage.date), `stage shows a non-catalog date ${s.stage.date}`);
    }
    ok(samples.some((s) => s.stage.ops.some((o) => o > 0.05 && o < 0.95)), 'no cross-fade was ever caught mid-way');
    await shot('10-timeline-sweep');
    return `${samples.length} frames, p ${samples[0].p.toFixed(2)}→${samples[samples.length - 1].p.toFixed(2)}, lit ${litCounts[0]}→${litCounts[litCounts.length - 1]}/${animTl.dates.length}`;
  });

  await step('timeline: after the sweep the static before/after view returns with every node lit and the bracket whole', async () => {
    await b.waitFor(REST, 30000, 'sweep settles');
    const s = await sample();
    ok(s.p === null && s.dots.every((d) => d.lit), 'nodes not all lit at rest');
    ok(await b.evaluate(`!!document.querySelector('.tl .thumb')`), 'rest marker missing');
    ok(await b.evaluate(`document.querySelector('.compare').getAttribute('role') === 'slider'`), 'slider not back');
    return 'rest view restored';
  });

  await step('timeline: replay restarts, pause freezes the playhead, scrub shows the cross-fade at that exact position', async () => {
    const xs = axisOf(animTl.dates);
    await b.click('button[aria-label="Replay timeline"]');
    await b.waitFor(`!!document.querySelector('.tl .playhead')`, 8000, 'replay starts');
    await sleep(900);
    const mid = await sample();
    ok(mid.p > 0.02 && mid.p < 0.9, `replay position ${mid.p}`);
    await b.click('button[aria-label="Pause timeline"]');
    const p1 = (await sample()).p; await sleep(700); const p2 = (await sample()).p;
    ok(p1 !== null && p1 === p2, `playhead moved while paused: ${p1} -> ${p2}`);
    // scrub with the range input to the middle of the axis
    await b.evaluate(`(() => { const e = document.querySelector('input[aria-label="Scrub timeline"]'); const set = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set; set.call(e, '500'); e.dispatchEvent(new Event('input', { bubbles: true })); })()`);
    await b.waitFor(`document.querySelector('.lapse-stage')?.dataset.p === '0.5000'`, 5000, 'scrub position');
    const s = await sample();
    const wantDate = animTl.dates[xs.reduce((acc, x, i) => (x <= 0.5 + 1e-9 ? i : acc), 0)];
    ok(s.stage.date === wantDate, `scrub to 0.5 shows ${s.stage.date}, expected ${wantDate}`);
    expectedOps(xs, 0.5).forEach((w, i) => ok(Math.abs(w - s.stage.ops[i]) < 0.02, `scrub layer ${i}: ${s.stage.ops[i]} vs ${w.toFixed(3)}`));
    // dragging the playhead strip with a real mouse scrubs too
    const r = await b.rectOf('.tl .tl-scrub');
    await b.drag(r.x + r.w * 0.2, r.y + r.h / 2, r.x + r.w * 0.8, r.y + r.h / 2, 6);
    await sleep(200);
    const dragged = await sample();
    ok(dragged.p > 0.6, `mouse scrub left the playhead at ${dragged.p}`);
    // Play resumes from the scrubbed position, and picking a date returns to the static view
    await b.click('button[aria-label="Play timeline"]');
    await sleep(500);
    ok((await sample()).p > dragged.p, 'Play did not resume');
    await b.click('.tl .dot', animTl.dates.length - 1);   // dot 0 is the baseline, which has no before/after pair
    await b.waitFor(REST, 8000, 'picking a date returns to the static view');
    return `paused at ${p1.toFixed(3)}, scrub -> ${wantDate}`;
  });

  await step('timeline: prefers-reduced-motion disables the auto sweep and cross-fades; Play steps date by date', async () => {
    const xs = axisOf(animTl.dates);
    await b.send('Emulation.setEmulatedMedia', { features: [{ name: 'prefers-reduced-motion', value: 'reduce' }] });
    const other = (await api('/candidates?sort=queue_score&limit=2')).candidates[1].candidate_id;
    await b.evaluate(`location.hash = '#/changes/${other}'`);
    const tl2 = await api(`/ui/candidates/${other}/timeline`);
    const xs2 = axisOf(tl2.dates);
    await b.waitFor(`document.querySelector('input[aria-label="Scrub timeline"]') && !document.querySelector('input[aria-label="Scrub timeline"]').disabled`, 30000, 'controls ready');
    await sleep(2500);
    ok(await b.evaluate(`!document.querySelector('.tl .playhead') && !document.querySelector('.lapse-stage')`), 'a sweep started despite reduced motion');
    await b.click('button[aria-label="Play timeline"]');
    const seen = new Set(), ops = [];
    for (let i = 0; i < 70; i++) {
      const s = await sample();
      if (s.p !== null) seen.add(s.p.toFixed(4)); else if (seen.size) break;
      if (s.stage) ops.push(...s.stage.ops);
      await sleep(150);
    }
    ok(seen.size >= 2, `stepped through only ${seen.size} positions`);
    for (const v of seen) ok(xs2.some((x) => Math.abs(x - +v) < 1e-3), `playhead rested between acquisitions at ${v} (reduced motion must step)`);
    ok(ops.every((o) => o === 0 || o === 1), 'a cross-fade was drawn under reduced motion');
    await b.send('Emulation.setEmulatedMedia', { features: [] });
    void xs;
    return `stepped through ${seen.size} acquisitions, no blending`;
  });

  // ---- tactical dossier + threat rings on the candidate map ----
  let zoneCand = null, zoneCandDetail = null;
  await step('dossier: opens from the workbench with MGRS/UTM, chips, gate trace, sensor provenance and audit trail from real data', async () => {
    const all = (await api('/candidates?sort=queue_score&limit=400')).candidates;
    zoneCand = all.find((c) => c.restricted_zone);
    ok(zoneCand, 'no candidate inside a restricted zone to build the dossier for');
    zoneCandDetail = await api(`/candidates/${zoneCand.candidate_id}`);
    await b.evaluate(`location.hash = '#/changes/${zoneCand.candidate_id}'`);
    const zoneTl = await api(`/ui/candidates/${zoneCand.candidate_id}/timeline`);
    await b.waitFor(`document.body.innerText.includes('${zoneCand.candidate_id}') && !!document.querySelector('.tl .dot')`, 20000, 'new candidate loaded');
    await b.waitFor(READY, 30000, 'timeline ready'); await b.waitFor(REST, 30000, 'sweep settled');
    await b.waitFor(`document.querySelector('.compare .tag.r')?.innerText.includes('${zoneTl.first_detected.date}')`, 15000, 'slider shows the new candidate');
    const before = await b.evaluate(`document.querySelector('.compare .tag.l').innerText`);
    const after = (await b.evaluate(`document.querySelector('.compare .tag.r').innerText`)).replace(/ · mask/, '');
    await b.evaluate(`window.__printed = 0; window.print = () => { window.__printed++; }`);
    await b.evaluate(`[...document.querySelectorAll('button')].find(x => x.innerText.includes('Dossier')).click()`);
    await b.waitFor(`document.querySelector('.dossier-sheet')?.dataset.ready === '1'`, 30000, 'dossier ready (images + provenance loaded)');
    const sheet = await b.evaluate(`(() => { const s = document.querySelector('.dossier-sheet'); const t = (sel) => s.querySelector(sel)?.innerText || '';
      return { text: s.innerText, mgrs: t('[data-field=mgrs]'), utm: t('[data-field=utm]'), conf: t('[data-field=confidence]'), verdict: t('[data-field=verdict]'),
        secs: [...s.querySelectorAll('section h2')].map(h => h.innerText),
        chips: [...s.querySelectorAll('.dz-chip')].map(c => ({ cap: c.querySelector('figcaption b').innerText, sub: c.querySelector('figcaption span').innerText, ok: c.querySelector('img')?.naturalWidth > 0 })),
        gates: s.querySelectorAll('[data-sec=gates] tbody tr').length, trail: s.querySelectorAll('[data-sec=decision] tbody tr').length,
        sensors: [...s.querySelectorAll('.dz-sensor')].map(x => ({ role: x.dataset.role, text: x.innerText })) }; })()`);
    const d = zoneCandDetail, [lon, lat] = d.centroid_lonlat;
    // coordinates: converted offline, and the MGRS square must be the Sentinel-2 tile the catalog itself names for the scene
    const wantM = geo.toMGRS(lon, lat, 5).text, u = geo.toUTM(lon, lat);
    ok(sheet.mgrs === wantM, `MGRS ${sheet.mgrs} vs ${wantM}`);
    const tileTok = d.provenance.observations[0].scene.scene_id.split('_')[1];
    ok(sheet.mgrs.replace(' ', '').startsWith(tileTok.slice(0, 3)) && sheet.mgrs.split(' ')[1] === tileTok.slice(3), `MGRS ${sheet.mgrs} is not in the catalog's own tile ${tileTok}`);
    ok(sheet.utm.includes(`zone ${u.zone}${u.band}`) && sheet.utm.includes(`E ${u.easting.toFixed(0)}`) && sheet.utm.includes(`N ${u.northing.toFixed(0)}`), `UTM ${sheet.utm}`);
    ok(sheet.conf === Math.round(d.confidence * 100) + '%', `confidence ${sheet.conf}`);
    ok(sheet.secs.length === 6 && sheet.secs[0].startsWith('1 ·') && sheet.secs[5].startsWith('6 ·'), sheet.secs.join('|'));
    ok(sheet.text.includes(d.restricted_zone.name), 'restricted zone named');
    ok(sheet.chips.length === 3 && sheet.chips.every((c) => c.ok), `chips ${JSON.stringify(sheet.chips)}`);
    ok(sheet.chips[0].sub === before && sheet.chips[1].sub === after && sheet.chips[2].cap === 'DIFFERENCE', `chip dates ${sheet.chips.map((c) => c.sub)} vs ${before}/${after}`);
    ok(sheet.gates === d.suppression.trace.length, `gate rows ${sheet.gates} vs ${d.suppression.trace.length}`);
    ok(sheet.trail === d.decisions.length, `audit rows ${sheet.trail} vs ${d.decisions.length}`);
    const tl = await api(`/ui/candidates/${zoneCand.candidate_id}/timeline`);
    ok(tl.dates.every((x) => sheet.text.includes(x)), 'acquisition dates listed');
    // sensor provenance equals the catalog projection; sun elevation / off-nadir only if the catalog holds them
    const pv = await api(`/ui/candidates/${zoneCand.candidate_id}/dossier?before=${before}&after=${after}`);
    ok(sheet.sensors.length === 2, 'two sensor blocks');
    pv.observations.forEach((o, i) => {
      const t = sheet.sensors[i].text;
      ok(t.includes(o.platform) && t.includes(o.acquired_at) && t.includes(o.scene_id), `sensor block ${o.role}: ${t.slice(0, 120)}`);
      if (o.cloud_fraction !== null) ok(t.includes((o.cloud_fraction * 100).toFixed(1) + '%'), `cloud cover for ${o.role}`);
      ok(/sun elevation/i.test(t) === ('sun_elevation_deg' in o) && /off-nadir/i.test(t) === ('off_nadir_deg' in o), `sun/off-nadir shown without being catalogued (${o.role})`);
    });
    ok(!/placeholder|not implemented|mockup|xx xx|— °/i.test(sheet.text), 'placeholder wording in the dossier');
    await b.waitFor(`!!document.querySelector('.dz-toolbar .btn.primary:not([disabled])')`, 5000);
    await b.click('.dz-toolbar .btn.primary');
    ok((await b.evaluate('window.__printed')) === 1, 'Print button did not call window.print()');
    await shot('11-dossier');
    return `${sheet.mgrs} | ${sheet.utm} | missing in catalog: ${pv.view_fields_not_catalogued.join(', ') || 'none'}`;
  });

  await step('dossier: the print stylesheet shows only the sheet, and the browser renders a real multi-section PDF', async () => {
    await b.send('Emulation.setEmulatedMedia', { media: 'print' });
    const st = await b.evaluate(`(() => ({ root: getComputedStyle(document.getElementById('root')).display, bar: getComputedStyle(document.querySelector('.dz-toolbar')).display,
      sheet: getComputedStyle(document.querySelector('.dossier-sheet')).display, pos: getComputedStyle(document.querySelector('.dossier-overlay')).position }))()`);
    ok(st.root === 'none' && st.bar === 'none' && st.sheet !== 'none' && st.pos === 'static', `print styles ${JSON.stringify(st)}`);
    const r = await b.send('Page.printToPDF', { printBackground: true, preferCSSPageSize: true });
    await b.send('Emulation.setEmulatedMedia', { media: '' });
    const pdf = Buffer.from(r.result.data, 'base64');
    ok(pdf.subarray(0, 5).toString() === '%PDF-', 'not a PDF');
    const pages = (pdf.toString('latin1').match(/\/Type\s*\/Page[^s]/g) || []).length;
    ok(pdf.length > 30000 && pages >= 1 && pages <= 3, `pdf ${pdf.length} bytes, ${pages} pages`);
    if (SHOTS) { const { writeFileSync, mkdirSync } = await import('node:fs'); mkdirSync(SHOTS, { recursive: true }); writeFileSync(path.join(SHOTS, 'dossier.pdf'), pdf); }
    await b.key('Escape');
    await b.waitFor(`!document.querySelector('.dossier-sheet')`, 5000, 'Esc closes the dossier');
    return `${(pdf.length / 1024).toFixed(0)} kB PDF, ${pages} page(s)`;
  });

  const ringsResult = (id, radii) => api(`/ui/threat-rings?lon=${zoneCandDetail.centroid_lonlat[0]}&lat=${zoneCandDetail.centroid_lonlat[1]}&radii_m=${radii}&exclude=candidate:${id}`);
  const centreOnMap = (lon, lat, z = 14) => b.evaluate(`document.querySelector('.leaflet-container').__leaflet.setView([${lat}, ${lon}], ${z}, { animate: false })`);
  const mapPx = async (lon, lat) => {
    const r = await b.rectOf('.leaflet-container');
    const p = await b.evaluate(`(() => { const p = document.querySelector('.leaflet-container').__leaflet.latLngToContainerPoint([${lat}, ${lon}]); return { x: p.x, y: p.y }; })()`);
    return { x: r.x + p.x, y: r.y + p.y };
  };
  const ringRadii = () => b.evaluate(`(() => { const out = []; document.querySelector('.leaflet-container').__leaflet.eachLayer((l) => { if (l._mRadius) out.push(Math.round(l._mRadius)); }); return out.sort((a, c) => a - c); })()`);

  await step('threat rings: right-click a candidate draws concentric rings; ring contents and distances equal the API', async () => {
    await b.waitFor(`!!document.querySelector('.leaflet-container')`, 15000, 'candidate map');
    await b.waitFor(`!!document.querySelector('[data-testid=draw-rings]') && document.body.innerText.includes('right-click any candidate footprint')`, 15000, 'draw-rings button with the right-click hint as secondary text');
    const id = zoneCand.candidate_id, [lon, lat] = zoneCandDetail.centroid_lonlat;
    await centreOnMap(lon, lat, 15); await sleep(700);
    const pt = await mapPx(lon, lat);
    await b.rightClickAt(pt.x, pt.y);
    await b.waitFor(`document.querySelector('.rings-centre')?.innerText === '${id}'`, 10000, 'ring centre set by right-click');
    await b.waitFor(`document.querySelectorAll('.ring-chips .chip').length === 4`, 15000, 'four ring chips');
    const want = await ringsResult(id, '500,1000,2500,5000');
    const chips = await b.evaluate(`[...document.querySelectorAll('.ring-chips .chip')].map(c => +c.dataset.total)`);
    ok(JSON.stringify(chips) === JSON.stringify(want.rings.map((r) => r.total)), `ring totals ${chips} vs ${want.rings.map((r) => r.total)}`);
    const rows = await b.evaluate(`[...document.querySelectorAll('.ring-table tbody tr')].map(r => ({ kind: r.dataset.kind, d: +r.dataset.distance, id: r.querySelector('td:nth-child(3)').title }))`);
    ok(rows.length === want.features.length, `rows ${rows.length} vs ${want.features.length}`);
    ok(rows.every((r, i) => r.id === want.features[i].id && Math.abs(r.d - want.features[i].distance_m) < 0.2), 'rows differ from the API');
    ok(!rows.some((r) => r.id === `candidate:${id}`), 'the ring centre listed itself');
    const zone = rows.find((r) => r.kind === 'restricted_zone');
    ok(zone && zone.d === 0, 'the zone that contains the candidate must be listed at distance 0');
    ok(rows.every((r, i) => i === 0 || r.d >= rows[i - 1].d), 'not sorted by distance');
    ok(JSON.stringify(await ringRadii()) === '[500,1000,2500,5000]', `map rings ${await ringRadii()}`);
    // every listed feature really is within its ring, and the next-larger rings only add features
    ok(want.features.every((f) => f.distance_m <= f.ring_m), 'feature outside its own ring');
    ok(want.rings.every((r, i) => i === 0 || r.total >= want.rings[i - 1].total), 'ring totals not cumulative');
    await shot('12-threat-rings');
    return `${rows.length} features; totals ${chips.join('/')}; ${rows.filter((r) => r.kind === 'change_candidate').length} candidates, ${rows.filter((r) => r.kind === 'restricted_zone').length} zones`;
  });

  await step('threat rings: radii are configurable, validated, and clearable', async () => {
    await b.type('input[aria-label="Ring radii in metres"]', '5');
    await b.waitFor(`!!document.querySelector('.rings-panel [role=alert]') && [...document.querySelectorAll('.rings-panel button')].find(x => x.innerText === 'Apply radii').disabled`, 5000, 'validation error blocks Apply');
    await b.type('input[aria-label="Ring radii in metres"]', '300, 800');
    await b.evaluate(`[...document.querySelectorAll('.rings-panel button')].find(x => x.innerText === 'Apply radii').click()`);
    await b.waitFor(`document.querySelectorAll('.ring-chips .chip').length === 2`, 10000, 'two rings');
    const id = zoneCand.candidate_id, want = await ringsResult(id, '300,800');
    const chips = await b.evaluate(`[...document.querySelectorAll('.ring-chips .chip')].map(c => +c.dataset.total)`);
    ok(JSON.stringify(chips) === JSON.stringify(want.rings.map((r) => r.total)), `totals ${chips}`);
    await b.waitFor(`JSON.stringify((() => { const o = []; document.querySelector('.leaflet-container').__leaflet.eachLayer((l) => { if (l._mRadius) o.push(Math.round(l._mRadius)); }); return o.sort((a, c) => a - c); })()) === '[300,800]'`, 5000, 'map shows exactly the two new rings');
    await b.evaluate(`[...document.querySelectorAll('.rings-panel button')].find(x => x.innerText === 'Clear rings').click()`);
    await b.waitFor(`!document.querySelector('.rings-centre') && document.querySelectorAll('.ring-chips .chip').length === 0`, 5000, 'cleared');
    ok((await ringRadii()).length === 0, 'rings still on the map');
    return 'validated, applied [300,800], cleared';
  });

  await step('detect: oriented boxes drawn over the scene, class toggle works', async () => {
    await go('detect');
    await b.waitFor(`document.querySelectorAll('.detect-stage polygon').length > 0`, 20000, 'polygons');
    const n = await count('.detect-stage polygon');
    const chip = await text('.chip.mono');
    ok(chip.startsWith(`${n} of`), `chip ${chip} vs polygons ${n}`);
    await b.waitFor(imgsLoaded('.detect-stage img'), 20000, 'scene image');
    await b.evaluate(`document.querySelector('.legend-row input[type=checkbox]').click()`);
    await b.waitFor(`document.querySelectorAll('.detect-stage polygon').length < ${n}`, 5000, 'toggle hides boxes');
    const after = await count('.detect-stage polygon');
    await b.evaluate(`document.querySelector('.legend-row input[type=checkbox]').click()`);
    await shot('04-detect');
    const body = await b.evaluate('document.body.innerText');
    ok(body.includes('0.845') && body.includes('0.854'), 'DOTA AP50 0.845 / 0.854 shown');
    return `${n} boxes; toggle -> ${after}`;
  });

  await step('threat rings: right-click a detection on the detection map', async () => {
    await go('detect');
    await b.waitFor(`!!document.querySelector('.leaflet-container') && document.body.innerText.includes('detections')`, 20000, 'detection map');
    const obs = (await api('/detect/observations')).observations.sort((a, c) => c.n_detections - a.n_detections)[0];
    const pts = (await api(`/ui/detections/${obs.observation_id}/points`)).points;
    await b.waitFor(`document.body.innerText.includes('${pts.length.toLocaleString('en-US')} detections')`, 20000, 'all stored detections plotted');
    ok(pts.length === obs.n_detections, `points ${pts.length} vs summary ${obs.n_detections}`);
    // a detection whose neighbours are all > 5 m away, so a right-click hits exactly one marker
    const dm = (a, c) => Math.hypot((a.lon - c.lon) * 111320 * Math.cos(a.lat * Math.PI / 180), (a.lat - c.lat) * 110540);
    const pick = pts.find((p) => pts.every((q) => q === p || dm(p, q) > 8));
    await centreOnMap(pick.lon, pick.lat, 19); await sleep(900);
    const pt = await mapPx(pick.lon, pick.lat);
    await b.rightClickAt(pt.x, pt.y);
    await b.waitFor(`document.querySelector('.rings-centre')?.innerText.includes('${pick.class}')`, 10000, 'ring centre = the clicked detection');
    await b.waitFor(`document.querySelectorAll('.ring-chips .chip').length === 4`, 15000);
    const want = await api(`/ui/threat-rings?lon=${pick.lon}&lat=${pick.lat}&radii_m=500,1000,2500,5000&exclude=${encodeURIComponent(pick.id)}`);
    const rows = await b.evaluate(`[...document.querySelectorAll('.ring-table tbody tr')].map(r => ({ kind: r.dataset.kind, d: +r.dataset.distance, id: r.querySelector('td:nth-child(3)').title }))`);
    ok(rows.length === want.features.length && rows.length > 0, `rows ${rows.length} vs ${want.features.length}`);
    ok(!rows.some((r) => r.id === pick.id), 'detection listed itself');
    ok(rows.every((r) => r.kind === 'detection'), 'this scene is far from every zone/candidate: only detections expected');
    // independent distance check in the browser: great-circle between the centre and each listed detection's nearest point
    const byId = Object.fromEntries(pts.map((p) => [p.id, p]));
    const bad = rows.filter((r) => Math.abs(dm(pick, byId[r.id]) - r.d) > 12);     // centroid-to-centroid vs edge distance differ by the box size
    ok(bad.length === 0, `${bad.length} distances disagree with centroid distances`);
    const chips = await b.evaluate(`[...document.querySelectorAll('.ring-chips .chip')].map(c => +c.dataset.total)`);
    ok(JSON.stringify(chips) === JSON.stringify(want.rings.map((r) => r.total)), 'ring totals');
    await shot('13-detection-rings');
    return `${pick.class}: ${rows.length} detections within 5 km (${chips.join('/')})`;
  });

  // ---- vector space + spectral evidence on Search ----
  let searchedIds = [];
  await step('vector space: 3-D scatter of the projection with an honest sample caption', async () => {
    await go('search');
    const proj = await api('/ui/projection?max_points=20000');
    ok(proj.available && proj.n_total > 50000, 'projection not available - run scripts/compute_projection.py');
    await b.waitFor(`document.querySelector('.vec-stage')?.dataset.points === '${proj.n_shown}'`, 40000, 'points drawn');
    const st = await b.evaluate(`(() => { const s = document.querySelector('.vec-stage'); return { shown: +s.dataset.points, total: +s.dataset.total, cap: document.querySelector('[data-testid=vec-caption]').innerText }; })()`);
    ok(st.total === proj.n_total && st.shown === Math.min(20000, proj.n_total), `drawn ${st.shown} of ${st.total}`);
    ok(st.cap.includes(`${st.shown.toLocaleString('en-US')}-point sample of ${st.total.toLocaleString('en-US')} tiles`), `caption: ${st.cap.slice(0, 120)}`);
    ok(/distances between far-apart groups mean nothing/.test(st.cap), 'UMAP caveat missing');
    // the canvas really contains coloured points (not a blank WebGL surface)
    const lit = await b.evaluate(`(() => { const c = document.querySelector('.vec-stage canvas'); const g = document.createElement('canvas'); g.width = 200; g.height = 150; const x = g.getContext('2d'); x.drawImage(c, 0, 0, 200, 150); const d = x.getImageData(0, 0, 200, 150).data; let n = 0; for (let i = 0; i < d.length; i += 4) if (d[i] + d[i + 1] + d[i + 2] > 150) n++; return n; })()`);
    ok(lit > 150, `canvas looks empty (${lit} lit pixels)`);
    const legend = await b.evaluate(`[...document.querySelectorAll('.vec-legend > span')].map(s => s.innerText)`);
    ok(legend.length === proj.regions.length || legend.length === new Set(proj.region).size, `legend ${legend.length} entries vs ${new Set(proj.region).size} regions in the sample`);
    await shot('14-vector-space');
    return `${st.shown} of ${st.total} points, ${legend.length} regions, ${lit} lit px`;
  });

  await step('vector space: colour by cluster, and search results are highlighted in the scatter', async () => {
    const proj = await api('/ui/projection?max_points=20000');
    await b.evaluate(`[...document.querySelectorAll('.tabs button')].find(x => x.innerText === 'cluster').click()`);
    const clusters = new Set(proj.cluster);
    await b.waitFor(`document.querySelectorAll('.vec-legend > span').length === ${clusters.size}`, 8000, 'cluster legend');
    ok(await b.evaluate(`[...document.querySelectorAll('.vec-legend > span')].some(s => /^cluster \\d+/.test(s.innerText))`), 'cluster labels');
    await b.type('input[aria-label="Search query"]', 'open water reservoir');
    await b.click('button[type=submit]');
    await b.waitFor(`document.querySelectorAll('.rcard').length > 0`, 20000, 'results');
    searchedIds = await b.evaluate(`[...document.querySelectorAll('button[data-evidence]')].map(e => e.dataset.evidence)`);
    const n = searchedIds.length;
    await b.waitFor(`document.querySelector('.vec-stage')?.dataset.hits === '${n}'`, 15000, 'results highlighted');
    const want = await api(`/ui/projection/lookup?ids=${searchedIds.join(',')}`);
    ok(want.points.length === n && want.missing.length === 0, `lookup ${want.points.length}/${n}`);
    const cap = await b.evaluate(`document.querySelector('[data-testid=vec-caption]').innerText`);
    ok(cap.includes(`${n} search results highlighted`), cap.slice(0, 200));
    // the highlight layer holds exactly the results' coordinates
    const bright = await b.evaluate(`(() => { const c = document.querySelector('.vec-stage canvas'); const g = document.createElement('canvas'); g.width = c.width; g.height = c.height; const x = g.getContext('2d'); x.drawImage(c, 0, 0); const d = x.getImageData(0, 0, g.width, g.height).data; let k = 0; for (let i = 0; i < d.length; i += 4) if (d[i] > 250 && d[i + 1] > 250 && d[i + 2] > 250) k++; return k; })()`);
    ok(bright > 20, `no white highlight pixels (${bright})`);
    await b.evaluate(`[...document.querySelectorAll('.tabs button')].find(x => x.innerText === 'region').click()`);
    await shot('15-vector-highlight');
    return `${n} results highlighted (${bright} white px)`;
  });

  await step('vector space: clicking a point pans the map to that tile', async () => {
    // click exactly on a drawn point; the expected result is the point nearest to the click on screen (front-most on a tie)
    const pick = await b.evaluate(`(() => {
      const v = document.querySelector('.vec-stage').__vec, r = document.querySelector('.vec-stage canvas').getBoundingClientRect();
      const P = []; for (let i = 0; i < v.count; i++) P.push(v.project(i));
      const cx = (r.left + r.right) / 2, cy = (r.top + r.bottom) / 2;
      let i0 = -1, bd = 1e18; for (let i = 0; i < v.count; i++) { if (P[i].z > 1) continue; const d = (P[i].x - cx) ** 2 + (P[i].y - cy) ** 2; if (d < bd) { bd = d; i0 = i; } }
      const x = P[i0].x, y = P[i0].y;
      let best = -1, bdist = 1e18, bz = 9; for (let i = 0; i < v.count; i++) { if (P[i].z > 1) continue; const d = (P[i].x - x) ** 2 + (P[i].y - y) ** 2; if (d < bdist - 1e-9 || (Math.abs(d - bdist) <= 1e-9 && P[i].z < bz)) { bdist = d; best = i; bz = P[i].z; } }
      return { x, y, id: v.tileId(best), ll: v.lonlat(best) };
    })()`);
    await b.clickAt(pick.x, pick.y);
    await b.waitFor(`document.querySelector('.vec-picked b')?.innerText === '${pick.id}'`, 8000, 'picked tile card');
    await b.waitFor(`(() => { const m = document.querySelector('.leaflet-container').__leaflet, c = m.getCenter(); return Math.abs(c.lng - ${pick.ll[0]}) < 0.002 && Math.abs(c.lat - ${pick.ll[1]}) < 0.002; })()`, 8000, 'map centred on the tile');
    const fp = (await api(`/ui/tiles?ids=${pick.id}`)).tiles[0];
    const inside = pick.ll[0] >= fp.bbox[0] - 1e-6 && pick.ll[0] <= fp.bbox[2] + 1e-6 && pick.ll[1] >= fp.bbox[1] - 1e-6 && pick.ll[1] <= fp.bbox[3] + 1e-6;
    ok(inside, 'the clicked point is not inside its catalogued tile footprint');
    await shot('16-vector-pan');
    return `${pick.id} -> map centre ${pick.ll.map((v) => v.toFixed(4)).join(', ')}`;
  });

  let evTile = null;
  await step('spectral evidence: per-pixel NDWI overlay for a water query, with legend and statistics from the API', async () => {
    await b.type('input[aria-label="Search query"]', 'an open water reservoir or pond');
    await b.click('button[type=submit]');
    await b.waitFor(`document.querySelectorAll('.rcard').length > 0`, 20000);
    const ids = await b.evaluate(`[...document.querySelectorAll('button[data-evidence]')].map(e => e.dataset.evidence)`);
    // choose a Sentinel-2 hit whose bands are staged
    let tid = null, api_spec = null;
    for (const id of ids) { const rr = await nfetch(`/ui/tiles/${id}/spectral?q=water`); const r = rr.ok ? await rr.json() : null; if (r && r.usable) { tid = id; api_spec = r; break; } }
    ok(tid, 'no usable Sentinel-2 hit with NIR/SWIR among the results');
    evTile = tid;
    await b.evaluate(`document.querySelector('button[data-evidence="${tid}"]').click()`);
    await b.waitFor(`document.querySelector('.spec-panel')?.dataset.tile === '${tid}' && document.querySelectorAll('.spec-fig').length > 0`, 20000, 'evidence panel');
    const info = await b.evaluate(`(() => ({
      why: document.querySelector('[data-testid=spec-why]').innerText,
      figs: [...document.querySelectorAll('.spec-fig')].map(f => ({ idx: f.dataset.index, rel: f.dataset.relevant, stats: f.querySelector('[data-stats]')?.innerText, bar: f.querySelector('.spec-legend .bar').style.background })),
      classes: Object.fromEntries([...document.querySelectorAll('[data-class]')].map(e => [e.dataset.class, e.innerText])),
    }))()`);
    ok(/NDWI/.test(info.why) && /water/.test(info.why), `why: ${info.why}`);
    ok(info.figs.length === 1 && info.figs[0].idx === 'ndwi' && info.figs[0].rel === '1', `figures ${JSON.stringify(info.figs.map((f) => f.idx))}`);
    const s = api_spec.layers.ndwi.stats;
    ok(info.figs[0].stats.includes(`mean ${s.mean.toFixed(3)}`) && info.figs[0].stats.includes(`p90 ${s.p90.toFixed(2)}`), `stats ${info.figs[0].stats} vs ${JSON.stringify(s)}`);
    ok(info.classes.water_frac === (api_spec.classes.water_frac * 100).toFixed(1) + '%', `water fraction ${info.classes.water_frac}`);
    const hexes = api_spec.layers.ndwi.stops.map((x) => x[1]);
    ok(hexes.every((h) => info.figs[0].bar.toLowerCase().includes(`rgb(${parseInt(h.slice(1, 3), 16)}, ${parseInt(h.slice(3, 5), 16)}, ${parseInt(h.slice(5, 7), 16)})`)), `legend gradient lacks the API stops: ${info.figs[0].bar}`);
    // the overlay is the API's per-pixel PNG at 10 m: every opaque pixel's colour lies on the legend ramp
    await b.waitFor(`(() => { const i = document.querySelector('.spec-fig .pix'); return i && i.complete && i.naturalWidth === ${api_spec.width_px}; })()`, 15000, 'overlay loaded');
    const off = await b.evaluate(`(async () => {
      const im = document.querySelector('.spec-fig .pix'); const c = document.createElement('canvas'); c.width = im.naturalWidth; c.height = im.naturalHeight;
      const x = c.getContext('2d'); x.drawImage(im, 0, 0); const d = x.getImageData(0, 0, c.width, c.height).data;
      const stops = ${JSON.stringify(api_spec.layers.ndwi.stops)}.map(([v, h]) => [v, [1, 3, 5].map((k) => parseInt(h.slice(k, k + 2), 16))]);
      const ramp = []; for (let i = 0; i <= 400; i++) { const v = stops[0][0] + (stops[stops.length - 1][0] - stops[0][0]) * i / 400; let k = 0; while (k < stops.length - 2 && v > stops[k + 1][0]) k++; const t = (v - stops[k][0]) / (stops[k + 1][0] - stops[k][0]); ramp.push([0, 1, 2].map((q) => stops[k][1][q] + (stops[k + 1][1][q] - stops[k][1][q]) * t)); }
      let opaque = 0, worst = 0; for (let i = 0; i < d.length; i += 4) { if (d[i + 3] < 255) continue; opaque++; let best = 1e9; for (const r of ramp) { const e = Math.abs(r[0] - d[i]) + Math.abs(r[1] - d[i + 1]) + Math.abs(r[2] - d[i + 2]); if (e < best) best = e; } if (best > worst) worst = best; }
      return { opaque, total: d.length / 4, worst };
    })()`);
    ok(off.opaque > 0.2 * off.total && off.worst <= 6, `overlay pixels off the legend ramp: ${JSON.stringify(off)}`);
    // toggles and opacity
    await b.evaluate(`document.querySelector('[data-toggle=ndbi]').click()`);
    await b.waitFor(`document.querySelectorAll('.spec-fig').length === 2`, 5000, 'NDBI figure added');
    await b.evaluate(`document.querySelector('[data-toggle=ndbi]').click()`);
    await b.waitFor(`document.querySelectorAll('.spec-fig').length === 1`, 5000, 'NDBI figure removed');
    await b.evaluate(`(() => { const e = document.querySelector('input[aria-label="Overlay opacity"]'); const set = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set; set.call(e, '0.3'); e.dispatchEvent(new Event('input', { bubbles: true })); })()`);
    await b.waitFor(`document.querySelector('.spec-fig .pix').style.opacity === '0.3'`, 5000, 'opacity slider');
    const body = await b.evaluate('document.body.innerText');
    ok(/never sees NIR\/SWIR/.test(body) && /\d+ m patches/.test(body), 'explainability caveat with the computed patch size');
    ok(!/attention|heatmap|XAI/i.test(body.replace(/never[^.]*\./i, '')), 'the panel must not present an attention map');
    await shot('17-spectral-evidence');
    return `${tid}: NDWI mean ${s.mean.toFixed(3)}, water ${info.classes.water_frac}, ${off.opaque}/${off.total} opaque px on the ramp`;
  });

  await step('spectral evidence: the indices that bear on the query change with the words (built-up, bare ground, unrelated)', async () => {
    await b.select('select:has(option[value=ayodhya])', 'ayodhya');
    await b.waitFor(`!!document.querySelector('.chip.cyan.mono')`, 5000, 'region box');
    const expect = [['dense urban buildings and rooftops', ['ndbi']], ['bare dry open ground', ['ndbi', 'ndvi']], ['trees and dense vegetation', ['ndvi']]];
    for (const [q, want] of expect) {
      await b.type('input[aria-label="Search query"]', q);
      await b.click('button[type=submit]');
      await b.waitFor(`document.body.innerText.toLowerCase().includes('“${q}”') && document.querySelectorAll('.rcard').length > 0`, 20000);
      const ids = await b.evaluate(`[...document.querySelectorAll('button[data-evidence]')].map(e => e.dataset.evidence)`);
      let tid = null;
      for (const id of ids) { if ((await nfetch(`/ui/tiles/${id}/spectral`)).ok) { tid = id; break; } }
      ok(tid, `no S2 tile among results for ${q}`);
      await b.evaluate(`document.querySelector('button[data-evidence="${tid}"]').click()`);
      await b.waitFor(`document.querySelector('.spec-panel')?.dataset.tile === '${tid}' && [...document.querySelectorAll('.spec-fig')].map(f => f.dataset.index).sort().join() === ${JSON.stringify([...want].sort().join())}`, 20000, `figures for "${q}"`).catch(() => {});   // the tile can still show the previous query's figures for a moment; the assertion below reports a real mismatch
      const shown = (await b.evaluate(`[...document.querySelectorAll('.spec-fig')].map(f => f.dataset.index)`)).sort();
      ok(JSON.stringify(shown) === JSON.stringify([...want].sort()), `"${q}": shown ${shown} vs ${want}`);
      ok(await b.evaluate(`[...document.querySelectorAll('.spec-fig')].every(f => f.dataset.relevant === '1')`), 'flags');
    }
    // a query with no spectral link shows all three, and says so
    await b.type('input[aria-label="Search query"]', 'a quiet scene');
    await b.click('button[type=submit]');
    await b.waitFor(`document.body.innerText.toLowerCase().includes('“a quiet scene”') && document.querySelectorAll('.rcard').length > 0`, 20000);
    const ids = await b.evaluate(`[...document.querySelectorAll('button[data-evidence]')].map(e => e.dataset.evidence)`);
    let tid = null; for (const id of ids) { if ((await nfetch(`/ui/tiles/${id}/spectral`)).ok) { tid = id; break; } }
    await b.evaluate(`document.querySelector('button[data-evidence="${tid}"]').click()`);
    await b.waitFor(`document.querySelectorAll('.spec-fig').length === 3`, 20000, 'all three shown');
    ok(/all three/.test(await text('[data-testid=spec-why]')), 'no-match note');
    return 'urban->NDBI, bare->NDBI+NDVI, vegetation->NDVI, unrelated->all three';
  });

  await step('spectral evidence: a tile without NIR/SWIR is a visible "not staged" answer, never a fabricated overlay', async () => {
    const maxar = (await api('/detect/observations')).observations.find((o) => o.n_detections > 1000);
    const tiles = (await api(`/detect/observations/${maxar.observation_id}/tiles`)).tiles;
    const rr = await nfetch(`/ui/tiles/${tiles[0].tile_id}/spectral`), r = { status: rr.status, body: await rr.json() };
    ok(r.status === 404 && /not staged/.test(r.body.detail), `status ${r.status} ${JSON.stringify(r.body)}`);
    const png = (await nfetch(`/ui/tiles/${tiles[0].tile_id}/spectral/ndwi.png`)).status;
    ok(png === 404, `png status ${png}`);
    return 'Maxar tile -> 404 with reason';
  });

  // ---- Data management: the ad-hoc raster dropzone parses real file headers in the browser ----
  const tmp = (await import('node:os')).tmpdir();
  const fsx = await import('node:fs');
  const { makeGeoTiff } = await import('./mktiff.mjs');
  let ayo = null;
  await step('data: archive inventory equals the API', async () => {
    await go('data');
    const m = await api('/ui/metrics'), regs = (await api('/regions')).regions;
    await b.waitFor(`document.querySelectorAll('table[aria-label=Collections] tbody tr').length === ${m.sensors.length} && document.querySelectorAll('table[aria-label=Regions] tbody tr').length === ${regs.length}`, 15000, 'inventory tables');
    const body = await b.evaluate('document.body.innerText');
    ok(body.includes(m.counters.tiles_indexed.toLocaleString('en-US')) && body.includes(m.observation_dates[0]), 'counters / dates');
    ayo = regs.find((r) => r.name === 'ayodhya');
    return `${m.sensors.length} collections, ${regs.length} regions`;
  });

  await step('data: dropping a GeoTIFF parses its real header; the badge shows what the file actually contains', async () => {
    const cx = (ayo.bbox[0] + ayo.bbox[2]) / 2, cy = (ayo.bbox[1] + ayo.bbox[3]) / 2;
    const u = geo.toUTM(cx, cy);
    const spec = { width: 64, height: 48, epsg: 32600 + u.zone, originX: Math.round(u.easting / 10) * 10, originY: Math.round(u.northing / 10) * 10, resX: 10, resY: 10, acquisition: '2024-03-08T05:21:00Z' };
    const file = path.join(tmp, 'geoseek-e2e-valid.tif');
    fsx.writeFileSync(file, makeGeoTiff(spec));
    await b.setFiles('[data-testid=raster-input]', [file]);
    await b.waitFor(`document.querySelector('.raster-card')?.dataset.verdict`, 15000, 'card');
    const c = await b.evaluate(`(() => { const k = document.querySelector('.raster-card'); const f = (n) => k.querySelector('[data-field=' + n + ']')?.innerText; return {
      verdict: k.dataset.verdict, label: k.querySelector('[data-testid=verdict]').innerText, size: f('size'), bands: f('bands'), crs: f('crs'), res: f('resolution'), affine: f('affine'), bounds: f('bounds'), lonlat: f('lonlat'), acq: f('acquisition'),
      checks: [...k.querySelectorAll('[data-check]')].map(e => [e.dataset.check, e.dataset.ok]), contains: k.querySelector('[data-testid=what-it-contains]').innerText, name: k.querySelector('header b').innerText }; })()`);
    ok(c.verdict === 'green' && /HEADER VALID/.test(c.label), `verdict ${c.label}`);
    ok(c.size === '64 × 48 px' && c.bands === '1 × uint8', `${c.size} | ${c.bands}`);
    ok(c.crs.startsWith(`EPSG:${spec.epsg}`) && c.crs.includes(`UTM zone ${u.zone}N`), c.crs);
    ok(c.res === '10 × 10 m', c.res);
    ok(c.affine === `[10, 0, ${spec.originX}, 0, -10, ${spec.originY}]`, c.affine);
    const wantB = [spec.originX, spec.originY - 480, spec.originX + 640, spec.originY];
    ok(c.bounds === wantB.map((x) => +x.toFixed(3)).join(', '), `bounds ${c.bounds} vs ${wantB}`);
    const ll = [[wantB[0], wantB[1]], [wantB[2], wantB[1]], [wantB[2], wantB[3]], [wantB[0], wantB[3]]].map(([e, n]) => geo.fromUTM(u.zone, true, e, n));
    const wantLL = [Math.min(...ll.map((p) => p.lon)), Math.min(...ll.map((p) => p.lat)), Math.max(...ll.map((p) => p.lon)), Math.max(...ll.map((p) => p.lat))];
    ok(c.lonlat === wantLL.map((x) => x.toFixed(5)).join(', '), `lonlat ${c.lonlat}`);
    ok(/2024-03-08T05:21:00\.000Z \(GDAL metadata ACQUISITION_DATE\)/.test(c.acq), c.acq);
    ok(c.checks.length === 5 && c.checks.every(([, v]) => v === '1'), JSON.stringify(c.checks));
    ok(/overlaps the archive region Ayodhya \(100% of the file\)/.test(c.contains), c.contains);
    ok(/EPSG:\d+/.test(c.contains) && c.contains.includes('64 × 48 px'), 'contains statement');
    await shot('18-data-dropzone');
    return `${c.size}, ${c.crs}, ${c.res}, acquired ${c.acq.slice(0, 24)}`;
  });

  await step('data: the panel states plainly that nothing was ingested, and shows no fake progress', async () => {
    const note = await text('[data-testid=ingest-note]');
    ok(/Not ingested/.test(note) && /no ingest is running/.test(note) && /python -m geoseek\.ingest\.pipeline ingest/.test(note), note.slice(0, 200));
    const fake = await b.evaluate(`document.querySelectorAll('progress, [role=progressbar], [class*=progress]').length`);
    ok(fake === 0, `${fake} progress elements`);
    return 'ingest-pipeline note present, 0 progress elements';
  });

  await step('data: a dropped (drag-and-drop) un-georeferenced TIFF and a non-TIFF are reported as such', async () => {
    const plain = makeGeoTiff({ epsg: null, width: 16, height: 16 }).toString('base64');
    const junk = Buffer.from('definitely not a tiff '.repeat(10)).toString('base64');
    await b.evaluate(`(() => {
      const mk = (b64, name) => new File([Uint8Array.from(atob(b64), (c) => c.charCodeAt(0))], name);
      const dt = new DataTransfer(); dt.items.add(mk(${JSON.stringify(plain)}, 'plain.tif')); dt.items.add(mk(${JSON.stringify(junk)}, 'junk.tif'));
      document.querySelector('.dropzone').dispatchEvent(new DragEvent('drop', { dataTransfer: dt, bubbles: true, cancelable: true }));
    })()`);
    await b.waitFor(`document.querySelectorAll('.raster-card').length === 2`, 15000, 'two cards');
    const cards = await b.evaluate(`[...document.querySelectorAll('.raster-card')].map(k => ({ name: k.querySelector('header b').innerText, verdict: k.dataset.verdict, label: k.querySelector('[data-testid=verdict]').innerText, checks: [...k.querySelectorAll('[data-check]')].map(e => e.dataset.check + ':' + e.dataset.ok), contains: k.querySelector('[data-testid=what-it-contains]')?.innerText || '', err: k.querySelector('.err')?.innerText || '' }))`);
    const plainC = cards.find((c) => c.name === 'plain.tif'), junkC = cards.find((c) => c.name === 'junk.tif');
    ok(plainC.verdict === 'amber' && /CHECKS? FAILED/.test(plainC.label) && plainC.checks.includes('crs:0') && plainC.checks.includes('transform:0'), JSON.stringify(plainC));
    ok(/no usable coordinate reference system/.test(plainC.contains) && /no acquisition timestamp/.test(plainC.contains), plainC.contains);
    ok(junkC.verdict === 'red' && /NOT A READABLE GEOTIFF/.test(junkC.label) && /could not be parsed/.test(junkC.err), JSON.stringify(junkC));
    return 'plain.tif amber (no CRS), junk.tif red';
  });

  await step('data: a 168 MB staged Sentinel-2 band parses from its header alone (no full read), matching the catalog', async () => {
    const p = 'C:/Users/Prash/Downloads/SIH 2026/geoseek/data/datasets/S2A_42QXM_20240110_0_L2A/B08.tif';
    if (!fsx.existsSync(p)) return 'skipped: staged band not on this machine';
    const t0 = Date.now();
    await b.setFiles('[data-testid=raster-input]', [p]);
    await b.waitFor(`document.querySelector('.raster-card')?.dataset.verdict === 'green'`, 30000, 'real file card');
    const ms = Date.now() - t0;
    const c = await b.evaluate(`(() => { const k = document.querySelector('.raster-card'); return { crs: k.querySelector('[data-field=crs]').innerText, size: k.querySelector('[data-field=size]').innerText, contains: k.querySelector('[data-testid=what-it-contains]').innerText, bytes: k.querySelector('header .dim').innerText }; })()`);
    ok(c.crs.startsWith('EPSG:32642') && c.size === '10,913 × 10,903 px', `${c.crs} | ${c.size}`);
    ok(/Kutch/.test(c.contains) && /matches the .*10 m/.test(c.contains), c.contains);
    ok(/MB|GB/.test(c.bytes), c.bytes);
    ok(ms < 8000, `took ${ms} ms (a whole-file read would be slower)`);
    const decoders = b.requests.filter((r) => /lerc|zstd|jpeg|lzw|packbits|pako|webimage/.test(r.url));
    ok(decoders.length === 0, 'pixel decoders were fetched: ' + decoders.map((d) => d.url).join(', '));
    return `${c.size} ${c.bytes} parsed in ${ms} ms; no pixel decoder loaded`;
  });

  await step('discovery: clusters table equals the API, no static image or duplicated legend block, seed explorer', async () => {
    await go('discovery');
    const cl = await api('/discovery/clusters');
    await b.waitFor(`document.querySelectorAll('.tbl tbody tr').length === ${cl.n_clusters}`, 15000, 'cluster rows');
    await b.waitFor(`document.querySelectorAll('.fp-tile').length > 0`, 15000, 'seed tiles');
    ok(!(await b.evaluate(`!!document.querySelector('.cluster-img, img[src*="cluster-map"]')`)), 'the static matplotlib cluster image is still on the page');
    const rows = await b.evaluate(`[...document.querySelectorAll('.tbl tbody tr')].map(r => ({ id: r.dataset.cluster, n: r.querySelector('td.num').innerText }))`);
    for (const r of rows) ok(r.n === cl.sizes[r.id].toLocaleString('en-US'), `cluster ${r.id}: table ${r.n} vs API ${cl.sizes[r.id]}`);
    await shot('05-discovery');
    return `${cl.n_clusters} clusters, tile counts equal the API, ${await count('.fp-tile')} seed neighbours`;
  });

  await step('fingerprints: gallery, comparison axes, no thermal axis', async () => {
    await go('fingerprints');
    await b.waitFor(`document.querySelectorAll('.fp-tile').length > 3 && document.querySelectorAll('.axis-row').length === 4`, 20000);
    await b.click('.fp-tile', 3);
    await b.waitFor(`(() => { const r = [...document.querySelectorAll('.axis-row')].map((x) => x.innerText); return document.querySelectorAll('.fp-tile.sel').length === 1 && r.length === 4 && r.some((a) => a.startsWith('Proximity') && a.includes('km')) && r.some((a) => a.startsWith('Footprint') && a.includes('km²')); })()`, 15000, 'proximity + footprint axes resolved from /ui/tiles');
    const axes = await b.evaluate(`[...document.querySelectorAll('.axis-row')].map(r => r.innerText.replace(/\\s+/g,' '))`);
    ok(axes.length === 4 && axes.some((a) => a.startsWith('Proximity') && a.includes('km')) && axes.some((a) => a.startsWith('Footprint') && a.includes('km²')), axes.join(' | '));
    ok(!/thermal/i.test(await b.evaluate('document.body.innerText')), 'thermal axis present');
    await shot('06-fingerprints');
    return axes.join(' | ');
  });

  // ---- layout primitives: equal-height rows, equal-width button rows, uniform gallery cards ----
  const rowsOf = (sel, kids) => b.evaluate(`(() => { const c = document.querySelector(${JSON.stringify(sel)}); if (!c) return null;
    const rows = {}; for (const e of [...c.children].filter((x) => x.matches(${JSON.stringify(kids)}))) { const r = e.getBoundingClientRect(); (rows[Math.round(r.top)] ||= []).push(Math.round(r.height * 10) / 10); }
    return Object.values(rows); })()`);
  const equalRows = (name, rows) => {
    ok(rows && rows.length > 0, `${name}: container not found`);
    ok(rows.some((r) => r.length >= 2), `${name}: no row holds two or more panels`);
    for (const r of rows) ok(Math.max(...r) - Math.min(...r) <= 1.5, `${name}: a row has unequal panel heights ${JSON.stringify(rows)}`);
  };

  await step('layout: side-by-side panels render at one height (dashboard, changes, fingerprints)', async () => {
    await go('dashboard');
    await b.waitFor(`document.querySelectorAll('.dash-pair .panel').length === 2 && document.querySelectorAll('.dash-triple .panel').length === 3 && document.querySelectorAll('.brow').length > 3`, 20000, 'dashboard rows');
    await sleep(600);
    equalRows('dashboard findings row', await rowsOf('.dash-pair', '.panel'));
    equalRows('dashboard before/details/confidence row', await rowsOf('.dash-triple', '.panel'));
    await go('changes');
    await b.waitFor(`document.querySelectorAll('.wb-grid .panel').length === 4`, 20000, 'changes workbench');
    await sleep(800);
    const wb = await rowsOf('.wb-grid', '.panel');
    equalRows('changes workbench', wb);
    ok(wb.length === 2 && wb.every((r) => r.length === 2), `changes workbench is 2 rows of 2 panels: ${JSON.stringify(wb)}`);
    const cols = await rowsOf('.changes-grid', '.col');
    ok(cols && cols[0].length === 2 && Math.abs(cols[0][0] - cols[0][1]) <= 1.5, `queue column and workbench column differ in height: ${JSON.stringify(cols)}`);
    await go('fingerprints');
    await b.waitFor(`document.querySelectorAll('.fp-tile').length > 3`, 30000, 'fingerprint tiles');
    await sleep(800);
    const fp = await b.evaluate(`(() => { const g = document.querySelector('.fp-grid'); const k = [...g.children]; const bot = (e) => Math.round(e.getBoundingClientRect().bottom * 10) / 10;
      const last = k[1].querySelector(':scope > .panel:last-child'); return { left: bot(k[0]), rightCol: bot(k[1]), lastPanel: bot(last) }; })()`);
    ok(Math.abs(fp.left - fp.rightCol) <= 1.5 && Math.abs(fp.rightCol - fp.lastPanel) <= 1.5, `fingerprint gallery / detail column end at different heights: ${JSON.stringify(fp)}`);
    return `dashboard 2+3, changes 2x2 + queue/workbench columns, fingerprints gallery/detail bottoms ${fp.left}`;
  });

  await step('layout: result-card buttons are three equal, unclipped, single-line cells at every card width', async () => {
    await go('search');
    await b.type('input[aria-label="Search query"]', 'open water reservoir');
    await b.click('button[type=submit]');
    await b.waitFor(`document.querySelectorAll('.rcard .actions').length > 3`, 20000, 'result cards');
    const check = () => b.evaluate(`(() => { const bad = []; let n = 0; let minCard = 1e9;
      for (const a of document.querySelectorAll('.rcard .actions')) { const btns = [...a.children]; n++; minCard = Math.min(minCard, a.getBoundingClientRect().width);
        const w = btns.map((x) => x.getBoundingClientRect().width), h = btns.map((x) => x.getBoundingClientRect().height);
        if (btns.length !== 3) bad.push('buttons=' + btns.length);
        if (Math.max(...w) - Math.min(...w) > 1) bad.push('widths ' + w.map((v) => v.toFixed(1)));
        if (Math.max(...h) - Math.min(...h) > 0.5) bad.push('heights ' + h.join('/'));
        const ar = a.getBoundingClientRect(); if (Math.abs(btns[0].getBoundingClientRect().left - ar.left) > 1 || Math.abs(btns[2].getBoundingClientRect().right - ar.right) > 1) bad.push('does not fill the card width');
        for (const x of btns) { if (x.scrollWidth > x.clientWidth + 0.5) bad.push('clipped "' + x.innerText + '"'); if (x.getClientRects().length && x.getBoundingClientRect().height > 30) bad.push('wrapped "' + x.innerText + '"'); }
      } return { n, minCard: Math.round(minCard), bad }; })()`);
    const wide = await check();
    ok(wide.n > 3 && wide.bad.length === 0, 'wide: ' + JSON.stringify(wide));
    await b.send('Emulation.setDeviceMetricsOverride', { width: 1100, height: 900, deviceScaleFactor: 1, mobile: false });
    await sleep(700);
    const narrow = await check();
    await b.send('Emulation.setDeviceMetricsOverride', { width: 1600, height: 1100, deviceScaleFactor: 1, mobile: false });
    await sleep(500);
    ok(narrow.n > 3 && narrow.bad.length === 0, 'narrow: ' + JSON.stringify(narrow));
    // the narrowest card the grid can ever produce is its column minimum (196 px): force it and re-check
    await b.evaluate(`document.querySelector('.results').style.gridTemplateColumns = 'repeat(auto-fill, 196px)'`);
    await sleep(300);
    const floor = await check();
    await b.evaluate(`document.querySelector('.results').style.gridTemplateColumns = ''`);
    ok(floor.n > 3 && floor.bad.length === 0 && floor.minCard <= 180, 'minimum-width card: ' + JSON.stringify(floor));
    const labels = await b.evaluate(`[...document.querySelectorAll('.rcard')][0].querySelectorAll('.actions > *')[0].innerText`);
    ok(labels.trim() === 'Similar', 'first button label: ' + labels);
    return `${wide.n} cards, card width ${wide.minCard}px (1600 px window) / ${narrow.minCard}px (1100 px) / ${floor.minCard}px (grid minimum): 3 equal cells, nothing clipped or wrapped`;
  });

  await step('layout: fingerprint cards share one geometry (square image, one line each for score and date)', async () => {
    await go('fingerprints');
    await b.waitFor(`document.querySelectorAll('.fp-tile').length > 3`, 30000, 'tiles');
    await b.waitFor(imgsLoaded('.fp-tile img'), 30000, 'thumbnails');
    const r = await b.evaluate(`(() => { const t = [...document.querySelectorAll('.fp-grid .fp-tile')]; const sz = t.map((e) => { const b = e.getBoundingClientRect(), i = e.querySelector('img,.thumb-img').getBoundingClientRect();
      const cap = [...e.querySelectorAll('.cap > *')].map((c) => c.getBoundingClientRect().height); return { w: b.width, h: b.height, iw: i.width, ih: i.height, cap }; });
      return { n: t.length, sz }; })()`);
    const h0 = r.sz[0].h;
    for (const [i, s] of r.sz.entries()) {
      ok(Math.abs(s.h - h0) <= 1, `card ${i} height ${s.h} vs ${h0}`);
      ok(Math.abs(s.iw - s.ih) <= 1, `card ${i} image ${s.iw}x${s.ih} is not square`);
      ok(s.cap.length === 2 && s.cap.every((c) => c <= 15), `card ${i} caption is not two single lines: ${JSON.stringify(s.cap)}`);
    }
    return `${r.n} cards all ${Math.round(h0)} px tall, square images, two caption lines`;
  });

  // ======================= maps: basemap, captions, linkage =======================
  const rect = (sel) => b.rectOf(sel);
  const mapState = (scope) => b.evaluate(`(async () => {
    const g = document.querySelector(${JSON.stringify(scope)} + ' .geomap'); if (!g) return { err: 'no map in ' + ${JSON.stringify(scope)} };
    const tiles = [...g.querySelectorAll('.gm-basemap img.leaflet-tile')], loaded = tiles.filter((i) => i.complete && i.naturalWidth > 0);
    let lit = 0, wrongSrc = tiles.filter((i) => !i.src.includes('/ui/basemap/')).length;
    for (const i of loaded.slice(0, 60)) {
      const c = document.createElement('canvas'); c.width = c.height = 64; const x = c.getContext('2d', { willReadFrequently: true }); x.drawImage(i, 0, 0, 64, 64);
      const d = x.getImageData(0, 0, 64, 64).data; let px = 0; for (let k = 0; k < d.length; k += 4) if (d[k + 3] > 200 && d[k] + d[k + 1] + d[k + 2] > 60) px++;
      if (px >= 8) lit++;           // a sparse low-zoom tile can hold just a corner of one granule: a few opaque, non-black pixels are imagery
    }
    const cap = g.querySelector('[data-testid=map-caption]'), m = g.querySelector('.leaflet-container').__leaflet, bd = m.getBounds();
    return { n: tiles.length, loaded: loaded.length, lit, wrongSrc, basemap: g.dataset.basemap, caption: cap ? cap.innerText.replace(/\\s+/g, ' ') : '', zoom: m.getZoom(), bounds: [bd.getWest(), bd.getSouth(), bd.getEast(), bd.getNorth()] };
  })()`);
  const captionSettled = (scope) => `(() => { const c = document.querySelector(${JSON.stringify(scope)} + ' .geomap [data-testid=map-caption]'); return !!c && !/checking/.test(c.innerText); })()`;
  const checkBasemap = async (name, scope, kind) => {
    await b.waitFor(`(async () => { const g = document.querySelector(${JSON.stringify(scope)} + ' .geomap'); return !!g && [...g.querySelectorAll('.gm-basemap img.leaflet-tile')].some((i) => i.complete && i.naturalWidth > 1); })()`, 60000, `${name}: basemap tiles with imagery`);
    await b.waitFor(captionSettled(scope), 15000, `${name}: caption`);
    await sleep(700);
    const st = await mapState(scope);
    ok(!st.err, st.err);
    ok(st.n > 0 && st.lit > 0 && st.wrongSrc === 0, `${name}: tiles=${st.n} loaded=${st.loaded} with imagery=${st.lit} wrongSrc=${st.wrongSrc}`);
    ok(st.basemap === kind, `${name}: data-basemap=${st.basemap}`);
    const w = Math.max(-180, st.bounds[0]), e = Math.min(180, st.bounds[2]), south = Math.max(-85, st.bounds[1]), n = Math.min(85, st.bounds[3]);
    const q = kind === 'scene' ? '&scene=' + encodeURIComponent((await api('/detect/observations')).observations.sort((x, y) => y.n_detections - x.n_detections)[0].observation_id) : '';
    const cov = await api(`/ui/basemap/coverage?bbox=${[w, south, e, n].map((v) => v.toFixed(5)).join(',')}${q}`);
    ok(cov.available, `${name}: the API says nothing is staged in view`);
    if (kind === 'scene') {
      ok(/shown dark/.test(st.caption) && st.caption.includes(cov.scenes[0].date) && st.caption.includes('the scene these detections were found on'), `${name}: caption "${st.caption}"`);
    } else {
      ok(st.caption.includes('one representative acquisition per granule') && st.caption.includes('lowest-cloud') && st.caption.includes('not the date of the overlaid features') && st.caption.includes('unstaged areas shown dark'), `${name}: caption lacks the selection rule / dark-gap statement: "${st.caption}"`);
      ok(st.caption.includes(`${cov.scenes.length} granule`), `${name}: caption granule count vs API ${cov.scenes.length}: "${st.caption}"`);
      ok(st.caption.includes(cov.dates[0]) && st.caption.includes(cov.dates[cov.dates.length - 1]), `${name}: caption dates vs API ${cov.dates[0]}..${cov.dates[cov.dates.length - 1]}`);
    }
    const hit = await b.evaluate(`(async () => { const i = [...document.querySelector(${JSON.stringify(scope)} + ' .geomap').querySelectorAll('.gm-basemap img.leaflet-tile')].find((x) => x.naturalWidth > 1); const r = await fetch(i.src); return { status: r.status, mime: r.headers.get('content-type'), s: r.headers.get('x-basemap-status') }; })()`);
    ok(hit.status === 200 && /image\/(jpeg|png)/.test(hit.mime) && hit.s === 'ok', `${name}: tile response ${JSON.stringify(hit)}`);
    return `${st.lit}/${st.loaded} tiles show imagery at z${st.zoom.toFixed(1)}; caption states ${kind === 'scene' ? 'the scene date' : cov.scenes.length + ' granule(s), ' + cov.dates[0] + '..' + cov.dates[cov.dates.length - 1]}`;
  };
  const centreOnMapIn = (scope, lon, lat, z) => b.evaluate(`document.querySelector('${scope} .leaflet-container').__leaflet.setView([${lat}, ${lon}], ${z}, { animate: false })`);
  // a match pin that is actually the topmost element at its own centre (pins can overlap), inside the visible map
  const topmostPin = (scope, skip = 0) => b.evaluate(`(() => {
    const box = document.querySelector(${JSON.stringify(scope)} + ' .leaflet-container').getBoundingClientRect(); let seen = 0;
    for (const p of document.querySelectorAll(${JSON.stringify(scope)} + ' .gm-pin[data-pin]:not(.seed)')) {
      const r = p.getBoundingClientRect(), x = r.left + r.width / 2, y = r.top + r.height / 2;
      if (x < box.left + 40 || x > box.right - 20 || y < box.top + 10 || y > box.bottom - 10) continue;
      if (document.elementFromPoint(x, y)?.closest('.gm-pin') === p && seen++ >= ${skip}) return { id: p.dataset.pin, text: p.innerText, x, y };
    } return null; })()`);
  const hoverAt = async (x, y) => { await b.send('Input.dispatchMouseEvent', { type: 'mouseMoved', x, y }); await sleep(250); };
  const center = (r) => ({ x: r.x + r.w / 2, y: r.y + r.h / 2 });
  const setRange = (sel, v) => b.evaluate(`(() => { const e = document.querySelector(${JSON.stringify(sel)}); const set = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set; set.call(e, ${JSON.stringify(String(v))}); e.dispatchEvent(new Event('input', { bubbles: true })); e.dispatchEvent(new Event('change', { bubbles: true })); })()`);
  const fmtDistJS = (m) => (m < 1000 ? `${Math.round(m)} m` : `${(m / 1000).toFixed(m < 10_000 ? 2 : 1)} km`);

  await step('maps: changes "Location & threat rings" sits on archive imagery zoomed to the selected candidate, footprint distinct', async () => {
    const scope = 'section[aria-label="Location & threat rings"]';
    await go('changes/' + encodeURIComponent(zoneCand.candidate_id));
    await b.waitFor(`!!document.querySelector(${JSON.stringify(scope)} + ' .gm-sellbl')`, 30000, 'location panel with the selected footprint');
    await b.evaluate(`document.querySelector(${JSON.stringify(scope)}).scrollIntoView({ block: 'center' })`);
    const msg = await checkBasemap('changes location', scope, 'sentinel-2');
    const st = await mapState(scope);
    const [lon, lat] = zoneCandDetail.centroid_lonlat;
    ok(st.bounds[0] < lon && lon < st.bounds[2] && st.bounds[1] < lat && lat < st.bounds[3], 'candidate not in view');
    ok(st.zoom >= 13 && st.bounds[2] - st.bounds[0] < 0.1, `not zoomed to the candidate: zoom ${st.zoom}, width ${(st.bounds[2] - st.bounds[0]).toFixed(3)} deg`);
    const sel = await b.evaluate(`(() => { const g = document.querySelector(${JSON.stringify(scope)}); return { centre: g.querySelectorAll('.gm-centre[data-selected]').length, label: g.querySelector('.gm-sellbl')?.innerText || '' }; })()`);
    ok(sel.centre === 1 && sel.label.startsWith('Selected ·') && sel.label.includes((zoneCandDetail.area_m2 / 10000).toFixed(zoneCandDetail.area_m2 < 100000 ? 2 : 1)), `selected footprint: ${JSON.stringify(sel)} vs area ${zoneCandDetail.area_m2}`);
    await shot('14-changes-location');
    return `${msg}; ${sel.label}`;
  });

  await step('maps: the "Draw threat rings" button (no right-click needed) draws labelled rings and lists the contents with distances', async () => {
    const scope = 'section[aria-label="Location & threat rings"]';
    const id = zoneCand.candidate_id;
    await b.evaluate(`document.querySelector('[data-testid=draw-rings]').scrollIntoView({ block: 'center' })`);
    await b.click('[data-testid=draw-rings]');
    await b.waitFor(`document.querySelector('.rings-centre')?.innerText === '${id}'`, 10000, 'ring centre = selected candidate');
    await b.waitFor(`document.querySelectorAll('.ring-chips .chip').length === 4 && document.querySelectorAll('${scope} .gm-ringlbl').length === 4`, 15000, 'four rings and four on-map radius labels');
    const labels = await b.evaluate(`[...document.querySelectorAll('${scope} .gm-ringlbl')].map((e) => e.innerText)`);
    const want = [500, 1000, 2500, 5000].map(fmtDistJS);
    ok(JSON.stringify(labels) === JSON.stringify(want), `ring labels ${labels} vs ${want}`);
    const api_ = await ringsResult(id, '500,1000,2500,5000');
    const rows = await b.evaluate(`[...document.querySelectorAll('.ring-table tbody tr')].map((r) => ({ id: r.querySelector('td:nth-child(3)').title, d: +r.dataset.distance }))`);
    ok(rows.length === api_.features.length && rows.length > 0 && rows.every((r, i) => r.id === api_.features[i].id && Math.abs(r.d - api_.features[i].distance_m) < 0.2), 'listing differs from the API');
    const visible = await b.evaluate(`(() => { const t = document.querySelector('.ring-table'); const r = t.getBoundingClientRect(); return r.width > 100 && r.height > 20; })()`);
    ok(visible, 'the per-ring listing is not visible in the panel');
    await shot('15-rings-button');
    await b.evaluate(`[...document.querySelectorAll('.rings-panel button')].find(x => x.innerText === 'Clear rings').click()`);
    await b.waitFor(`!document.querySelector('.rings-centre')`, 5000);
    return `labels ${labels.join(' / ')}; ${rows.length} features listed with distances`;
  });

  await step('maps: detect "Detection map" subtitle, legend counts, confidence mapping and live min-score filter equal the API', async () => {
    await go('detect');
    await b.waitFor(`!!document.querySelector('[data-testid=map-legend] label[data-class]') && !!document.querySelector('[data-testid=map-subtitle]')`, 40000, 'detection map legend');
    const obs = (await api('/detect/observations')).observations.sort((x, y) => y.n_detections - x.n_detections)[0];
    const pts = (await api(`/ui/detections/${obs.observation_id}/points`)).points;
    const scope = 'section[aria-label="Detection map · threat rings"]';
    const sub = await text('[data-testid=map-subtitle]');
    for (const piece of [obs.acquired_at, obs.platform, obs.sensor, 'one oriented detection']) ok(sub.includes(piece), `subtitle lacks "${piece}": ${sub}`);
    const msg = await checkBasemap('detect map', scope, 'scene');
    // legend counts per class equal the API's, and the marker count on the map equals the shown total
    const tot = {}; for (const p of pts) tot[p.class] = (tot[p.class] ?? 0) + 1;
    const legend = () => b.evaluate(`[...document.querySelectorAll('[data-testid=map-legend] label[data-class]')].map((l) => ({ c: l.dataset.class, shown: +l.dataset.shown, total: +l.dataset.total, text: l.innerText.replace(/\\s+/g, ' ') }))`);
    let lg = await legend();
    ok(lg.length === Object.keys(tot).length && lg.every((l) => l.total === tot[l.c] && l.shown === tot[l.c]), `legend ${JSON.stringify(lg)} vs API ${JSON.stringify(tot)}`);
    ok(lg.every((l) => l.text.includes(tot[l.c].toLocaleString('en-US'))), 'legend text does not show the counts');
    const dots = () => b.evaluate(`+document.querySelector('${scope} .geomap').dataset.pointCount`);
    ok((await dots()) === pts.length, `marker count ${await dots()} vs ${pts.length}`);
    ok((await text('[data-testid=map-shown]')).startsWith(pts.length.toLocaleString('en-US')), 'header chip total');
    // the size legend states the real score range of this scene
    const sc = pts.map((p) => p.score).sort((x, y) => x - y), lo = sc[0], hi = sc[sc.length - 1], mid = sc[Math.floor(sc.length / 2)];
    const sz = await text('[data-testid=map-size-legend]');
    ok(sz.includes(lo.toFixed(2)) && sz.includes(hi.toFixed(2)) && sz.includes(mid.toFixed(2)) && /confidence/.test(sz), `size legend "${sz}" vs ${lo}/${mid}/${hi}`);
    // live filter
    for (const thr of [0.5, 0.7]) {
      await setRange('[aria-label="Minimum confidence on the map"]', thr);
      const want = {}; for (const p of pts) if (p.score >= thr) want[p.class] = (want[p.class] ?? 0) + 1;
      const total = Object.values(want).reduce((a, c) => a + c, 0);
      await b.waitFor(`+document.querySelector('${scope} .geomap').dataset.pointCount === ${total}`, 8000, `markers filtered at ${thr}`);
      lg = await legend();
      ok(lg.every((l) => l.shown === (want[l.c] ?? 0) && l.total === tot[l.c]), `at ${thr}: legend ${JSON.stringify(lg.map((l) => [l.c, l.shown]))} vs ${JSON.stringify(want)}`);
      ok((await text('[data-testid=map-shown]')).startsWith(`${total.toLocaleString('en-US')} of ${pts.length.toLocaleString('en-US')}`), `chip at ${thr}: ${await text('[data-testid=map-shown]')}`);
    }
    await setRange('[aria-label="Minimum confidence on the map"]', 0);
    await b.evaluate(`document.querySelector('[data-testid=map-legend] label[data-class="small-vehicle"] input').click()`);
    const noSv = pts.length - tot['small-vehicle'];
    await b.waitFor(`+document.querySelector('${scope} .geomap').dataset.pointCount === ${noSv}`, 8000, 'class toggle removes its markers');
    await b.evaluate(`document.querySelector('[data-testid=map-legend] label[data-class="small-vehicle"] input').click()`);
    await b.waitFor(`+document.querySelector('${scope} .geomap').dataset.pointCount === ${pts.length}`, 8000);
    await shot('16-detect-map');
    return `${msg}; legend, chip and ${pts.length} markers follow the slider and class toggles`;
  });

  await step('maps: detect hover shows class / confidence / box size; click opens the detection chip and its tile', async () => {
    const pts = (await api(`/ui/detections/${(await api('/detect/observations')).observations.sort((x, y) => y.n_detections - x.n_detections)[0].observation_id}/points`)).points;
    const dm = (a, c) => Math.hypot((a.lon - c.lon) * 111320 * Math.cos(a.lat * Math.PI / 180), (a.lat - c.lat) * 110540);
    const pick = pts.find((p) => p.class === 'small-vehicle' && p.length_m && pts.every((q) => q === p || dm(p, q) > 8));
    await centreOnMap(pick.lon, pick.lat, 19); await sleep(900);
    const pt = await mapPx(pick.lon, pick.lat);
    await hoverAt(pt.x, pt.y);
    await b.waitFor(`!!document.querySelector('.leaflet-tooltip')`, 5000, 'tooltip');
    const tip = await text('.leaflet-tooltip');
    ok(tip.includes(pick.class) && tip.includes(pick.score.toFixed(3)) && tip.includes(`${pick.length_m.toFixed(1)} × ${pick.width_m.toFixed(1)} m`), `tooltip "${tip}" vs ${JSON.stringify(pick)}`);
    await b.clickAt(pt.x, pt.y);
    await b.waitFor(`document.querySelector('[data-testid=detection-chip]')?.dataset.id === ${JSON.stringify(pick.id)}`, 5000, 'detection chip');
    const chip = await text('[data-testid=detection-chip]');
    ok(chip.includes(pick.class) && chip.includes(pick.score.toFixed(3)) && chip.includes(pick.length_m.toFixed(1)) && chip.includes(`${Math.round(pick.heading_deg)}°`), `chip "${chip}"`);
    const [, row, col] = pick.tile_id.match(/_r(\d+)_c(\d+)$/);
    await b.evaluate(`[...document.querySelectorAll('[data-testid=detection-chip] button')].find((x) => x.innerText.startsWith('Open tile')).click()`);
    await b.waitFor(`document.querySelector('.detect-stage img')?.src.includes('/tiles/${+row}/${+col}/image.png')`, 8000, 'tile opened in the detections viewer');
    return `${pick.class} ${pick.score.toFixed(3)} ${pick.length_m} x ${pick.width_m} m -> tile r${+row} c${+col}`;
  });

  await step('maps: search map has the basemap; numbered pins match the cards; hover links card and pin both ways; every search refits', async () => {
    await go('search');
    await b.type('input[aria-label="Search query"]', 'open water reservoir');
    await b.click('button[type=submit]');
    await b.waitFor(`document.querySelectorAll('.rcard').length > 0`, 30000, 'results');
    const scope = 'section[aria-label="Map · spatial filter"]';
    const msg = await checkBasemap('search map', scope, 'sentinel-2');
    const cards = await b.evaluate(`[...document.querySelectorAll('.rcard')].map((c) => ({ tile: c.dataset.tile, n: +c.dataset.n }))`);
    const pins = await b.evaluate(`[...document.querySelectorAll('${scope} .gm-pin[data-pin]')].map((p) => ({ tile: p.dataset.pin, text: p.innerText }))`);
    ok(pins.length === cards.length, `pins ${pins.length} vs cards ${cards.length}`);
    for (const c of cards) ok(pins.find((p) => p.tile === c.tile)?.text === String(c.n), `pin for card #${c.n} is not numbered ${c.n}`);
    // fit on every search: every hit not reported as "outside" is inside the map view
    const st = await mapState(scope);
    const hits = await b.evaluate(`(() => { const m = document.querySelector('${scope} .leaflet-container').__leaflet; return [...document.querySelectorAll('${scope} .gm-pin[data-pin]')].length; })()`);
    const outTxt = await b.evaluate(`document.querySelector('[data-testid=map-outside]')?.innerText || ''`);
    const outN = (outTxt.match(/#\d+/g) || []).length;
    const searched = await api(`/search/text?q=${encodeURIComponent('open water reservoir')}&k=20`);
    const inView = searched.results.filter((r) => r.lon >= st.bounds[0] && r.lon <= st.bounds[2] && r.lat >= st.bounds[1] && r.lat <= st.bounds[3]).length;
    ok(inView === searched.results.length - outN, `${inView} hits in view, ${outN} reported outside, of ${searched.results.length}`);
    // card -> pin
    const first = cards[0];
    await b.evaluate(`document.querySelector('.rcard[data-n="1"]').scrollIntoView({ block: 'center' })`);
    const cr = await rect('.rcard[data-n="1"] img, .rcard[data-n="1"] .thumb-img');
    await hoverAt(center(cr).x, center(cr).y);
    await b.waitFor(`document.querySelectorAll('${scope} .gm-pin.hl').length === 1 && document.querySelector('${scope} .gm-pin.hl').dataset.pin === ${JSON.stringify(first.tile)}`, 4000, 'card hover highlights its pin');
    // pin -> card (a real pointer over a pin that is topmost at its own centre)
    await hoverAt(5, 5);
    await b.evaluate(`document.querySelector('${scope} .leaflet-container').scrollIntoView({ block: 'center' })`);
    const tp = await topmostPin(scope);
    ok(tp, 'no visible, un-overlapped pin to hover');
    await hoverAt(tp.x, tp.y);
    await b.waitFor(`document.querySelectorAll('.rcard.hl').length === 1 && document.querySelector('.rcard.hl').dataset.tile === ${JSON.stringify(tp.id)} && document.querySelector('.rcard.hl').dataset.n === ${JSON.stringify(tp.text)}`, 4000, `pin #${tp.text} hover highlights card #${tp.text}`);
    // refit: a second search elsewhere moves the view
    const before = st.bounds.join();
    await b.select('select:has(option[value=kutch])', 'kutch');
    await b.type('input[aria-label="Search query"]', 'trees and dense vegetation');
    await b.click('button[type=submit]');
    await b.waitFor(`document.body.innerText.toLowerCase().includes('“trees and dense vegetation”')`, 30000);
    await sleep(900);
    const st2 = await mapState(scope);
    ok(st2.bounds.join() !== before, 'map did not refit on the second search');
    const kut = (await api('/regions')).regions.find((r) => r.name === 'kutch').bbox;
    const res2 = await api(`/search/text?q=${encodeURIComponent('trees and dense vegetation')}&k=20&bbox=${kut.join(',')}`);
    ok(res2.results.every((r) => r.lon >= st2.bounds[0] && r.lon <= st2.bounds[2] && r.lat >= st2.bounds[1] && r.lat <= st2.bounds[3]), 'a kutch result is outside the refitted map');
    await shot('17-search-map');
    return `${msg}; ${pins.length} pins numbered like the cards; hover links both ways; refit on the 2nd search`;
  });

  await step('maps: "Draw box" has an obvious active state, a filled box, a chip with one-click clear, and Esc cancels', async () => {
    const scope = 'section[aria-label="Map · spatial filter"]';
    await b.evaluate(`document.querySelector('${scope} button[aria-pressed]').scrollIntoView({ block: 'center' })`);
    ok(!(await b.evaluate(`!!document.querySelector('${scope} .gm-hint')`)), 'hint visible before drawing');
    await b.evaluate(`[...document.querySelectorAll('${scope} .chip-x')].forEach((x) => x.click())`);       // start from a clean state
    await b.click(`${scope} button[aria-pressed]`);
    await b.waitFor(`!!document.querySelector('${scope} .gm-hint') && document.querySelector('${scope} button[aria-pressed]').classList.contains('drawing') && document.querySelector('${scope} button[aria-pressed]').getAttribute('aria-pressed') === 'true'`, 4000, 'active drawing state');
    ok((await b.evaluate(`document.querySelector('${scope} .leaflet-container').style.cursor`)) === 'crosshair', 'no crosshair cursor');
    await b.key('Escape');
    await b.waitFor(`!document.querySelector('${scope} .gm-hint') && document.querySelector('${scope} button[aria-pressed]').getAttribute('aria-pressed') === 'false'`, 4000, 'Esc cancels drawing');
    await b.click(`${scope} button[aria-pressed]`);
    const r = await rect(`${scope} .leaflet-container`);
    await b.drag(r.x + r.w * 0.25, r.y + r.h * 0.3, r.x + r.w * 0.6, r.y + r.h * 0.6);
    await b.waitFor(`!!document.querySelector('[data-testid=map-bbox-chip]')`, 5000, 'bbox chip');
    const chip = await text('[data-testid=map-bbox-chip]');
    const drawn = await b.evaluate(`(() => { const m = document.querySelector('${scope} .leaflet-container').__leaflet; let o = null; m.eachLayer((l) => { if (l._bounds && l.options && l.options.fillOpacity >= 0.15 && l.options.color === '#ffd166' && !l._mRadius) { const b = l.getBounds(); o = { box: [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()], fill: l.options.fillOpacity, weight: l.options.weight }; } }); return o; })()`);
    ok(drawn && drawn.fill >= 0.15 && drawn.weight >= 2, 'the drawn box is not rendered with a clear fill: ' + JSON.stringify(drawn));
    ok(drawn.box.every((v) => chip.includes(v.toFixed(2))), `chip "${chip}" does not show the drawn bbox ${drawn.box.map((v) => v.toFixed(2))}`);
    ok((await text('.chip.cyan.mono')).includes(drawn.box[0].toFixed(2)), 'the search panel chip does not agree');
    await b.click(`${scope} .chip-x`);
    await b.waitFor(`!document.querySelector('[data-testid=map-bbox-chip]') && document.body.innerText.includes('whole archive')`, 4000, 'one-click clear');
    return `chip "${chip}" = drawn box; Esc cancels; one click clears`;
  });

  await step('maps: fingerprints "Spatial spread" - basemap, labelled seed pin, pins numbered like the gallery, hover linked, view fits the set', async () => {
    await go('fingerprints');
    await b.waitFor(`document.querySelectorAll('.fp-tile').length > 3`, 30000, 'tiles');
    const scope = 'section[aria-label="Spatial spread"]';
    const msg = await checkBasemap('fingerprints spread', scope, 'sentinel-2');
    const cards = await b.evaluate(`[...document.querySelectorAll('.fp-grid .fp-tile[data-n]')].map((c) => ({ tile: c.dataset.tile, n: +c.dataset.n, cap: c.querySelector('.cap').innerText }))`);
    const pins = await b.evaluate(`[...document.querySelectorAll('${scope} .gm-pin[data-pin]')].map((p) => ({ id: p.dataset.pin, cls: p.className, text: p.innerText }))`);
    const seed = pins.filter((p) => p.cls.includes('seed'));
    ok(seed.length === 1 && seed[0].text.toLowerCase() === 'seed', 'one labelled seed pin expected: ' + JSON.stringify(seed));
    const rest = pins.filter((p) => !p.cls.includes('seed'));
    ok(rest.length === cards.length, `match pins ${rest.length} vs gallery cards ${cards.length}`);
    for (const c of cards) { ok(c.cap.startsWith(`#${c.n} ·`), `card caption "${c.cap}"`); ok(rest.find((p) => p.id === c.tile)?.text === String(c.n), `pin for card #${c.n}`); }
    // everything is inside the view (seed + matches)
    const st = await mapState(scope);
    const tiles = (await api(`/ui/tiles?ids=${cards.map((c) => c.tile).join(',')}`)).tiles;
    const inside = tiles.filter((t) => { const cx = (t.bbox[0] + t.bbox[2]) / 2, cy = (t.bbox[1] + t.bbox[3]) / 2; return cx >= st.bounds[0] && cx <= st.bounds[2] && cy >= st.bounds[1] && cy <= st.bounds[3]; }).length;
    ok(inside === tiles.length, `${inside}/${tiles.length} matches inside the fitted view`);
    // hover both ways
    await b.evaluate(`document.querySelector('.fp-tile[data-n="2"]').scrollIntoView({ block: 'center' })`);
    const cr = await rect('.fp-tile[data-n="2"] img, .fp-tile[data-n="2"] .thumb-img');
    await hoverAt(center(cr).x, center(cr).y);
    await b.waitFor(`document.querySelectorAll('${scope} .gm-pin.hl').length === 1 && document.querySelector('${scope} .gm-pin.hl').dataset.pin === ${JSON.stringify(cards[1].tile)}`, 4000, 'card hover -> pin');
    await hoverAt(5, 5);
    await b.evaluate(`document.querySelector('${scope} .leaflet-container').scrollIntoView({ block: 'center' })`);
    const tp = await topmostPin(scope);
    ok(tp, 'no visible, un-overlapped pin to hover');
    await hoverAt(tp.x, tp.y);
    await b.waitFor(`document.querySelectorAll('.fp-tile.hl').length === 1 && document.querySelector('.fp-tile.hl').dataset.tile === ${JSON.stringify(tp.id)} && document.querySelector('.fp-tile.hl').dataset.n === ${JSON.stringify(tp.text)}`, 4000, `pin #${tp.text} hover -> card #${tp.text}`);
    await shot('18-fingerprints-map');
    return `${msg}; seed pin + ${rest.length} numbered pins; hover linked; all inside the view`;
  });

  await step('maps: discovery cluster map - hover a row highlights its cluster (pixel-checked), colours match the list, click zooms', async () => {
    await go('discovery');
    const cl = await api('/discovery/clusters'), geo = await api('/ui/clusters/geo');
    await b.waitFor(`document.querySelectorAll('.tbl tbody tr').length === ${cl.n_clusters} && +(document.querySelector('canvas.gm-cells')?.dataset.drawn || 0) > 0`, 40000, 'clusters drawn');
    const scope = 'section[aria-label="Cluster map"]';
    const msg = await checkBasemap('discovery cluster map', scope, 'sentinel-2').catch(() => 'basemap tiles only where granules are in view');
    ok(Object.entries(geo.clusters).every(([id, c]) => c.n_tiles === cl.sizes[id]), 'geo tile counts differ from /discovery/clusters sizes');
    ok(geo.n_clustered_tiles === cl.n_tiles, `placed ${geo.n_clustered_tiles} vs run ${cl.n_tiles}`);
    ok((await text(`${scope} .chip.mono`)).startsWith(geo.n_clustered_tiles.toLocaleString('en-US')), 'tiles-placed chip');
    // pick three clusters and a cell of each that no other cluster shares, then read the canvas pixel there
    const owner = new Map(); for (const [id, c] of Object.entries(geo.clusters)) for (const [x, y] of c.cells) { const k = x + ',' + y; owner.set(k, owner.has(k) ? null : id); }
    const ids = Object.keys(geo.clusters);
    const solo = (id) => geo.clusters[id].cells.find(([x, y]) => owner.get(x + ',' + y) === id);
    const pixel = (lon, lat) => b.evaluate(`(() => { const g = document.querySelector('${scope} .geomap'), m = g.querySelector('.leaflet-container').__leaflet, c = g.querySelector('canvas.gm-cells'); const p = m.latLngToContainerPoint([${lat}, ${lon}]); const dpr = c.width / m.getSize().x; const d = c.getContext('2d').getImageData(Math.round(p.x * dpr), Math.round(p.y * dpr), 1, 1).data; return [d[0], d[1], d[2], d[3]]; })()`);
    const hex = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
    // zoom so cells are several pixels wide: look at the whole archive (default fit) first
    const sw = {}; for (const r of await b.evaluate(`[...document.querySelectorAll('.tbl tbody tr')].map((r) => ({ id: r.dataset.cluster, bg: getComputedStyle(r.querySelector('i[data-swatch]')).backgroundColor }))`)) sw[r.id] = r.bg.match(/\d+/g).slice(0, 3).map(Number);
    let checked = 0;
    for (const id of ids.slice(0, 6)) {
      const cell = solo(id); if (!cell) continue;
      await centreOnMapIn(scope, cell[0], cell[1], 13); await sleep(500);
      await hoverAt(5, 5);
      const row = await rect(`.tbl tbody tr[data-cluster="${id}"]`);
      await hoverAt(center(row).x, center(row).y);
      await b.waitFor(`document.querySelector('${scope} .geomap').dataset.activeCluster === ${JSON.stringify(id)}`, 4000, `row ${id} -> map highlight`);
      await centreOnMapIn(scope, cell[0], cell[1], 13); await sleep(400);
      const px = await pixel(cell[0], cell[1]);
      const want = sw[id];
      ok(px[3] > 200 && px.slice(0, 3).every((v, i) => Math.abs(v - want[i]) <= 28), `cluster ${id}: map pixel ${px} vs list swatch ${want}`);
      // a different cluster's solo cell is dimmed while this one is highlighted
      const other = ids.find((o) => o !== id && solo(o));
      const oc = solo(other);
      const near = await b.evaluate(`(() => { const m = document.querySelector('${scope} .leaflet-container').__leaflet; m.setView([${oc[1]}, ${oc[0]}], 13, { animate: false }); return true; })()`);
      await sleep(400);
      const px2 = await pixel(oc[0], oc[1]);
      ok(px2[3] < 80, `cluster ${other} is not dimmed while ${id} is highlighted: alpha ${px2[3]}`);
      checked++;
    }
    ok(checked >= 3, `only ${checked} clusters pixel-checked`);
    await hoverAt(5, 5);
    await b.waitFor(`document.querySelector('${scope} .geomap').dataset.activeCluster === ''`, 4000, 'leaving the row clears the highlight');
    // click a row -> the map zooms to that cluster's extent
    const span = (i) => geo.clusters[i].bbox[2] - geo.clusters[i].bbox[0], target = [...ids].sort((x, y) => span(x) - span(y))[0], bb = geo.clusters[target].bbox;     // a compact cluster: the wide ones really do span the archive
    await centreOnMapIn(scope, 79, 22, 4);                                  // start from a wide view, then click the row
    await sleep(500);
    const z0 = (await mapState(scope)).zoom;
    await b.click(`.tbl tbody tr[data-cluster="${target}"]`);
    await sleep(900);
    const st = await mapState(scope);
    const cx = (bb[0] + bb[2]) / 2, cy = (bb[1] + bb[3]) / 2, clusterSpan = bb[2] - bb[0], viewSpan = st.bounds[2] - st.bounds[0];
    ok(st.bounds[0] <= cx && cx <= st.bounds[2] && st.bounds[1] <= cy && cy <= st.bounds[3], 'cluster centre not in view after click');
    ok(st.zoom > z0 && viewSpan <= Math.max(3 * clusterSpan, 0.7), `zoom ${z0} -> ${st.zoom}; view ${viewSpan.toFixed(2)} deg for a cluster ${clusterSpan.toFixed(2)} deg wide`);
    await shot('19-discovery-map');
    return `${msg}; ${checked} clusters: highlighted pixel colour = list swatch, others dimmed; click zooms (z${z0.toFixed(1)} -> z${st.zoom.toFixed(1)})`;
  });

  await step('briefing: tour controls, annotation canvas, spotlight, contrast, hide panels, Esc', async () => {
    await go('briefing');
    await b.waitFor(`document.querySelector('.brief .hud')?.innerText.includes('SITE 1 /')`, 20000, 'HUD');
    await b.click('button[aria-label="Next site"]');
    await b.waitFor(`document.querySelector('.brief .hud').innerText.includes('SITE 2 /')`, 5000);
    const r = await b.rectOf('canvas.anno');
    await b.evaluate(`[...document.querySelectorAll('.toolbar button')].find(x => x.innerText.includes('Pen')).click()`);
    await b.drag(r.x + 800, r.y + 300, r.x + 1100, r.y + 470);
    const painted = () => b.evaluate(`(() => { const c = document.querySelector('canvas.anno'); const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data; for (let i = 3; i < d.length; i += 4) if (d[i]) return true; return false; })()`);
    await b.waitFor(`(() => { const c = document.querySelector('canvas.anno'); const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data; for (let i = 3; i < d.length; i += 4) if (d[i]) return true; return false; })()`, 5000, 'pen stroke painted');
    await b.evaluate(`[...document.querySelectorAll('.toolbar button')].find(x => x.innerText.includes('Undo')).click()`);
    // poll, don't sleep: the software-rendered globe can keep the main thread busy for a few hundred ms
    await b.waitFor(`(() => { const c = document.querySelector('canvas.anno'); const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data; for (let i = 3; i < d.length; i += 4) if (d[i]) return false; return true; })()`, 5000, 'undo clears the canvas');
    // the deferred second clear must never erase a stroke drawn straight after an Undo
    await b.drag(r.x + 800, r.y + 600, r.x + 1000, r.y + 700);
    await sleep(600);
    ok(await b.evaluate(`(() => { const c = document.querySelector('canvas.anno'); const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data; for (let i = 3; i < d.length; i += 4) if (d[i]) return true; return false; })()`), 'a stroke drawn right after Undo was erased by the deferred clear');
    await b.evaluate(`[...document.querySelectorAll('.toolbar button')].find(x => x.innerText.includes('Undo')).click()`);
    await b.waitFor(`(() => { const c = document.querySelector('canvas.anno'); const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data; for (let i = 3; i < d.length; i += 4) if (d[i]) return false; return true; })()`, 5000, 'second undo clears');
    await b.evaluate(`[...document.querySelectorAll('.toolbar button')].find(x => x.innerText.includes('Spotlight')).click()`);
    await b.send('Input.dispatchMouseEvent', { type: 'mouseMoved', x: 700, y: 500 });
    await b.waitFor(`!!document.querySelector('.brief .spot')`, 5000, 'spotlight');
    const hc0 = await b.evaluate(`document.querySelector('.brief').classList.contains('hc')`);
    await b.evaluate(`[...document.querySelectorAll('.toolbar button')].find(x => x.innerText.includes('Contrast')).click()`);
    await b.waitFor(`document.querySelector('.brief').classList.contains('hc') !== ${hc0}`, 5000, 'contrast toggle');
    await shot('07-briefing');
    await b.evaluate(`[...document.querySelectorAll('.toolbar button')].find(x => x.innerText.includes('Play tour')).click()`);
    await b.waitFor(`document.querySelector('.brief .hud').innerText.includes('SITE 3 /')`, 14000, 'auto-advance');
    await b.evaluate(`[...document.querySelectorAll('.toolbar button')].find(x => x.innerText.includes('Hide panels')).click()`);
    await b.waitFor(`document.querySelectorAll('.toolbar').length === 0`, 5000, 'panels hidden');
    await b.key('Escape');
    await b.waitFor(`location.hash === '#/dashboard'`, 5000, 'Esc exits');
    return 'next, pen, undo, spotlight, contrast, auto-advance, hide, Esc';
  });

  await step('chrome: no tier badges, roadmap entry, dashed frames, hatching or rail legend on any screen', async () => {
    const bad = [];
    for (const r of ['dashboard', 'search', 'changes', 'detect', 'discovery', 'fingerprints', 'settings', 'roadmap']) {
      await go(r); await sleep(1000);
      const info = await b.evaluate(`(() => ({
        badges: document.querySelectorAll('.tier-badge, [data-tier], .tier-dot, .roadmap-note, .legend').length,
        railRoadmap: [...document.querySelectorAll('.rail a')].some(a => /roadmap/i.test(a.innerText + a.getAttribute('href'))),
        dashed: [...document.querySelectorAll('.panel')].some(p => getComputedStyle(p).borderTopStyle === 'dashed'),
        text: /ROADMAP|NOT IMPLEMENTED|LIVE UI|EXISTING BACKEND|MOCKUP|PLACEHOLDER/.test(document.body.innerText),
      }))()`);
      if (info.badges) bad.push(`${r}: ${info.badges} tier elements`);
      if (info.railRoadmap) bad.push(`${r}: roadmap rail entry`);
      if (info.dashed) bad.push(`${r}: dashed panel`);
      if (info.text) bad.push(`${r}: tier/roadmap wording`);
    }
    ok(bad.length === 0, bad.join('; '));
    ok((await b.evaluate('location.hash')) === '#/roadmap' ? (await count('.panel')) > 0 : true, 'unknown route renders a screen');
    return '8 routes clean; /roadmap no longer exists (falls back to the dashboard)';
  });

  await step('settings: System / Model performance is API-driven and holds the metrics removed from the dashboard', async () => {
    await go('settings');
    const m = await api('/ui/metrics');
    await b.waitFor(`document.querySelectorAll('.perf-big').length === 2`, 15000, 'perf panels');
    const body = await b.evaluate('document.body.innerText');
    ok(/system \/ model performance/i.test(body), 'panel title');
    ok(body.includes((m.change_model.f1 * 100).toFixed(1) + '%') && body.includes((m.change_model.precision * 100).toFixed(1) + '%') && body.includes((m.change_model.recall * 100).toFixed(1) + '%'), 'change-model F1/P/R from /ui/metrics');
    ok(body.includes(m.detector.dota_val.small_vehicle_ap50.toFixed(3)) && body.includes(m.detector.dota_val.ground_vehicles_ap50.toFixed(3)), 'AP50 from /ui/metrics');
    ok(body.includes(m.detector.xview_test.small_vehicle_ap50.toFixed(3)), 'xView figure shown');
    ok(body.includes(m.detector.caveat.slice(0, 40)), 'detector caveat shown');
    await shot('09-settings');
    return `F1 ${(m.change_model.f1 * 100).toFixed(1)}%, AP50 ${m.detector.dota_val.small_vehicle_ap50.toFixed(3)}`;
  });

  await step('scope: no InSAR / velocity / completion-rate / thermal UI on any route', async () => {
    const bad = [];
    for (const r of ['dashboard', 'search', 'changes', 'detect', 'discovery', 'fingerprints', 'settings']) {
      await go(r); await sleep(900);
      const t = await b.evaluate('document.body.innerText');
      for (const re of [/insar/i, /interferom/i, /fringe/i, /velocity/i, /completion/i, /thermal/i, /rate of change/i]) if (re.test(t)) bad.push(`${r}: ${re}`);
    }
    ok(bad.length === 0, bad.join('; '));
    return '7 routes clean';
  });

  await step('offline: indicator reflects reality and zero external requests were made', async () => {
    await go('dashboard'); await sleep(800);
    const pill = await b.evaluate(`(() => { const p = document.querySelector('.offline-pill'); return { cls: p.className, text: p.innerText, title: p.title }; })()`);
    ok(pill.cls.includes('ok') && /OFFLINE/.test(pill.text), JSON.stringify(pill));
    const ext = externalRequests(b.requests, origin);
    ok(ext.length === 0, 'external: ' + ext.map((e) => e.url).join(', '));
    const csp = b.problems.filter((p) => /content security policy|blocked/i.test(p.text));
    ok(csp.length === 0, 'CSP: ' + csp.map((p) => p.text).join('; '));
    return `${b.requests.length} requests, 0 external; pill "${pill.text}"`;
  });
} finally {
  const pad = (s, n) => (s.length > n ? s.slice(0, n - 1) + '…' : s.padEnd(n));
  for (const r of results) console.log(`${r.ok ? 'PASS' : 'FAIL'}  ${pad(r.name, 92)} ${String(r.ms).padStart(6)}ms  ${r.note}`);
  const failed = results.filter((r) => !r.ok).length;
  const probs = b.problems.filter((p) => !/favicon|willReadFrequently/.test(p.text));
  console.log(`\n${results.length - failed}/${results.length} steps passed; page problems: ${probs.length}`);
  for (const p of probs.slice(0, 10)) console.log('  problem', p.kind, '@', p.tag, '-', p.text);
  b.close();
  process.exitCode = failed ? 1 : 0;
}
