// End-to-end interaction test for the React console (Tier 1 + Tier 2 + Tier 3 chrome), driven with REAL mouse/keyboard
// input in headless Chrome. Every number the UI shows is compared against the backend's own response.
//
// !! Confirm / reject / reopen append PERMANENT rows to the analyst audit table. Point this at a scratch backend
// !! (DATABASE_URL=sqlite:///<copy of tiles.sqlite>), never at the production catalog. The script refuses to run the
// !! decision step unless --allow-writes is passed.
//
//   node tools/e2e-tier1.mjs --base http://127.0.0.1:8001/react/ --allow-writes [--shots DIR]
import path from 'node:path';
import { externalRequests, launch, sleep } from './cdp.mjs';

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
const go = async (route) => { await b.evaluate(`location.hash = '#/${route}'`); await sleep(400); };
const imgsLoaded = (sel) => `(() => { const l = [...document.querySelectorAll(${JSON.stringify(sel)})]; return l.length > 0 && l.filter(i => i.complete && i.naturalWidth > 0).length >= Math.ceil(l.length * 0.7); })()`;
const shot = (n) => (SHOTS ? b.shot(path.join(SHOTS, `${n}.png`)) : Promise.resolve());
const text = (sel, nth = 0) => b.evaluate(`(document.querySelectorAll(${JSON.stringify(sel)})[${nth}]?.innerText || '')`);
const has = (s) => `document.body.innerText.toLowerCase().includes(${JSON.stringify(s.toLowerCase())})`;
const count = (sel) => b.evaluate(`document.querySelectorAll(${JSON.stringify(sel)}).length`);

try {
  await b.nav(BASE);
  await b.waitFor(`document.querySelectorAll('.stat').length >= 8 && !document.querySelector('.stats .skeleton')`, 30000, 'stat cards');

  await step('dashboard: stat cards equal the API (no hardcoded numbers)', async () => {
    const m = await api('/ui/metrics'), lat = await api('/ui/latency');
    const stats = await b.evaluate(`[...document.querySelectorAll('.stat')].map(s => ({ l: s.querySelector('.label').innerText, v: s.querySelector('.value').innerText }))`);
    const by = Object.fromEntries(stats.map((s) => [s.l.trim().toLowerCase(), s.v]));
    ok(by['tiles indexed'] === m.counters.tiles_indexed.toLocaleString('en-US'), `tiles ${by['tiles indexed']}`);
    ok(by['scenes'] === String(m.counters.scenes), 'scenes');
    ok(by['regions'] === String(m.counters.regions), 'regions');
    ok(by['sensors'] === String(m.counters.sensors), 'sensors');
    ok(by['change detection f1'] === (m.change_model.f1 * 100).toFixed(1) + '%', `F1 ${by['change detection f1']}`);
    ok(by['object detection ap50'].startsWith(m.detector.dota_val.small_vehicle_ap50.toFixed(3)), `AP50 ${by['object detection ap50']}`);
    const apSub = await b.evaluate(`[...document.querySelectorAll('.stat')].find(s => s.innerText.includes('OBJECT DETECTION')).querySelector('.sub').innerText`);
    ok(apSub.includes(m.detector.dota_val.ground_vehicles_ap50.toFixed(3)), 'ground-vehicle AP50 shown');
    ok(Math.abs(parseFloat(by['search latency']) - lat.median_ms) < Math.max(40, lat.median_ms), `latency ${by['search latency']} vs ${lat.median_ms}`);
    ok(!stats.some((s) => s.v.includes('—')), 'a card shows a placeholder');
    await shot('01-dashboard');
    return `F1 ${by['change detection f1']}, AP50 ${by['object detection ap50'].replace(/\s+/g, ' ')}, tiles ${by['tiles indexed']}, latency ${by['search latency']}`;
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
    await b.waitFor(imgsLoaded('.compare img'), 20000, 'compare images');
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
      await b.waitFor(`document.body.innerText.includes('confirmed')`);
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

  await step('discovery: clusters, map image and seed explorer', async () => {
    await go('discovery');
    const cl = await api('/discovery/clusters');
    await b.waitFor(`document.querySelectorAll('.tbl tbody tr').length === ${cl.n_clusters}`, 15000, 'cluster rows');
    await b.waitFor(imgsLoaded('.cluster-img'), 15000, 'cluster map');
    await b.waitFor(`document.querySelectorAll('.fp-tile').length > 0`, 15000, 'seed tiles');
    await shot('05-discovery');
    return `${cl.n_clusters} clusters, ${await count('.fp-tile')} seed neighbours`;
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

  await step('roadmap: all five mockups carry the persistent badge, note and dashed frame', async () => {
    for (const id of ['dossier', 'ingest', 'vector3d', 'threat', 'xai']) {
      await go(`roadmap/${id}`);
      await b.waitFor(`!!document.querySelector('.panel.t-roadmap')`, 8000, id);
      const info = await b.evaluate(`(() => { const p = document.querySelector('.panel.t-roadmap'); return { badge: p.querySelector('header .tier-badge.roadmap')?.innerText, note: p.querySelector('.roadmap-note')?.innerText, border: getComputedStyle(p).borderTopStyle }; })()`);
      ok(/ROADMAP — NOT IMPLEMENTED/.test(info.badge || ''), `${id}: badge`);
      ok((info.note || '').length > 20, `${id}: note`);
      ok(info.border === 'dashed', `${id}: border ${info.border}`);
    }
    await go('roadmap/xai');
    await b.waitFor(`/~[0-9]+ m patches/.test(document.querySelector('.roadmap-note')?.innerText || '')`, 15000, 'XAI patch size computed from the catalog');
    const xai = await text('.roadmap-note');
    ok(/ViT-L\/14/.test(xai) && /ViT-B\/32/.test(xai) && /~\d+ m patches/.test(xai), 'XAI note: ' + xai);
    await go('roadmap/xai'); await sleep(1200); await shot('08-roadmap-xai');
    return xai.replace(/\s+/g, ' ');
  });

  await step('scope: no InSAR / velocity / completion-rate / thermal UI on any route', async () => {
    const bad = [];
    for (const r of ['dashboard', 'search', 'changes', 'detect', 'discovery', 'fingerprints', 'roadmap/dossier', 'roadmap/ingest', 'roadmap/vector3d', 'roadmap/threat', 'roadmap/xai']) {
      await go(r); await sleep(900);
      const t = await b.evaluate('document.body.innerText');
      for (const re of [/insar/i, /interferom/i, /fringe/i, /velocity/i, /completion/i, /thermal/i, /rate of change/i]) if (re.test(t)) bad.push(`${r}: ${re}`);
    }
    ok(bad.length === 0, bad.join('; '));
    return '11 routes clean';
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
