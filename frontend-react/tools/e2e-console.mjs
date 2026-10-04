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
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

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
  // ---- opening sequence: it must be tested first, on a fresh profile, before the rest of the run switches it off ----
  const INTRO = `!!document.querySelector('[data-testid=intro]')`;
  const NO_INTRO = `!document.querySelector('[data-testid=intro]')`;
  const resetFlags = () => b.evaluate(`(() => { try { sessionStorage.clear(); localStorage.removeItem('geoseek.intro.off'); } catch (e) {} return true; })()`);
  await b.nav(BASE);
  await step('intro: plays on first load, skip control visible from the first frame, app already live underneath, ends itself in 2-4 s', async () => {
    await b.waitFor(INTRO, 30000, 'intro');
    const t0 = Date.now();
    const first = await b.evaluate(`(() => { const r = document.querySelector('[data-testid=intro-skip]').getBoundingClientRect(); const cs = getComputedStyle(document.querySelector('[data-testid=intro]')); return { skipVisible: r.width > 0 && r.top >= 0 && r.bottom <= innerHeight, never: !!document.querySelector('[data-testid=intro-never]'), opacity: Number(cs.opacity), app: !!document.querySelector('.app .rail'), word: document.querySelector('[data-testid=intro-word]').innerText.split('\\n').join(''), tag: document.querySelector('[data-testid=intro-tagline]').innerText }; })()`);
    ok(first.skipVisible && first.never, 'skip / never-again controls are not visible on the first observed frame');
    ok(first.app, 'the app is not mounted under the intro (it would be blocked, not overlaid)');
    ok(first.word === 'GEOSEEK' && /offline/i.test(first.tag) && !/\n/.test(first.tag), `word "${first.word}" tag "${first.tag}"`);
    await b.waitFor(NO_INTRO, 9000, 'intro to end by itself');
    const ms = Date.now() - t0;
    ok(ms >= 1800 && ms <= 4300, `intro ran ${ms} ms`);
    return `ran ${ms} ms (spec 2-4 s), wordmark ${first.word}, tagline one line`;
  });
  await step('intro: not repeated on reload in the same session', async () => {
    await b.nav(BASE); await b.waitFor(`!!document.querySelector('.app .rail')`, 30000, 'app'); await sleep(900);
    ok(await b.evaluate(NO_INTRO), 'the intro came back within the same session');
    return 'session flag respected';
  });
  await step('intro: a click skips it', async () => {
    await resetFlags(); await b.nav(BASE); await b.waitFor(INTRO, 30000, 'intro');
    await b.clickAt(150, 150);
    await b.waitFor(NO_INTRO, 1500, 'dissolve after a click');
    return 'clicked away';
  });
  await step('intro: any key skips it', async () => {
    await resetFlags(); await b.nav(BASE); await b.waitFor(INTRO, 30000, 'intro');
    await b.key('x');
    await b.waitFor(NO_INTRO, 1500, 'dissolve after a key');
    return 'key skip';
  });
  await step('intro: "don\'t show again" is saved to localStorage and honoured; the checkbox itself does not skip', async () => {
    await resetFlags(); await b.nav(BASE); await b.waitFor(INTRO, 30000, 'intro');
    await b.click('[data-testid=intro-never]'); await sleep(300);
    ok(await b.evaluate(INTRO), 'ticking the box skipped the intro');
    ok(await b.evaluate(`localStorage.getItem('geoseek.intro.off') === '1'`), 'preference not persisted');
    await b.click('[data-testid=intro-skip]');
    await b.waitFor(NO_INTRO, 1500, 'skip');
    await b.evaluate(`sessionStorage.clear(); true`);                      // a NEW session: only the stored preference can keep it away
    await b.nav(BASE); await b.waitFor(`!!document.querySelector('.app .rail')`, 30000, 'app'); await sleep(900);
    ok(await b.evaluate(NO_INTRO), 'intro shown despite the stored preference');
    return 'persisted in localStorage, suppressed in a new session';
  });
  await step('intro: never shown on a deep link', async () => {
    await resetFlags(); await b.nav('about:blank'); await sleep(200);
    await b.nav(BASE + '#/data'); await b.waitFor(`!!document.querySelector('.app .rail')`, 30000, 'app'); await sleep(900);
    ok(await b.evaluate(NO_INTRO), 'intro played over a deep link to #/data');
    await b.nav('about:blank'); await sleep(200);
    await b.nav(BASE + '#/changes/2019_2026_004510'); await b.waitFor(`!!document.querySelector('.app .rail')`, 30000, 'app'); await sleep(900);
    ok(await b.evaluate(NO_INTRO), 'intro played over a deep link to a candidate');
    return '#/data and #/changes/<id> go straight to the route';
  });
  await step('intro + logo: prefers-reduced-motion gives a short still intro and a still logo', async () => {
    await b.send('Emulation.setEmulatedMedia', { features: [{ name: 'prefers-reduced-motion', value: 'reduce' }] });
    await resetFlags(); await b.nav('about:blank'); await sleep(200); await b.nav(BASE);
    await b.waitFor(INTRO, 30000, 'intro');
    const t0 = Date.now();
    ok(await b.evaluate(`document.querySelector('[data-testid=intro]').classList.contains('reduced')`), 'reduced class');
    await b.waitFor(NO_INTRO, 5000, 'reduced intro to end');
    const ms = Date.now() - t0;
    ok(ms <= 2300, `reduced-motion intro ran ${ms} ms`);
    const logo = await b.evaluate(`(() => { const s = document.querySelector('.rail .logo svg'); return { orbit: s.dataset.orbit, anim: s.querySelectorAll('animateMotion').length }; })()`);
    ok(logo.orbit === 'still' && logo.anim === 0, JSON.stringify(logo));
    await b.send('Emulation.setEmulatedMedia', { features: [{ name: 'prefers-reduced-motion', value: 'no-preference' }] });
    return `reduced intro ${ms} ms; logo not animated`;
  });
  await b.evaluate(`(() => { try { localStorage.setItem('geoseek.intro.off', '1'); } catch (e) {} return true; })()`);   // the rest of the run is about the console
  await b.nav('about:blank'); await sleep(200);
  await b.nav(BASE);
  await b.waitFor(`document.querySelectorAll('.stat').length >= 7 && !document.querySelector('.stats .skeleton')`, 30000, 'stat cards');

  await step('logo: the rail logo is animated slowly and continuously (the satellite flies its orbit), no new asset', async () => {
    const q = `(() => { const s = document.querySelector('.rail .logo svg'); const g = s.querySelector('[data-testid=logo-sat-back]').getBoundingClientRect(); return { orbit: s.dataset.orbit, anim: s.querySelectorAll('animateMotion').length, t: s.getCurrentTime(), x: g.left + g.width / 2, y: g.top + g.height / 2, imgs: document.querySelectorAll('.rail .logo img').length }; })()`;
    const a = await b.evaluate(q); await sleep(4000); const c = await b.evaluate(q);
    ok(a.orbit === 'running' && a.anim === 2 && a.imgs === 0, JSON.stringify(a));
    ok(c.t - a.t > 3.2, `animation clock advanced ${(c.t - a.t).toFixed(2)} s in 4 s`);
    const d = Math.hypot(c.x - a.x, c.y - a.y);
    ok(d > 0.8 && d < 14, `satellite moved ${d.toFixed(2)} px in 4 s (expected a slow drift: >0.8 px, <14 px)`);
    return `satellite moved ${d.toFixed(1)} px in 4 s of a 48 s lap`;
  });

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
    await b.waitFor(`document.querySelectorAll('[data-testid=regions-panel] [data-region]').length === ${m.findings_by_region.length}`, 10000, 'one row per region');
    const rows = await b.evaluate(`({
      regions: [...document.querySelectorAll('[data-testid=regions-panel] [data-region]')].map(r => ({ name: r.dataset.region, candidates: r.querySelector('[data-field=candidates]')?.innerText ?? null, text: r.innerText.replace(/\\s+/g, ' ').trim() })),
      types: [...document.querySelectorAll('.dash-pair .panel')].pop() ? [...[...document.querySelectorAll('.dash-pair .panel')].pop().querySelectorAll('.brow')].map(r => r.innerText.replace(/\\s+/g, ' ').trim()) : [] })`);
    const total = m.findings_by_region.reduce((s, r) => s + r.candidates, 0);
    ok(rows.regions.length === m.findings_by_region.length, `region rows ${rows.regions.length}`);
    const withCount = rows.regions.filter((r) => r.candidates !== null);
    ok(withCount.length === m.findings_by_region.filter((r) => r.catalog.analysed).length, 'only analysed regions show a finding count');
    ok(withCount[0].candidates.replace(/\D/g, '') === String(Math.max(...m.findings_by_region.map((r) => r.candidates))), `top region row: ${withCount[0].text}`);
    ok(total === m.counters.change_candidates, `regions sum ${total} vs candidates ${m.counters.change_candidates}`);
    const typeTotal = Object.values(m.findings_by_type).reduce((a, c) => a + c, 0);
    ok(rows.types.length === Object.keys(m.findings_by_type).length && typeTotal === m.counters.change_candidates, 'type rows / total');
    ok(rows.types.some((r) => r.includes(String(Math.max(...Object.values(m.findings_by_type))))), 'top type count shown');
    return `${withCount[0].text.slice(0, 40)} | ${rows.types[0]}`;
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
    // Opening a candidate (re)starts the on-load sweep 0.5-1.3 s later (measured idle / under CPU load), and the sweep hides this slider.
    // A "ready and at rest" reading taken before that moment belongs to the PREVIOUS candidate, so first let the new sweep begin (if it
    // is going to: it has already run when this step is reached late), then wait for it to finish.
    await b.waitFor(`!!document.querySelector('.lapse-stage')`, 2500, 'new candidate sweep starts').catch(() => {});
    await b.waitFor(READY, 30000, 'timeline imagery preloaded'); await sleep(300);
    await b.waitFor(REST, 30000, 'on-load timeline sweep settles to the before/after slider');
    const r = await b.rectOf('.compare');
    const read = () => b.evaluate(`+document.querySelector('.compare').getAttribute('aria-valuenow')`);
    await b.drag(r.x + r.w * 0.5, r.y + r.h * 0.5, r.x + r.w * 0.85, r.y + r.h * 0.5);
    await b.waitFor(`+document.querySelector('.compare').getAttribute('aria-valuenow') >= 78`, 5000, 'slider follows the drag');
    // the page may still be working through the queued pointer events (the logo animates continuously): read the value only once it has stopped moving
    let v1 = await read();
    for (let i = 0; i < 20; i++) { await sleep(250); const v = await read(); if (v === v1) break; v1 = v; }
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

  // ---------------------------------------------------------------- explainability + suppression pipeline
  const ID0 = (await b.evaluate('location.hash')).split('/').pop();   // later steps expect this candidate to be open
  const FMT_PTS = (x) => (x == null || Math.abs(x) < 0.05 ? 'no reduction' : `${x >= 0 ? '+' : '−'}${Math.abs(x).toFixed(1)} pts`);

  await step('explain: "Why this was flagged" lead sentence, evidence order, weights and costs equal /ui/candidates/{id}/explain', async () => {
    const id = (await b.evaluate('location.hash')).split('/').pop();
    const ex = await api(`/ui/candidates/${id}/explain`);
    await b.waitFor(`!!document.querySelector('[data-testid="why-lead"]') && document.querySelectorAll('.ev-row[data-term]').length === ${ex.evidence.terms.length}`, 20000, 'why panel');
    ok((await text('[data-testid="why-lead"]')).trim() === ex.lead.headline, `lead sentence differs: ${await text('[data-testid="why-lead"]')}`);
    ok((await text('[data-testid="why-persistence"]')).trim() === ex.lead.persistence, 'persistence sentence');
    const rows = await b.evaluate(`[...document.querySelectorAll('.ev-row[data-term]')].map(r => ({ term: r.dataset.term, eff: r.querySelector('[data-effect]').innerText, weak: r.classList.contains('weak'), share: r.querySelectorAll('.meter .mono')[0].innerText }))`);
    ok(JSON.stringify(rows.map((r) => r.term)) === JSON.stringify(ex.evidence.terms.map((t) => t.name)), 'evidence order = API order');
    ex.evidence.terms.forEach((t, i) => {
      ok(rows[i].eff === FMT_PTS(t.effect_points), `${t.name} cost: ${rows[i].eff} vs ${FMT_PTS(t.effect_points)}`);
      ok(rows[i].weak === (t.strength === 'weak'), `${t.name} weak flag`);
      ok(rows[i].share === `${(t.weight_share * 100).toFixed(0)}%`, `${t.name} weight share ${rows[i].share}`);
    });
    const f = ex.evidence.terms.map((t) => t.factor);
    ok(f.every((v, i) => i === 0 || f[i - 1] <= v), 'terms are ordered by realised effect, largest first');
    const sum = await text('[data-testid="why-sum"]');
    ok(sum.includes(`${(ex.evidence.stored_confidence * 100).toFixed(1)}%`) && sum.includes(`${(ex.evidence.recomputed_confidence * 100).toFixed(1)}%`), `confidence sum line: ${sum}`);
    ok(ex.evidence.reproduces && /✓/.test(sum), 'recomputed value reproduces the stored confidence');
    for (const m of ex.evidence.multipliers) ok((await b.evaluate(`document.querySelector('[data-mult="${m.name}"]')?.innerText || ''`)).includes(`×${m.factor.toFixed(2)}`), `multiplier ${m.name}`);
    await shot('10-explain-why');
    return `${ex.evidence.terms.length} terms; largest effect: ${ex.evidence.terms[0].name} ${FMT_PTS(ex.evidence.terms[0].effect_points)}`;
  });

  await step('explain: spectral deltas vs thresholds, and absent / weak evidence is as plain as strong (SAR missing stays prominent)', async () => {
    const id = (await b.evaluate('location.hash')).split('/').pop();
    const ex = await api(`/ui/candidates/${id}/explain`);
    const rows = await b.evaluate(`[...document.querySelectorAll('.why-spec tbody tr[data-index]')].map(r => ({ i: r.dataset.index, cells: [...r.querySelectorAll('td')].map(c => c.innerText.replace(/\\s+/g, ' ')) }))`);
    ok(rows.length === 3, 'one row per index');
    for (const a of ex.spectral.anomalies) {
      const r = rows.find((x) => x.i === a.index);
      const sg = (v) => `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(2)}`;
      ok(r.cells[1] === sg(a.delta) && r.cells[2] === sg(a.seasonal) && r.cells[3] === sg(a.anomaly), `${a.index}: ${r.cells.slice(1, 4)} vs ${sg(a.delta)},${sg(a.seasonal)},${sg(a.anomaly)}`);
    }
    for (const t of ex.spectral.tests.filter((x) => x.decisive)) ok((await b.evaluate('document.querySelector(".why-spec").innerText')).replace(/\s+/g, ' ').includes(t.text), `threshold shown: ${t.text}`);
    const gaps = await text('[data-testid="why-gaps"]');
    ok(ex.weak_or_absent.length > 0 && ex.weak_or_absent.every((w) => gaps.replace(/\s+/g, ' ').includes(w.text.slice(0, 40))), 'every weak / missing item from the API is listed');
    if (!ex.sar.available) {
      ok(/No Sentinel-1 coverage for this candidate/.test(gaps), 'SAR absence stated in the panel');
      ok((await b.evaluate(`document.querySelector('[data-gap="sar"]')?.className`)) === 'sar', 'SAR absence is styled as the prominent item');
    }
    ok(/context only/.test(gaps) || !ex.terrain, 'terrain is labelled as context, not evidence');
    return `${ex.weak_or_absent.length} weak / missing items; SAR ${ex.sar.available ? 'present' : 'absent'}`;
  });

  await step('explain: the raw decomposition sits behind a toggle and shows the engine\'s own numbers', async () => {
    const id = (await b.evaluate('location.hash')).split('/').pop();
    const ex = await api(`/ui/candidates/${id}/explain`);
    ok((await count('[data-testid="why-raw"]')) === 0, 'raw decomposition is hidden by default');
    await b.click('[data-testid="why-raw-toggle"]');
    await b.waitFor(`!!document.querySelector('[data-testid="why-raw"]')`, 5000, 'raw panel');
    ok((await count('[data-testid="why-raw"] .why-lines li')) === ex.evidence.breakdown_lines.length, 'every breakdown line shown');
    const raw = await text('[data-testid="why-raw"]');
    for (const t of ex.evidence.terms) ok(raw.includes(t.factor.toFixed(4)) && raw.includes(t.value.toFixed(4)), `raw numbers for ${t.name}`);
    await b.click('[data-testid="why-raw-toggle"]');
    await b.waitFor(`!document.querySelector('[data-testid="why-raw"]')`, 5000, 'raw panel closes');
    return 'toggle opens and closes';
  });

  await step('explain: "Show … on the pixels" opens the index maps for the after-date tile with the candidate footprint outlined', async () => {
    const id = (await b.evaluate('location.hash')).split('/').pop();
    const ex = await api(`/ui/candidates/${id}/explain`);
    ok(!!ex.overlay, 'candidate has an overlay tile');
    await b.click('[data-testid="why-pixels"]');
    await b.waitFor(`document.querySelectorAll('.spec-fig').length === ${Math.max(1, ex.overlay.focus_indices.length || 3)} && !!document.querySelector('[data-testid="footprint-outline"]')`, 30000, 'index maps with outline');
    const shown = await b.evaluate(`[...document.querySelectorAll('.spec-fig')].map(f => f.dataset.index)`);
    ok(ex.overlay.focus_indices.length === 0 || JSON.stringify(shown.sort()) === JSON.stringify([...ex.overlay.focus_indices].sort()), `decisive indices shown first: ${shown}`);
    const pts = await b.evaluate(`document.querySelector('[data-testid="footprint-outline"] polygon').getAttribute('points').trim().split(/\\s+/).length`);
    ok(pts === ex.overlay.geometry.coordinates[0].length, `outline has ${pts} vertices, geometry has ${ex.overlay.geometry.coordinates[0].length}`);
    ok((await text('.spec-panel')).length > 0 && /decisive for this change type/.test(await b.evaluate('document.body.innerText')) === (ex.overlay.focus_indices.length > 0), 'decisive tag matches the API focus');
    await b.waitFor(imgsLoaded('.spec-img img.pix'), 30000, 'overlay image');
    ok((await b.evaluate(`document.querySelector('.spec-panel').getAttribute('data-tile')`)) === ex.overlay.tile_id, 'tile id equals API overlay tile');
    await shot('11-explain-pixels');
    await b.click('[data-testid="why-pixels"]');
    await b.waitFor(`!document.querySelector('.spec-panel')`, 5000, 'maps close');
    return `${ex.overlay.tile_id} ${ex.overlay.focus_indices.join('+') || 'all indices'}`;
  });

  await step('trace: every stage the candidate went through, in order, with verdict and effect from the API', async () => {
    const id = (await b.evaluate('location.hash')).split('/').pop();
    const ex = await api(`/ui/candidates/${id}/explain`);
    const rows = await b.evaluate(`[...document.querySelectorAll('[data-testid="trace-list"] .tr-row')].map(r => ({ id: r.dataset.stage, verdict: r.dataset.verdict, eff: r.querySelector('.eff').innerText }))`);
    ok(JSON.stringify(rows.map((r) => r.id)) === JSON.stringify(ex.trace.map((s) => s.id)), `stage order ${rows.map((r) => r.id)}`);
    ok(JSON.stringify(rows.map((r) => r.verdict)) === JSON.stringify(ex.trace.map((s) => s.verdict)), 'verdicts equal the API');
    ex.trace.forEach((s, i) => {
      const f = s.effect?.factor;
      if (f != null && Math.abs(f - 1) >= 5e-4) ok(rows[i].eff.includes(`×${f.toFixed(2)}`), `${s.id} effect ${rows[i].eff} vs ×${f.toFixed(2)}`);
      else if (s.effect == null || f == null) ok(/no effect/.test(rows[i].eff), `${s.id}: ${rows[i].eff}`);
    });
    ok(rows.length === 8 && rows[0].id === 'morphology' && rows[7].id === 'sar', 'eight stages, morphology first, SAR last');
    return rows.map((r) => `${r.id}:${r.verdict}`).join(' ');
  });

  await step('trace: a demoted candidate explains why, and says it was demoted rather than dropped', async () => {
    const list = await api('/candidates?persistence=transient&sort=confidence&limit=1');
    ok(list.total > 0, 'there is a transient candidate');
    const id = list.candidates[0].candidate_id;
    await go(`changes/${id}`);
    const ex = await api(`/ui/candidates/${id}/explain`);
    await b.waitFor(`!!document.querySelector('[data-testid="trace-demoted"]')`, 20000, 'demotion note');
    const note = await text('[data-testid="trace-demoted"]');
    const pers = ex.trace.find((s) => s.id === 'persistence');
    ok(/Demoted, not dropped/.test(note) && note.includes(pers.detail.slice(0, 30)), `note: ${note}`);
    ok(note.includes(`${Math.round(ex.evidence.stored_confidence * 100)}%`), 'confidence in the note equals the API');
    ok(pers.verdict === 'demoted' && pers.effect.points < 0, 'API marks the stage demoted with a negative effect');
    const lead = await text('[data-testid="why-persistence"]');
    ok(/Time is against it/.test(lead) && lead.includes(`×${(ex.evidence.multipliers.find((m) => m.name === 'persistence_penalty').factor).toFixed(2)}`), `lead: ${lead}`);
    await shot('12-explain-demoted');
    return `${id}: ${pers.verdict}, ${pers.effect.points} pts`;
  });

  await step('explain: an unclassified candidate says no rule matched and starts no index as "decisive"', async () => {
    const list = await api('/candidates?change_type=other&limit=1');
    ok(list.total > 0, 'an unclassified candidate exists');
    const id = list.candidates[0].candidate_id;
    await go(`changes/${id}`);
    await b.waitFor(`document.querySelector('[data-testid="why-lead"]')?.innerText.includes('no spectral rule matched')`, 20000, 'unclassified lead');
    ok(/no rule matched/.test(await text('.why-spec')), 'table says no rule matched');
    await b.click('[data-testid="why-pixels"]');
    await b.waitFor(`document.querySelectorAll('.spec-fig').length === 3`, 30000, 'all three index maps');
    ok(!/decisive for this change type/.test(await b.evaluate('document.body.innerText')), 'nothing is starred as decisive');
    await b.click('[data-testid="why-pixels"]');
    return id;
  });

  await step('pipeline: funnel counts, per-gate removals and shares equal /ui/pipeline/funnel; nothing typed in', async () => {
    const f = await api('/ui/pipeline/funnel');
    await go('pipeline');
    await b.waitFor(`document.querySelectorAll('[data-testid="funnel"] .fn-row').length === ${f.pairs.find((p) => p.name === f.span_pair).stages.length + 2}`, 20000, 'funnel rows');
    const span = f.pairs.find((p) => p.name === f.span_pair);
    const rows = await b.evaluate(`[...document.querySelectorAll('[data-testid="funnel"] .fn-row')].map(r => ({ s: r.dataset.stage, removed: +r.dataset.removed, remaining: +r.dataset.remaining, input: +r.dataset.input, text: r.querySelector('.cnt').innerText }))`);
    ok(rows[0].s === 'raw' && rows[0].remaining === span.raw, `raw ${rows[0].remaining} vs ${span.raw}`);
    span.stages.forEach((s, i) => {
      ok(rows[i + 1].s === s.rule && rows[i + 1].removed === s.removed && rows[i + 1].remaining === s.remaining, `${s.rule}: ${JSON.stringify(rows[i + 1])}`);
      if (s.removed > 0) ok(rows[i + 1].text.includes(s.removed.toLocaleString('en-US')) && rows[i + 1].text.includes(`${(s.share_of_raw * 100).toFixed(1)}%`), `${s.rule} text ${rows[i + 1].text}`);
    });
    const last = rows[rows.length - 1];
    ok(last.s === 'survivors' && last.remaining === span.survivors, 'survivors row');
    // the funnel's end is the queue the analyst actually sees
    const q = await api('/candidates?limit=1');
    ok(span.survivors === q.total, `funnel survivors ${span.survivors} = queue total ${q.total}`);
    ok(span.raw - span.suppressed === span.survivors && span.consistent, 'API says gate counts add up');
    const cards = await b.evaluate(`[...document.querySelectorAll('.stat .value')].map(v => v.innerText)`);
    ok(cards.includes(span.raw.toLocaleString('en-US')) && cards.includes(span.survivors.toLocaleString('en-US')) && cards.includes(span.suppressed.toLocaleString('en-US')), `stat cards ${cards}`);
    await shot('13-pipeline');
    return `${span.raw} → ${span.survivors}; ${span.stages.map((s) => `${s.rule} −${s.removed}`).join(', ')}`;
  });

  await step('pipeline: choosing another pair changes every figure (proof they are read, not typed); zoom rescales to the size floor', async () => {
    const f = await api('/ui/pipeline/funnel');
    const other = f.pairs.find((p) => p.name !== f.span_pair);
    await b.evaluate(`(() => { const e = document.querySelector('[data-testid="pair-select"]'); const set = Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value').set; set.call(e, ${JSON.stringify(other.name)}); e.dispatchEvent(new Event('change', {bubbles:true})); })()`);
    await b.waitFor(`document.querySelector('[data-testid="funnel"] .fn-row[data-stage="raw"]')?.dataset.remaining === '${other.raw}'`, 8000, 'other pair raw');
    const rows = await b.evaluate(`[...document.querySelectorAll('[data-testid="funnel"] .fn-row')].map(r => ({ s: r.dataset.stage, removed: +r.dataset.removed }))`);
    other.stages.forEach((s, i) => ok(rows[i + 1].removed === s.removed, `${other.name} ${s.rule}: ${rows[i + 1].removed} vs ${s.removed}`));
    ok(/computed for the span pair/.test(await b.evaluate('document.body.innerText')), 'persistence / SAR sections say they are span-only');
    await b.click('[data-testid="funnel-zoom"]');
    await b.waitFor(`document.querySelector('[data-testid="funnel"] .fn-row[data-stage="raw"]')?.innerText.includes('reached the checks')`, 5000, 'zoom');
    const w = await b.evaluate(`(() => { const r = document.querySelector('[data-testid="funnel"] .fn-row[data-stage="survivors"] .keep').getBoundingClientRect(); const t = document.querySelector('[data-testid="funnel"] .fn-row[data-stage="survivors"] .fn-track').getBoundingClientRect(); return r.width / t.width; })()`);
    ok(Math.abs(w - other.survivors / other.stages[0].remaining) < 0.02, `zoomed survivors bar ${w.toFixed(3)} vs ${(other.survivors / other.stages[0].remaining).toFixed(3)}`);
    await b.click('[data-testid="funnel-zoom"]');
    await b.evaluate(`(() => { const e = document.querySelector('[data-testid="pair-select"]'); const set = Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value').set; set.call(e, ${JSON.stringify(f.span_pair)}); e.dispatchEvent(new Event('change', {bubbles:true})); })()`);
    return `${other.name}: ${other.raw} → ${other.survivors}`;
  });

  await step('pipeline: post-gate stages (typing, persistence, SAR, bands), labelled-benchmark table and the Ayodhya-calibration caveat come from the API', async () => {
    const f = await api('/ui/pipeline/funnel');
    await b.waitFor(`!!document.querySelector('[data-testid="post-gate"] [data-class]')`, 8000, 'post-gate');
    const pe = f.post_gate.persistence;
    for (const [k, n] of Object.entries(pe.by_class)) {
      const cells = await b.evaluate(`(() => { const r = document.querySelector('[data-testid="post-gate"] tr[data-class="${k}"]'); return r ? [...r.querySelectorAll('td')].map(c => c.innerText) : null; })()`);
      ok(cells && cells[1] === n.toLocaleString('en-US'), `class ${k}: ${cells}`);
    }
    const body = await text('[data-testid="post-gate"]');
    ok(body.includes(pe.contradicted.toLocaleString('en-US')) && body.includes(f.post_gate.survivors.toLocaleString('en-US')), 'demoted / survivor counts');
    if (!f.post_gate.sar.available) ok(/No Sentinel-1 corroboration in this run/.test(body), 'SAR absence stated on the funnel screen');
    for (const [k, n] of Object.entries(f.post_gate.confidence_bands)) ok(body.includes(`${k} ${n.toLocaleString('en-US')}`.replace(/\s+/g, ' ')) || body.replace(/\s+/g, ' ').includes(`${k} ${n.toLocaleString('en-US')}`), `band ${k} ${n}`);
    const lab = f.labelled_benchmark;
    const t = await text('[data-testid="labelled"]');
    ok(lab.available && lab.stages.every((s) => t.includes(s.f1.toFixed(3))), 'labelled-benchmark F1 values equal the API');
    ok(t.includes(lab.caveat.slice(0, 40)), 'labelled benchmark carries its own caveat');
    ok(/calibrated on the Ayodhya AOI/.test(await text('[data-testid="funnel-scope"]')), 'gate effects are scoped to Ayodhya');
    const gc = await text('.gate-cards');
    for (const g of f.gates) ok(gc.includes(g.what.slice(0, 40)), `gate card for ${g.rule}`);
    ok(/costs F1 off-region/.test(gc) === lab.stages.some((s) => s.stage === 'phenology' && s.delta_f1 <= -0.005), 'the F1-cost warning appears exactly where the benchmark shows a cost');
    return `${Object.keys(pe.by_class).length} persistence classes; benchmark ΔF1 ${lab.stages.map((s) => s.delta_f1).filter((x) => x != null).join(', ')}`;
  });

  await step('pipeline: rejected components are stated as not retained, what would be needed is named, demoted sample is inspectable', async () => {
    const f = await api('/ui/pipeline/funnel');
    ok(f.rejected.retained === false, 'API says rejected components are not retained');
    const rej = await text('[data-testid="rejected"]');
    ok(/Not retained\./.test(rej) && rej.includes(f.rejected.to_emit.slice(0, 40)), `rejected panel: ${rej.slice(0, 120)}`);
    ok(!/browse rejected|download rejected/i.test(rej) && (await count('[data-testid="rejected"] a, [data-testid="rejected"] button')) === 0, 'no control implies rejected components can be recovered');
    const n = await count('[data-testid="demoted-sample"] tbody tr');
    ok(n === f.demoted_sample.length && n > 0, `${n} demoted sample rows`);
    const first = f.demoted_sample[0];
    ok((await text('[data-testid="demoted-sample"] tbody tr')).includes(first.candidate_id) && (await text('[data-testid="demoted-sample"] tbody tr')).includes(first.reason.slice(0, 20)), 'sample row shows id and reason');
    await b.click('[data-testid="demoted-sample"] tbody tr');
    await b.waitFor(`location.hash === '#/changes/${first.candidate_id}'`, 8000, 'opens the candidate');
    await b.waitFor(`!!document.querySelector('[data-testid="trace-demoted"]')`, 20000, 'its trace says demoted');
    return `${first.candidate_id}: ${first.reason}`;
  });

  await step('explain: back on the original candidate the workbench is intact (compare slider, timeline, panels)', async () => {
    await go(`changes/${ID0}`);
    // leaving the screen reset the queue filter; later steps (filtered export) expect the construction filter, as before
    await b.waitFor(`document.querySelectorAll('.tbl tbody tr').length > 0`, 20000, 'queue');
    await b.evaluate(`(() => { const s = document.querySelector('.filters select'); const set = Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value').set; set.call(s, 'construction'); s.dispatchEvent(new Event('change', {bubbles:true})); })()`);
    const nConstruction = (await api('/candidates?change_type=construction&limit=1')).total;
    await b.waitFor(`document.body.innerText.includes('of ${nConstruction}')`, 10000, 'filtered total');
    await b.waitFor(`document.querySelector('.compare') && ${imgsLoaded('.compare img').replace(/\n/g, ' ')}`, 30000, 'compare images');
    await b.waitFor(`!!document.querySelector('[data-testid="why-lead"]')`, 20000, 'why panel');
    return ID0;
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
      document.querySelector('.vec-stage').scrollIntoView({ block: 'center' });      // the stage can sit below the fold; the click is a raw screen point
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

  await step('data: inventory tables keep one line per row on fixed columns, GSD rounded with the exact value on hover, dates as chips', async () => {
    await go('data');
    const m = await api('/ui/metrics'), regs = (await api('/regions')).regions;
    await b.waitFor(`document.querySelectorAll('table[aria-label=Collections] tbody tr').length === ${m.sensors.length} && document.querySelectorAll('table[aria-label=Regions] tbody tr').length === ${regs.length}`, 15000, 'tables');
    const lay = await b.evaluate(`(() => {
      const tbl = (n) => document.querySelector('table[aria-label=' + n + ']');
      const info = (t) => ({
        fixed: getComputedStyle(t).tableLayout,
        heads: [...t.querySelectorAll('th')].map((h) => ({ t: h.innerText, h: h.getBoundingClientRect().height, clipped: h.scrollWidth > h.clientWidth + 1 })),
        rows: [...t.querySelectorAll('tbody tr')].map((r) => ({ h: r.getBoundingClientRect().height, cells: [...r.children].map((c) => ({ x: Math.round(c.getBoundingClientRect().left), txt: c.innerText, title: c.title, cut: c.scrollWidth > c.clientWidth + 1 })) })),
      });
      const d = document.querySelector('[data-testid=inventory-dates]');
      return { c: info(tbl('Collections')), r: info(tbl('Regions')), dates: { h: d.getBoundingClientRect().height, chips: [...d.querySelectorAll('.chip')].map((x) => x.innerText), w: d.getBoundingClientRect().width, panelW: d.closest('.body').clientWidth } };
    })()`);
    for (const [name, t] of [['collections', lay.c], ['regions', lay.r]]) {
      ok(t.fixed === 'fixed', `${name}: table-layout ${t.fixed}`);
      ok(t.heads.every((h) => h.h < 34 && !h.clipped), `${name}: a header wraps or is clipped: ${JSON.stringify(t.heads)}`);
      const hs = t.rows.map((r) => r.h); ok(Math.max(...hs) - Math.min(...hs) < 3 && Math.max(...hs) < 40, `${name}: rows differ in height (a cell wrapped): ${hs}`);
      for (let k = 0; k < t.rows[0].cells.length; k++) ok(new Set(t.rows.map((r) => r.cells[k].x)).size === 1, `${name}: column ${k} is not aligned`);
    }
    const r1 = (v) => (v >= 1 ? +v.toFixed(1) : +v.toFixed(2));
    m.sensors.forEach((s, i) => {
      const cells = lay.c.rows[i].cells, gsd = cells[2], sensor = cells[1];
      ok(cells[0].txt === s.collection_id && !cells[0].cut, `collection id "${cells[0].txt}" wraps or is cut`);
      ok(gsd.txt === `${r1(s.native_gsd_m)} m` && gsd.title === `${s.native_gsd_m} m`, `GSD ${gsd.txt} / title ${gsd.title} for ${s.native_gsd_m}`);
      ok(sensor.title === `${s.platform} · ${s.sensor}`, `sensor tooltip ${sensor.title}`);
    });
    ok(!lay.c.rows.some((r) => /\d\.\d{3,}/.test(r.cells[2].txt)), 'a GSD is shown at full float precision');
    ok(lay.dates.chips.length === m.observation_dates.length && m.observation_dates.every((d, i) => lay.dates.chips[i] === d), `date chips ${lay.dates.chips}`);
    ok(lay.dates.h < 70 && lay.dates.w > lay.dates.panelW * 0.9, `dates block is ${lay.dates.h}px tall, ${lay.dates.w}px wide of ${lay.dates.panelW}`);
    ok(lay.r.rows.every((r, i) => r.cells[2].title.includes(regs[i].bbox[0].toFixed(3))), 'region bbox tooltips');
    await shot('24-data-inventory');
    return `${m.sensors.length} collection rows + ${regs.length} region rows, one line each, columns aligned; GSD ${lay.c.rows.map((r) => r.cells[2].txt).join(' / ')}; ${lay.dates.chips.length} date chips in ${Math.round(lay.dates.h)}px`;
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

  await step('data: a 168 MB staged Sentinel-2 band: header parsed at once, then a decimated pixel preview, matching the catalog', async () => {
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
    ok(ms < 8000, `header took ${ms} ms (a whole-file read would be slower)`);
    // the pixels are decoded afterwards as a decimated preview, and the caption says so
    await b.waitFor(`document.querySelector('[data-testid=preview-canvas]')?.dataset.drawn`, 90000, 'decimated preview of the 168 MB band');
    const cap = await text('[data-testid=preview-caption]');
    ok(/decimation/.test(cap) && /10913 × 10903/.test(cap), cap.slice(0, 260));
    return `${c.size} ${c.bytes}: header in ${ms} ms, then a decimated preview (${await b.evaluate(`document.querySelector('[data-testid=preview-canvas]').dataset.w + 'x' + document.querySelector('[data-testid=preview-canvas]').dataset.h`)})`;
  });

  // ============================== raster preview, upload-vs-archive comparison, Temporal tab ==============================
  const here = path.dirname(fileURLToPath(import.meta.url));
  const PY = process.env.GEOSEEK_PY || 'C:/AnacondaPython/anaconda3/envs/geoseek/python.exe';
  let fx = null;
  try {
    const dir = fsx.mkdtempSync(path.join(tmp, 'geoseek-fx-')), envRoot = path.dirname(PY);
    execFileSync(PY, [path.join(here, '..', '..', 'tests', 'raster_fixtures.py'), dir], { stdio: 'pipe', env: { ...process.env, GDAL_DATA: `${envRoot}/Library/share/gdal`, PROJ_LIB: `${envRoot}/Library/share/proj`, PYTHONIOENCODING: 'utf-8' } });
    fx = { dir, ref: JSON.parse(fsx.readFileSync(path.join(dir, 'reference.json'), 'utf8')) };
  } catch (e) { fx = null; }
  const F = (k) => fx.ref.files[k].path;
  const drop = async (paths, ready = 1) => { await go('data'); await b.setFiles('[data-testid=raster-input]', paths); await b.waitFor(`document.querySelectorAll('[data-testid=preview-canvas][data-drawn]:not([data-drawn=""])').length >= ${ready}`, 40000, 'preview canvas'); };
  const pixelAt = (r, c, nth = 0) => b.evaluate(`Array.from(document.querySelectorAll('[data-testid=preview-canvas]')[${nth}].getContext('2d').getImageData(${c}, ${r}, 1, 1).data)`);
  const f4 = (v) => (Number.isFinite(v) ? (Math.abs(v) >= 100 ? v.toFixed(1) : v.toFixed(4)) : '—');

  await step('data: the dropzone caption is accurate - pixels ARE read here now, nothing leaves the machine', async () => {
    await go('data');
    const cap = await text('[data-testid=dropzone-caption]');
    ok(!/pixel data is not read/i.test(cap), cap);
    ok(/pixels are read in this browser/i.test(cap) && /nothing leaves this machine/i.test(cap) && /nothing is added to the archive/i.test(cap), cap);
    return cap.slice(0, 150);
  });

  await step('data: a real 4-band GeoTIFF previews as true colour (B04/B03/B02); every canvas pixel equals the numpy-stretched value; the percentiles are stated', async () => {
    if (!fx) return 'skipped: fixtures could not be generated (needs the geoseek python env and the staged scenes)';
    await drop([F('2019-03-30')]);
    const sel = await b.evaluate(`({ r: document.querySelector('select[aria-label="Band shown as red"]').value, g: document.querySelector('select[aria-label="Band shown as green"]').value, bl: document.querySelector('select[aria-label="Band shown as blue"]').value, mode: document.querySelector('[data-testid=preview-canvas]').dataset.mode })`);
    ok(sel.mode === 'rgb' && sel.r === '2' && sel.g === '1' && sel.bl === '0', JSON.stringify(sel));
    const samples = fx.ref.files['2019-03-30'].composite_samples;
    ok(samples.length >= 3, 'fixture samples');
    for (const sm of samples) { const px = await pixelAt(sm.row, sm.col); ok(px[0] === sm.rgb[0] && px[1] === sm.rgb[1] && px[2] === sm.rgb[2] && px[3] === 255, `pixel (${sm.row},${sm.col}) = ${px} but numpy says ${sm.rgb}`); }
    const cap = await text('[data-testid=preview-caption]');
    ok(/2nd and 98th percentile/.test(cap) && /nothing is uploaded/i.test(cap) && /native resolution, 800 × 800 px/.test(cap) && /NoData value \(0\)/.test(cap), cap.slice(0, 420));
    const st = fx.ref.files['2019-03-30'].bands;
    ok(cap.includes(`R ${(+st.B04.p2.toPrecision(4)) >= 100 ? st.B04.p2.toFixed(0) : st.B04.p2.toPrecision(4)}`), `stated red stretch low ${st.B04.p2}: ${cap.slice(0, 300)}`);
    await shot('30-data-preview');
    return `${samples.length} canvas pixels equal numpy exactly; caption states 2nd–98th percentile and the per-band values`;
  });

  await step('data: band selector - single band grey view and other stretches re-render the canvas', async () => {
    if (!fx) return 'skipped';
    const before = await b.evaluate(`document.querySelector('[data-testid=preview-canvas]').dataset.drawn`);
    await b.select('[data-testid=preview-mode]', 'grey');
    await b.waitFor(`document.querySelector('[data-testid=preview-canvas]').dataset.mode === 'grey' && document.querySelector('[data-testid=preview-canvas]').dataset.drawn !== ${JSON.stringify(before)}`, 8000, 'grey');
    const g = await pixelAt(400, 400); ok(g[0] === g[1] && g[1] === g[2] && g[3] === 255, `not grey: ${g}`);
    await b.select('[data-testid=preview-stretch]', '0-100');
    await b.waitFor(`/0th and 100th percentile/.test(document.querySelector('[data-testid=preview-caption]').innerText)`, 8000, 'min-max caption');
    const px = await pixelAt(400, 400);
    await b.select('[data-testid=preview-mode]', 'rgb'); await b.select('[data-testid=preview-stretch]', '2-98');
    await b.waitFor(`document.querySelector('[data-testid=preview-canvas]').dataset.mode === 'rgb' && /2nd and 98th/.test(document.querySelector('[data-testid=preview-caption]').innerText)`, 8000, 'back to rgb');
    return `grey (r=g=b) and 0–100 stretch re-render; sample ${px.slice(0, 3)}`;
  });

  await step('data: the footprint is drawn on the archive basemap, zoomed to the file bounds (not an empty graticule)', async () => {
    if (!fx) return 'skipped';
    await b.waitFor(`document.querySelector('.raster-map .geomap')?.dataset.basemap === 'sentinel-2'`, 10000, 'basemap');
    await b.waitFor(`[...document.querySelectorAll('.raster-map .gm-basemap img')].filter(i => i.complete && i.naturalWidth > 0).length >= 1`, 30000, 'basemap tiles');
    const m = await b.evaluate(`(() => { const el = document.querySelector('.raster-map [role=application]'); const mp = el.__leaflet; const v = mp.getBounds(); const ll = document.querySelector('[data-field=lonlat]').innerText.split(',').map(Number);
      return { z: mp.getZoom(), view: [v.getWest(), v.getSouth(), v.getEast(), v.getNorth()], ll, paths: (() => { let n = 0; mp.eachLayer((l) => { if (l.options && l.options.color === '#f5a524' && l.getBounds) n++; }); return n; })(), cap: document.querySelector('.raster-map [data-testid=map-caption]')?.innerText || '', note: document.querySelector('[data-testid=footprint-note]')?.innerText || '' }; })()`);
    ok(m.ll[0] >= m.view[0] && m.ll[2] <= m.view[2] && m.ll[1] >= m.view[1] && m.ll[3] <= m.view[3], `footprint ${m.ll} outside view ${m.view}`);
    ok((m.view[2] - m.view[0]) / (m.ll[2] - m.ll[0]) < 6, 'map is not zoomed to the file');
    ok(m.z >= 11, `zoom ${m.z}`); ok(m.paths >= 1, 'no footprint outline'); ok(/Local archive imagery/.test(m.cap) && /Amber outline/.test(m.note), m.cap.slice(0, 120));
    await shot('31-data-footprint-basemap');
    return `zoom ${m.z}, view ${(m.view[2] - m.view[0]).toFixed(3)}° wide around a ${(m.ll[2] - m.ll[0]).toFixed(3)}° footprint, tiles loaded`;
  });

  await step('data: upload vs archive comparison is labelled an indicative visual difference, names the archive date, and is a real image difference', async () => {
    if (!fx) return 'skipped';
    ok(await b.evaluate(`!!document.querySelector('[data-testid=compare-banner]')`), 'banner');
    await b.waitFor(`document.querySelector('[data-testid=compare-acq-select]') && document.querySelector('[data-testid=compare-acq-select]').options.length > 0`, 20000, 'archive acquisitions');
    ok((await b.evaluate(`document.querySelector('[data-testid=compare-acq-select]').value`)) === '2019-03-30', 'nearest acquisition to a 2019-03-30 file is not 2019-03-30');
    await b.click('[data-testid=compare-run]');
    await b.waitFor(`!!document.querySelector('[data-testid=compare-grid]')`, 90000, 'comparison');
    await sleep(500);
    const r = await b.evaluate(`(() => { const root = document.querySelector('[data-testid=raster-compare]'); const alpha = (id) => { const c = document.querySelector('[data-testid=' + id + ']'); const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data; let n = 0, nonWhite = 0; for (let i = 3; i < d.length; i += 4) if (d[i] > 0) { n++; if (d[i - 3] < 240 || d[i - 1] < 240) nonWhite++; } return { n, nonWhite, w: c.width }; };
      return { text: root.innerText, banner: document.querySelector('[data-testid=compare-banner]').innerText, caveats: [...document.querySelectorAll('[data-testid=compare-caveats] li')].map(l => l.innerText), acq: document.querySelector('[data-testid=compare-acq]').innerText, nums: document.querySelector('[data-testid=compare-numbers]').innerText, method: document.querySelector('[data-testid=compare-method]').innerText,
        file: alpha('compare-canvas-file'), arch: alpha('compare-canvas-archive'), diff: alpha('compare-canvas-diff'), order: !!(document.querySelector('[data-testid=compare-banner]').compareDocumentPosition(document.querySelector('[data-testid=compare-grid]')) & Node.DOCUMENT_POSITION_FOLLOWING) }; })()`);
    ok(/Indicative visual difference — not pipeline output/.test(r.banner), r.banner.slice(0, 120));
    for (const w of ['coordinate system', 'pixel size', 'radiometric scaling', 'registration']) ok(r.banner.includes(w) || r.caveats.join(' ').toLowerCase().includes(w), `caveat for ${w} missing`);
    ok(r.caveats.length === 4 && r.order, 'four caveats, banner above the images');
    ok(/acquisition used: 2019-03-30/.test(r.acq), r.acq);
    ok(/Web-Mercator grid/.test(r.method) && /nearest-neighbour/.test(r.method) && /Rec\. 601/.test(r.method) && /2nd–98th percentile/.test(r.method), 'method not stated');
    ok(!/confidence|candidate|gate|persisten|suppress|detected|first detected/i.test(r.text), 'pipeline wording leaked into the comparison panel: ' + (r.text.match(/confidence|candidate|gate|persisten|suppress|detected/i) || [])[0]);
    ok(r.file.n > 20000 && r.arch.n > 20000 && r.diff.n > 20000 && r.diff.nonWhite > 500, `canvases look empty: ${JSON.stringify([r.file, r.arch, r.diff])}`);
    const corr = Number(r.nums.match(/correlation\s+(-?\d+\.\d+)/)[1]);
    ok(corr > 0.8, `same-date archive imagery should correlate strongly with the file, got ${corr}`);
    // choosing the 2024 acquisition must give a visibly worse match for this 2019 file: the difference is measuring something
    await b.select('[data-testid=compare-acq-select]', '2024-03-08');
    await b.waitFor(`!document.querySelector('[data-testid=compare-grid]')`, 5000, 'result cleared on a new choice');
    await b.click('[data-testid=compare-run]'); await b.waitFor(`!!document.querySelector('[data-testid=compare-grid]')`, 90000, 'second comparison');
    const r2 = await b.evaluate(`({ acq: document.querySelector('[data-testid=compare-acq]').innerText, nums: document.querySelector('[data-testid=compare-numbers]').innerText })`);
    const corr2 = Number(r2.nums.match(/correlation\s+(-?\d+\.\d+)/)[1]);
    ok(/acquisition used: 2024-03-08/.test(r2.acq) && corr2 < corr - 0.15, `2024 archive vs 2019 file: ${corr2} vs ${corr}`);
    await shot('32-data-compare');
    return `labelled indicative / not pipeline output; archive date named; correlation ${corr} against 2019 imagery vs ${corr2} against 2024 imagery`;
  });

  await step('data: a footprint with no archive coverage says so plainly (no images, no numbers)', async () => {
    const file = path.join(tmp, 'geoseek-e2e-uk.tif');
    fsx.writeFileSync(file, makeGeoTiff({ width: 64, height: 48, epsg: 32630, originX: 500000, originY: 5700000, resX: 10, resY: 10, acquisition: '2024-03-08T05:21:00Z' }));
    await go('data'); await b.setFiles('[data-testid=raster-input]', [file]);
    await b.waitFor(`!!document.querySelector('[data-testid=compare-none]')`, 30000, 'no-coverage statement');
    const t = await text('[data-testid=compare-none]');
    ok(/no imagery over this footprint/i.test(t) && /nothing to compare/i.test(t), t);
    ok(await b.evaluate(`!document.querySelector('[data-testid=compare-run]') && !document.querySelector('[data-testid=compare-grid]')`), 'a comparison control is offered with no coverage');
    return t.slice(0, 120);
  });

  await step('data: a ZSTD-compressed file (WebAssembly decoder, refused by the page CSP) is explained, not left as "Failed to fetch"; header and footprint still work', async () => {
    if (!fx) return 'skipped';
    await go('data'); await b.setFiles('[data-testid=raster-input]', [F('zstd')]);
    await b.waitFor(`/WebAssembly/.test(document.querySelector('.raster-card .err')?.innerText || '')`, 20000, 'explanation');
    const err = await text('.raster-card .err');
    ok(/ZSTD-compressed/.test(err) && /DEFLATE or LZW/.test(err) && !/Failed to fetch/.test(err), err);
    ok(await b.evaluate(`document.querySelector('.raster-card').dataset.verdict === 'green' && !document.querySelector('[data-testid=preview-canvas]')`), 'header verdict or preview state');
    // the CSP refusing the wasm is the behaviour under test here, so those console lines are not a regression for the offline check at the end
    for (let i = b.problems.length - 1; i >= 0; i--) if (/application\/wasm/.test(b.problems[i].text)) b.problems.splice(i, 1);
    return 'explains the ZSTD limit and the fix; header verdict still green';
  });

  // ---------------- Temporal: archive mode ----------------
  await step('temporal: archive mode - every chart mark equals /ui/temporal/archive; calendar axis; observations are marks, never joined', async () => {
    await go('temporal');
    const t = await api('/ui/temporal/archive');
    await b.waitFor(`document.querySelectorAll('[data-testid=chart-counts] [data-testid=seg]').length > 0 && document.querySelectorAll('[data-testid=cum-point]').length > 0 && document.querySelectorAll('[data-testid=sar-point]').length > 0`, 30000, 'charts');
    ok(await b.evaluate(`document.querySelector('.rail a[href="#/temporal"]')?.getAttribute('aria-current') === 'page'`), 'rail entry not active');
    ok(await b.evaluate(`document.body.innerText.includes('ARCHIVE MODE') && !document.querySelector('[data-testid=profile-chart]') && !document.querySelector('[data-testid=temporal-upload]')`), 'modes are mixed on one screen');
    const dates = await b.evaluate(`[...document.querySelectorAll('[data-testid=chart-counts] [data-testid=acq-point]')].map(e => e.dataset.date)`);
    ok(JSON.stringify(dates) === JSON.stringify(t.dates) && dates.length === 5, `dates ${dates}`);
    // counts per interval and type
    const segs = await b.evaluate(`[...document.querySelectorAll('[data-testid=chart-counts] [data-testid=seg]')].map(e => [e.dataset.interval, e.dataset.type, Number(e.dataset.n)])`);
    let marks = 0;
    for (const iv of t.intervals) for (const [ty, n] of Object.entries(iv.stored.by_type)) { if (!n) continue; marks++; const g = segs.find((x) => x[0] === iv.id && x[1] === ty); ok(g && g[2] === n, `${iv.id} ${ty}: chart ${g?.[2]} vs API ${n}`); }
    ok(segs.length === marks, `${segs.length} segments for ${marks} non-zero API cells`);
    const totals = await b.evaluate(`[...document.querySelectorAll('[data-testid=chart-counts] [data-testid=interval-bar]')].map(e => [e.dataset.interval, Number(e.dataset.total)])`);
    t.intervals.forEach((iv, i) => ok(totals[i][0] === iv.id && totals[i][1] === iv.stored.n, `total ${iv.id}`));
    // cumulative area
    const cum = await b.evaluate(`[...document.querySelectorAll('[data-testid=cum-point]')].map(e => [e.dataset.type, e.dataset.date, Number(e.dataset.areaM2), Number(e.dataset.n)])`);
    ok(cum.length === t.types.length * t.cumulative.length, `${cum.length} cumulative marks`);
    for (const [ty, dt, a, n] of cum) { const c = t.cumulative.find((x) => x.date === dt); ok(Math.abs((c.area_m2_by_type[ty] ?? 0) - a) < 0.5 && (c.n_by_type[ty] ?? 0) === n, `cumulative ${ty} ${dt}: ${a} vs ${c.area_m2_by_type[ty]}`); }
    // persistence + SAR
    const ps = await b.evaluate(`[...document.querySelectorAll('[data-testid=pseg]')].map(e => [e.dataset.interval, e.dataset.class, Number(e.dataset.n)])`);
    for (const iv of t.intervals) for (const [k, n] of Object.entries(iv.stored.persistence)) { if (k === 'none') continue; const g = ps.find((x) => x[0] === iv.id && x[1] === k); ok(g && g[2] === n, `persistence ${iv.id} ${k}`); }
    const held = await b.evaluate(`[...document.querySelectorAll('[data-testid=pers-bar]')].map(e => [Number(e.dataset.held), Number(e.dataset.demoted)])`);
    t.intervals.forEach((iv, i) => ok(held[i][0] === iv.stored.held && held[i][1] === iv.stored.demoted, `held/demoted ${iv.id}`));
    const sar = await b.evaluate(`[...document.querySelectorAll('[data-testid=sar-point]')].map(e => [e.dataset.interval, Number(e.dataset.n), e.dataset.covered])`);
    t.intervals.forEach((iv, i) => ok(sar[i][1] === iv.stored.n && Number(sar[i][2]) === iv.stored.sar_available, `sar ${iv.id}`));
    ok((await text('[data-testid=sar-note]')).includes(t.sar.note), 'SAR note from the API not shown');
    // real calendar axis: acquisition circle x positions are proportional to elapsed days
    const xs = await b.evaluate(`[...document.querySelectorAll('[data-testid=chart-counts] [data-testid=acq-point] circle')].map(c => Number(c.getAttribute('cx')))`);
    const day = (a, c) => (Date.parse(c) - Date.parse(a)) / 86400000;
    const gx = xs.slice(1).map((x, i) => x - xs[i]), gd = t.dates.slice(1).map((d, i) => day(t.dates[i], d));
    gx.forEach((g, i) => ok(Math.abs(g / gx[0] - gd[i] / gd[0]) < 0.01, `axis gap ${i}: ${g / gx[0]} vs ${gd[i] / gd[0]}`));
    ok(Math.abs(gx[1] / gx[2] - gd[1] / gd[2]) < 0.02 && gx[1] > gx[2] * 2.5, 'the 2021→2024 gap is not ~3x the 2024→2025 gap');
    const joined = await b.evaluate(`document.querySelectorAll('[data-testid=chart-counts] polyline, [data-testid=chart-counts] path, [data-testid=chart-cumulative] polyline, [data-testid=chart-cumulative] path, [data-testid=chart-sar] polyline, [data-testid=chart-sar] path').length`);
    ok(joined === 0, `${joined} line/path elements draw a series between observations`);
    ok((await text('[data-testid=cum-note]')).includes('not joined'), 'cumulative note');
    // not-localised rows
    const un = await b.evaluate(`[...document.querySelectorAll('[data-testid=unplaced-row]')].map(e => [e.dataset.id, Number(e.dataset.n)])`);
    for (const m of t.multi_interval) ok(un.some((u) => u[0] === m.id && u[1] === m.n), `unplaced ${m.id}`);
    ok(un.some((u) => u[0] === 'none' && u[1] === t.no_interval.n), 'unplaced none');
    await shot('33-temporal-archive');
    return `${marks} count segments, ${cum.length} cumulative marks, ${ps.length} persistence segments, ${sar.length} SAR marks all equal the API; axis gaps ∝ days; 0 joining lines`;
  });

  await step('temporal: clicking a mark opens exactly those candidates in Changes (count equals the API for the same filter)', async () => {
    const t = await api('/ui/temporal/archive');
    const iv = t.intervals[0], want = (await api(`/candidates?first_detected=${iv.id}&change_type=water_gain&limit=1`)).total;
    ok(want === iv.stored.by_type.water_gain && want > 0, `api ${want} vs stored ${iv.stored.by_type.water_gain}`);
    await b.click(`[data-testid=chart-counts] [data-testid=seg][data-interval="${iv.id}"][data-type=water_gain]`);
    await b.waitFor(`location.hash === '#/changes' && !!document.querySelector('[data-testid=changes-origin]')`, 10000, 'Changes with the filter chip');
    await b.waitFor(`document.querySelector('.queue-col')?.innerText.includes(' of ${want}')`, 10000, `queue total ${want}`);
    const chip = await text('[data-testid=changes-origin]');
    ok(chip.includes(iv.from) && chip.includes(iv.to), chip);
    const ids = await b.evaluate(`[...document.querySelectorAll('.queue-col tbody tr')].length`);
    ok(ids === Math.min(40, want), `${ids} rows`);
    // a cumulative mark opens the union of the intervals up to that date
    await go('temporal');
    await b.waitFor(`document.querySelectorAll('[data-testid=cum-point]').length > 0`, 15000, 'charts');
    const upTo = t.intervals.slice(0, 3).map((x) => x.id).join(','), want2 = (await api(`/candidates?first_detected=${upTo}&change_type=water_gain&limit=1`)).total;
    await b.click(`[data-testid=cum-point][data-type=water_gain][data-date="${t.intervals[2].to}"]`);
    await b.waitFor(`location.hash === '#/changes' && document.querySelector('.queue-col')?.innerText.includes(' of ${want2}')`, 10000, `cumulative queue ${want2}`);
    // the Clear button drops the Temporal filter
    await b.click('[data-testid=changes-origin] button');
    await b.waitFor(`!document.querySelector('[data-testid=changes-origin]')`, 5000, 'chip cleared');
    return `${want} candidates (first-detected ${iv.from} → ${iv.to}, water gain); cumulative mark → ${want2}`;
  });

  await step('temporal: pair-run basis is a separate, unclickable series; a sub-region shows its own counts and withholds the whole-area ones', async () => {
    await go('temporal');
    await b.waitFor(`document.querySelectorAll('[data-testid=chart-counts] [data-testid=seg]').length > 0`, 15000, 'charts');
    const t = await api('/ui/temporal/archive');
    await b.select('[data-testid=temporal-basis]', 'pair_run');
    await b.waitFor(`document.querySelector('[data-testid=chart-counts]').dataset.basis === 'pair_run'`, 8000, 'pair-run basis');
    const tot = await b.evaluate(`[...document.querySelectorAll('[data-testid=chart-counts] [data-testid=interval-bar]')].map(e => Number(e.dataset.total))`);
    t.intervals.forEach((iv, i) => ok(tot[i] === Object.values(iv.pair_run.by_type).reduce((a, c) => a + c, 0), `pair-run total ${iv.id}: ${tot[i]}`));
    ok(await b.evaluate(`document.querySelectorAll('[data-testid=chart-counts] [data-testid=seg][role=button]').length === 0 && document.querySelectorAll('[data-testid=unplaced]').length === 0`), 'pair-run bars are clickable or the stored-only strip is shown');
    ok(/cannot be opened in Changes/.test(await text('[data-testid=basis-note]')), 'basis note');
    await b.select('[data-testid=temporal-basis]', 'stored');
    await b.select('[data-testid=temporal-region]', 'kutch');
    await b.waitFor(`/0 stored candidates in scope/.test(document.querySelector('[data-testid=temporal-scope]')?.innerText || '')`, 10000, 'kutch scope');
    const k = await api(`/ui/temporal/archive?bbox=${(await api('/regions')).regions.find((r) => r.name === 'kutch').bbox.join(',')}`);
    ok(k.totals.stored_in_scope === 0 && k.intervals.every((iv) => iv.pair_run === null), 'API for kutch');
    ok(await b.evaluate(`document.querySelector('[data-testid=temporal-basis] option[value=pair_run]').disabled`), 'pair-run option should be disabled for a region');
    await b.select('[data-testid=temporal-region]', 'ayodhya');
    await b.waitFor(`document.querySelectorAll('[data-testid=chart-counts] [data-testid=seg]').length > 0`, 10000, 'ayodhya');
    return 'pair-run totals equal the report; Kutch → 0 stored, pair runs withheld';
  });

  // ---------------- Temporal: upload mode ----------------
  const uploadClean = `!document.querySelector('[data-testid=profile-chart]') && !document.querySelector('[data-testid=seg]') && !document.querySelector('[data-testid=chart-counts]')`;
  await step('temporal upload mode: with no file it refuses to render and says one date cannot produce a series', async () => {
    await b.nav('about:blank'); await sleep(200); await b.nav(BASE + '#/temporal/upload');
    await b.waitFor(`!!document.querySelector('[data-testid=upload-unavailable]')`, 30000, 'unavailable panel');
    const t = await text('[data-testid=upload-unavailable]');
    ok(/One date cannot produce a temporal series/.test(t) && /missing/.test(t), t.slice(0, 200));
    ok(await b.evaluate(uploadClean), 'a chart was drawn without data');
    return 'no chart, requirements listed';
  });

  await step('temporal upload mode: ONE file is refused (nothing is drawn), with the missing requirements named', async () => {
    if (!fx) return 'skipped';
    await drop([F('2019-03-30')]);
    await go('temporal/upload');
    await b.waitFor(`!!document.querySelector('[data-testid=upload-unavailable]')`, 15000, 'unavailable panel');
    const t = await text('[data-testid=upload-unavailable]');
    ok(/1 loaded/.test(t) && /Two or more GeoTIFFs/.test(t) && /different acquisition dates/.test(t), t.slice(0, 400));
    ok(await b.evaluate(uploadClean), 'a profile was drawn from a single file');
    const banner = await text('[data-testid=upload-banner]');
    for (const w of ['SPECTRAL PROFILE', 'not change detection', 'not run', 'not co-registered', 'not comparable across sensors']) ok(banner.toLowerCase().includes(w.toLowerCase()), `banner lacks "${w}"`);
    return 'single file: refused, banner states what this is not';
  });

  await step('temporal upload mode: a second file with the SAME date is refused; entering a different date for it enables the profile', async () => {
    if (!fx) return 'skipped';
    await drop([F('2019-03-30'), F('3band')], 2);
    await go('temporal/upload');
    await b.waitFor(`!!document.querySelector('[data-testid=upload-unavailable]')`, 15000, 'unavailable');
    ok(/dated|date/i.test(await text('[data-testid=upload-unavailable]')), 'no date requirement');
    const dateInput = `[data-testid=upload-files] input[type=date]`;
    await b.type(dateInput, '2019-03-30');
    await b.waitFor(`/1 distinct/.test(document.querySelector('[data-testid=upload-unavailable]')?.innerText || '')`, 8000, 'one distinct date');
    ok(await b.evaluate(uploadClean), 'profile drawn from two files with the same date');
    await b.type(dateInput, '2024-03-08');
    await b.waitFor(`!!document.querySelector('[data-testid=profile-chart]')`, 30000, 'profile after a second date');
    const note = await text('[data-testid=upload-files]');
    ok(/entered by you/.test(note), 'the analyst-entered date is not labelled as such');
    // bands are matched by role across a 4-band (B02,B03,B04,B08) and a 3-band (B04,B03,B02) file
    await b.select('[data-testid=profile-metric]', 'red');
    await b.waitFor(`document.querySelectorAll('[data-testid=profile-point]').length === 2`, 15000, 'two red points');
    return 'same date refused; analyst-entered date accepted and labelled; red band matched across 4-band and 3-band files';
  });

  await step('temporal upload mode: two dated overlapping files → a SPECTRAL PROFILE whose numbers equal the numpy/pyproj reference; no candidates, scores or pipeline output', async () => {
    if (!fx) return 'skipped';
    await drop([F('2019-03-30'), F('2024-03-08')], 2);
    await go('temporal/upload');
    await b.waitFor(`!!document.querySelector('[data-testid=profile-chart]')`, 40000, 'profile chart');
    await b.waitFor(`document.querySelectorAll('[data-testid=profile-table] tbody tr').length >= 12`, 20000, 'profile table');
    const opts = await b.evaluate(`[...document.querySelectorAll('[data-testid=profile-metric] option')].map(o => o.value)`);
    ok(['blue', 'green', 'red', 'nir', 'ndvi', 'ndwi'].every((k) => opts.includes(k)) && !opts.includes('ndbi'), `metrics ${opts}`);
    const prof = fx.ref.profile;
    const note = await text('[data-testid=profile-note]');
    ok(note.includes(`zoom ${prof.z}`) && note.includes(prof.cells.toLocaleString('en-US')), `grid / cell count in note: ${note.slice(0, 260)}`);
    let n = 0, worst = 0;
    for (const date of ['2019-03-30', '2024-03-08']) {
      const file = path.basename(fx.ref.files[date].path);
      for (const [metric, w] of Object.entries(prof.files[date])) {
        const cells = await b.evaluate(`(() => { const r = document.querySelector('[data-testid=profile-table] tr[data-metric="${metric}"][data-file="${file}"]'); return r ? [...r.children].map(c => c.innerText) : null; })()`);
        ok(cells, `no table row for ${file} ${metric}`);
        const [, , , nvalid, mean, std, p10, p50, p90] = cells;
        ok(nvalid === w.n.toLocaleString('en-US'), `${file} ${metric}: n ${nvalid} vs ${w.n}`);
        for (const [got, want, nm] of [[mean, w.mean, 'mean'], [std, w.std, 'std'], [p10, w.p10, 'p10'], [p50, w.p50, 'p50'], [p90, w.p90, 'p90']]) { ok(got === f4(want), `${file} ${metric} ${nm}: shows ${got}, numpy ${f4(want)}`); n++; }
      }
    }
    // the chart marks carry the same means; marks are not joined
    await b.select('[data-testid=profile-metric]', 'ndvi');
    await b.waitFor(`document.querySelector('[data-testid=profile-chart]').dataset.metric === 'ndvi'`, 5000, 'ndvi');
    const pts = await b.evaluate(`[...document.querySelectorAll('[data-testid=profile-point]')].map(e => [e.dataset.date, Number(e.dataset.mean), Number(e.dataset.n)])`);
    ok(pts.length === 2, `${pts.length} marks`);
    for (const [dt, mean] of pts) ok(Math.abs(mean - prof.files[dt].ndvi.mean) < 2e-6, `ndvi mark ${dt}: ${mean} vs ${prof.files[dt].ndvi.mean}`);
    ok(await b.evaluate(`document.querySelectorAll('[data-testid=profile-chart] polyline, [data-testid=profile-chart] path').length === 0`), 'the profile joins observations with a line');
    // everything except the banner (which says, in so many words, what is NOT done here)
    const all = await b.evaluate(`(() => { const c = document.querySelector('[data-testid=temporal-upload]').cloneNode(true); c.querySelector('[data-testid=upload-banner]').remove(); return c.innerText; })()`);
    ok(!/confiden|candidate|first detected|\bpersistent\b|\bprogressive\b|\bgate/i.test(all), 'pipeline-like output appears in upload mode: ' + (all.match(/confiden|candidate|first detected|persistent|progressive|gate/i) || [])[0]);
    ok(await b.evaluate(`!document.querySelector('[data-testid=seg]') && !document.querySelector('[data-testid=cum-point]')`), 'archive-mode marks present in upload mode');
    await shot('34-temporal-upload');
    return `${n} table statistics (n, mean, std, p10, p50, p90 × 6 metrics × 2 files) equal numpy/pyproj exactly as displayed; ndvi marks within 2e-6; no joining line`;
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
    // overlapping pins are grouped into one count badge that lists the member tiles and their card numbers
    const clusters = await b.evaluate(`[...document.querySelectorAll('${scope} .gm-pin.cluster')].map((p) => ({ ids: p.dataset.cluster.split(' '), nums: p.dataset.numbers.split(' '), count: +p.dataset.count, text: p.innerText }))`);
    ok(clusters.every((k) => k.count === k.ids.length && k.text === String(k.count) && k.ids.length >= 2), 'a cluster badge does not show its member count');
    ok(pins.length + clusters.reduce((n, k) => n + k.count, 0) === cards.length, `pins ${pins.length} + clustered ${clusters.reduce((n, k) => n + k.count, 0)} vs cards ${cards.length}`);
    for (const c of cards) {
      const one = pins.find((p) => p.tile === c.tile);
      const grp = clusters.find((k) => k.ids.includes(c.tile));
      ok(one ? one.text === String(c.n) && !grp : grp && grp.nums[grp.ids.indexOf(c.tile)] === String(c.n), `card #${c.n} is not shown as pin #${c.n} or inside a cluster that lists #${c.n}`);
    }
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
    await b.waitFor(`document.querySelectorAll('${scope} .gm-pin.hl').length === 1 && (() => { const h = document.querySelector('${scope} .gm-pin.hl'); return h.dataset.pin === ${JSON.stringify(first.tile)} || (h.dataset.cluster || '').split(' ').includes(${JSON.stringify(first.tile)}); })()`, 4000, 'card hover highlights its pin (or the cluster holding it)');
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
    const tiles = (await api(`/ui/tiles?ids=${cards.map((c) => c.tile).join(',')}`)).tiles;
    // the map refits after its data arrives: let that settle (poll up to 6 s) instead of reading the bounds in the middle of it
    let inside = 0;
    for (let i = 0; i < 24; i++) {
      const st = await mapState(scope);
      inside = tiles.filter((t) => { const cx = (t.bbox[0] + t.bbox[2]) / 2, cy = (t.bbox[1] + t.bbox[3]) / 2; return cx >= st.bounds[0] && cx <= st.bounds[2] && cy >= st.bounds[1] && cy <= st.bounds[3]; }).length;
      if (inside === tiles.length) break;
      await sleep(250);
    }
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
    // the zoom frames the middle 95% of the cluster's tiles (stray tiles far away do not stretch it): the tile-weighted median must be in view
    const wmed = (axis) => { const cs = [...geo.clusters[target].cells].sort((p, q) => p[axis] - q[axis]), tot = cs.reduce((a, c) => a + c[2], 0); let acc = 0; for (const c of cs) { acc += c[2]; if (acc >= tot / 2) return c[axis]; } };
    const cx = wmed(0), cy = wmed(1), clusterSpan = bb[2] - bb[0], viewSpan = st.bounds[2] - st.bounds[0];
    ok(st.bounds[0] <= cx && cx <= st.bounds[2] && st.bounds[1] <= cy && cy <= st.bounds[3], 'cluster centre not in view after click');
    ok(st.zoom > z0 && viewSpan <= Math.max(3 * clusterSpan, 0.7), `zoom ${z0} -> ${st.zoom}; view ${viewSpan.toFixed(2)} deg for a cluster ${clusterSpan.toFixed(2)} deg wide`);
    await shot('19-discovery-map');
    return `${msg}; ${checked} clusters: highlighted pixel colour = list swatch, others dimmed; click zooms (z${z0.toFixed(1)} -> z${st.zoom.toFixed(1)})`;
  });

  // ---- Discovery: the screen explains the clustering (computed reading, per-cluster summary, region x cluster matrix) ----
  await step('discovery: the interpretation line is counted from /discovery/clusters, not templated', async () => {
    await go('dashboard');
    await go('discovery');
    const cl = await api('/discovery/clusters');
    await b.waitFor(`!!document.querySelector('[data-testid=insight-counts]')`, 20000, 'interpretation');
    const share = (id) => { const r = cl.region_purity[id]; return Math.max(...Object.values(r.regions)) / cl.sizes[id]; };
    const ids = Object.keys(cl.sizes), conc = ids.filter((i) => share(i) >= 0.8).length;
    const line = await text('[data-testid=insight-counts]');
    ok(line.includes(`${cl.n_clusters} clusters over ${cl.n_tiles.toLocaleString('en-US')} tiles`), `counts line: ${line}`);
    ok(line.includes(`${conc} sit 80% or more inside a single region`) && line.includes(`${ids.length - conc} are spread across several regions`), `concentrated/spread vs API (${conc}/${ids.length - conc}): ${line}`);
    const fact = await text('[data-testid=insight-fact]'), kind = await b.evaluate(`document.querySelector('[data-testid=insight-fact]').dataset.kind`);
    const m = fact.match(/^(\d+) of (\d+) clusters whose closest concept is “(.+?)” each sit/);
    ok(m, `fact line: ${fact}`);
    const same = ids.filter((i) => cl.cluster_concepts[i][0][0] === m[3]), sameConc = same.filter((i) => share(i) >= 0.8);
    ok(+m[2] === same.length && +m[1] === sameConc.length, `"${m[3]}": text says ${m[1]} of ${m[2]}, API says ${sameConc.length} of ${same.length}`);
    const regs = new Set(sameConc.map((i) => cl.region_purity[i].dominant_region));
    ok(regs.size >= 2 && kind === 'concept-splits-by-region', `fact kind ${kind}, regions ${[...regs]}`);
    const chips = await b.evaluate(`[...document.querySelectorAll('[data-insight-cluster]')].map((c) => c.dataset.insightCluster)`);
    ok(chips.length === sameConc.length && chips.every((c) => sameConc.includes(c)), `chips ${chips} vs ${sameConc}`);
    await shot('20-discovery-insight');
    return `${cl.n_clusters} clusters, ${conc} concentrated / ${ids.length - conc} spread; fact: ${m[1]} of ${m[2]} "${m[3]}" clusters in ${regs.size} regions`;
  });

  await step('discovery: hover previews a cluster, click selects it, others dim (not hide), Esc clears; summary equals the API', async () => {
    await go('dashboard');
    await hoverAt(5, 5);
    await go('discovery');
    const cl = await api('/discovery/clusters'), geo = await api('/ui/clusters/geo');
    await b.waitFor(`document.querySelectorAll('tr[data-cluster]').length === ${cl.n_clusters} && +(document.querySelector('canvas.gm-cells')?.dataset.drawn || 0) > 0`, 40000, 'clusters drawn');
    const scope = 'section[aria-label="Cluster map"]';
    ok(await b.evaluate(`document.querySelector('[data-testid=cluster-summary]').dataset.cluster === ''`), 'summary is not empty by default');
    ok((await b.evaluate(`document.querySelector('${scope} .geomap').dataset.activeCluster`)) === '', 'default state dims something');
    // default: every cluster is legible in the list (a swatch per row), all drawn at the same strength
    ok((await count('tr[data-cluster] i[data-swatch]')) === cl.n_clusters, 'a cluster has no legend swatch');
    const id = Object.keys(cl.sizes).sort((x, y) => cl.sizes[y] - cl.sizes[x])[3], other = Object.keys(cl.sizes).find((o) => o !== id);
    const row = center(await rect(`tr[data-cluster="${id}"]`));
    await hoverAt(row.x, row.y);
    await b.waitFor(`document.querySelector('[data-testid=cluster-summary]').dataset.cluster === ${JSON.stringify(id)}`, 5000, 'hover -> summary');
    ok((await b.evaluate(`document.querySelector('[data-testid=cluster-summary]').dataset.pinned`)) === '0', 'a hover preview is marked as selected');
    const f = (n) => b.evaluate(`document.querySelector('[data-testid=cluster-summary] [data-field=${n}]')?.innerText || ''`);
    const rp = cl.region_purity[id], total = Object.values(cl.sizes).reduce((a, c) => a + c, 0);
    ok((await f('tiles')) === cl.sizes[id].toLocaleString('en-US'), `tiles ${await f('tiles')} vs ${cl.sizes[id]}`);
    ok((await f('share')) === `${((cl.sizes[id] / total) * 100).toFixed(1)}%`, `share ${await f('share')}`);
    ok((await f('concept')).startsWith(cl.cluster_concepts[id][0][0]), `concept ${await f('concept')}`);
    const spans = Object.values(rp.regions).filter((n) => n / cl.sizes[id] >= 0.05).length, touched = Object.keys(rp.regions).length;
    ok((await f('spans')).startsWith(`${spans} of `) && (await f('spans')).includes(`${touched} touched`), `regions spanned: ${await f('spans')} vs ${spans}/${touched}`);
    ok(/km/.test(await f('extent')), `extent ${await f('extent')}`);
    const mixTop = await b.evaluate(`document.querySelector('[data-testid=cluster-summary] .mixrow').dataset.region`);
    ok(mixTop === rp.dominant_region, `top region ${mixTop} vs ${rp.dominant_region}`);
    await b.waitFor(`(() => { const t = [...document.querySelectorAll('[data-testid=cluster-examples] img')]; return t.length > 0 && t.every((i) => i.complete && i.naturalWidth > 0); })()`, 20000, 'example thumbnails');
    const ex = await b.evaluate(`[...document.querySelectorAll('[data-testid=cluster-examples] figure')].map((e) => e.dataset.tile)`);
    const ge = geo.clusters[id].examples;
    ok(ex.length >= 3 && ex.length <= 4 && JSON.stringify(ex) === JSON.stringify(ge.map((e) => e.tile_id)), `examples ${ex.length} vs API ${ge.length}`);
    ok(ge.every((e) => rp.regions[e.region] > 0), 'an example comes from a region the cluster has no tiles in');
    await shot('21-discovery-hover');
    // moving away from the row clears the preview
    await hoverAt(5, 5);
    await b.waitFor(`document.querySelector('[data-testid=cluster-summary]').dataset.cluster === ''`, 5000, 'leaving clears the preview');
    // click -> selected: map zoomed to it, summary pinned, the rest dimmed
    const z0 = (await mapState(scope)).zoom;
    await b.clickAt(row.x, row.y); await sleep(900);
    ok((await b.evaluate(`document.querySelector('[data-testid=cluster-summary]').dataset.pinned`)) === '1', 'a click did not pin the summary');
    ok((await b.evaluate(`document.querySelector('${scope} .geomap').dataset.activeCluster`)) === id, 'map not on the selected cluster');
    ok((await mapState(scope)).zoom > z0, 'map did not zoom to the selected cluster');
    // dim, do not hide: a cell of another cluster is still drawn, at low alpha; the selected one is lit
    const owner = new Map(); for (const [c, g] of Object.entries(geo.clusters)) for (const [x, y] of g.cells) { const k = x + ',' + y; owner.set(k, owner.has(k) ? null : c); }
    const pixel = (lon, lat) => b.evaluate(`(() => { const g = document.querySelector('${scope} .geomap'), m = g.querySelector('.leaflet-container').__leaflet, c = g.querySelector('canvas.gm-cells'); const p = m.latLngToContainerPoint([${lat}, ${lon}]); const dpr = c.width / m.getSize().x; return [...c.getContext('2d').getImageData(Math.round(p.x * dpr), Math.round(p.y * dpr), 1, 1).data]; })()`);
    const mine = geo.clusters[id].cells.find(([x, y]) => owner.get(x + ',' + y) === id), theirs = geo.clusters[other].cells.find(([x, y]) => owner.get(x + ',' + y) === other);
    ok(mine && theirs, 'no cell owned by exactly one cluster');
    await centreOnMapIn(scope, mine[0], mine[1], 13); await sleep(500);
    const lit = await pixel(mine[0], mine[1]);
    await centreOnMapIn(scope, theirs[0], theirs[1], 13); await sleep(500);
    const dim = await pixel(theirs[0], theirs[1]);
    ok(lit[3] > 200, `selected cluster alpha ${lit[3]}`);
    ok(dim[3] > 5 && dim[3] < 80, `another cluster should be dimmed but present: alpha ${dim[3]}`);
    // hovering a different row previews it; leaving returns to the selection
    const orow = center(await rect(`tr[data-cluster="${other}"]`));
    await hoverAt(orow.x, orow.y);
    await b.waitFor(`document.querySelector('[data-testid=cluster-summary]').dataset.cluster === ${JSON.stringify(other)}`, 5000, 'preview another cluster');
    await hoverAt(5, 5);
    await b.waitFor(`document.querySelector('[data-testid=cluster-summary]').dataset.cluster === ${JSON.stringify(id)}`, 5000, 'back to the selection');
    await shot('22-discovery-selected');
    await b.key('Escape');
    await b.waitFor(`document.querySelector('${scope} .geomap').dataset.activeCluster === '' && document.querySelector('[data-testid=cluster-summary]').dataset.cluster === ''`, 5000, 'Esc clears');
    return `cluster ${id}: summary = API (tiles, share, concept, ${spans}/${touched} regions, ${ex.length} examples); hover previews, click pins+zooms (z${z0.toFixed(1)}->), others dimmed not hidden (alpha ${dim[3]}), Esc clears`;
  });

  await step('discovery: the region matrix agrees with the API and is linked; clicking a map cell selects its cluster', async () => {
    await go('dashboard');
    await hoverAt(5, 5);
    await go('discovery');
    const cl = await api('/discovery/clusters'), geo = await api('/ui/clusters/geo');
    await b.waitFor(`document.querySelectorAll('tr[data-cluster]').length === ${cl.n_clusters} && +(document.querySelector('canvas.gm-cells')?.dataset.drawn || 0) > 0`, 40000, 'clusters drawn');
    const scope = 'section[aria-label="Cluster map"]';
    ok((await count('tr[data-mx-cluster]')) === cl.n_clusters, 'matrix rows');
    const regs = [...new Set(Object.values(cl.region_purity).flatMap((r) => Object.keys(r.regions)))];
    ok((await count('th.mx-col')) === regs.length, `matrix columns ${await count('th.mx-col')} vs ${regs.length}`);
    const cells = await b.evaluate(`[...document.querySelectorAll('tr[data-mx-cluster]')].flatMap((r) => [...r.querySelectorAll('td')].map((t) => ({ c: r.dataset.mxCluster, g: t.dataset.region, n: +t.dataset.n, s: +t.dataset.share })))`);
    for (const c of cells) {
      const want = cl.region_purity[c.c].regions[c.g] ?? 0;
      ok(c.n === want && Math.abs(c.s - want / cl.sizes[c.c]) < 1e-3, `matrix ${c.c}/${c.g}: ${c.n} vs ${want}`);
    }
    for (const c of Object.keys(cl.sizes)) ok(Math.abs(cells.filter((x) => x.c === c).reduce((a, x) => a + x.s, 0) - 1) < 0.01, `row ${c} does not sum to 100%`);
    const groups = await b.evaluate(`[...document.querySelectorAll('.mx-group th')].map((t) => t.innerText.split(' ·')[0].trim())`);
    ok(new Set(groups).size === groups.length && groups.length === new Set(Object.values(cl.cluster_concepts).map((c) => c[0][0])).size, `concept groups ${groups.length}`);
    // hover a matrix row -> the map lights the cluster and the list row highlights
    const mid = Object.keys(cl.sizes)[2], mrow = center(await rect(`tr[data-mx-cluster="${mid}"]`));
    await hoverAt(mrow.x, mrow.y);
    await b.waitFor(`document.querySelector('${scope} .geomap').dataset.activeCluster === ${JSON.stringify(mid)} && !!document.querySelector('tr[data-cluster="${mid}"].hl')`, 5000, 'matrix hover -> map + list');
    await hoverAt(5, 5);
    // a map cell that only one cluster owns: hover previews, click selects that cluster
    const owner = new Map(); for (const [c, g] of Object.entries(geo.clusters)) for (const [x, y] of g.cells) { const k = x + ',' + y; owner.set(k, owner.has(k) ? null : c); }
    const pick = Object.keys(geo.clusters).find((c) => c !== mid && geo.clusters[c].cells.some(([x, y]) => owner.get(x + ',' + y) === c));
    const cell = geo.clusters[pick].cells.find(([x, y]) => owner.get(x + ',' + y) === pick);
    await b.evaluate(`document.querySelector('${scope}').scrollIntoView({ block: 'center' })`);
    await centreOnMapIn(scope, cell[0], cell[1], 13); await sleep(700);
    const pt = await b.evaluate(`(() => { const e = document.querySelector('${scope} .leaflet-container'), m = e.__leaflet, r = e.getBoundingClientRect(), p = m.latLngToContainerPoint([${cell[1]}, ${cell[0]}]); return { x: r.left + p.x, y: r.top + p.y }; })()`);
    await hoverAt(pt.x, pt.y);
    await b.waitFor(`document.querySelector('[data-testid=cluster-summary]').dataset.cluster === ${JSON.stringify(pick)}`, 5000, 'hovering a map cell previews its cluster');
    await b.clickAt(pt.x, pt.y);
    await b.waitFor(`document.querySelector('[data-testid=cluster-summary]').dataset.pinned === '1' && document.querySelector('[data-testid=cluster-summary]').dataset.cluster === ${JSON.stringify(pick)}`, 5000, 'map click selects');
    ok(await b.evaluate(`!!document.querySelector('tr[data-cluster="${pick}"].sel') && !!document.querySelector('tr[data-mx-cluster="${pick}"].sel')`), 'list and matrix rows do not show the selection');
    await b.click('[data-testid=show-all]'); await sleep(400);
    ok((await b.evaluate(`document.querySelector('${scope} .geomap').dataset.activeCluster`)) === '', 'Show all clusters did not clear');
    // the honest caption is still there
    const cap = await text(`${scope} [data-testid=map-caption]`);
    ok(/one representative acquisition per granule/.test(cap) && /unstaged areas shown dark/.test(cap), `caption: ${cap}`);
    await shot('23-discovery-matrix');
    return `matrix ${cl.n_clusters} x ${regs.length} cells = API counts, rows sum to 100%, ${groups.length} concept groups; map cell click selects ${pick}; caption intact`;
  });

  await step('shell: India flag by the clock (inline SVG, 3 bands, 24-spoke chakra), logo mark, wordmark animation + reduced motion', async () => {
    await go('dashboard');
    const sh = await b.evaluate(`(() => {
      const f = document.querySelector('[data-testid=india-flag]'), clock = document.querySelector('.clock'), fr = f.getBoundingClientRect(), cr = clock.getBoundingClientRect();
      const rects = [...f.querySelectorAll(':scope > rect')].map((r) => r.getAttribute('fill'));
      const g = f.querySelector('g'), spokes = g.querySelectorAll('line').length;
      const lg = document.querySelector('[data-testid=logo]'), svg = lg.querySelector('svg');
      const w = document.querySelector('[data-testid=wordmark]'), cs = getComputedStyle(w);
      const fixed = [...svg.querySelectorAll('[fill],[stroke]')].map((e) => e.getAttribute('fill') || e.getAttribute('stroke')).filter((v) => v && /^#/.test(v) && !/^#(fff|000)/i.test(v)).length;
      return { flag: { w: fr.width, h: fr.height, right: fr.right, clockLeft: cr.left, midY: fr.top + fr.height / 2, clockMidY: cr.top + cr.height / 2, viewBox: f.getAttribute('viewBox'), tag: f.tagName, rects, spokes, chakra: g.getAttribute('stroke'), imgs: document.querySelectorAll('header.topbar img').length },
        logo: { w: svg.getBoundingClientRect().width, h: svg.getBoundingClientRect().height, text: lg.innerText.trim(), fixed, role: svg.getAttribute('role') },
        word: { clip: cs.backgroundClip || cs.webkitBackgroundClip, anim: cs.animationName, dur: cs.animationDuration, iter: cs.animationIterationCount, img: cs.backgroundImage, text: w.textContent } };
    })()`);
    ok(sh.flag.tag === 'svg' && sh.flag.imgs === 0, 'the flag is not an inline SVG');
    ok(Math.abs(sh.flag.w / sh.flag.h - 1.5) < 0.02 && sh.flag.viewBox === '0 0 36 24', `flag proportions ${sh.flag.w}x${sh.flag.h}`);
    ok(sh.flag.rects.join() === '#FF9933,#FFFFFF,#138808', `bands ${sh.flag.rects}`);
    ok(sh.flag.spokes === 24 && sh.flag.chakra === '#000080', `chakra: ${sh.flag.spokes} spokes, ${sh.flag.chakra}`);
    ok(sh.flag.right <= sh.flag.clockLeft && sh.flag.clockLeft - sh.flag.right < 24 && Math.abs(sh.flag.midY - sh.flag.clockMidY) < 6, `flag is not beside the clock: gap ${sh.flag.clockLeft - sh.flag.right}`);
    ok(sh.flag.h >= 20 && sh.flag.h <= 32, `flag height ${sh.flag.h}`);
    ok(sh.logo.w >= 28 && sh.logo.h >= 28 && sh.logo.text === '' && sh.logo.role === 'img', `logo ${sh.logo.w}x${sh.logo.h} text "${sh.logo.text}"`);
    ok(sh.logo.fixed === 0, 'the logo uses a fixed colour: it must be one-ink (currentColor)');
    ok(/text/.test(sh.word.clip) && sh.word.anim === 'wordmark-drift' && sh.word.iter === 'infinite' && /rgb\(245, 154, 59\)/.test(sh.word.img), `wordmark ${JSON.stringify(sh.word)}`);
    const secs = parseFloat(sh.word.dur); ok(secs >= 20, `wordmark loop is ${secs}s - too fast for a permanent bar`);
    ok(sh.word.text === 'GeoSeek', 'wordmark text');
    // it really moves, slowly: the gradient offset advances over a few seconds
    const pos = () => b.evaluate(`getComputedStyle(document.querySelector('[data-testid=wordmark]')).backgroundPositionX`);
    const p0 = await pos(); await sleep(2500); const p1 = await pos();
    ok(p0 !== p1, `wordmark gradient does not move (${p0} -> ${p1})`);
    // reduced motion: the same gradient, not moving
    await b.send('Emulation.setEmulatedMedia', { features: [{ name: 'prefers-reduced-motion', value: 'reduce' }] }); await sleep(400);
    const rm = await b.evaluate(`(() => { const cs = getComputedStyle(document.querySelector('[data-testid=wordmark]')); return { anim: cs.animationName, pos: cs.backgroundPositionX, img: cs.backgroundImage }; })()`);
    await sleep(1200); const rm2 = await pos();
    await b.send('Emulation.setEmulatedMedia', { features: [{ name: 'prefers-reduced-motion', value: 'no-preference' }] });
    ok(rm.anim === 'none' && rm.pos === rm2 && /gradient/.test(rm.img), `reduced motion: ${JSON.stringify(rm)} / ${rm2}`);
    const ext = externalRequests(b.requests, origin);
    ok(ext.length === 0, 'external requests: ' + ext.map((r) => r.url).join(', '));
    return `flag ${Math.round(sh.flag.w)}x${Math.round(sh.flag.h)} inline SVG, bands ${sh.flag.rects.join('/')}, ${sh.flag.spokes} spokes; logo ${Math.round(sh.logo.w)}px one-ink; wordmark ${secs}s loop moves (${p0} -> ${p1}), still under reduced motion`;
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
    for (const r of ['dashboard', 'search', 'changes', 'pipeline', 'detect', 'discovery', 'fingerprints', 'settings', 'roadmap']) {
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
    return '9 routes clean; /roadmap no longer exists (falls back to the dashboard)';
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
    for (const r of ['dashboard', 'search', 'changes', 'pipeline', 'temporal', 'detect', 'discovery', 'fingerprints', 'settings']) {
      await go(r); await sleep(900);
      const t = await b.evaluate('document.body.innerText');
      for (const re of [/insar/i, /interferom/i, /fringe/i, /velocity/i, /completion/i, /thermal/i, /rate of change/i]) if (re.test(t)) bad.push(`${r}: ${re}`);
    }
    ok(bad.length === 0, bad.join('; '));
    return '9 routes clean';
  });


  // ---------------------------------------------------------------- "Why this was flagged": ONE control per screen, and it must reveal the explanation
  await step('why: the single "Why this was flagged" control reveals the explanation on the Dashboard and on Changes (content visible, not just a button)', async () => {
    const inView = (sel) => `(() => { const e = document.querySelector(${JSON.stringify(sel)}); const m = document.querySelector('main.main'); if (!e || !m) return false; const r = e.getBoundingClientRect(), v = m.getBoundingClientRect(); return r.width > 0 && r.height > 0 && getComputedStyle(e).visibility !== 'hidden' && r.top < v.bottom - 40 && r.bottom > v.top + 40; })()`;
    const whyButtons = `[...document.querySelectorAll('button')].filter((x) => x.innerText.startsWith('Why this was flagged')).length`;
    const resetScroll = `(() => { document.querySelector('main.main').scrollTop = 0; return true; })()`;
    // ---- Dashboard: the explanation is collapsed until the control is used
    await go('dashboard');
    await b.waitFor(`!!document.querySelector('[data-testid=why-open]')`, 30000, 'the control on the dashboard');
    await sleep(800);
    ok((await b.evaluate(whyButtons)) === 1, 'there must be exactly one "Why this was flagged" control on the dashboard');
    ok(!(await b.evaluate(`!!document.getElementById('why-section')`)), 'the explanation should start collapsed on the dashboard');
    ok((await b.evaluate(`document.querySelector('[data-testid=why-open]').getAttribute('aria-expanded')`)) === 'false', 'control should report aria-expanded=false');
    const dashId = await b.evaluate(`document.querySelector('.sel-head b.mono').innerText`);
    const exD = await api(`/ui/candidates/${dashId}/explain`);
    await b.evaluate(resetScroll);
    await b.click('[data-testid=why-open]');
    await b.waitFor(inView('[data-testid="why-lead"]'), 8000, 'dashboard: the lead sentence is on screen after the click');
    await b.waitFor(inView('[data-testid="trace-list"]'), 8000, 'dashboard: the gate trace is on screen after the click');
    ok((await text('[data-testid="why-lead"]')).trim() === exD.lead.headline, 'dashboard: the revealed sentence differs from /ui/candidates/{id}/explain');
    ok((await b.evaluate(`document.querySelectorAll('.ev-row[data-term]').length`)) === exD.evidence.terms.length, 'dashboard: evidence rows differ from the API');
    ok((await b.evaluate(`document.querySelector('[data-testid=why-open]').getAttribute('aria-expanded')`)) === 'true', 'control should report aria-expanded=true once open');
    ok((await b.evaluate(`document.activeElement && document.activeElement.id`)) === 'why-section', 'focus should move to the explanation');
    await shot('why-dashboard-open');
    await b.click('[data-testid=why-open]');
    await b.waitFor(`!document.getElementById('why-section') && document.querySelector('[data-testid=why-open]').getAttribute('aria-expanded') === 'false'`, 4000, 'dashboard: a second click collapses it');
    // ---- Changes: the explanation is always mounted below the workbench; the control must bring it into view
    await go(`changes/${dashId}`);
    await b.waitFor(`!!document.querySelector('[data-testid=why-open]') && !!document.querySelector('[data-testid="why-lead"]')`, 30000, 'the control and the explanation on changes');
    await sleep(800);
    ok((await b.evaluate(whyButtons)) === 1, 'there must be exactly one "Why this was flagged" control on changes');
    await b.evaluate(resetScroll);
    await sleep(300);
    ok(!(await b.evaluate(inView('[data-testid="why-lead"]'))), 'precondition: the explanation starts below the fold on changes');
    await b.click('[data-testid=why-open]');
    await b.waitFor(inView('[data-testid="why-lead"]'), 8000, 'changes: the lead sentence is on screen after the click');
    ok((await text('[data-testid="why-lead"]')).trim() === exD.lead.headline, 'changes: the sentence differs from the API');
    ok((await b.evaluate(`document.activeElement && document.activeElement.id`)) === 'why-section', 'changes: focus should move to the explanation');
    await shot('why-changes-scrolled');
    return `one control per screen; both reveal "${exD.lead.headline.slice(0, 50)}…" (${exD.evidence.terms.length} evidence rows, ${exD.trace.length} gate stages)`;
  });

  // ---------------------------------------------------------------- search map legibility
  await step('maps: search map outlines every catalogued region from the API, names them, lists them, and the caption states the count', async () => {
    const scope = 'section[aria-label="Map · spatial filter"]';
    await go('search');
    await b.waitFor(`!!document.querySelector('${scope} .geomap [data-testid=map-caption]')`, 30000, 'search map');
    await b.waitFor(captionSettled(scope), 15000, 'caption');
    await sleep(600);
    const regs = (await api('/regions')).regions;
    const outl = await b.evaluate(`(() => { const m = document.querySelector('${scope} .leaflet-container').__leaflet; const o = []; m.eachLayer((l) => { if (l.getBounds && l.options && l.options.weight === 1.8 && l.options.interactive) { const x = l.getBounds(); o.push([x.getWest(), x.getSouth(), x.getEast(), x.getNorth()]); } }); return o; })()`);
    ok(outl.length === regs.length, `${outl.length} outlines for ${regs.length} catalogued regions`);
    for (const r of regs) ok(outl.some((x) => x.every((v, i) => Math.abs(v - r.bbox[i]) < 1e-6)), `no outline equal to the catalog box of ${r.name}`);
    const lbl = await b.evaluate(`(() => { const g = document.querySelector('${scope} .geomap'); return { count: +g.dataset.regionCount, named: +g.dataset.regionLabels, names: [...g.querySelectorAll('.gm-reglbl')].map((e) => e.dataset.region) }; })()`);
    ok(lbl.count === regs.length, `data-region-count ${lbl.count} vs ${regs.length}`);
    ok(lbl.named >= 3 && lbl.names.length === lbl.named, `${lbl.names.length} names drawn, ${lbl.named} reported`);
    ok(lbl.names.every((n) => regs.some((r) => r.name === n)), 'a drawn name is not a catalogued region');
    ok(new Set(lbl.names).size === lbl.names.length, 'a region name is drawn twice');
    const chips = await b.evaluate(`[...document.querySelectorAll('[data-testid=map-regions] .chip')].map((c) => c.innerText)`);
    ok(chips.length === regs.length, `${chips.length} region chips for ${regs.length} regions`);
    const cap = await text(`${scope} [data-testid=map-caption]`);
    ok(cap.includes(`${regs.length} catalogued regions`) && /outside the archive, not a failed load/.test(cap), `caption does not state the count / meaning of the dark: "${cap}"`);
    ok(/one representative acquisition per granule/.test(cap) && /unstaged areas shown dark/.test(cap), 'the caption lost the imagery-selection statement');
    // a region chip frames that region, from the catalog box
    const k = regs.find((r) => r.name === 'kutch');
    await b.evaluate(`[...document.querySelectorAll('[data-testid=map-regions] .chip')].find((c) => c.innerText === 'Kutch').click()`);
    await sleep(700);
    const bd = await b.evaluate(`(() => { const x = document.querySelector('${scope} .leaflet-container').__leaflet.getBounds(); return [x.getWest(), x.getSouth(), x.getEast(), x.getNorth()]; })()`);
    ok(bd[0] <= k.bbox[0] && bd[1] <= k.bbox[1] && bd[2] >= k.bbox[2] && bd[3] >= k.bbox[3], `the Kutch chip did not frame its box: view ${bd.map((v) => v.toFixed(2))} vs ${k.bbox}`);
    await shot('search-map-regions');
    return `${outl.length} outlines = catalog boxes; ${lbl.named} names drawn, ${chips.length} chips; caption states ${regs.length} regions`;
  });

  await step('maps: a result far from the rest is left out of the default frame, named, and one click zooms out to all results (and back)', async () => {
    const scope = 'section[aria-label="Map · spatial filter"]';
    const Q = 'airport runway and parked aircraft';
    await go('search');
    await b.type('input[aria-label="Search query"]', Q);
    await b.click('button[type=submit]');
    await b.waitFor(`document.querySelectorAll('.rcard').length > 0`, 60000, 'results');
    await b.waitFor(`!!document.querySelector('[data-testid=map-frame-toggle]')`, 8000, 'the zoom-out control');
    const hits = (await api(`/search/text?q=${encodeURIComponent(Q)}&k=20`)).results;
    const far = hits.filter((r) => r.lon < -100);
    ok(far.length > 0 && far.length < hits.length, `fixture: expected the Los Angeles hits among India results (${far.length}/${hits.length})`);
    const view = () => b.evaluate(`(() => { const x = document.querySelector('${scope} .leaflet-container').__leaflet.getBounds(); return [x.getWest(), x.getSouth(), x.getEast(), x.getNorth()]; })()`);
    const inside = (v, r) => r.lon >= v[0] && r.lon <= v[2] && r.lat >= v[1] && r.lat <= v[3];
    await sleep(800);
    let v = await view();
    ok(far.every((r) => !inside(v, r)), 'the default frame still includes the far-off hits (it is framing an ocean)');
    ok(hits.filter((r) => r.lon >= -100).every((r) => inside(v, r)), 'the default frame misses a result of the main group');
    const warn = await text('[data-testid=map-outside]');
    ok(far.every((r) => warn.includes('#' + (hits.indexOf(r) + 1))) && /Zoom out to all 20 results/.test(warn), `warning "${warn}"`);
    await b.click('[data-testid=map-frame-toggle]');
    await sleep(900);
    v = await view();
    ok(hits.every((r) => inside(v, r)), `after "Zoom out to all results" a hit is still outside the view ${v.map((x) => x.toFixed(1))}`);
    ok(/Back to the main group/.test(await text('[data-testid=map-frame-toggle]')), 'the control did not offer the way back');
    await shot('search-map-all-results');
    await b.click('[data-testid=map-frame-toggle]');
    await sleep(900);
    v = await view();
    ok(far.every((r) => !inside(v, r)) && /Zoom out to all 20 results/.test(await text('[data-testid=map-frame-toggle]')), 'the way back did not restore the main-group frame');
    return `${far.length} Los Angeles hit(s) left out of the default frame; one click frames all ${hits.length}, one click returns`;
  });


  // ---------------------------------------------------------------- Dashboard "Findings by region": measured figures only
  await step('dashboard: findings by region - two labelled groups, no zero counts, and EVERY displayed figure equals /ui/metrics (cross-checked against /regions and the tile total)', async () => {
    await go('dashboard');
    await b.waitFor(`!!document.querySelector('[data-testid=regions-panel]') && document.querySelectorAll('[data-testid=regions-panel] [data-region]').length >= 2`, 30000, 'regions panel');
    await sleep(500);
    const m = await api('/ui/metrics');
    const regs = (await api('/regions')).regions;
    const num = (t) => Number(String(t).replace(/[^\d]/g, ''));
    const R = m.findings_by_region;
    ok(R.length === regs.length && R.every((r) => r.catalog), 'every region must carry catalog figures from the API');
    const dom = await b.evaluate(`(() => {
      const out = { heads: [], analysed: [], rest: [] };
      for (const g of document.querySelectorAll('[data-testid=regions-panel] section[data-group]')) {
        out.heads.push(g.querySelector('.reg-h').innerText.replace(/\\s+/g, ' ').trim());
        const key = g.dataset.group === 'analysed' ? 'analysed' : 'rest';
        for (const row of g.querySelectorAll('[data-region]')) {
          const f = (n) => row.querySelector('[data-field=' + n + ']')?.innerText ?? null;
          out[key].push({ region: row.dataset.region, candidates: f('candidates'), tiles: f('tiles'), obs: f('observations'), acq: f('acquisitions'), sensor: f('sensor'),
            first: row.querySelector('[data-first]')?.innerText ?? null, last: row.querySelector('[data-last]')?.innerText ?? null,
            types: [...row.querySelectorAll('[data-type]')].map((c) => [c.dataset.type, c.querySelector('[data-field=type-count]').innerText]), text: row.innerText.replace(/\\s+/g, ' ') });
        }
      }
      out.why = document.querySelector('[data-testid=regions-why]')?.innerText ?? null;
      out.caption = document.querySelector('[data-testid=regions-caption]')?.innerText ?? '';
      out.statSub = [...document.querySelectorAll('.stats')].map((x) => x.innerText).join(' ');
      return out;
    })()`);
    ok(/^CHANGE ANALYSIS COMPLETE\s*\d+$/i.test(dom.heads[0].replace(/\s+/g, ' ')) || /change analysis complete/i.test(dom.heads[0]), `first group header: ${dom.heads[0]}`);
    ok(/indexed and searchable · change analysis not yet run/i.test(dom.heads[1]), `second group header: ${dom.heads[1]}`);
    // membership: the groups are exactly the API's analysed / not-analysed split, every region once
    const wantA = R.filter((r) => r.catalog.analysed).map((r) => r.name).sort(), wantB = R.filter((r) => !r.catalog.analysed).map((r) => r.name).sort();
    ok(JSON.stringify(dom.analysed.map((x) => x.region).sort()) === JSON.stringify(wantA), `analysed group ${dom.analysed.map((x) => x.region)} vs API ${wantA}`);
    ok(JSON.stringify(dom.rest.map((x) => x.region).sort()) === JSON.stringify(wantB), `not-run group ${dom.rest.map((x) => x.region)} vs API ${wantB}`);
    ok(dom.analysed.length + dom.rest.length === regs.length, 'a region is missing or listed twice');
    // no zeros: a region that was never analysed shows NO finding count at all
    for (const x of dom.rest) ok(x.candidates === null && !/candidate/i.test(x.text), `un-analysed region ${x.region} shows a finding count: "${x.text}"`);
    ok(!dom.rest.some((x) => /(^| )0 (candidates|findings)/.test(x.text)), 'a zero finding count is displayed');
    // every displayed number equals the API's
    const fmtDate = (d) => d;
    for (const x of [...dom.analysed, ...dom.rest]) {
      const r = R.find((q) => q.name === x.region), c = r.catalog, p = c.sensors[0];
      ok(num(x.tiles) === c.n_tiles, `${x.region}: tiles shown ${x.tiles} vs API ${c.n_tiles}`);
      ok(num(x.obs) === r.n_observations, `${x.region}: observations shown ${x.obs} vs API ${r.n_observations}`);
      ok(num(x.acq) === p.n_acquisitions, `${x.region}: acquisitions shown ${x.acq} vs API ${p.n_acquisitions}`);
      ok(x.first === fmtDate(p.first_date) && (p.n_acquisitions > 1 ? x.last === p.last_date : x.last === null), `${x.region}: dates ${x.first} → ${x.last} vs API ${p.first_date} → ${p.last_date}`);
      ok(c.sensors.every((sn) => x.sensor.includes((sn.platform || 'unknown').replace(' Open Data Program', ''))), `${x.region}: sensor "${x.sensor}"`);
      // independent sources: /regions agrees on the observation count, and the catalog's date range brackets the stack's
      ok(regs.find((q) => q.name === x.region).n_observations === r.n_observations, `${x.region}: /regions disagrees on observations`);
      ok(c.first_date <= p.first_date && c.last_date >= p.last_date, `${x.region}: overall range does not bracket the sensor range`);
    }
    for (const x of dom.analysed) {
      const r = R.find((q) => q.name === x.region);
      ok(num(x.candidates) === r.candidates, `${x.region}: candidates shown ${x.candidates} vs API ${r.candidates}`);
      ok(JSON.stringify(x.types.map((t) => [t[0], num(t[1])]).sort()) === JSON.stringify(Object.entries(r.by_type).sort()), `${x.region}: breakdown ${JSON.stringify(x.types)} vs API ${JSON.stringify(r.by_type)}`);
      ok(x.types.reduce((n, t) => n + num(t[1]), 0) === r.candidates, `${x.region}: the breakdown does not add up to its candidates`);
      ok(r.candidates === (await api(`/candidates?limit=1&bbox=${r.bbox.join(',')}`)).total, `${x.region}: candidates differ from the /candidates bbox query`);
    }
    // the totals agree with the independent counters
    ok(R.reduce((n, r) => n + r.catalog.n_tiles, 0) === m.counters.tiles_indexed, `region tiles ${R.reduce((n, r) => n + r.catalog.n_tiles, 0)} vs tiles_indexed ${m.counters.tiles_indexed}`);
    ok(R.reduce((n, r) => n + r.candidates, 0) <= m.counters.change_candidates, 'region candidates exceed the candidate total');
    ok(m.counters.regions === regs.length, 'region counter vs /regions');
    // the "why" line: every number in it is recomputed here from the same catalog fields
    const A = R.filter((r) => r.catalog.analysed), O = R.filter((r) => !r.catalog.analysed);
    ok(dom.why, 'the explanation line is missing');
    for (const r of A) {
      const p = r.catalog.sensors[0];
      ok(dom.why.includes(`has ${p.n_acquisitions} ${(p.platform || '').replace(' Open Data Program', '')} acquisition${p.n_acquisitions === 1 ? '' : 's'} (${p.first_date} → ${p.last_date}`) && dom.why.includes(`${p.n_coregistered} of ${p.n_observations} with a co-registration record`), `why line misstates ${r.name}: ${dom.why}`);
    }
    const ps = O.map((r) => r.catalog.sensors[0]), rng = (v) => (Math.min(...v) === Math.max(...v) ? `${Math.min(...v)}` : `${Math.min(...v)}–${Math.max(...v)}`);
    const days = ps.map((p) => Math.round((Date.parse(p.last_date) - Date.parse(p.first_date)) / 86400000));
    ok(dom.why.includes(`The other ${O.length} regions hold ${rng(ps.map((p) => p.n_acquisitions))} acquisitions each, spanning ${rng(days)} days, with ${ps.reduce((n, p) => n + p.n_coregistered, 0)} of ${ps.reduce((n, p) => n + p.n_observations, 0)} observations carrying a co-registration record`), `why line numbers differ from the API: ${dom.why}`);
    // the caption still names the AOI and MGRS square, from the API
    ok(m.change_pipeline_aoi && dom.caption.includes(m.change_pipeline_aoi) && /MGRS/.test(dom.caption), `caption: ${dom.caption}`);
    ok(new RegExp(`${A.length}\\s*analysed for change`, 'i').test(dom.statSub.replace(/\s+/g, ' ')), 'the Regions stat card disagrees with the analysed count');
    await shot('dashboard-regions');
    return `${A.length} analysed (${A.map((r) => r.candidates).join('/')} candidates) + ${O.length} not run; ${R.length * 5} displayed figures match the API; tiles total ${m.counters.tiles_indexed}`;
  });

  // ---------------------------------------------------------------- Search map furniture
  await step('maps: search map carries a degree graticule with labels, a scale bar and a north indicator, all consistent with the view', async () => {
    const scope = 'section[aria-label="Map · spatial filter"]';
    await go('search');
    await b.waitFor(`!!document.querySelector('${scope} .gm-deg') && !!document.querySelector('${scope} .leaflet-control-scale-line')`, 30000, 'graticule labels and scale bar');
    await sleep(700);
    const st = await b.evaluate(`(() => {
      const g = document.querySelector('${scope} .geomap'), m = g.querySelector('.leaflet-container').__leaflet, bd = m.getBounds(), c = m.getCenter();
      const sc = g.querySelector('.leaflet-control-scale-line');
      return { bounds: [bd.getWest(), bd.getSouth(), bd.getEast(), bd.getNorth()], lat: c.lat, z: m.getZoom(), step: +g.dataset.graticuleStep, nLabels: +g.dataset.graticuleLabels,
        lats: [...g.querySelectorAll('.gm-deg.lat')].map((e) => e.innerText), lons: [...g.querySelectorAll('.gm-deg.lon')].map((e) => e.innerText),
        scaleText: sc.innerText, scalePx: parseFloat(sc.style.width), north: g.querySelector('.gm-north')?.getAttribute('aria-label') || null,
        arrow: !!g.querySelector('.gm-north-arrow'), lines: 0 };
    })()`);
    ok(st.north === 'North is up' && st.arrow, 'no north indicator');
    ok(st.nLabels > 0 && st.lats.length > 0 && st.lons.length > 0, `degree labels: ${st.lats.length} lat, ${st.lons.length} lon`);
    const val = (t) => (Number(t.replace(/°.*/, '')) * (/[SW]$/.test(t) ? -1 : 1));
    for (const t of st.lats) ok(/^\d+(\.\d+)?°[NS]$/.test(t) && val(t) >= st.bounds[1] - 1e-6 && val(t) <= st.bounds[3] + 1e-6 && Math.abs(Math.round(val(t) / st.step) * st.step - val(t)) < 1e-6, `latitude label "${t}" is outside the view or off the ${st.step}° grid`);
    for (const t of st.lons) ok(/^\d+(\.\d+)?°[EW]$/.test(t) && val(t) >= st.bounds[0] - 1e-6 && val(t) <= st.bounds[2] + 1e-6 && Math.abs(Math.round(val(t) / st.step) * st.step - val(t)) < 1e-6, `longitude label "${t}" is outside the view or off the ${st.step}° grid`);
    // the scale bar is physically right: the length it prints equals its pixel width x metres per pixel at the view centre (Web Mercator)
    const mm = /^(\d+(?:\.\d+)?)\s*(km|m)$/.exec(st.scaleText.trim());
    ok(mm, `scale text "${st.scaleText}"`);
    const metres = Number(mm[1]) * (mm[2] === 'km' ? 1000 : 1), mpp = (40075016.686 * Math.cos((st.lat * Math.PI) / 180)) / (256 * Math.pow(2, st.z));
    ok(Math.abs(st.scalePx * mpp - metres) / metres < 0.03, `scale bar ${st.scalePx}px x ${mpp.toFixed(1)} m/px = ${(st.scalePx * mpp).toFixed(0)} m, but it prints ${st.scaleText}`);
    // it follows the view: zoom in one level and the graticule step and scale text both change
    await b.evaluate(`document.querySelector('${scope} .leaflet-container').__leaflet.setView([27, 82], 8, { animate: false })`);
    await sleep(900);
    const st2 = await b.evaluate(`(() => { const g = document.querySelector('${scope} .geomap'); return { step: +g.dataset.graticuleStep, scale: g.querySelector('.leaflet-control-scale-line').innerText, lats: [...g.querySelectorAll('.gm-deg.lat')].map((e) => e.innerText) }; })()`);
    ok(st2.step < st.step && st2.scale !== st.scaleText && st2.lats.length > 0, `at z8 the graticule did not refine: ${JSON.stringify(st2)}`);
    await shot('search-map-cartography');
    return `grid ${st.step}° with ${st.lats.length}+${st.lons.length} labels; scale "${st.scaleText}" verified against Web Mercator; north indicator; refines to ${st2.step}° at z8`;
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
