// Before/after screenshots of the Dashboard "Findings by region" panel and the Search "Map · spatial filter" panel.
// Real headless Chrome against a running backend; read-only (never presses Confirm / Reject).
//   node tools/shots-regions-map.mjs --base http://127.0.0.1:8001/react/ --out <dir> --tag before|after
import path from 'node:path';
import { writeFileSync, mkdirSync } from 'node:fs';
import { launch, sleep } from './cdp.mjs';

const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, all) => {
  if (a.startsWith('--')) acc.push([a.slice(2), all[i + 1] && !all[i + 1].startsWith('--') ? all[i + 1] : 'true']);
  return acc;
}, []));
const BASE = args.base || 'http://127.0.0.1:8001/react/';
const OUT = args.out || 'shots';
const TAG = args.tag || 'shot';
const QUERIES = (args.queries || 'an open water reservoir or pond|airport runway and parked aircraft').split('|');
mkdirSync(OUT, { recursive: true });
const b = await launch({ width: 1600, height: 1100 });
async function clip(name, sel) {
  const r = await b.evaluate(`(() => { const e = document.querySelector(${JSON.stringify(sel)}); if (!e) return null; e.scrollIntoView({ block: 'start' }); const x = e.getBoundingClientRect(); return { x: x.left, y: x.top, w: x.width, h: x.height }; })()`);
  if (!r) { console.log('MISSING', name, sel); return; }
  await sleep(500);
  const res = await b.send('Page.captureScreenshot', { format: 'png', clip: { x: r.x, y: r.y, width: Math.max(1, r.w), height: Math.max(1, r.h), scale: 1 } });
  writeFileSync(path.join(OUT, `${TAG}-${name}.png`), Buffer.from(res.result.data, 'base64'));
  console.log('shot', name, Math.round(r.w), 'x', Math.round(r.h));
}
try {
  await b.nav(BASE);
  await sleep(1500);
  await b.evaluate(`try{localStorage.setItem('geoseek.intro.off','1')}catch(e){}`);
  await b.nav(BASE); await sleep(1500);
  await b.evaluate(`location.hash='#/dashboard'`);
  await b.waitFor(`!!document.querySelector('section[aria-label="Findings by region"]') && document.querySelectorAll('section[aria-label="Findings by region"] .brow, section[aria-label="Findings by region"] .reg-row').length > 3`, 40000, 'regions panel');
  await sleep(1500);
  await clip('regions-panel', 'section[aria-label="Findings by region"]');
  await clip('regions-pair', '.dash-pair');
  await b.evaluate(`location.hash='#/search'`);
  await b.waitFor(`!!document.querySelector('[data-testid=map-caption]')`, 30000, 'search map');
  await sleep(2500);
  await clip('map-empty', 'section[aria-label="Map · spatial filter"]');
  let i = 0;
  for (const q of QUERIES) {
    i++;
    await b.type('input[aria-label="Search query"]', q);
    await b.click('button[type=submit]');
    await b.waitFor(`document.querySelectorAll('.rcard').length > 0`, 60000, 'results');
    await sleep(3500);
    await clip(`map-q${i}`, 'section[aria-label="Map · spatial filter"]');
  }
} finally { console.log('problems', b.problems.filter((p) => !/404.*thumbnail/.test(p.text)).length); b.close(); }
