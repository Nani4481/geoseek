// Screenshots of the explainability panel, pipeline trace and the suppression funnel. Real headless Chrome, real backend,
// read-only (it never presses Confirm / Reject).
//
//   node tools/shots-explain.mjs --base http://127.0.0.1:8001/app/ --out <dir> [--id <candidate id>]
import path from 'node:path';
import { writeFileSync, mkdirSync } from 'node:fs';
import { launch, sleep } from './cdp.mjs';

const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, all) => {
  if (a.startsWith('--')) acc.push([a.slice(2), all[i + 1] && !all[i + 1].startsWith('--') ? all[i + 1] : 'true']);
  return acc;
}, []));
const BASE = args.base || 'http://127.0.0.1:8001/app/';
const OUT = args.out || 'shots';
mkdirSync(OUT, { recursive: true });
const b = await launch({ width: 1600, height: 1400 });
const panel = (title) => `section[aria-label="${title}"]`;
async function clip(name, sel) {
  const r = await b.evaluate(`(() => { const e = document.querySelector(${JSON.stringify(sel)}); if (!e) return null; e.scrollIntoView({block:'start'}); const x = e.getBoundingClientRect(); return {x:x.left,y:x.top,w:x.width,h:x.height}; })()`);
  if (!r) { console.log('MISSING', name, sel); return; }
  await sleep(300);
  const res = await b.send('Page.captureScreenshot', { format: 'png', clip: { x: r.x, y: r.y, width: Math.max(1, r.w), height: Math.max(1, r.h), scale: 1 } });
  writeFileSync(path.join(OUT, `${name}.png`), Buffer.from(res.result.data, 'base64'));
  console.log('shot', name, Math.round(r.w), 'x', Math.round(r.h));
}
try {
  await b.nav(BASE);
  await sleep(1500);
  await b.evaluate(`location.hash = '#/changes${args.id ? '/' + args.id : ''}'`);
  await b.waitFor(`!!document.querySelector('[data-testid="why-lead"]') && !!document.querySelector('[data-testid="trace-list"]')`, 40000, 'why + trace');
  await sleep(1500);
  await clip('why-panel', panel('Why this was flagged'));
  await clip('trace-panel', panel('Pipeline trace · how it got through'));
  await b.click('[data-testid="why-raw-toggle"]');
  await sleep(300);
  await clip('why-panel-raw', panel('Why this was flagged'));
  await b.click('[data-testid="why-pixels"]');
  await b.waitFor(`document.querySelectorAll('.spec-img img.pix').length > 0 && !!document.querySelector('[data-testid="footprint-outline"]')`, 30000, 'index maps with the outline').catch((e) => console.log('pixels:', e.message));
  await sleep(2500);
  await clip('why-pixels', 'section[aria-label^="Index maps behind"]');
  await b.evaluate(`location.hash = '#/pipeline'`);
  await b.waitFor(`!!document.querySelector('[data-testid="funnel"]')`, 30000, 'funnel');
  await sleep(800);
  await b.shot(path.join(OUT, 'pipeline-screen.png'));
  await b.click('[data-testid="funnel-zoom"]');
  await sleep(400);
  await clip('pipeline-funnel-zoom', panel('False-alarm suppression · what was filtered out before you saw it'));
} catch (e) { console.log('FAILED', e.message); }
finally { await b.close(); }
