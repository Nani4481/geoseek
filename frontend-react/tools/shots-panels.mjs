// Panel screenshots for the layout / basemap pass. Real headless Chrome, real backend; one PNG per panel (cropped to the
// element's box) plus one per screen. Read-only: it never presses Confirm / Reject.
//
//   node tools/shots-panels.mjs --base http://127.0.0.1:8001/react/ --out <dir>
import path from 'node:path';
import { writeFileSync, mkdirSync } from 'node:fs';
import { launch, sleep } from './cdp.mjs';

const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, all) => {
  if (a.startsWith('--')) acc.push([a.slice(2), all[i + 1] && !all[i + 1].startsWith('--') ? all[i + 1] : 'true']);
  return acc;
}, []));
const BASE = args.base || 'http://127.0.0.1:8001/react/';
const OUT = args.out || 'shots';
const W = Number(args.width || 1600), H = Number(args.height || 2600);
mkdirSync(OUT, { recursive: true });

const b = await launch({ width: W, height: H });
const done = [], failed = [];
const panel = (title) => `section[aria-label="${title}"]`;

async function clip(name, sel) {
  const r = await b.evaluate(`(() => { const e = document.querySelector(${JSON.stringify(sel)}); if (!e) return null; e.scrollIntoView({block:'start'}); const x = e.getBoundingClientRect(); return {x:x.left,y:x.top,w:x.width,h:x.height}; })()`);
  if (!r) { failed.push(`${name}: no element ${sel}`); return; }
  await sleep(250);
  const res = await b.send('Page.captureScreenshot', { format: 'png', clip: { x: r.x, y: r.y, width: Math.max(1, r.w), height: Math.max(1, r.h), scale: 1 } });
  writeFileSync(path.join(OUT, `${name}.png`), Buffer.from(res.result.data, 'base64'));
  done.push(name);
}
const go = async (route) => { await b.evaluate(`location.hash = '#/${route}'`); await sleep(500); };
const settle = (ms = 2500) => sleep(ms);

try {
  await b.nav(BASE);
  await b.waitFor(`document.querySelectorAll('.stat').length >= 7 && !document.querySelector('.stats .skeleton')`, 40000, 'dashboard');
  await b.waitFor(`document.querySelectorAll('.brow').length > 3`, 15000, 'breakdown');
  await b.waitFor(`document.querySelectorAll('.compare img').length >= 2`, 30000, 'compare').catch(() => {});
  await settle(3000);
  await b.shot(path.join(OUT, 'dashboard-screen.png'));
  await clip('dashboard-breakdown-row', '.dash-pair');
  await clip('dashboard-triple-row', '.dash-triple');

  await go('changes');
  await b.waitFor(`document.querySelectorAll('.tbl tbody tr').length > 3 && !!document.querySelector('.leaflet-container')`, 30000, 'changes');
  await b.waitFor(`document.querySelectorAll('.compare img').length >= 2`, 30000, 'compare').catch(() => {});
  await settle(3500);
  await b.shot(path.join(OUT, 'changes-screen.png'));
  await clip('changes-workbench-grid', '.wb-grid');
  await clip('changes-location-panel', panel('Location & threat rings'));

  await go('search');
  await b.waitFor(`!!document.querySelector('input[aria-label="Search query"]')`, 15000, 'search');
  await b.type('input[aria-label="Search query"]', 'open water reservoir');
  await b.click('button[type=submit]');
  await b.waitFor(`document.querySelectorAll('.rcard').length > 0`, 30000, 'results');
  await b.waitFor(`(() => { const l = [...document.querySelectorAll('.rcard img')]; return l.length > 0 && l.every(i => i.complete && i.naturalWidth > 0); })()`, 30000, 'thumbs').catch(() => {});
  await settle(2500);
  await b.shot(path.join(OUT, 'search-screen.png'));
  await clip('search-result-cards', '.results');
  await clip('search-map-panel', panel('Map · spatial filter'));

  await go('detect');
  await b.waitFor(`!!document.querySelector('.detect-stage img') && document.querySelectorAll('.leaflet-container').length > 0`, 40000, 'detect');
  await settle(4000);
  await b.shot(path.join(OUT, 'detect-screen.png'));
  await clip('detect-map-panel', panel('Detection map · threat rings'));

  await go('discovery');
  await b.waitFor(`!!document.querySelector('.disc-grid')`, 20000, 'discovery');
  await settle(3000);
  await b.shot(path.join(OUT, 'discovery-screen.png'));
  await clip('discovery-cluster-map-panel', panel('Cluster map'));

  await go('fingerprints');
  await b.waitFor(`document.querySelectorAll('.fp-tile').length > 3`, 40000, 'fingerprints');
  await b.waitFor(`(() => { const l = [...document.querySelectorAll('.fp-tile img')]; return l.length > 0 && l.every(i => i.complete && i.naturalWidth > 0); })()`, 30000, 'thumbs').catch(() => {});
  await settle(3000);
  await b.shot(path.join(OUT, 'fingerprints-screen.png'));
  await clip('fingerprints-grid', '.fp-grid');
  await clip('fingerprints-spatial-spread', panel('Spatial spread'));
} catch (e) {
  failed.push(String(e.message || e));
} finally {
  console.log(JSON.stringify({ out: OUT, done, failed, problems: b.problems.slice(0, 12) }, null, 1));
  b.close();
}
