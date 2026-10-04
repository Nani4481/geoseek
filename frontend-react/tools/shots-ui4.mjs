// Before/after screenshots of the Discovery and Data screens (and the top bar). Real headless Chrome, real backend, read-only.
//
//   node tools/shots-ui4.mjs --base http://127.0.0.1:8001/app/ --out <dir> [--select <cluster id>]
import path from 'node:path';
import { writeFileSync } from 'node:fs';
import { launch, sleep } from './cdp.mjs';

const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, all) => {
  if (a.startsWith('--')) acc.push([a.slice(2), all[i + 1] && !all[i + 1].startsWith('--') ? all[i + 1] : 'true']);
  return acc;
}, []));
const BASE = args.base || 'http://127.0.0.1:8001/app/';
const OUT = args.out || 'shots';
const b = await launch({ width: 1600, height: 1900 });
try {
  await b.nav(BASE);
  await sleep(1500);
  await b.evaluate(`location.hash = '#/data'`);
  await b.waitFor(`document.querySelectorAll('table[aria-label="Collections"] tbody tr').length > 0 && document.querySelectorAll('table[aria-label="Regions"] tbody tr').length > 0`, 30000, 'inventory tables');
  await sleep(800);
  await b.shot(path.join(OUT, 'data.png'));
  await b.evaluate(`location.hash = '#/discovery'`);
  await b.waitFor(`document.querySelectorAll('tr[data-cluster]').length > 0 && Number(document.querySelector('canvas.gm-cells')?.dataset.drawn) > 0`, 60000, 'cluster map drawn');
  await sleep(6000);                                    // basemap tiles
  await b.shot(path.join(OUT, 'discovery.png'));
  if (args.select) {
    await b.click(`tr[data-cluster="${args.select}"]`);
    await b.waitFor(`document.querySelectorAll('.csum-tile img').length === 0 || [...document.querySelectorAll('.csum-tile img')].every((i) => i.complete && i.naturalWidth > 0)`, 20000, 'example thumbnails').catch(() => {});
    await sleep(6000);
    await b.shot(path.join(OUT, `discovery-selected-${args.select}.png`));
  }
  // the top bar at 3x: flag, clock, logo and wordmark
  const bar = await b.evaluate(`(() => { const r = document.querySelector('header.topbar').getBoundingClientRect(); const l = document.querySelector('.rail .logo').getBoundingClientRect(); return { x: 0, y: 0, w: innerWidth, h: r.height, logo: { x: l.left, y: l.top, w: l.width, h: l.height } }; })()`);
  for (const [name, c] of [['topbar', { x: bar.x, y: bar.y, w: bar.w, h: bar.h }], ['logo', { x: bar.logo.x - 6, y: bar.logo.y - 6, w: bar.logo.w + 12, h: bar.logo.h + 12 }]]) {
    const res = await b.send('Page.captureScreenshot', { format: 'png', clip: { x: c.x, y: c.y, width: c.w, height: c.h, scale: 3 } });
    writeFileSync(path.join(OUT, `${name}.png`), Buffer.from(res.result.data, 'base64'));
  }
  console.log('problems:', JSON.stringify(b.problems.filter((p) => !/404/.test(p.text)).slice(0, 5)));
} catch (e) { console.log('FAILED', e.message); }
finally { b.close(); }
