// Before/after screenshots: the Dashboard candidate workbench (Why-this-was-flagged control) and the Search map.
// Read-only (never presses Confirm / Reject).   node tools/shots-why-map.mjs --base http://127.0.0.1:8001/react/ --out <dir> --tag before|after
import path from 'node:path';
import { mkdirSync } from 'node:fs';
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
const f = (n) => path.join(OUT, `${TAG}-${n}.png`);
try {
  await b.nav(BASE);
  await sleep(1500);
  await b.evaluate(`try{localStorage.setItem('geoseek.intro.off','1')}catch(e){}`);
  await b.nav(BASE); await sleep(1500);

  // ---- candidate workbench on the Dashboard
  await b.evaluate(`location.hash='#/dashboard'`);
  await b.waitFor(`[...document.querySelectorAll('button')].some(x=>x.innerText.startsWith('Why this was flagged'))`, 40000, 'why control');
  await sleep(2000);
  const wb = `(() => { const e=document.querySelector('.dash-triple'); e.scrollIntoView({block:'start'}); const r=e.getBoundingClientRect(); return {x:r.left,y:r.top,w:r.width,h:r.height}; })()`;
  await b.evaluate(wb); await sleep(500); await b.shot(f('workbench'));
  // click the control(s) with a real mouse event and capture the result
  const btn = `(() => { const e=[...document.querySelectorAll('button')].filter(x=>x.innerText.startsWith('Why this was flagged'))[0]; e.scrollIntoView({block:'center'}); const x=e.getBoundingClientRect(); return {x:x.left+x.width/2,y:x.top+x.height/2}; })()`;
  const p = await b.evaluate(btn); await b.clickAt(p.x, p.y); await sleep(1800);
  await b.shot(f('workbench-after-click'));

  // ---- Search map, one shot per query
  await b.evaluate(`location.hash='#/search'`);
  await b.waitFor(`!!document.querySelector('[data-testid=map-caption]')`, 30000, 'search map');
  await sleep(2500);
  await b.shot(f('search-empty'));
  let i = 0;
  for (const q of QUERIES) {
    i++;
    await b.type('input[aria-label="Search query"]', q);
    await b.click('button[type=submit]');
    await b.waitFor(`document.querySelectorAll('.rcard').length > 0`, 60000, 'results');
    await sleep(3500);
    await b.evaluate(`document.querySelector('.geomap').scrollIntoView({block:'start'})`);
    await sleep(500);
    await b.shot(f(`search-q${i}`));
    console.log('q' + i, q, JSON.stringify(await b.evaluate(`({pins: document.querySelectorAll('.gm-pin').length, outside: document.querySelector('[data-testid=map-outside]')?.innerText||null, caption: document.querySelector('[data-testid=map-caption]')?.innerText})`)));
  }
} finally { console.log('problems', JSON.stringify(b.problems.slice(0, 5))); b.close(); }
