// Screenshots of the intro, the animated rail logo, the Data screen with a real raster dropped on it, and both Temporal modes.
// Real headless Chrome, real backend, read-only. Needs the fixtures from `python tests/raster_fixtures.py <dir>`.
//
//   node tools/shots-temporal.mjs --base http://127.0.0.1:8001/app/ --out <dir> --fixtures <dir>
import path from 'node:path';
import { launch, sleep } from './cdp.mjs';

const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, all) => {
  if (a.startsWith('--')) acc.push([a.slice(2), all[i + 1] && !all[i + 1].startsWith('--') ? all[i + 1] : 'true']);
  return acc;
}, []));
const BASE = args.base || 'http://127.0.0.1:8001/app/';
const OUT = args.out || 'shots';
const FX = args.fixtures;
const b = await launch({ width: 1500, height: 1000 });
try {
  await b.nav(BASE);
  await sleep(700); await b.shot(path.join(OUT, 'intro-0.7s.png'));
  await sleep(700); await b.shot(path.join(OUT, 'intro-1.4s.png'));
  await sleep(800); await b.shot(path.join(OUT, 'intro-2.2s.png'));
  await sleep(1600);
  await b.evaluate(`document.querySelector('[data-testid=intro-never]')?.click(); true`);
  await b.waitFor(`document.querySelectorAll('.stat').length >= 7`, 30000, 'dashboard');
  await b.shot(path.join(OUT, 'dashboard.png'));

  await b.evaluate(`location.hash = '#/temporal'`);
  await b.waitFor(`document.querySelectorAll('[data-testid=seg]').length > 0 && document.querySelectorAll('[data-testid=cum-point]').length > 0`, 30000, 'archive charts');
  await sleep(500);
  await b.send('Emulation.setDeviceMetricsOverride', { width: 1500, height: 2600, deviceScaleFactor: 1, mobile: false });
  await sleep(500); await b.shot(path.join(OUT, 'temporal-archive.png'));
  await b.send('Emulation.setDeviceMetricsOverride', { width: 1500, height: 1000, deviceScaleFactor: 1, mobile: false });

  if (FX) {
    await b.evaluate(`location.hash = '#/data'`);
    await b.waitFor(`!!document.querySelector('[data-testid=raster-input]')`, 15000, 'input');
    await b.setFiles('[data-testid=raster-input]', [path.join(FX, 'ayodhya_2019-03-30_4band.tif'), path.join(FX, 'ayodhya_2024-03-08_4band.tif')]);
    await b.waitFor(`document.querySelectorAll('[data-testid=preview-canvas]').length === 2`, 30000, 'previews');
    await sleep(3000);
    await b.send('Emulation.setDeviceMetricsOverride', { width: 1500, height: 2400, deviceScaleFactor: 1, mobile: false });
    await sleep(600); await b.shot(path.join(OUT, 'data-preview.png'));
    await b.evaluate(`document.querySelector('[data-testid=compare-run]').click(); true`);
    await b.waitFor(`!!document.querySelector('[data-testid=compare-grid]')`, 60000, 'compare');
    await sleep(1500); await b.shot(path.join(OUT, 'data-compare.png'));
    await b.send('Emulation.setDeviceMetricsOverride', { width: 1500, height: 1000, deviceScaleFactor: 1, mobile: false });
    await b.evaluate(`location.hash = '#/temporal/upload'`);
    await b.waitFor(`!!document.querySelector('[data-testid=profile-chart]')`, 30000, 'profile');
    await sleep(600); await b.shot(path.join(OUT, 'temporal-upload.png'));
  }
  console.log('problems:', JSON.stringify(b.problems.filter((p) => !/404/.test(p.text)).slice(0, 8)));
} catch (e) { console.log('FAILED', e.message); await b.shot(path.join(OUT, 'FAILED.png')).catch(() => {}); }
finally { b.close(); }
