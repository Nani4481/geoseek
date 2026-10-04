// Runtime offline audit. Drives headless Chrome over the DevTools protocol, loads the built React console from the
// running backend, visits every route, and records EVERY network request the page makes. Exit code 1 if any request
// left the page's own origin or a CSP block was raised. This is the runtime counterpart of the static URL scan in
// tests/test_frontend_offline.py.
//
//   node tools/verify-offline.mjs [--base http://127.0.0.1:8000/app/] [--routes a,b/c] [--shots DIR]
//                                 [--width 1600] [--height 1000] [--settle 2500] [--out report.json]
import { writeFileSync } from 'node:fs';
import path from 'node:path';
import { externalRequests, launch, sleep } from './cdp.mjs';

const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, all) => {
  if (a.startsWith('--')) acc.push([a.slice(2), all[i + 1] && !all[i + 1].startsWith('--') ? all[i + 1] : 'true']);
  return acc;
}, []));
const BASE = args.base || 'http://127.0.0.1:8000/app/';
const ROUTES = (args.routes || '').split(',').filter(Boolean);
const SETTLE = Number(args.settle || 2500);
const origin = new URL(BASE).origin;

const b = await launch({ width: Number(args.width || 1600), height: Number(args.height || 1000) });
const visited = [];
try {
  b.setTag('load');
  await b.nav(BASE);
  await sleep(SETTLE);
  if (args.shots) await b.shot(path.join(args.shots, '00-load.png'));
  for (const route of ROUTES) {
    b.setTag(route);
    await b.evaluate(`location.hash = '#/${route}'`);
    await sleep(SETTLE);
    visited.push({ route, text: ((await b.evaluate('document.body.innerText.slice(0,120)')) || '').replace(/\s+/g, ' ') });
    if (args.shots) await b.shot(path.join(args.shots, `route-${route.replace(/[^\w-]/g, '_')}.png`));
  }
} finally {
  const external = externalRequests(b.requests, origin);
  const csp = b.problems.filter((p) => /content security policy|csp|blocked/i.test(p.text));
  const report = {
    base: BASE, routes: ROUTES, total_requests: b.requests.length, external_requests: external, csp_or_blocked: csp,
    problems: b.problems, visited, checked_at: new Date().toISOString(),
  };
  if (args.out) writeFileSync(args.out, JSON.stringify(report, null, 2));
  console.log(`requests: ${report.total_requests} (EXTERNAL ${external.length}); CSP blocks: ${csp.length}; page problems: ${b.problems.length}`);
  for (const e of external) console.log('  EXTERNAL', e.type, e.url, '@', e.tag);
  for (const p of b.problems.slice(0, 12)) console.log('  problem', p.kind, '@', p.tag, '-', p.text);
  b.close();
  process.exitCode = external.length || csp.length ? 1 : 0;
}
