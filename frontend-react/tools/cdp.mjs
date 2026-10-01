// Minimal headless-Chrome driver over the DevTools protocol (Node's built-in WebSocket; no npm dependency).
// Shared by verify-offline.mjs (network audit) and e2e-tier1.mjs (interaction test).
import { spawn } from 'node:child_process';
import { existsSync, mkdirSync, mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';

export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function findChrome() {
  const cands = [process.env.CHROME_PATH, 'C:/Program Files/Google/Chrome/Application/chrome.exe',
    'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe', 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
    '/usr/bin/google-chrome', '/usr/bin/chromium', '/usr/bin/chromium-browser'].filter(Boolean);
  return cands.find((p) => existsSync(p));
}

export async function launch({ width = 1600, height = 1000 } = {}) {
  const chromePath = findChrome();
  if (!chromePath) throw new Error('no Chrome/Edge found; set CHROME_PATH');
  const port = 9300 + Math.floor(Math.random() * 600);
  const profile = mkdtempSync(path.join(tmpdir(), 'geoseek-cdp-'));
  const proc = spawn(chromePath, [
    '--headless=new', `--remote-debugging-port=${port}`, `--user-data-dir=${profile}`, '--no-first-run', '--disable-extensions',
    '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist', `--window-size=${width},${height}`, 'about:blank',
  ], { stdio: 'ignore' });

  let wsUrl = null;
  for (let i = 0; i < 80 && !wsUrl; i++) {
    try { wsUrl = (await (await fetch(`http://127.0.0.1:${port}/json`)).json()).find((p) => p.type === 'page')?.webSocketDebuggerUrl; } catch { /* starting */ }
    if (!wsUrl) await sleep(250);
  }
  if (!wsUrl) throw new Error('Chrome DevTools endpoint never came up');
  const ws = new WebSocket(wsUrl);
  await new Promise((r, j) => { ws.onopen = r; ws.onerror = j; });

  let nextId = 1, tag = 'boot';
  const pending = new Map(), requests = [], problems = [], downloads = [];
  ws.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); return; }
    const p = m.params;
    if (m.method === 'Network.requestWillBeSent') requests.push({ url: p.request.url, type: p.type, tag });
    else if (m.method === 'Runtime.exceptionThrown') problems.push({ kind: 'exception', tag, text: p.exceptionDetails.exception?.description || p.exceptionDetails.text });
    else if (m.method === 'Runtime.consoleAPICalled' && p.type === 'error') problems.push({ kind: 'console.error', tag, text: p.args.map((a) => a.value ?? a.description).join(' ').slice(0, 300) });
    else if (m.method === 'Log.entryAdded' && ['error', 'warning'].includes(p.entry.level)) problems.push({ kind: 'log:' + p.entry.level, tag, text: (p.entry.text + ' ' + (p.entry.url || '')).slice(0, 300) });
    else if (m.method === 'Network.loadingFailed' && p.errorText !== 'net::ERR_ABORTED') problems.push({ kind: 'loadingFailed', tag, text: `${p.errorText} ${p.blockedReason || ''}` });
    else if (m.method === 'Page.downloadWillBegin') downloads.push({ url: p.url, name: p.suggestedFilename, tag });
  };
  const send = (method, params = {}) => new Promise((res) => { const id = nextId++; pending.set(id, res); ws.send(JSON.stringify({ id, method, params })); });
  for (const d of ['Page', 'Network', 'Runtime', 'Log']) await send(`${d}.enable`);
  await send('Emulation.setDeviceMetricsOverride', { width, height, deviceScaleFactor: 1, mobile: false });
  await send('Network.setCacheDisabled', { cacheDisabled: true });
  await send('Browser.setDownloadBehavior', { behavior: 'deny' }).catch(() => {});

  const evaluate = async (expression) => {
    const r = await send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true });
    if (r.result?.exceptionDetails) throw new Error('eval failed: ' + (r.result.exceptionDetails.exception?.description || r.result.exceptionDetails.text));
    return r.result?.result?.value;
  };
  const waitFor = async (expr, timeout = 15000, label = expr) => {
    const t0 = Date.now();
    while (Date.now() - t0 < timeout) { try { const v = await evaluate(expr); if (v) return v; } catch { /* retry */ } await sleep(150); }
    throw new Error(`timed out waiting for: ${label}`);
  };
  // real input events (not synthetic DOM events): pointer capture, Leaflet and canvas handlers all see them
  const rectOf = async (sel, nth = 0) => {
    const r = await evaluate(`(() => { const e = document.querySelectorAll(${JSON.stringify(sel)})[${nth}]; if (!e) return null; e.scrollIntoView({block:'center'}); const b = e.getBoundingClientRect(); return {x:b.left,y:b.top,w:b.width,h:b.height}; })()`);
    if (!r) throw new Error(`no element for ${sel}[${nth}]`);
    return r;
  };
  const mouse = (type, x, y, extra = {}) => send('Input.dispatchMouseEvent', { type, x, y, button: 'left', buttons: type === 'mouseReleased' ? 0 : 1, clickCount: 1, ...extra });
  const clickAt = async (x, y) => { await send('Input.dispatchMouseEvent', { type: 'mouseMoved', x, y }); await mouse('mousePressed', x, y); await mouse('mouseReleased', x, y); };
  const click = async (sel, nth = 0) => { const r = await rectOf(sel, nth); await clickAt(r.x + r.w / 2, r.y + r.h / 2); };
  const drag = async (x0, y0, x1, y1, steps = 8) => {
    await send('Input.dispatchMouseEvent', { type: 'mouseMoved', x: x0, y: y0 });
    await mouse('mousePressed', x0, y0);
    for (let i = 1; i <= steps; i++) await send('Input.dispatchMouseEvent', { type: 'mouseMoved', x: x0 + ((x1 - x0) * i) / steps, y: y0 + ((y1 - y0) * i) / steps, button: 'left', buttons: 1 });
    await mouse('mouseReleased', x1, y1);
  };
  const type = async (sel, text) => {
    await click(sel);
    await evaluate(`(() => { const e = document.querySelector(${JSON.stringify(sel)}); const set = Object.getOwnPropertyDescriptor(Object.getPrototypeOf(e), 'value').set; set.call(e, ${JSON.stringify(text)}); e.dispatchEvent(new Event('input', {bubbles:true})); })()`);
  };
  const select = (sel, value) => evaluate(`(() => { const e = document.querySelector(${JSON.stringify(sel)}); const set = Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value').set; set.call(e, ${JSON.stringify(value)}); e.dispatchEvent(new Event('change', {bubbles:true})); })()`);
  const key = async (k) => { await send('Input.dispatchKeyEvent', { type: 'keyDown', key: k, code: k }); await send('Input.dispatchKeyEvent', { type: 'keyUp', key: k, code: k }); };
  const shot = async (file) => { mkdirSync(path.dirname(file), { recursive: true }); const r = await send('Page.captureScreenshot', { format: 'png' }); writeFileSync(file, Buffer.from(r.result.data, 'base64')); };
  const nav = async (url) => { await send('Page.navigate', { url }); };

  return {
    send, evaluate, waitFor, sleep, rectOf, mouse, clickAt, click, drag, type, select, key, shot, nav,
    requests, problems, downloads, setTag: (t) => { tag = t; },
    close: () => { try { ws.close(); } catch { /* */ } proc.kill(); },
  };
}

export function externalRequests(requests, origin) {
  return requests.filter((r) => !(r.url.startsWith(origin) || r.url.startsWith('data:') || r.url.startsWith('blob:') || r.url.startsWith('about:')));
}
