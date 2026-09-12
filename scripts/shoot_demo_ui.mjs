// Presentation layer - screenshot the overview, the collapsed / expanded
// candidate detail, and each step of the guided demo. Drives the system Chrome
// with Playwright (no browser download) and hard-fails on any off-origin
// request, exactly like scripts/shoot_analyst_ui.mjs. Usage:
//   node scripts/shoot_demo_ui.mjs [baseURL] [outDir]
import { createRequire } from "node:module";
import { execSync } from "node:child_process";
import { mkdirSync } from "node:fs";

const require = createRequire(import.meta.url);
process.env.NODE_PATH = execSync("npm root -g").toString().trim();
require("module").Module._initPaths();
const { chromium } = require("playwright");

const BASE = process.argv[2] || "http://127.0.0.1:8077";
const OUT = process.argv[3] || "data/change_model/ui_screens";
mkdirSync(OUT, { recursive: true });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const browser = await chromium.launch({ channel: "chrome", headless: true });
const ctx = await browser.newContext({ viewport: { width: 1500, height: 1000 }, deviceScaleFactor: 2 });
await ctx.route("**/*", (route) => {
  const u = new URL(route.request().url());
  if (u.origin !== new URL(BASE).origin) {
    console.error("BLOCKED off-origin request:", u.href);
    return route.abort();
  }
  return route.continue();
});
const page = await ctx.newPage();
page.on("console", (m) => { if (m.type() === "error") console.error("[console.error]", m.text()); });
page.on("pageerror", (e) => console.error("[pageerror]", e.message));

async function settleImages() {
  await page.evaluate(async () => {
    document.querySelectorAll("img[loading]").forEach((i) => i.removeAttribute("loading"));
    const H = document.body.scrollHeight;
    for (let y = 0; y <= H; y += 700) { window.scrollTo(0, y); await new Promise((r) => setTimeout(r, 60)); }
    window.scrollTo(0, 0);
  });
  await page.waitForFunction(() => [...document.images].every((i) => i.complete), { timeout: 20000 }).catch(() => {});
  await sleep(400);
}
async function shot(name, full = true) {
  await settleImages();
  await page.screenshot({ path: `${OUT}/${name}.png`, fullPage: full });
  console.log("saved", `${OUT}/${name}.png`);
}

// ---------------------------------------------------------------- 1. OVERVIEW
await page.goto(`${BASE}/app/#/overview`, { waitUntil: "networkidle" });
await page.waitForSelector("#ov-counters .stat");
await page.waitForSelector("#ov-featured .ff-card");
await page.waitForFunction(() => {
  const im = [...document.querySelectorAll("#ov-featured img")];
  return im.length >= 2 && im.every((x) => x.complete && x.naturalWidth > 0);
}, { timeout: 20000 });
await sleep(700);
await shot("08_overview");

// ---------------------------------------------------------------- 2. DETAIL (collapsed)
// open the top water-gain candidate straight from the API so this is deterministic
const waterId = await page.evaluate(async () => {
  const r = await fetch("/presentation/summary");
  return (await r.json()).demo.water_gain_candidate_id;
});
await page.goto(`${BASE}/app/#/detail/${waterId}`, { waitUntil: "networkidle" });
await page.waitForSelector("#d_body:not(.hidden)");
await page.waitForFunction(() => {
  const im = [...document.querySelectorAll("#d_imgs img")];
  return im.length === 3 && im.every((x) => x.complete && x.naturalWidth > 0);
}, { timeout: 20000 });
// disclosures must start collapsed
await page.waitForFunction(() =>
  [...document.querySelectorAll("#view-detail details.disclosure")].every((d) => !d.open));
await sleep(500);
await shot("09_candidate_detail_collapsed");

// ---------------------------------------------------------------- 3. DETAIL (expanded)
await page.click("#d_why > summary");
await page.click("#d_provwrap > summary");
await page.waitForSelector("#d_trace tr");
await page.waitForSelector("#d_prov .obs");
await sleep(500);
await shot("10_candidate_detail_expanded");

// ---------------------------------------------------------------- 4. GUIDED DEMO
await page.goto(`${BASE}/app/#/overview`, { waitUntil: "networkidle" });
await page.waitForSelector("#ov-counters .stat");
await page.click("#demoBtn");
await page.waitForSelector("#demo:not([hidden])");

const stepWaits = [
  async () => page.waitForFunction(() => {
    const i = [...document.querySelectorAll("#s_cards .card img")].slice(0, 6);
    return i.length && i.every((x) => x.complete);
  }, { timeout: 20000 }),
  async () => page.waitForFunction(() => {
    const i = [...document.querySelectorAll("#d_imgs img")];
    return i.length === 3 && i.every((x) => x.complete && x.naturalWidth > 0);
  }, { timeout: 20000 }),
  async () => page.waitForFunction(() => {
    const i = [...document.querySelectorAll("#disc_results .card img")].slice(0, 6);
    return i.length && i.every((x) => x.complete);
  }, { timeout: 20000 }),
  async () => page.waitForSelector("#ov-counters .stat"),
];

for (let s = 0; s < 4; s++) {
  await page.waitForFunction((n) => document.querySelector("#demo-step")
    && document.querySelector("#demo-step").textContent.includes(`Step ${n} /`), s + 1, { timeout: 15000 });
  await stepWaits[s]().catch((e) => console.warn(`step ${s + 1} wait:`, e.message));
  await sleep(900);
  await shot(`1${s + 1}_demo_step${s + 1}`);
  if (s < 3) {
    await page.click("#demo-next");
    await sleep(400);
  }
}
// exit cleanly - Esc must tear the overlay down
await page.keyboard.press("Escape");
await page.waitForSelector("#demo", { state: "hidden" });
console.log("demo overlay dismissed with Esc");

await browser.close();
console.log("DONE");
