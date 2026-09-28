// Screenshot the 7 analyst-UI screens (post-rebuild) with Playwright driving
// the system Chrome (no browser download). Usage:
//   node scripts/shoot_analyst_ui.mjs [baseURL] [outDir]
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
// hard-fail if the page reaches for anything off-origin
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
  });
  await page.waitForFunction(() => [...document.images].every((i) => i.complete), { timeout: 20000 }).catch(() => {});
  await sleep(400);
}

async function shot(name) {
  await settleImages();
  await page.screenshot({ path: `${OUT}/${name}.png` });
  console.log("saved", `${OUT}/${name}.png`);
}

// 1. OVERVIEW
await page.goto(`${BASE}/app/#/overview`, { waitUntil: "networkidle" });
await page.waitForSelector(".ov-inspector-list", { timeout: 15000 });
await sleep(800);
await shot("01_overview");

// 2. REVIEW QUEUE
await page.goto(`${BASE}/app/#/queue`, { waitUntil: "networkidle" });
await page.waitForSelector(".rq-row", { timeout: 15000 });
await sleep(500);
await shot("02_review_queue");

// 3. CANDIDATE DETAIL - open the first queue row
await page.click(".rq-row:first-child");
await page.waitForSelector("#screen-candidate .compare-surface img", { timeout: 15000 });
await page.waitForFunction(() => {
  const imgs = [...document.querySelectorAll("#screen-candidate .compare-surface img")];
  return imgs.length === 2 && imgs.every((i) => i.complete && i.naturalWidth > 0);
}, { timeout: 20000 });
await sleep(600);
await shot("03_candidate_detail");

// record a confirm decision, screenshot the decided state
await page.click("#cd-confirm");
await page.waitForSelector(".tinted-box.agree, .tinted-box.disagree", { timeout: 10000 });
await sleep(400);
await shot("04_candidate_detail_confirmed");

// reopen it (tests the new decision value end to end)
await page.click("#cd-reopen");
await page.waitForSelector("#cd-confirm", { timeout: 10000 });
await sleep(400);
await shot("05_candidate_detail_reopened");

// 4. SEARCH
await page.goto(`${BASE}/app/#/search`, { waitUntil: "networkidle" });
await page.fill("#se-input", "a river with sandbars");
await page.click("#se-go");
await page.waitForSelector(".se-card", { timeout: 15000 });
await sleep(500);
await shot("06_search");

// 5. OBJECT DETECTION
await page.goto(`${BASE}/app/#/detect`, { waitUntil: "networkidle" });
await page.waitForSelector("#screen-detect .compare-surface", { timeout: 15000 });
await sleep(1200);
await shot("07_object_detection");

// 6. DISCOVERY
await page.goto(`${BASE}/app/#/discovery`, { waitUntil: "networkidle" });
await page.waitForSelector(".ds-tile-grid", { timeout: 15000 });
await sleep(800);
await shot("08_discovery");

// 7. WATCH AREAS
await page.goto(`${BASE}/app/#/watch`, { waitUntil: "networkidle" });
await page.waitForSelector(".wa-form", { timeout: 15000 });
await sleep(500);
await shot("09_watch_areas");

await browser.close();
console.log("done");
