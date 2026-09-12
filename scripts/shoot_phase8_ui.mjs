// Phase 8 - screenshot the terrain evidence panel + the Watch Areas view
// (Step B / Step C). Same Playwright-driving-system-Chrome harness as
// scripts/shoot_analyst_ui.mjs. Usage:
//   node scripts/shoot_phase8_ui.mjs [baseURL] [outDir]
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
const ctx = await browser.newContext({ viewport: { width: 1500, height: 1100 }, deviceScaleFactor: 2 });
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
async function shot(name) {
  await settleImages();
  await page.screenshot({ path: `${OUT}/${name}.png`, fullPage: true });
  console.log("saved", `${OUT}/${name}.png`);
}

// 1. CANDIDATE DETAIL - terrain evidence panel (Step B)
// pick a queue candidate with terrain (all of them have it) and open the
// "why did the system flag this" disclosure so the terrain table is visible.
await page.goto(`${BASE}/app/#/queue`, { waitUntil: "networkidle" });
await page.waitForSelector("#q_table tbody tr", { timeout: 15000 });
await page.click("#q_table tbody tr:first-child");
await page.waitForSelector("#d_body:not(.hidden)");
await page.evaluate(() =>
  document.querySelectorAll("#view-detail details.disclosure").forEach((d) => (d.open = true)));
await page.waitForSelector("#d_terrain tr", { timeout: 15000 });
await page.locator("#d_terrainwrap").scrollIntoViewIfNeeded();
await sleep(500);
await shot("08_candidate_detail_terrain_evidence");

// 2. WATCH AREAS (Step C) - list + the fired notification, linking to a candidate
await page.goto(`${BASE}/app/#/watch`, { waitUntil: "networkidle" });
await page.waitForSelector("#w_table tbody tr", { timeout: 15000 });
await page.waitForFunction(() => (document.querySelectorAll("#n_list > div").length > 0), { timeout: 15000 }).catch(() => {});
await sleep(500);
await shot("09_watch_areas_and_notification");

await browser.close();
console.log("DONE");
