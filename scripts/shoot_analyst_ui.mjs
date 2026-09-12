// Phase 6 - screenshot the four analyst views with Playwright driving the
// system Chrome (no browser download). Usage:
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
  // force lazy <img> to load by scrolling the whole page, then wait for decode
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

// 1. QUEUE
await page.goto(`${BASE}/app/#/queue`, { waitUntil: "networkidle" });
await page.waitForSelector("#q_table tbody tr", { timeout: 15000 });
await sleep(600);
await shot("01_review_queue");

// filtered queue (construction, conf >= 0.9)
await page.selectOption("#f_type", "construction");
await page.fill("#f_conf", "0.9");
await page.click("#f_go");
await page.waitForFunction(() => document.querySelectorAll("#q_table tbody tr").length > 0 && !document.querySelector("#q_table .spinner"));
await sleep(500);
await shot("02_review_queue_filtered");

// 2. DETAIL - click the top row
await page.selectOption("#f_type", "");
await page.fill("#f_conf", "");
await page.click("#f_go");
await page.waitForFunction(() => !document.querySelector("#q_table .spinner"));
await page.click("#q_table tbody tr:first-child");
await page.waitForSelector("#d_body:not(.hidden)");
// the presentation layer collapses evidence + provenance behind disclosures;
// open them so the analyst screenshots still show the full detail.
await page.evaluate(() =>
  document.querySelectorAll("#view-detail details.disclosure").forEach((d) => (d.open = true)));
await page.waitForSelector("#d_imgs img");
await page.waitForFunction(() => {
  const imgs = [...document.querySelectorAll("#d_imgs img")];
  return imgs.length === 3 && imgs.every((i) => i.complete && i.naturalWidth > 0);
}, { timeout: 20000 });
await page.waitForSelector("#d_trace tr");
await sleep(800);
await shot("03_candidate_detail");

// detail - switch the BEFORE date to 2021 to show the date selector working
await page.click('#d_dates button[data-d="2021"]');
await sleep(1200);
await shot("04_candidate_detail_2021");

// record a decision so the audit table shows a row (idempotent-ish: the log is
// append-only, so re-runs just add history - that is the point of the panel)
await page.fill("#d_note", "Reservoir fill confirmed against the 2021 frame; SAR VV drop corroborates.");
await page.click("#d_confirm");
await page.waitForFunction(() => [...document.querySelectorAll("#d_hist tr td")].some((t) => t.textContent.includes("confirm")));
await sleep(600);
await page.locator("#d_hist").scrollIntoViewIfNeeded();
await shot("05_candidate_detail_after_decision");

// also reject one, on a different candidate, so /audit shows a confirm AND a
// reject (best-effort - never let this block the screenshots)
try {
  await page.goto(`${BASE}/app/#/queue`, { waitUntil: "networkidle" });
  await page.selectOption("#f_type", "other");
  await page.click("#f_go");
  await page.waitForFunction(() => !document.querySelector("#q_table .spinner") && document.querySelectorAll("#q_table tbody tr").length, { timeout: 8000 });
  const rejectId = await page.$eval("#q_table tbody tr:first-child", (tr) => tr.dataset.id);
  await page.click("#q_table tbody tr:first-child");
  await page.waitForFunction((id) => location.hash.endsWith(id) &&
    document.querySelector("#d_head .mono") && document.querySelector("#d_head .mono").textContent === id,
    rejectId, { timeout: 10000 });
  await page.evaluate(() =>
    document.querySelectorAll("#view-detail details.disclosure").forEach((d) => (d.open = true)));
  await page.waitForSelector("#d_trace tr", { timeout: 8000 });
  await page.fill("#d_note", "Weak spectral support and 'other' typing - not a defensible structural change.");
  await page.click("#d_reject");
  await page.waitForFunction(() => [...document.querySelectorAll("#d_hist tr td")].some((t) => t.textContent.includes("reject")), { timeout: 8000 });
  console.log("recorded a reject on candidate", rejectId);
} catch (e) { console.warn("reject demo skipped:", e.message); }

// 3. SEARCH
await page.goto(`${BASE}/app/#/search`, { waitUntil: "networkidle" });
await page.fill("#q", "a sandy braided riverbed");
await page.click("#s_go");
await page.waitForSelector("#s_cards .card img");
await page.waitForFunction(() => {
  const i = [...document.querySelectorAll("#s_cards .card img")].slice(0, 6);
  return i.length && i.every((x) => x.complete);
}, { timeout: 20000 });
await sleep(800);
await shot("06_search");

// 4. DISCOVERY
await page.goto(`${BASE}/app/#/discovery`, { waitUntil: "networkidle" });
await page.fill("#disc_seed", "2019_2026_004510");   // Phase 8: 5-date candidate id scheme (was 2019_2024_*)
await page.click("#disc_go");
await page.waitForSelector("#disc_results .card img");
await page.waitForFunction(() => {
  const i = [...document.querySelectorAll("#disc_results .card img")].slice(0, 6);
  return i.length && i.every((x) => x.complete);
}, { timeout: 20000 });
await page.waitForFunction(() => { const m = document.querySelector("#disc_map"); return m && m.complete && m.naturalWidth > 0; }, { timeout: 20000 });
await sleep(800);
await shot("07_discovery");

await browser.close();
console.log("DONE");
