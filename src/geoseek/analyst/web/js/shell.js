// shell.js - top bar + left rail + global keyboard navigation.
import { api } from "./api-client.js";
import { loadRegions, humanizeRegionName } from "./regions.js";
import { go, currentScreen } from "./router.js";

const RAIL_ITEMS = [
  { code: "OV", label: "Overview", screen: "overview", key: "1" },
  { code: "CD", label: "Candidate", screen: "candidate", key: "2" },
  { code: "SE", label: "Search", screen: "search", key: "3" },
  { code: "RQ", label: "Review Queue", screen: "queue", key: "4" },
  { code: "OD", label: "Object Detect", screen: "detect", key: "5" },
  { code: "DS", label: "Discovery", screen: "discovery", key: "6" },
  { code: "WA", label: "Watch Areas", screen: "watch", key: "7" },
];

const WINDOWS = [
  { label: "30 days", days: 30 },
  { label: "90 days", days: 90 },
  { label: "6 months", days: 182 },
  { label: "12 months", days: 365 },
];

export const shellState = {
  regions: [],
  aoiIdx: 0,
  windowIdx: 1,
  latestObsDate: null,
  undecidedCount: 0,
  onAoiChange: [],
  onWindowChange: [],
};

export function currentAoi() {
  return shellState.regions[shellState.aoiIdx] || null;
}

export function currentWindowDays() {
  return WINDOWS[shellState.windowIdx].days;
}

// The "time window" filter is relative to the archive's own latest
// observation date, not the OS wall clock - this dataset's acquisitions run
// 2019-2026 on a fixed schedule, so "last 30 days" has to mean the 30 days
// before the newest imagery actually staged, not today's real calendar date.
export function windowDateStart() {
  if (!shellState.latestObsDate) return null;
  const end = new Date(shellState.latestObsDate + "T00:00:00Z");
  const start = new Date(end.getTime() - currentWindowDays() * 86400000);
  return start.toISOString().slice(0, 10);
}

function renderRail() {
  const nav = document.getElementById("rail-nav");
  nav.innerHTML = RAIL_ITEMS.map((it) => `
    <button class="rail-btn" data-screen="${it.screen}" title="${it.label} (${it.key})">
      <span class="rail-btn-code">${it.code}</span>
      <span class="rail-btn-label">${it.label}</span>
      ${it.screen === "queue" ? '<span class="rail-btn-badge" id="rail-badge-queue"></span>' : ""}
    </button>`).join("");
  nav.querySelectorAll(".rail-btn").forEach((btn) => {
    btn.addEventListener("click", () => go(btn.dataset.screen));
  });
}

function renderTopbar() {
  document.getElementById("tb-wordmark").innerHTML = `
    <div class="tb-wordmark-mark"></div>
    <div class="tb-wordmark-lockup">
      <div class="tb-wordmark-title">GEOSEEK</div>
      <div class="tb-wordmark-sub">v1.0 &middot; AIR-GAPPED</div>
    </div>`;

  document.getElementById("tb-analyst").innerHTML = `
    <div class="tb-analyst-lockup">
      <div class="tb-analyst-name">LOCAL ANALYST</div>
      <div class="tb-analyst-role">IMAGERY ANALYST &middot; OFFLINE SEAT</div>
    </div>`;
  // no backend authentication/identity endpoint exists - this is fixed
  // chrome text, not a per-analyst value read from anywhere.

  renderAoiField();
  renderWindowField();
}

function renderAoiField() {
  const el = document.getElementById("tb-aoi");
  const r = currentAoi();
  const name = r ? humanizeRegionName(r.name) : "—";
  const n = shellState.regions.length;
  el.innerHTML = `
    <div>
      <div class="t-eyebrow">Area of interest</div>
      <div class="tb-field-value-row">
        <span class="tb-field-value">${name}</span>
        <span class="tb-code-chip">AOI-${String(shellState.aoiIdx + 1).padStart(2, "0")}</span>
      </div>
    </div>
    <div class="tb-stepper" id="tb-aoi-stepper">
      <button data-dir="-1" aria-label="Previous area of interest">◀</button>
      <span class="tb-stepper-value">${n ? shellState.aoiIdx + 1 : 0} / ${n}</span>
      <button data-dir="1" aria-label="Next area of interest">▶</button>
    </div>`;
  el.querySelector('[data-dir="-1"]').addEventListener("click", () => stepAoi(-1));
  el.querySelector('[data-dir="1"]').addEventListener("click", () => stepAoi(1));
}

function stepAoi(dir) {
  const n = shellState.regions.length;
  if (!n) return;
  shellState.aoiIdx = (shellState.aoiIdx + dir + n) % n;
  renderAoiField();
  shellState.onAoiChange.forEach((cb) => cb(currentAoi()));
}

function renderWindowField() {
  const el = document.getElementById("tb-window");
  el.innerHTML = `
    <div>
      <div class="t-eyebrow">Time window</div>
      <div class="tb-field-value-row">
        <span class="tb-field-value">${WINDOWS[shellState.windowIdx].label}</span>
      </div>
    </div>
    <div class="tb-stepper" id="tb-window-stepper">
      <button data-dir="-1" aria-label="Shorter time window">−</button>
      <button data-dir="1" aria-label="Longer time window">+</button>
    </div>`;
  el.querySelector('[data-dir="-1"]').addEventListener("click", () => stepWindow(-1));
  el.querySelector('[data-dir="1"]').addEventListener("click", () => stepWindow(1));
}

function stepWindow(dir) {
  shellState.windowIdx = Math.min(WINDOWS.length - 1, Math.max(0, shellState.windowIdx + dir));
  renderWindowField();
  shellState.onWindowChange.forEach((cb) => cb(currentWindowDays()));
}

export async function refreshQueueBadge() {
  try {
    const body = await api.listCandidates({ decision: "undecided", limit: 1 });
    shellState.undecidedCount = body.total;
    const badge = document.getElementById("rail-badge-queue");
    if (badge) badge.textContent = body.total > 0 ? String(body.total) : "";
  } catch (e) { /* badge is best-effort */ }
}

function isTypingTarget(el) {
  return el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.isContentEditable);
}

export function initKeyboardNav({ onConfirm, onReject } = {}) {
  document.addEventListener("keydown", (e) => {
    if (isTypingTarget(document.activeElement)) return;
    const rail = RAIL_ITEMS.find((it) => it.key === e.key);
    if (rail) { go(rail.screen); return; }
    if (e.key === "Escape") { go("overview"); return; }
    if ((e.key === "c" || e.key === "C") && onConfirm) { onConfirm(); return; }
    if ((e.key === "x" || e.key === "X") && onReject) { onReject(); return; }
  });
}

export async function initShell() {
  renderRail();
  renderTopbar();
  try {
    shellState.regions = await loadRegions(api);
  } catch (e) {
    shellState.regions = [];
  }
  renderAoiField();
  try {
    const pres = await api.presentationSummary();
    const dates = pres.observation_dates || [];
    shellState.latestObsDate = dates.length ? dates[dates.length - 1] : null;
  } catch (e) { /* window filter degrades to "no lower bound" without this */ }
  refreshQueueBadge();
}
