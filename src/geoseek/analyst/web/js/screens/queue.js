import { api } from "../api-client.js";
import { registerScreen, go } from "../router.js";
import { loadRegions } from "../regions.js";
import { toCandidateSummaryViewModel } from "../viewmodels/candidate.js";
import { setBrowseOrder } from "./candidate.js";
import { refreshQueueBadge } from "../shell.js";

let root;
const state = { confBand: "all", status: "pending", sort: "priority", year: null, rows: [], counters: {}, years: [] };

const SORT_MAP = { priority: "queue_score", confidence: "confidence", size: "area_m2", newest: "-rank" };
const STATUS_MAP = { pending: "undecided", confirmed: "confirm", rejected: "reject", all: null };

function mount() {
  root = document.getElementById("screen-queue");
  root.classList.add("rq-root");
}

function filterChip(group, value, label, active) {
  return `<button class="chip ${active ? "active" : ""}" data-group="${group}" data-value="${value}">${label}</button>`;
}

function headerHtml() {
  return `
    <div class="rq-header fixed">
      <div>
        <div class="t-screen-title">Review Queue</div>
        <div class="t-body" style="color:var(--ink-3); margin-top:4px;">Worked top-down. The order puts large, well-evidenced, repeat-sighted changes first.</div>
      </div>
      <div class="rq-counters">
        <div class="rq-counter"><div class="t-stat" style="color:var(--amber);" id="rq-c-pending">–</div><div class="t-micro">Awaiting review</div></div>
        <div class="rq-counter"><div class="t-stat" style="color:var(--green);" id="rq-c-confirmed">–</div><div class="t-micro">Confirmed</div></div>
        <div class="rq-counter"><div class="t-stat" style="color:var(--red);" id="rq-c-rejected">–</div><div class="t-micro">Rejected</div></div>
        <div class="rq-counter"><div class="t-stat" id="rq-c-highconf">–</div><div class="t-micro">High confidence</div></div>
      </div>
    </div>
    <div class="rq-filters fixed">
      <div class="rq-filter-group">
        <span class="t-micro">Confidence</span>
        ${["all", "high", "medium", "low"].map((v) => filterChip("conf", v, v.toUpperCase(), state.confBand === v)).join("")}
      </div>
      <div class="rq-filter-group">
        <span class="t-micro">Status</span>
        ${["pending", "confirmed", "rejected", "all"].map((v) => filterChip("status", v, v.toUpperCase(), state.status === v)).join("")}
      </div>
      <div class="rq-filter-group">
        <span class="t-micro">Sort</span>
        ${["priority", "confidence", "size", "newest"].map((v) => filterChip("sort", v, v.toUpperCase(), state.sort === v)).join("")}
      </div>
      <div class="rq-filter-group">
        <span class="t-micro">Year</span>
        <div class="year-tabs">
          <button class="year-tab ${!state.year ? "active" : ""}" data-year="">ALL</button>
          ${state.years.map((y) => `<button class="year-tab ${state.year === y ? "active" : ""}" data-year="${y}">${y}</button>`).join("")}
        </div>
      </div>
    </div>`;
}

function rowHtml(vm) {
  const gateSquares = vm.gates.map((g) =>
    `<span class="gate-square ${g.pass ? "pass" : "fail"}" title="${g.label} — ${g.pass ? "clear" : "flagged"}"></span>`).join("");
  const meterColor = { high: "var(--band-high)", medium: "var(--band-medium)", low: "var(--band-low)" }[vm.band];
  const decisionPillClass = vm.decision === "confirm" ? "decision-confirm" : vm.decision === "reject" ? "decision-reject" : "decision-pending";
  const decisionLabel = vm.decision === "confirm" ? "CONFIRMED" : vm.decision === "reject" ? "REJECTED" : "PENDING";
  return `
    <button class="rq-row" data-id="${vm.id}">
      <div class="rq-thumb">${vm.thumbUrl
        ? `<img src="${vm.thumbUrl}" alt="" loading="lazy">`
        : `<span class="rq-thumb-empty">No preview</span>`}</div>
      <div class="min0">
        <div class="t-row-title" style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${vm.locationName}</div>
        <div class="t-meta">${vm.id}</div>
      </div>
      <div class="t-body" style="color:var(--ink-2);">${vm.changeTypeLabel}</div>
      <div class="t-chip">${(vm.window[0]||"").slice(5)} → ${(vm.window[1]||"").slice(5)}</div>
      <div class="t-chip">${vm.areaHectares.toFixed(1)} ha</div>
      <div class="rq-confidence">
        <div class="confidence-meter-track"><div class="confidence-meter-fill" style="width:${vm.confidencePct}%; background:${meterColor};"></div></div>
        <span style="color:${meterColor}; font:600 11px var(--font-mono);">${vm.confidencePct}</span>
      </div>
      <div class="gate-squares">${gateSquares}</div>
      <div class="pill ${decisionPillClass}">${decisionLabel}</div>
    </button>`;
}

function tableHtml() {
  return `
    <div class="rq-table-head">
      <div></div>
      <div class="t-micro">Location</div>
      <div class="t-micro">What changed</div>
      <div class="t-micro">Between</div>
      <div class="t-micro">Size</div>
      <div class="t-micro">Confidence</div>
      <div class="t-micro">Evidence</div>
      <div class="t-micro">Decision</div>
    </div>
    <div class="rq-table-body scroll-pane" id="rq-rows"></div>
    <div class="rq-footer fixed t-small" id="rq-footer" style="color:var(--ink-dim);"></div>`;
}

async function loadAndRender() {
  const rowsHost = document.getElementById("rq-rows");
  if (rowsHost) rowsHost.innerHTML = `<div class="loading-state">Loading candidates from the local index…</div>`;
  try {
    if (!state.years.length) {
      const pres = await api.presentationSummary();
      state.years = Array.from(new Set((pres.observation_dates || []).map((d) => d.slice(0, 4)))).sort();
    }
    const [regions, allDecided, body] = await Promise.all([
      loadRegions(api),
      api.listCandidates({ limit: 1, decision: "confirm" }),
      api.listCandidates({
        decision: STATUS_MAP[state.status] || undefined,
        min_confidence: state.confBand === "high" ? 0.75 : state.confBand === "medium" ? 0.52 : undefined,
        sort: SORT_MAP[state.sort], year: state.year || undefined, limit: 100,
      }),
    ]);
    const [confirmedTotal, rejectedTotal, undecidedTotal, highConfTotal] = await Promise.all([
      api.listCandidates({ decision: "confirm", year: state.year || undefined, limit: 1 }).then((b) => b.total),
      api.listCandidates({ decision: "reject", year: state.year || undefined, limit: 1 }).then((b) => b.total),
      api.listCandidates({ decision: "undecided", year: state.year || undefined, limit: 1 }).then((b) => b.total),
      api.listCandidates({ min_confidence: 0.75, year: state.year || undefined, limit: 1 }).then((b) => b.total),
    ]);
    document.getElementById("rq-c-pending").textContent = undecidedTotal;
    document.getElementById("rq-c-confirmed").textContent = confirmedTotal;
    document.getElementById("rq-c-rejected").textContent = rejectedTotal;
    document.getElementById("rq-c-highconf").textContent = highConfTotal;

    let rows = state.confBand === "low"
      ? body.candidates.filter((c) => (c.confidence || 0) < 0.52)
      : body.candidates;
    const vms = rows.map((r) => toCandidateSummaryViewModel(r, regions, api));
    state.rows = vms;
    rowsHost.innerHTML = vms.length
      ? vms.map(rowHtml).join("")
      : state.year
        ? `<div class="empty-state"><div class="empty-state-title">No observations for ${state.year}</div>Data ingestion pending for this year.</div>`
        : `<div class="empty-state"><div class="empty-state-title">Nothing matches these filters</div>
            Checked: status "${state.status}", confidence "${state.confBand}". Try widening the status filter or clearing the confidence filter.</div>`;
    rowsHost.querySelectorAll(".rq-row").forEach((btn) =>
      btn.addEventListener("click", () => {
        setBrowseOrder(vms.map((v) => v.id));
        go("candidate", btn.dataset.id);
      }));

    const overallTotal = await api.listCandidates({ limit: 1 }).then((b) => b.total);
    document.getElementById("rq-footer").textContent =
      `SHOWING ${vms.length} OF ${body.total} FILTERED · ${overallTotal} CANDIDATES IN ARCHIVE`;
  } catch (e) {
    rowsHost.innerHTML = `
      <div class="error-state">
        <div class="error-state-msg">Could not load the review queue.</div>
        <button class="btn" id="rq-retry">Retry</button>
        <details style="margin-top:10px;"><summary class="t-small">Details</summary><div class="error-detail">${e.message}</div></details>
      </div>`;
    document.getElementById("rq-retry")?.addEventListener("click", loadAndRender);
  }
}

function wireFilters() {
  root.querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      const { group, value } = chip.dataset;
      if (group === "conf") state.confBand = value;
      if (group === "status") state.status = value;
      if (group === "sort") state.sort = value;
      root.querySelector(".rq-filters").outerHTML = headerFiltersOnly();
      wireFilters();
      loadAndRender();
    });
  });
  root.querySelectorAll(".year-tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.year = btn.dataset.year || null;
      root.querySelector(".rq-filters").outerHTML = headerFiltersOnly();
      wireFilters();
      loadAndRender();
    });
  });
}
function headerFiltersOnly() {
  const div = document.createElement("div");
  div.innerHTML = headerHtml();
  return div.querySelector(".rq-filters").outerHTML;
}

function render() {
  root.innerHTML = headerHtml() + tableHtml();
  wireFilters();
}

async function show() {
  if (!root.querySelector(".rq-table-body")) render();
  await loadAndRender();
  refreshQueueBadge();
}

registerScreen("queue", { mount, show });
