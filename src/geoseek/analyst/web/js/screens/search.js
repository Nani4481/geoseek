import { api } from "../api-client.js";
import { registerScreen } from "../router.js";
import { loadRegions } from "../regions.js";
import { toSearchResultViewModel } from "../viewmodels/search.js";

let root;
let lastQuery = "";
let facets = { sensor: null };

function mount() {
  root = document.getElementById("screen-search");
  root.classList.add("se-root");
  root.innerHTML = `
    <div class="se-bar fixed">
      <div class="se-field">
        <span class="t-chip" style="color:var(--cyan);">FIND</span>
        <input id="se-input" type="text" placeholder="describe what you are looking for, in plain words">
        <span class="t-micro" id="se-scope"></span>
      </div>
      <button class="btn solid-cyan" id="se-go">Search</button>
    </div>
    <div class="se-suggestions fixed" id="se-suggestions"></div>
    <div class="se-results">
      <div class="se-results-left scroll-pane">
        <div class="se-summary t-small" id="se-summary" style="color:var(--ink-dim);"></div>
        <div class="se-grid" id="se-grid"></div>
      </div>
      <div class="se-facets scroll-pane" id="se-facets"></div>
    </div>`;

  const SUGGESTIONS = [
    "temporary vehicle park on bare ground", "new earth berm around a compound",
    "aircraft on an unpaved apron", "flooded field beside a road",
    "freshly graded construction pad", "small boats moored at a jetty",
  ];
  document.getElementById("se-suggestions").innerHTML = SUGGESTIONS.map((s) =>
    `<button class="chip" data-q="${s}"><span style="color:var(--ink-ghost);">TRY</span> ${s}</button>`).join("");
  root.querySelectorAll("[data-q]").forEach((b) =>
    b.addEventListener("click", () => { document.getElementById("se-input").value = b.dataset.q; runSearch(); }));

  document.getElementById("se-go").addEventListener("click", runSearch);
  document.getElementById("se-input").addEventListener("keydown", (e) => { if (e.key === "Enter") runSearch(); });
}

function resultCardHtml(vm) {
  return `
    <div class="se-card">
      <div class="se-card-img">
        <img src="${vm.thumbnailUrl}" alt="" loading="lazy">
        <span class="badge-plate se-rank" style="color:var(--cyan);">RANK ${String(vm.rank).padStart(2, "0")}</span>
        <span class="badge-plate se-date">${vm.date}</span>
      </div>
      <div class="se-card-body">
        <div class="t-row-title" style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${vm.locationName}</div>
        <div class="t-meta">${vm.sensor} · ${vm.tileId}</div>
        <hr class="hairline-rule" style="margin:6px 0;">
        <div class="t-small" style="color:var(--ink-3);">${vm.reason}</div>
      </div>
    </div>`;
}

async function runSearch() {
  const q = document.getElementById("se-input").value.trim();
  if (!q) return;
  lastQuery = q;
  const grid = document.getElementById("se-grid");
  grid.innerHTML = `<div class="loading-state">Searching the local tile index…</div>`;
  try {
    const [body, regions] = await Promise.all([
      api.searchText(q, { k: 40, sensor: facets.sensor || undefined }),
      loadRegions(api),
    ]);
    const vms = body.results.map((r, i) => toSearchResultViewModel(r, i + 1, regions, api));
    document.getElementById("se-summary").innerHTML =
      `Tiles that match "${q}" — best first <span style="float:right;">SEARCHED LOCALLY IN ${(body.latency_ms / 1000).toFixed(2)} S</span>`;
    grid.innerHTML = vms.length ? vms.map(resultCardHtml).join("") :
      `<div class="empty-state"><div class="empty-state-title">No matches</div>Checked the full local tile index for "${q}". Try a shorter or more general description.</div>`;
  } catch (e) {
    grid.innerHTML = `
      <div class="error-state">
        <div class="error-state-msg">Search failed.</div>
        <button class="btn" id="se-retry">Retry</button>
        <details style="margin-top:10px;"><summary class="t-small">Details</summary><div class="error-detail">${e.message}</div></details>
      </div>`;
    document.getElementById("se-retry")?.addEventListener("click", runSearch);
  }
}

async function show() {
  try {
    const stats = await api.stats();
    document.getElementById("se-scope").textContent =
      `${(stats.index.tiles || 0).toLocaleString()} TILES · ${(stats.index.collections || []).length} COLLECTIONS`;
  } catch (e) { /* scope readout is decorative-only */ }
  document.getElementById("se-facets").innerHTML = `
    <div class="t-panel-title" style="padding:14px 14px 8px;">Narrow it down</div>
    <div class="se-facet-group" style="padding:0 14px 14px;">
      <div class="t-micro" style="margin-bottom:6px;">Sensor</div>
      ${["sentinel-2", "sentinel-1"].map((s) => `<button class="chip" data-sensor="${s}" style="margin:2px 4px 2px 0;">${s}</button>`).join("")}
    </div>
    <div class="tinted-box neutral" style="margin:0 14px 14px;">
      <div class="tinted-box-title" style="font-size:11px;">How matching works</div>
      <div class="tinted-box-text">Results are ordered by how closely each tile resembles your description. Order is a guide, not a measurement — read the reason on each tile and judge the imagery yourself.</div>
    </div>`;
  document.querySelectorAll("[data-sensor]").forEach((b) =>
    b.addEventListener("click", () => {
      facets.sensor = facets.sensor === b.dataset.sensor ? null : b.dataset.sensor;
      document.querySelectorAll("[data-sensor]").forEach((x) => x.classList.toggle("active", x.dataset.sensor === facets.sensor));
      if (lastQuery) runSearch();
    }));
}

registerScreen("search", { mount, show });
