import { api } from "../api-client.js";
import { registerScreen, go } from "../router.js";
import { loadRegions, humanizeRegionName } from "../regions.js";
import { toOverviewViewModel } from "../viewmodels/overview.js";
import { buildActivityFeed } from "../viewmodels/activity.js";
import { setBrowseOrder } from "./candidate.js";
import { shellState, currentAoi } from "../shell.js";

let root;
let layersOn = { changes: true, grid: true };
let yearFilter = null; // null = all years

function zoneRectHtml(zone, bbox) {
  if (!bbox) return "";
  const tl = pctPos(zone.min_lon, zone.max_lat, bbox);
  const br = pctPos(zone.max_lon, zone.min_lat, bbox);
  const left = Math.min(tl.x, br.x), top = Math.min(tl.y, br.y);
  const width = Math.abs(br.x - tl.x), height = Math.abs(br.y - tl.y);
  if (width <= 0 || height <= 0) return "";
  return `
    <div class="ov-zone-rect" style="left:${left}%; top:${top}%; width:${width}%; height:${height}%;">
      <span class="ov-zone-label">${zone.name}</span>
    </div>`;
}

function mount() {
  root = document.getElementById("screen-overview");
  root.classList.add("ov-root");
  shellState.onAoiChange.push(() => show());
}

function pctPos(lon, lat, bbox) {
  const [w, s, e, n] = bbox;
  const x = ((lon - w) / (e - w)) * 100;
  const y = 100 - ((lat - s) / (n - s)) * 100;
  return { x: Math.min(98, Math.max(2, x)), y: Math.min(98, Math.max(2, y)) };
}

function markerHtml(f, bbox, idx) {
  if (f.lon == null || !bbox) return "";
  const { x, y } = pctPos(f.lon, f.lat, bbox);
  const color = `var(--ct-${["water_gain","water_loss","construction","clearance","road"].includes(f.changeTypeCode) ? f.changeTypeCode : "other"})`;
  const alertTitle = f.restrictedZone ? ` — RESTRICTED ZONE: ${f.restrictedZone.name}` : "";
  return `
    <button class="ov-marker" data-id="${f.id}" style="left:${x}%; top:${y}%;" title="${f.locationName} — ${f.changeTypeLabel}${alertTitle}">
      ${f.restrictedZone ? '<span class="ov-marker-alert-ring"></span>' : ""}
      <span class="ov-marker-ring" style="border-color:${color};"></span>
      <span class="ov-marker-dot" style="background:${color}; box-shadow:0 0 8px ${color};"></span>
      <span class="ov-marker-label badge-plate">${String(idx + 1).padStart(2, "0")} ${f.changeTypeLabel}</span>
    </button>`;
}

function feedRowHtml(item) {
  return `
    <div class="feed-row">
      <span class="feed-dot ${item.tone}"></span>
      <div class="feed-body">
        <div class="feed-text ${item.tone}">${item.text}</div>
        <div class="feed-timestamp">${item.tsLabel}</div>
      </div>
    </div>`;
}

function yearTabsHtml(years) {
  const options = [{ label: "ALL", value: null }, ...years.map((y) => ({ label: y, value: y }))];
  return `
    <div class="year-tabs">
      ${options.map((o) => `<button class="year-tab ${yearFilter === o.value ? "active" : ""}" data-year="${o.value || ""}">${o.label}</button>`).join("")}
    </div>`;
}

function statsBarHtml(vm) {
  const stats = [
    { label: "Active watch areas", value: vm.counters.watchAreas },
    { label: "Candidates", value: vm.counters.candidates },
    { label: "Restricted zone alerts", value: vm.counters.restrictedZoneAlerts },
    { label: "Review rate", value: `${vm.counters.reviewRatePct}%` },
  ];
  return `
    <div class="ov-stats-bar">
      ${stats.map((s) => `
        <div class="ov-stat-item">
          <div class="ov-stat-value">${s.value}</div>
          <div class="t-micro">${s.label}</div>
        </div>`).join("")}
      <div class="ov-stats-spacer"></div>
      <div class="ov-stats-years">${yearTabsHtml(vm.years)}</div>
    </div>`;
}

function inspectorRowHtml(f) {
  const gates = ["quality", "registration", "radiometric", "phenology", "morphology"];
  const squares = gates.map((g) =>
    `<span class="gate-square ${f.gates[g] !== false ? "pass" : "fail"}"></span>`).join("");
  const sarLabel = !f.sarAvailable ? "NO RADAR" : (f.sarVerdict === "agree" || f.sarVerdict === "agrees") ? "RADAR AGREES" : "RADAR DIFFERS";
  return `
    <button class="ov-finding" data-id="${f.id}">
      <div class="ov-finding-thumb">${f.thumbUrl
        ? `<img src="${f.thumbUrl}" alt="" loading="lazy">`
        : `<span class="ov-finding-thumb-empty">No preview</span>`}</div>
      <div class="min0" style="flex:1;">
        <div class="ov-finding-top">
          <span class="t-row-title">${f.changeTypeLabel}</span>
          <span class="pill band-${f.band}">${f.band.toUpperCase()}</span>
        </div>
        <div class="t-meta" style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${f.locationName}</div>
        <div class="t-small" style="color:var(--ink-dim); display:flex; gap:8px; margin-top:2px;">
          <span>${f.areaHectares.toFixed(1)} ha</span>
          <span>${(f.window[0]||"").slice(5)}→${(f.window[1]||"").slice(5)}</span>
          <span style="color:var(--band-${f.band});">${f.confidencePct}</span>
        </div>
        <div class="ov-finding-footer">
          <div class="gate-squares">${squares}</div>
          <span class="t-micro">${sarLabel}</span>
        </div>
      </div>
    </button>`;
}

function render(vm, bbox) {
  // presentation_summary's own featured[].imagery URLs are already built
  // server-side with the year-format date the imagery endpoint expects
  // (candidates' own earliest_supported/window fields are full ISO dates,
  // not years, and would 400 if passed directly).
  const bgUrl = vm.featured[0] ? vm.featured[0].imagery.overlay : null;

  root.innerHTML = `
    ${statsBarHtml(vm)}
    <div class="ov-map">
      ${bgUrl ? `<img class="ov-base-img" src="${bgUrl}" alt="">` : `<div class="ov-base-img ov-base-empty"></div>`}
      <div class="ov-vignette"></div>
      <div class="ov-graticule ${layersOn.grid ? "" : "hidden"}"></div>
      <div class="ov-zones">${(vm.restrictedZones || []).map((z) => zoneRectHtml(z, bbox)).join("")}</div>
      <div class="ov-markers ${layersOn.changes ? "" : "hidden"}" id="ov-markers"></div>
      <div class="ov-aoi-boundary">
        <span class="ov-aoi-label">AOI BOUNDARY · AOI-${String(shellState.aoiIdx + 1).padStart(2, "0")}</span>
      </div>

      <div class="ov-panel panel ov-layers">
        <div class="panel-header t-eyebrow">Analysis layers</div>
        <div class="ov-layer-row" data-layer="changes">
          <span class="ov-layer-swatch" style="background:${layersOn.changes ? "var(--amber)" : "transparent"};"></span>
          <span class="t-body" style="color:${layersOn.changes ? "var(--ink)" : "var(--ink-faint)"};">Detected changes</span>
          <span class="t-micro" style="margin-left:auto;">${vm.findings.length} SHOWN</span>
        </div>
        <div class="ov-layer-row" data-layer="grid">
          <span class="ov-layer-swatch" style="background:${layersOn.grid ? "var(--cyan)" : "transparent"};"></span>
          <span class="t-body" style="color:${layersOn.grid ? "var(--ink)" : "var(--ink-faint)"};">Map graticule</span>
        </div>
        <div class="ov-layers-footer t-small" style="color:var(--ink-dim);">
          Optical multispectral imagery<br>${vm.findings[0] ? "ACQ " + vm.findings[0].window[1] : ""}
        </div>
      </div>

      <div class="ov-panel panel ov-activity">
        <div class="feed-live">
          <span class="feed-live-dot"></span>
          <span class="feed-live-label">LIVE ACTIVITY FEED</span>
        </div>
        <div class="feed-list">
          ${vm.activity.length ? vm.activity.map(feedRowHtml).join("") :
            `<div class="empty-state">No recorded activity yet.</div>`}
        </div>
      </div>

      <div class="ov-panel panel ov-scan">
        <div class="panel-header" style="display:flex; justify-content:space-between; align-items:center;">
          <span class="t-eyebrow">Area scan</span>
          <span class="pill band-high">INDEXED</span>
        </div>
        <div style="padding:10px 11px;">
          <div class="t-small" style="color:var(--ink-dim);">${vm.tilesIndexed ?? vm.counters.tiles} tiles indexed in the local archive.</div>
        </div>
      </div>

      <div class="ov-panel panel ov-inspector">
        <div class="panel-header">
          <div style="display:flex; justify-content:space-between; align-items:baseline;">
            <span class="t-panel-title">Priority findings</span>
            <span class="t-micro">${vm.findings.length} OVER THRESHOLD</span>
          </div>
          <div class="t-small" style="color:var(--ink-3); margin-top:4px;">
            Ranked by how much the ground changed and how well the evidence holds up. Open one to see the proof.
          </div>
        </div>
        <div class="ov-inspector-list scroll-pane">
          ${vm.findings.length ? vm.findings.map(inspectorRowHtml).join("") :
            yearFilter
              ? `<div class="empty-state"><div class="empty-state-title">No observations for ${yearFilter}</div>Data ingestion pending for this year in the current area of interest.</div>`
              : `<div class="empty-state">No findings meet the priority threshold for this area of interest.</div>`}
        </div>
        <div class="ov-inspector-footer">
          <button class="btn flex1" id="ov-open-queue">Open review queue</button>
          <button class="btn tone-cyan flex1" id="ov-inspect-top">Inspect top finding</button>
        </div>
      </div>
    </div>

    <div class="readout-strip">
      <span>AOI ${bbox ? bbox.map((v) => v.toFixed(3)).join(", ") : "—"}</span>
      <span class="readout-sep">|</span>
      <span class="${vm.counters.candidates ? "" : ""}" style="color:var(--amber);">${vm.counters.candidates} CANDIDATES</span>
      <span class="readout-sep">|</span>
      <span>${(vm.tilesIndexed ?? vm.counters.tiles ?? 0).toLocaleString()} TILES</span>
      <span class="readout-sep">|</span>
      <span style="color:var(--green);">${vm.counters.decided} DECIDED</span>
      <span class="readout-sep">|</span>
      <span class="readout-scalebar"></span>
    </div>`;

  root.querySelector("#ov-markers").innerHTML = vm.findings.map((f, i) => markerHtml(f, bbox, i)).join("");
  root.querySelectorAll(".ov-marker, .ov-finding").forEach((el) =>
    el.addEventListener("click", () => {
      setBrowseOrder(vm.findings.map((f) => f.id));
      go("candidate", el.dataset.id);
    }));
  root.querySelectorAll(".ov-layer-row").forEach((row) =>
    row.addEventListener("click", () => {
      const key = row.dataset.layer;
      layersOn[key] = !layersOn[key];
      render(vm, bbox);
    }));
  root.querySelector("#ov-open-queue")?.addEventListener("click", () => go("queue"));
  root.querySelector("#ov-inspect-top")?.addEventListener("click", () => {
    if (vm.findings[0]) { setBrowseOrder(vm.findings.map((f) => f.id)); go("candidate", vm.findings[0].id); }
  });
  root.querySelectorAll(".year-tab").forEach((btn) =>
    btn.addEventListener("click", () => {
      yearFilter = btn.dataset.year || null;
      show();
    }));
}

async function show() {
  root.innerHTML = `<div class="loading-state">Loading the overview from the local index…</div>`;
  try {
    const aoi = currentAoi();
    const [presentation, regions, stats, candidatesBody, zonesBody, feedCandidatesBody, auditBody] = await Promise.all([
      api.presentationSummary(),
      loadRegions(api),
      api.stats(),
      api.listCandidates({ bbox: aoi ? aoi.bbox.join(",") : undefined, sort: "queue_score", limit: 9, year: yearFilter || undefined }),
      api.restrictedZones(),
      api.listCandidates({ sort: "queue_score", limit: 30 }),
      api.audit({ limit: 20 }),
    ]);
    const vm = toOverviewViewModel(presentation, candidatesBody, regions, stats, api);
    vm.restrictedZones = zonesBody.zones || [];
    vm.activity = buildActivityFeed({
      candidates: feedCandidatesBody.candidates || [],
      auditRows: auditBody.decisions || [],
      regions,
    });
    render(vm, aoi ? aoi.bbox : null);
  } catch (e) {
    root.innerHTML = `
      <div class="error-state">
        <div class="error-state-msg">Could not load the overview.</div>
        <button class="btn" id="ov-retry">Retry</button>
        <details style="margin-top:10px;"><summary class="t-small">Details</summary><div class="error-detail">${e.message}</div></details>
      </div>`;
    root.querySelector("#ov-retry")?.addEventListener("click", show);
  }
}

registerScreen("overview", { mount, show });
