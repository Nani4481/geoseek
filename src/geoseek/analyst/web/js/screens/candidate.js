import { api } from "../api-client.js";
import { registerScreen, go } from "../router.js";
import { loadRegions } from "../regions.js";
import { loadZones, zoneBboxByName } from "../zones.js";
import { toCandidateDetailViewModel } from "../viewmodels/candidate.js";
import { mountCompareSlider, compareSurfaceHtml } from "../compare-slider.js";
import { shellState, refreshQueueBadge } from "../shell.js";
import { Terrain3D } from "../components/terrain-3d.js";
import { mountCandidateMap } from "../components/candidate-map.js";
import { mountGateChart } from "../components/mini-chart.js";

let root;
let currentId = null;
let currentVm = null;
let currentIds = []; // the browsing order set by whatever screen deep-linked here
let sliderCtl = null;
let terrain = null;
let candidateMap = null;
let gateChart = null;
let selectedBeforeYear = null;

function mount() {
  root = document.getElementById("screen-candidate");
  root.classList.add("cd-root");
}

function bandColorVar(band) {
  return { high: "var(--band-high)", medium: "var(--band-medium)", low: "var(--band-low)" }[band];
}

function ringSvg(pct, band) {
  const r = 50, c = 2 * Math.PI * r;
  const dash = (pct / 100) * c;
  return `
    <svg class="confidence-ring" viewBox="0 0 120 120">
      <g transform="rotate(-90 60 60)">
        <circle cx="60" cy="60" r="${r}" fill="none" stroke="rgba(255,255,255,0.09)" stroke-width="9"/>
        <circle cx="60" cy="60" r="${r}" fill="none" stroke="${bandColorVar(band)}" stroke-width="9"
          stroke-dasharray="${dash} ${c}" stroke-linecap="butt"/>
      </g>
    </svg>`;
}

function gateRowsHtml(gates) {
  return gates.map((g) => `
    <div class="gate-row">
      <div class="gate-icon ${g.pass ? "pass" : "fail"}">${g.pass ? "✓" : "!"}</div>
      <div class="gate-text min0">
        <div class="gate-label">${g.label}</div>
        <div class="gate-note">${g.note}</div>
      </div>
    </div>`).join("");
}

function reasonRowsHtml(reasons) {
  const markChar = { plus: "+", minus: "−", neutral: "~" };
  return reasons.map((r) => `
    <div class="reason-row">
      <div class="reason-mark ${r.sign}">${markChar[r.sign]}</div>
      <div class="reason-text">${r.text}</div>
    </div>`).join("");
}

function alertBannerHtml(vm) {
  if (!vm.restrictedZone) return "";
  const isCritical = vm.restrictedZone.alert_level === "critical";
  return `
    <div class="alert-banner fixed ${isCritical ? "" : "warning"}">
      <span class="alert-banner-icon">⚠</span>
      <div class="alert-banner-text">
        <div class="alert-banner-title">RESTRICTED ZONE ALERT — ${vm.changeTypeLabel} detected inside ${vm.restrictedZone.name}</div>
        <div class="alert-banner-sub">Alert level: ${vm.restrictedZone.alert_level.toUpperCase()} · Flagged for immediate review</div>
      </div>
    </div>`;
}

function yearTabsHtml(vm) {
  const years = vm.imagery.beforeYears;
  if (!years.length) return "";
  const active = selectedBeforeYear || vm.imagery.defaultBeforeYear;
  return `
    <div class="cd-year-tabs">
      <span class="t-micro" style="margin-right:8px;">Compare from</span>
      <div class="year-tabs">
        ${years.map((y) => `<button class="year-tab ${y === active ? "active" : ""}" data-year="${y}">${y}</button>`).join("")}
      </div>
    </div>`;
}

function timelineHtml(timeline) {
  return timeline.map((t) => `
    <div class="cd-timeline-bar cd-timeline-${t.state}" title="${t.date} · ${t.state === "present" ? "change present" : t.state === "entering" ? "change window" : "before change"}">
      <div class="cd-timeline-tick">${t.date.slice(5)}</div>
    </div>`).join("");
}

function decisionFooterHtml(vm) {
  if (vm.effectiveDecision === "undecided") {
    return `
      <div class="cd-decision-buttons">
        <button class="btn tone-green large flex1" id="cd-confirm">Confirm &middot; C</button>
        <button class="btn tone-red large flex1" id="cd-reject">Reject &middot; X</button>
      </div>
      <div class="t-small" style="color:var(--ink-dim); margin-top:8px;">
        Your decision is recorded against ${vm.id} with the imagery dates and the evidence shown above.
      </div>`;
  }
  const label = vm.effectiveDecision === "confirm" ? "CONFIRMED AS A REAL CHANGE" : "REJECTED — NOT A REAL CHANGE";
  const cd = vm.currentDecision;
  return `
    <div class="tinted-box ${vm.effectiveDecision === "confirm" ? "agree" : "disagree"}">
      <div class="tinted-box-title">${label}</div>
      <div class="t-small" style="color:var(--ink-dim);">
        LOGGED ${cd ? cd.created_at : ""}${cd && cd.analyst ? " · " + cd.analyst : ""}
      </div>
    </div>
    <button class="btn flex1" id="cd-reopen" style="margin-top:8px; width:100%;">Reopen for review</button>`;
}

function render(vm) {
  const beforeYear = selectedBeforeYear || vm.imagery.defaultBeforeYear;
  const beforeUrl = api.candidateImageryUrl(vm.id, { date: beforeYear, view: "rgb" });

  root.innerHTML = `
    <div class="cd-left scroll-pane">
      ${alertBannerHtml(vm)}
      <div class="cd-header fixed">
        <button class="btn" id="cd-back">← MAP</button>
        <div class="cd-header-title">
          <span class="t-screen-title">${vm.changeTypeLabel}</span>
          <span class="cd-id" style="color:var(--cyan); font-family:var(--font-mono); font-size:13px; margin-left:10px;">${vm.id}</span>
          <div class="t-small" style="color:var(--ink-dim);">${vm.locationName}</div>
        </div>
        <div class="cd-header-nav">
          <button class="btn" id="cd-prev" aria-label="Previous candidate">◀</button>
          <button class="btn" id="cd-next" aria-label="Next candidate">▶</button>
        </div>
      </div>

      <div class="cd-verdict fixed">
        <div class="t-eyebrow" style="color:var(--amber);">What the system thinks happened</div>
        <div class="t-verdict" style="margin-top:6px;">${vm.verdictSentence}</div>
        <div class="cd-facts">
          <div><div class="t-micro">Change type</div><div class="cd-fact-value">${vm.changeTypeLabel}</div></div>
          <div><div class="t-micro">Area</div><div class="cd-fact-value">${vm.areaHectares.toFixed(1)} ha</div></div>
          <div><div class="t-micro">Earliest seen</div><div class="cd-fact-value">${vm.earliestSupported[0] || "—"}</div></div>
          <div><div class="t-micro">Seen again on</div><div class="cd-fact-value">${vm.earliestSupported[1] || "—"}</div></div>
          <div><div class="t-micro">Priority band</div><div class="cd-fact-value">${vm.band.toUpperCase()}</div></div>
        </div>
      </div>

      <div class="cd-evidence">
        <div class="cd-evidence-head">
          <span class="t-section-title">Visual evidence &middot; drag to compare</span>
          ${yearTabsHtml(vm)}
        </div>
        <hr class="hairline-rule">
        ${compareSurfaceHtml({
          baseHtml: `<img src="${beforeUrl}" alt="Before imagery">`,
          afterHtml: `<img src="${vm.imagery.afterUrl}" alt="After imagery with change overlay">`,
          cornerTL: `AFTER &middot; ${vm.afterDate}`,
          cornerTR: `BEFORE &middot; ${vm.beforeDate}`,
        })}
        <div class="cd-resolution-caption t-small">Optical multispectral imagery &middot; Sentinel-2, ~10m/px</div>
      </div>

      <div class="cd-terrain-section fixed">
        <span class="t-section-title">Terrain change model</span>
        <div class="cd-terrain-canvas-wrap">
          <span class="cd-terrain-axis left">BEFORE &rarr;</span>
          <canvas id="cd-terrain-3d" width="500" height="320"></canvas>
          <span class="cd-terrain-axis right">&rarr; AFTER</span>
        </div>
        <div class="terrain-summary t-small" id="cd-terrain-summary"></div>
      </div>

      <div class="cd-map-section fixed">
        <span class="t-section-title">Location &middot; drag to pan, scroll to zoom</span>
        <div class="geo-map" id="cd-map"></div>
      </div>

      <div class="cd-timeline fixed">
        <div class="cd-timeline-head">
          <span class="t-section-title">When it was visible</span>
        </div>
        <div class="cd-timeline-bars">${timelineHtml(vm.timeline)}</div>
      </div>
    </div>

    <div class="cd-right scroll-pane">
      <div class="cd-ring-block fixed">
        <div class="confidence-ring-wrap">
          <div class="confidence-ring-outer">
            ${ringSvg(vm.confidencePct, vm.band)}
            <div class="confidence-ring-center">
              <div class="confidence-ring-value">${vm.confidencePct}</div>
              <div class="confidence-ring-suffix">OF 100</div>
            </div>
          </div>
          <div>
            <div class="pill band-${vm.band}">${vm.band.toUpperCase()} CONFIDENCE</div>
            <div class="confidence-meaning">${vm.bandMeaning}</div>
          </div>
        </div>
      </div>

      <div class="cd-reasons fixed">
        <div class="t-eyebrow">Why the score is what it is</div>
        <div>${reasonRowsHtml(vm.reasons)}</div>
      </div>

      <div class="cd-gates fixed">
        <div class="cd-gates-head">
          <span class="t-eyebrow">Evidence check</span>
          <span class="t-micro">${vm.gatesPassCount} OF 5 CLEAR</span>
        </div>
        <div class="cd-gate-chart-wrap"><canvas id="cd-gate-chart" height="90"></canvas></div>
        <div>${gateRowsHtml(vm.gates)}</div>
      </div>

      <div class="cd-sar fixed">
        <div class="tinted-box ${vm.sar.state === "agree" ? "agree" : vm.sar.state === "disagree" ? "disagree" : "neutral"}">
          <div class="tinted-box-title">${vm.sar.title}</div>
          <div class="tinted-box-text">${vm.sar.text}</div>
        </div>
      </div>

      <div class="cd-source fixed">
        <div class="t-eyebrow">Source observations</div>
        <div class="cd-kv"><span class="t-micro">Before</span><span class="t-meta">${vm.provenance.beforeDate || "—"}</span></div>
        <div class="cd-kv"><span class="t-micro">After</span><span class="t-meta">${vm.provenance.afterDate || "—"}${vm.provenance.afterGsd ? " · " + vm.provenance.afterGsd + "m GSD" : ""}</span></div>
        <div class="cd-kv"><span class="t-micro">Scene</span><span class="t-meta">${vm.provenance.sceneId || "—"}</span></div>
        <div class="cd-kv"><span class="t-micro">Tile</span><span class="t-meta">${vm.provenance.tileId || "—"}</span></div>
      </div>

      <div class="cd-decision fixed">${decisionFooterHtml(vm)}</div>
    </div>`;

  const surface = root.querySelector(".compare-surface");
  if (sliderCtl) sliderCtl.destroy();
  sliderCtl = mountCompareSlider(surface, { initial: 50, ariaLabel: "Compare before and after imagery" });

  if (terrain) terrain.destroy();
  terrain = new Terrain3D(root.querySelector("#cd-terrain-3d"));
  terrain.setData(vm.terrain.changeType, vm.terrain.magnitude);
  root.querySelector("#cd-terrain-summary").textContent = terrain.summary;
  terrain.animate();
  terrain.enableAutoRotate();

  if (candidateMap) candidateMap.destroy();
  candidateMap = mountCandidateMap(root.querySelector("#cd-map"), {
    geometry: vm.geometry,
    centroidLonLat: vm.centroidLonLat,
    imageUrl: vm.imagery.afterUrl,
    restrictedZone: vm.restrictedZoneBbox,
  });
  if (candidateMap.marker) {
    candidateMap.marker.bindPopup(`
      <strong>${vm.changeTypeLabel}</strong><br>
      ${vm.confidencePct}% confidence &middot; ${vm.band.toUpperCase()}<br>
      ${vm.effectiveDecision === "undecided" ? "Undecided" : vm.effectiveDecision.toUpperCase()}`);
  }

  if (gateChart) gateChart.destroy();
  gateChart = mountGateChart(root.querySelector("#cd-gate-chart"), vm.gates);

  root.querySelector("#cd-back").addEventListener("click", () => go("overview"));
  root.querySelector("#cd-prev").addEventListener("click", () => stepCandidate(-1));
  root.querySelector("#cd-next").addEventListener("click", () => stepCandidate(1));

  const confirmBtn = root.querySelector("#cd-confirm");
  const rejectBtn = root.querySelector("#cd-reject");
  const reopenBtn = root.querySelector("#cd-reopen");
  if (confirmBtn) confirmBtn.addEventListener("click", () => submitDecision("confirm"));
  if (rejectBtn) rejectBtn.addEventListener("click", () => submitDecision("reject"));
  if (reopenBtn) reopenBtn.addEventListener("click", () => submitDecision("reopen"));

  root.querySelectorAll(".cd-year-tabs .year-tab").forEach((btn) =>
    btn.addEventListener("click", () => {
      selectedBeforeYear = btn.dataset.year;
      render(vm);
    }));
}

async function submitDecision(decision) {
  if (!currentId) return;
  try {
    await api.postDecision(currentId, { decision, note: "", analyst: "LOCAL ANALYST" });
    await show(currentId); // re-fetch: shows the freshly-appended decision state
    refreshQueueBadge();
  } catch (e) {
    showToast(`Could not record decision: ${e.message}`);
  }
}

function showToast(msg) {
  const t = document.getElementById("toast");
  t.textContent = msg;
  t.classList.add("show");
  setTimeout(() => t.classList.remove("show"), 4000);
}

async function stepCandidate(dir) {
  if (!currentIds.length) return go("candidate", currentId);
  const idx = currentIds.indexOf(currentId);
  const next = currentIds[(idx + dir + currentIds.length) % currentIds.length];
  go("candidate", next);
}

export function setBrowseOrder(ids) { currentIds = ids || []; }

async function show(id) {
  if (!id) {
    root.innerHTML = `<div class="empty-state"><div class="empty-state-title">No candidate selected</div>
      Open a finding from the Overview, Review Queue, or Search to see its evidence here.</div>`;
    return;
  }
  currentId = id;
  selectedBeforeYear = null;
  root.innerHTML = `<div class="loading-state">Loading candidate ${id} from the local index…</div>`;
  try {
    const [detail, regions, zones] = await Promise.all([api.getCandidate(id), loadRegions(api), loadZones(api)]);
    currentVm = toCandidateDetailViewModel(detail, regions, api);
    currentVm.restrictedZoneBbox = currentVm.restrictedZone ? zoneBboxByName(zones, currentVm.restrictedZone.name) : null;
    render(currentVm);
  } catch (e) {
    root.innerHTML = `
      <div class="error-state">
        <div class="error-state-msg">Could not load candidate ${id}.</div>
        <button class="btn" id="cd-retry">Retry</button>
        <details style="margin-top:10px;"><summary class="t-small">Details</summary>
          <div class="error-detail">${e.message}</div></details>
      </div>`;
    root.querySelector("#cd-retry")?.addEventListener("click", () => show(id));
  }
}

registerScreen("candidate", { mount, show });

export function openCandidate(id, browseIds) {
  if (browseIds) setBrowseOrder(browseIds);
  go("candidate", id);
}

export function wireGlobalDecisionKeys() {
  return {
    onConfirm: () => { if (currentScreenIsCandidate() && currentVm && currentVm.effectiveDecision === "undecided") submitDecision("confirm"); },
    onReject: () => { if (currentScreenIsCandidate() && currentVm && currentVm.effectiveDecision === "undecided") submitDecision("reject"); },
  };
}
function currentScreenIsCandidate() {
  return location.hash.startsWith("#/candidate");
}
