import { api } from "../api-client.js";
import { registerScreen } from "../router.js";
import { toModelInfoViewModel, classToGroupName, groupColor, contextualNotes } from "../viewmodels/detection.js";
import { mountCompareSlider, compareSurfaceHtml } from "../compare-slider.js";

let root;
let sliderCtl = null;
let sceneResizeObserver = null;
// "boxes" matches the chip marked active in the initial markup below - they
// used to disagree, so the very first render showed confidence labels
// cluttering every box while the UI implied "boxes only" was selected.
let state = { observationId: null, row: null, col: null, mode: "boxes", tileW: null, tileH: null };

function mount() {
  root = document.getElementById("screen-detect");
  root.classList.add("od-root");
}

// The annotation layer draws boxes as percentages of the tile's own pixel
// grid, which only lines up with the visible image when the surface's box
// has *exactly* the tile's aspect ratio. Leaving that to CSS (height:100% +
// max-width:100% + `aspect-ratio`) only holds while the flex column happens
// to be wide enough; once the available width is the tighter constraint,
// max-width clamps the box's width but not its height, so the surface's
// ratio drifts from the tile's and object-fit:contain letterboxes the <img>
// inside a box the annotation layer doesn't know about - detections then
// read as drifted past the visible photo's edge. Sizing the surface with
// explicit pixel dimensions computed the same way "contain" would keeps the
// two in lockstep at every viewport size.
function fitSurfaceToScene() {
  const scene = document.getElementById("od-scene");
  const surface = scene && scene.querySelector(".compare-surface");
  if (!surface || !state.tileW || !state.tileH) return;
  const availW = scene.clientWidth;
  const availH = scene.clientHeight;
  const ratio = state.tileW / state.tileH;
  let w = availW, h = w / ratio;
  if (h > availH) { h = availH; w = h * ratio; }
  surface.style.width = `${Math.floor(w)}px`;
  surface.style.height = `${Math.floor(h)}px`;
}

function boxHtml(d, tileW, tileH, mode) {
  // polygon_px is already the 4 corners of the oriented box (rotated rectangle),
  // so its true width/height/angle come from the corner edges themselves - not
  // from the axis-aligned bbox of those corners (which is larger than the box
  // whenever it's rotated) combined with a second rotation, which used to
  // double up the rotation and push boxes/labels outside the image.
  const pts = d.polygon_px;
  const cx = pts.reduce((s, p) => s + p[0], 0) / pts.length;
  const cy = pts.reduce((s, p) => s + p[1], 0) / pts.length;
  const w = Math.hypot(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]);
  const h = Math.hypot(pts[2][0] - pts[1][0], pts[2][1] - pts[1][1]);
  const angle = Math.atan2(pts[1][1] - pts[0][1], pts[1][0] - pts[0][0]) * (180 / Math.PI);
  const color = groupColor(classToGroupName(d.class));
  const leftPct = (cx / tileW) * 100, topPct = (cy / tileH) * 100;
  const wPct = (w / tileW) * 100, hPct = (h / tileH) * 100;
  const box = `
    <div class="od-box" style="left:${leftPct}%; top:${topPct}%; width:${wPct}%; height:${hPct}%;
      border-color:${color}; box-shadow:0 0 6px ${color}44; transform: translate(-50%,-50%) rotate(${angle}deg);"></div>`;
  // The label used to be a child of the rotated box, so it inherited that
  // rotation and swung out to the side (sometimes past the image edge) for
  // any steeply-angled detection. Positioned as its own sibling here, at the
  // box's centroid in plain (unrotated) image space, it always reads upright
  // and sits just above the box regardless of the detection's heading.
  const label = mode === "boxes-confidence"
    ? `<span class="od-box-label" style="left:${leftPct}%; top:${topPct}%; color:${color};">${d.score.toFixed(2)}</span>`
    : "";
  return box + label;
}

async function loadTile(obsId, row, col) {
  const host = document.getElementById("od-scene");
  host.innerHTML = `<div class="loading-state">Loading tile ${row},${col}…</div>`;
  try {
    const detail = await api.detectTileDetections(obsId, row, col);
    const imgUrl = api.detectTileImageUrl(obsId, row, col);
    host.innerHTML = compareSurfaceHtml({
      baseHtml: `<img src="${imgUrl}" alt="Raw imagery">`,
      afterHtml: `<div class="od-annotation-layer" id="od-annotation">${detail.detections.map((d) => boxHtml(d, detail.width, detail.height, state.mode)).join("")}</div>`,
      cornerTL: "DETECTIONS DRAWN",
      cornerTR: "IMAGERY ONLY",
    });
    state.tileW = detail.width;
    state.tileH = detail.height;
    const surface = host.querySelector(".compare-surface");
    fitSurfaceToScene();
    if (sliderCtl) sliderCtl.destroy();
    sliderCtl = mountCompareSlider(surface, { initial: 62, ariaLabel: "Compare imagery and detections" });
    if (sceneResizeObserver) sceneResizeObserver.disconnect();
    sceneResizeObserver = new ResizeObserver(() => fitSurfaceToScene());
    sceneResizeObserver.observe(host);
    renderLegend(detail.detections);
    renderContextNotes(detail.detections);
  } catch (e) {
    host.innerHTML = `<div class="error-state"><div class="error-state-msg">Could not load this tile's detections.</div>
      <details><summary class="t-small">Details</summary><div class="error-detail">${e.message}</div></details></div>`;
  }
}

function renderLegend(detections) {
  const groups = [...new Set(detections.map((d) => classToGroupName(d.class)))];
  document.getElementById("od-legend").innerHTML = groups.map((g) =>
    `<span class="od-legend-item"><span class="od-legend-swatch" style="border-color:${groupColor(g)};"></span>${g}</span>`).join("");
}

function renderContextNotes(detections) {
  const notes = contextualNotes(detections);
  const host = document.getElementById("od-context-notes");
  if (!host) return;
  host.innerHTML = notes.length
    ? notes.map((n) => `<div class="od-context-note">${n}</div>`).join("")
    : "";
}

async function renderRight(obsId) {
  const [{ observations }, modelInfo] = await Promise.all([api.detectObservations(), api.detectModelInfo()]);
  const obs = observations.find((o) => o.observation_id === obsId);
  const vm = toModelInfoViewModel(modelInfo);
  const grouped = {};
  if (obs && obs.by_class) {
    for (const [cls, n] of Object.entries(obs.by_class)) {
      const g = classToGroupName(cls);
      grouped[g] = (grouped[g] || 0) + n;
    }
  }
  const countsHtml = vm.groups.map((g) => `
    <div class="od-count-card">
      <span class="od-count-swatch" style="border-color:${g.color};"></span>
      <div class="min0" style="flex:1;">
        <div class="t-row-title">${g.name}</div>
        <div class="t-small" style="color:var(--ink-dim);">${g.foundPct != null ? g.foundPct + "% found · " + g.falsePct + "% false" : "no measured operating point"}</div>
      </div>
      <div class="t-stat">${grouped[g.name] || 0}</div>
    </div>`).join("");

  const reliabilityHtml = vm.groups.map((g) => `
    <div class="od-rel-row">
      <div class="t-row-title" style="color:${g.color};">${g.name}</div>
      <div class="t-meta">${g.foundPct != null ? g.foundPct + "% found · " + g.falsePct + "% false" : "—"}</div>
      ${g.caveats.length ? `<div class="t-small" style="color:var(--ink-dim); margin-top:2px;">${g.caveats[0]}</div>` : ""}
    </div>`).join("");

  document.getElementById("od-right").innerHTML = `
    <div class="od-right-section">
      <div class="t-eyebrow">Counts in this scene</div>
      ${countsHtml || `<div class="t-small" style="color:var(--ink-dim);">No detections recorded for this observation.</div>`}
    </div>
    <div class="od-right-section">
      <div class="t-eyebrow">How reliable these counts are</div>
      <div class="t-small" style="color:var(--ink-3); margin-bottom:8px;">Measured against a held-out labeled test set. Read these before you report a number.</div>
      ${reliabilityHtml}
    </div>
    <div class="od-right-section">
      <div class="tinted-box warn">
        <div class="tinted-box-title">Known blind spots</div>
        <div class="tinted-box-text">Objects under netting, in deep shadow, or smaller than a few pixels are routinely missed. Counts are a floor, not a total.</div>
      </div>
    </div>`;
}

async function show() {
  root.innerHTML = `
    <div class="od-left">
      <div class="od-header fixed">
        <div>
          <span class="t-screen-title">Assets on the ground</span>
          <div class="t-small" id="od-provenance" style="color:var(--ink-dim); margin-top:3px;"></div>
        </div>
        <div class="od-chips">
          <button class="chip active" data-mode="boxes">BOXES ONLY</button>
          <button class="chip" data-mode="boxes-confidence">BOXES + CONFIDENCE</button>
        </div>
      </div>
      <select id="od-observation-select" class="od-select"></select>
      <div class="od-scene" id="od-scene"></div>
      <div class="od-legend" id="od-legend"></div>
      <div class="od-context-notes" id="od-context-notes"></div>
    </div>
    <div class="od-right scroll-pane" id="od-right"></div>`;

  root.querySelectorAll("[data-mode]").forEach((b) =>
    b.addEventListener("click", () => {
      root.querySelectorAll("[data-mode]").forEach((x) => x.classList.remove("active"));
      b.classList.add("active");
      state.mode = b.dataset.mode;
      if (state.observationId) loadTile(state.observationId, state.row, state.col);
    }));

  try {
    const { observations } = await api.detectObservations();
    if (!observations.length) {
      root.querySelector("#od-scene").innerHTML = `<div class="empty-state"><div class="empty-state-title">No detection runs staged</div>
        No observation has stored oriented-box detections yet.</div>`;
      return;
    }
    const select = document.getElementById("od-observation-select");
    select.innerHTML = observations.map((o) => `<option value="${o.observation_id}">${o.aoi_name} (${o.n_detections} detections)</option>`).join("");
    select.addEventListener("change", () => selectObservation(select.value));
    await selectObservation(observations[0].observation_id);
  } catch (e) {
    root.querySelector("#od-scene").innerHTML = `<div class="error-state"><div class="error-state-msg">Could not load observations.</div>
      <details><summary class="t-small">Details</summary><div class="error-detail">${e.message}</div></details></div>`;
  }
}

async function selectObservation(obsId) {
  state.observationId = obsId;
  const { tiles } = await api.detectTiles(obsId);
  if (!tiles.length) {
    document.getElementById("od-scene").innerHTML = `<div class="empty-state">No tiles with detections in this observation.</div>`;
    return;
  }
  const top = tiles[0];
  state.row = top.row; state.col = top.col;
  document.getElementById("od-provenance").textContent = `SCENE ${obsId} · tile row ${top.row}, col ${top.col}`;
  await loadTile(obsId, top.row, top.col);
  renderRight(obsId);
}

registerScreen("detect", { mount, show });
