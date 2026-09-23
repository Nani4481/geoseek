/* geoseek analyst - offline single-page UI. Vanilla JS, no dependencies, no
   network beyond this same FastAPI process. Every fetch() is an absolute path
   on the current origin; there is no <script src>, no CDN, no web map tile. */
"use strict";

const TYPES = ["water_gain", "water_loss", "construction", "clearance", "road", "other"];
const CTYPE_COLOR = {
  water_gain: "#1f6feb", water_loss: "#7b3fbf", construction: "#d98324",
  clearance: "#b5850b", road: "#6b7280", other: "#3f9e6b",
};
const AOI_FALLBACK = [82.0124, 26.3613, 82.8459, 27.1098];
const CTYPE_HUMAN = {
  water_gain: "New open water / flooding", water_loss: "Water body shrank or dried",
  construction: "New built-up surface", clearance: "Vegetation or land cleared",
  road: "New road / linear corridor", other: "Surface change (unclassified)",
};
let PRES = null;          // /presentation/summary, fetched once at boot
let OBS_DATES = [];        // e.g. ["2019-03-30","2021-03-04","2024-03-08"]
let REGIONS = null;       // /regions, fetched once and cached: [{name, bbox, n_observations}]
const REGION_PALETTE = ["#1f6feb", "#7b3fbf", "#d98324", "#b5850b", "#3f9e6b", "#6b7280",
  "#c93c37", "#0aa1a3", "#8a5a2b"];

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
const esc = (s) => String(s == null ? "" : s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const num = (v, d = 2) => (v == null || isNaN(v)) ? "–" : Number(v).toFixed(d);

let LAST_LATENCY = {};
async function api(path, opts) {
  const t0 = performance.now();
  const r = await fetch(path, opts);
  const ms = performance.now() - t0;
  LAST_LATENCY[path.split("?")[0]] = ms;
  setChip(`${path.split("?")[0]}  ${ms.toFixed(0)} ms`);
  if (!r.ok) {
    let detail = r.statusText;
    try { detail = (await r.json()).detail || detail; } catch (e) {}
    throw new Error(`${r.status} ${detail}`);
  }
  const ct = r.headers.get("content-type") || "";
  return ct.includes("json") ? r.json() : r;
}
function setChip(t) { $("#latencyChip").textContent = t; }
function toast(t) {
  const el = $("#toast"); el.textContent = t; el.classList.add("show");
  clearTimeout(toast._t); toast._t = setTimeout(() => el.classList.remove("show"), 2600);
}

/* ---------------------------------------------------------------- map */
class CoordMap {
  constructor(canvas, onPick, opts) {
    this.c = canvas; this.ctx = canvas.getContext("2d");
    this.items = []; this.bbox = AOI_FALLBACK; this.onPick = onPick; this.sel = null;
    this.opts = opts || {};
    canvas.addEventListener("click", (e) => this._click(e));
    if (this.opts.onDrawRect) {
      canvas.style.cursor = "crosshair";
      canvas.addEventListener("mousedown", (e) => this._dragStart(e));
      canvas.addEventListener("mousemove", (e) => this._dragMove(e));
      window.addEventListener("mouseup", (e) => this._dragEnd(e));
    }
    new ResizeObserver(() => this.draw()).observe(canvas);
  }
  _dragStart(e) {
    const r = this.c.getBoundingClientRect();
    this._drag = { x0: e.clientX - r.left, y0: e.clientY - r.top, x1: null, y1: null };
  }
  _dragMove(e) {
    if (!this._drag) return;
    const r = this.c.getBoundingClientRect();
    this._drag.x1 = e.clientX - r.left; this._drag.y1 = e.clientY - r.top;
    this.draw();
  }
  _dragEnd() {
    if (!this._drag) return;
    const d = this._drag; this._drag = null;
    const moved = d.x1 != null && (Math.abs(d.x1 - d.x0) > 4 || Math.abs(d.y1 - d.y0) > 4);
    this.draw();
    if (!moved) return;
    const p = this._proj();
    const lon0 = p.lon(Math.min(d.x0, d.x1)), lon1 = p.lon(Math.max(d.x0, d.x1));
    const lat0 = p.lat(Math.max(d.y0, d.y1)), lat1 = p.lat(Math.min(d.y0, d.y1));
    this.opts.onDrawRect([lon0, lat0, lon1, lat1]);
  }
  setData(items, bbox) {
    this.items = items || [];
    if (bbox) this.bbox = bbox;
    else if (this.items.length) this.bbox = this._autobbox();
    this.draw();
  }
  select(id) { this.sel = id; this.draw(); }
  _autobbox() {
    let w = 180, s = 90, e = -180, n = -90;
    for (const it of this.items) {
      const pts = it.ring || [[it.lon, it.lat]];
      for (const [x, y] of pts) { w = Math.min(w, x); e = Math.max(e, x); s = Math.min(s, y); n = Math.max(n, y); }
    }
    if (!(e > w)) return AOI_FALLBACK;
    const px = (e - w) * 0.06 + 1e-4, py = (n - s) * 0.06 + 1e-4;
    return [w - px, s - py, e + px, n + py];
  }
  _proj() {
    const dpr = window.devicePixelRatio || 1;
    const W = this.c.clientWidth, H = this.c.clientHeight;
    this.c.width = W * dpr; this.c.height = H * dpr;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const [w, s, e, n] = this.bbox;
    const sx = W / (e - w), sy = H / (n - s), k = Math.min(sx, sy);
    const ox = (W - k * (e - w)) / 2, oy = (H - k * (n - s)) / 2;
    return {
      W, H,
      x: (lon) => ox + (lon - w) * k,
      y: (lat) => H - oy - (lat - s) * k,
      lon: (x) => w + (x - ox) / k,
      lat: (y) => s + (H - oy - y) / k,
    };
  }
  draw() {
    const ctx = this.ctx, p = this._proj();
    const css = getComputedStyle(document.body);
    ctx.clearRect(0, 0, p.W, p.H);
    ctx.fillStyle = css.getPropertyValue("--panel"); ctx.fillRect(0, 0, p.W, p.H);
    // graticule
    ctx.strokeStyle = css.getPropertyValue("--line");
    ctx.fillStyle = css.getPropertyValue("--muted");
    ctx.lineWidth = 1; ctx.font = "10px system-ui, sans-serif";
    const [w, s, e, n] = this.bbox;
    const stepLon = niceGraticuleStep(e - w), stepLat = niceGraticuleStep(n - s);
    const decLon = Math.max(0, -Math.floor(Math.log10(stepLon))), decLat = Math.max(0, -Math.floor(Math.log10(stepLat)));
    let lastLabelY = -Infinity;
    for (let lon = Math.ceil(w / stepLon) * stepLon; lon < e; lon += stepLon) {
      const x = p.x(lon); ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, p.H); ctx.stroke();
      ctx.fillText(lon.toFixed(decLon) + "°E", x + 2, p.H - 3);
    }
    for (let lat = Math.ceil(s / stepLat) * stepLat; lat < n; lat += stepLat) {
      const y = p.y(lat); ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(p.W, y); ctx.stroke();
      // skip a label that would overlap the previous one (happens if stepLat
      // maps to less than a text-line's worth of pixels at extreme aspect ratios)
      if (Math.abs(y - lastLabelY) >= 11) { ctx.fillText(lat.toFixed(decLat) + "°N", 3, y - 3); lastLabelY = y; }
    }
    this._screen = [];
    for (const it of this.items) {
      const selected = it.id != null && it.id === this.sel;
      if (it.ring) {
        const scr = it.ring.map(([lo, la]) => [p.x(lo), p.y(la)]);
        this._screen.push({ it, scr, poly: true });
        ctx.beginPath();
        scr.forEach(([x, y], i) => i ? ctx.lineTo(x, y) : ctx.moveTo(x, y));
        ctx.closePath();
        ctx.fillStyle = hexA(it.color, selected ? 0.5 : 0.22);
        ctx.strokeStyle = it.color; ctx.lineWidth = selected ? 2.5 : 1.2;
        ctx.fill(); ctx.stroke();
      } else {
        const x = p.x(it.lon), y = p.y(it.lat);
        this._screen.push({ it, scr: [[x, y]], poly: false });
        ctx.beginPath(); ctx.arc(x, y, selected ? 7 : 4.5, 0, 7);
        ctx.fillStyle = it.color; ctx.fill();
        if (selected) { ctx.strokeStyle = css.getPropertyValue("--ink"); ctx.lineWidth = 2; ctx.stroke(); }
      }
    }
    if (this._drag && this._drag.x1 != null) {
      const x0 = Math.min(this._drag.x0, this._drag.x1), x1 = Math.max(this._drag.x0, this._drag.x1);
      const y0 = Math.min(this._drag.y0, this._drag.y1), y1 = Math.max(this._drag.y0, this._drag.y1);
      const accent = css.getPropertyValue("--accent").trim();
      ctx.fillStyle = hexA(accent, 0.15);
      ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
      ctx.strokeStyle = accent; ctx.lineWidth = 1.5; ctx.setLineDash([5, 3]);
      ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
      ctx.setLineDash([]);
    }
  }
  _click(e) {
    if (!this._screen) return;
    const r = this.c.getBoundingClientRect();
    const mx = e.clientX - r.left, my = e.clientY - r.top;
    let hit = null;
    for (const s of this._screen) {
      if (s.poly) { if (pointInPoly(mx, my, s.scr)) hit = s.it; }
      else { const [x, y] = s.scr[0]; if (Math.hypot(x - mx, y - my) < 9) hit = s.it; }
    }
    if (hit && this.onPick) this.onPick(hit);
  }
}
function hexA(hex, a) {
  const m = hex.replace("#", ""); const bi = parseInt(m.length === 3 ? m.split("").map(c => c + c).join("") : m, 16);
  return `rgba(${(bi >> 16) & 255},${(bi >> 8) & 255},${bi & 255},${a})`;
}
function niceGraticuleStep(range, targetLines = 6) {
  // A fixed 0.1deg graticule step looked fine at AOI scale (~1deg spans) but
  // produces hundreds of sub-pixel-spaced grid lines - and as many overlapping
  // labels stacked at the left edge - once results are scattered across a much
  // wider bbox (e.g. search hits spanning several regions). Pick a "nice"
  // (1/2/5 x10^n) step sized so roughly `targetLines` gridlines fit the span,
  // the standard adaptive-graticule approach, instead of a constant.
  if (!(range > 0)) return 0.1;
  const raw = range / targetLines;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const nice = norm < 1.5 ? 1 : norm < 3.5 ? 2 : norm < 7.5 ? 5 : 10;
  return nice * mag;
}
function pointInPoly(x, y, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if (((yi > y) !== (yj > y)) && (x < ((xj - xi) * (y - yi)) / (yj - yi) + xi)) inside = !inside;
  }
  return inside;
}
/* ---------------------------------------------------------------- regions
   /regions once, cached; backs the region picker on the watch-area form and
   the Queue/Search AOI filters so nobody has to type a raw bbox string. */
function bboxStr(bbox) { return bbox.map(x => Number(x.toFixed(5))).join(","); }
async function loadRegions() {
  if (REGIONS) return REGIONS;
  try { REGIONS = (await api("/regions")).regions || []; }
  catch (e) { REGIONS = []; }
  return REGIONS;
}
function fillRegionSelect(selectEl, { includeBbox = true } = {}) {
  loadRegions().then(regions => {
    const extra = regions.map(r =>
      `<option value="${esc(r.name)}">${esc(r.name)} (${r.n_observations} obs)</option>`).join("");
    selectEl.insertAdjacentHTML("beforeend", extra);
  });
}
function regionByName(name) { return (REGIONS || []).find(r => r.name === name); }
function legend(elId, types) {
  $(elId).innerHTML = types.map(t => `<span><i style="background:${CTYPE_COLOR[t] || "#888"}"></i>${t}</span>`).join("");
}

/* ---------------------------------------------------------------- routing */
const views = ["overview", "search", "queue", "detail", "discovery", "detect", "watch"];
function go(route) { location.hash = "#/" + route; }
function router() {
  const parts = (location.hash || "#/overview").slice(2).split("/");
  const route = views.includes(parts[0]) ? parts[0] : "overview";
  views.forEach(v => $("#view-" + v).classList.toggle("active", v === route));
  $$("nav button").forEach(b => b.classList.toggle("active", b.dataset.route === route));
  if (route === "overview") ensureOverview();
  if (route === "search") ensureSearch();
  if (route === "queue") ensureQueue();
  if (route === "discovery") ensureDiscovery();
  if (route === "detect") ensureDetect();
  if (route === "watch") ensureWatch();
  if (route === "detail" && parts[1]) openDetail(decodeURIComponent(parts[1]));
}
window.addEventListener("hashchange", router);
$$("nav button").forEach(b => b.addEventListener("click", () => go(b.dataset.route)));

/* ---------------------------------------------------------------- OVERVIEW
   Demo-facing landing view. Read-only projection of /presentation/summary +
   the same /candidates footprints the Review Queue uses. */
let overviewMap, overviewInit = false;
const nf = (n) => (n == null ? "–" : Number(n).toLocaleString());

function persistenceHuman(persistence, nLater) {
  const obs = nLater === 1 ? "observation" : "observations";
  return ({
    persistent: `Confirmed across ${nLater} later ${obs}`,
    progressive: `Grew steadily across ${nLater} later ${obs}`,
    recent: "Only visible in the most recent interval",
    transient: "Appeared, then reverted — treat with caution",
    inconsistent: "Flickers across dates — low trust",
  })[persistence] || "Single before / after pair only";
}
function laterObsCount(traj) {
  const end = ((traj && traj.earliest_supported_change) || {}).window;
  const e = end ? end[1] : null;
  const dates = OBS_DATES.length ? OBS_DATES
    : Array.from(new Set([].concat(...((traj && traj.intervals) || []).map(i => i.window || [])))).sort();
  const n = e ? dates.filter(d => d >= e).length : 0;
  return n || 1;
}
function confBand(v) { v = +v || 0; return v >= 0.85 ? "high" : (v >= 0.60 ? "medium" : "low"); }

function ensureOverview() {
  if (overviewInit) { return; }
  overviewInit = true;
  overviewMap = new CoordMap($("#ov_map"), (it) => { if (it.id) go("detail/" + it.id); });
  $("#ov_legend").innerHTML = TYPES.map(t =>
    `<span><i style="background:${CTYPE_COLOR[t] || "#888"}"></i>${esc(CTYPE_HUMAN[t] || t)}</span>`).join("");
  $("#ov-go").addEventListener("click", overviewSearch);
  $("#ov-q").addEventListener("keydown", e => { if (e.key === "Enter") overviewSearch(); });
  loadOverview();
}
function overviewSearch() {
  const q = $("#ov-q").value.trim();
  if (q) $("#q").value = q;
  go("search");
}
async function loadOverview() {
  try {
    if (!PRES) PRES = await api("/presentation/summary");
    OBS_DATES = PRES.observation_dates || OBS_DATES;
    renderOverviewCounters(PRES);
    renderFeatured(PRES.featured || []);
    prewarm(featuredUrls(PRES.featured || []));
    renderLatestAlerts(PRES.latest_alerts || []);
  } catch (e) {
    $("#ov-featured").innerHTML = `<span class="muted">${esc(e.message)}</span>`;
  }
  try {
    const d = await api("/candidates?limit=400&sort=queue_score");
    overviewMap.setData((d.candidates || []).map(c => ({
      id: c.candidate_id, ring: c.geometry && c.geometry.coordinates[0],
      lon: c.centroid_lonlat[0], lat: c.centroid_lonlat[1],
      color: CTYPE_COLOR[c.change_type] || "#888",
    })), AOI_FALLBACK);
  } catch (e) { /* map is a nicety; counters + cards already rendered */ }
}
function renderOverviewCounters(p) {
  const c = p.counters || {};
  if (p.aoi) $("#ov-aoi").textContent = p.aoi;
  const dates = p.observation_dates || [];
  if (dates.length) {
    const obs = dates.length === 1 ? "observation" : `${dates.length} observations`;
    $("#ov_mapnote").innerHTML =
      `Every flagged change over the ${esc(dates[0])} &rarr; ${esc(dates[dates.length - 1])} window ` +
      `(${obs}), coloured by type. Offline canvas in EPSG:4326 &mdash; no web map tiles. ` +
      `Click a footprint to open it.`;
  }
  const cells = [
    ["change_candidates", "change candidates", true],
    ["tiles_indexed", "tiles indexed", false],
    ["regions", "regions", false],
    ["scenes", "satellite scenes", false],
    ["high_confidence", "high-confidence", false],
  ];
  $("#ov-counters").innerHTML = cells.map(([k, label, accent]) =>
    `<div class="stat${accent ? " accent" : ""}"><div class="n">${nf(c[k])}</div><div class="l">${label}</div></div>`
  ).join("");
}
function renderLatestAlerts(alerts) {
  const el = $("#ov-alerts");
  if (!alerts.length) { el.innerHTML = `<span class="muted">no alerts yet &mdash; define a watch area to start monitoring</span>`; return; }
  el.innerHTML = alerts.map(n => `
    <div class="row alertrow" data-ids="${esc(n.candidate_ids.join(","))}" data-name="${esc(n.watch_name || n.watch_id)}">
      <span class="band ${n.severity || "low"}">${(n.severity || "low").toUpperCase()}</span>
      <b>${esc(n.watch_name || n.watch_id)}</b>
      <span class="muted">${n.candidate_ids.length} candidate(s) &middot; obs ${esc(n.observation_id)}${n.observation_date ? " (" + esc(n.observation_date) + ")" : ""}</span>
      <span class="muted small">${esc((n.created_at || "").replace("T", " ").slice(0, 16))}</span>
    </div>`).join("");
  $$("#ov-alerts .alertrow").forEach(row => row.addEventListener("click", () =>
    openQueueForIds(row.dataset.ids.split(","), `${row.dataset.ids.split(",").length} candidate(s) from alert "${row.dataset.name}"`)));
}
function featuredUrls(featured) {
  const u = [];
  for (const f of featured) {
    if (!f.imagery) continue;
    u.push(f.imagery.before + "&scale=2", f.imagery.after + "&scale=2");
  }
  return u;
}
function renderFeatured(featured) {
  if (!featured.length) { $("#ov-featured").innerHTML = `<span class="muted">no featured findings</span>`; return; }
  $("#ov-featured").innerHTML = featured.map(f => {
    const band = (f.confidence_band || confBand(f.confidence)).toLowerCase();
    return `<figure class="ff-card" data-id="${esc(f.candidate_id)}">
      <div class="ff-imgs">
        <div class="ff-im"><img loading="lazy" src="${esc(f.imagery.before)}&scale=2" alt="before"><span>before · ${esc(f.before_date)}</span></div>
        <div class="ff-im"><img loading="lazy" src="${esc(f.imagery.after)}&scale=2" alt="after"><span>after · ${esc(f.after_date)}</span></div>
      </div>
      <figcaption>
        <div class="ff-badges">
          <span class="badge b-${esc(f.change_type)}">${esc(f.change_type_human || CTYPE_HUMAN[f.change_type] || f.change_type)}</span>
          <span class="band ${band}">${band[0].toUpperCase() + band.slice(1)} confidence</span>
        </div>
        <p>${esc(f.caption)}</p>
        <div class="ff-sub">confidence ${num(f.confidence)} · ${esc(f.persistence_human)}</div>
      </figcaption>
    </figure>`;
  }).join("");
  $$("#ov-featured .ff-card").forEach(el =>
    el.addEventListener("click", () => go("detail/" + el.dataset.id)));
}

/* fire-and-forget: pull demo-path imagery into the HTTP cache so the guided
   walkthrough paints instantly. Same-origin only. */
const _prewarmed = new Set();
function prewarm(urls) {
  for (const u of urls) {
    if (!u || _prewarmed.has(u)) continue;
    _prewarmed.add(u);
    fetch(u, { cache: "force-cache" }).catch(() => {});
  }
}

/* ---------------------------------------------------------------- SEARCH */
let searchMap, searchInit = false;
function ensureSearch() {
  if (searchInit) return; searchInit = true;
  searchMap = new CoordMap($("#s_map"), (it) => { if (it.tile_id) toast("tile " + it.tile_id); });
  legend("#s_legend", ["other"]);
  $("#s_legend").innerHTML = `<span><i style="background:#1f6feb"></i>search hit (size ∝ score)</span>`;
  $("#s_go").addEventListener("click", runSearch);
  $("#q").addEventListener("keydown", e => { if (e.key === "Enter") runSearch(); });
  $("#s_upload").addEventListener("click", () => $("#s_file").click());
  $("#s_file").addEventListener("change", runImageSearch);
  fillRegionSelect($("#s_region"));
  $("#s_region").addEventListener("change", (e) => {
    const r = regionByName(e.target.value);
    $("#s_bbox").value = r ? bboxStr(r.bbox) : "";
    runSearch();
  });
  runSearch();
}
function searchFilterQS() {
  const p = new URLSearchParams();
  if ($("#s_bbox").value.trim()) p.set("bbox", $("#s_bbox").value.trim());
  if ($("#s_from").value) p.set("date_start", $("#s_from").value);
  if ($("#s_to").value) p.set("date_end", $("#s_to").value);
  if ($("#s_sensor").value) p.set("sensor", $("#s_sensor").value);
  if ($("#s_cloud").value !== "") p.set("max_cloud_fraction", (Number($("#s_cloud").value) / 100).toFixed(3));
  return p;
}
async function runSearch() {
  const p = searchFilterQS(); p.set("q", $("#q").value || "water"); p.set("k", "30");
  $("#s_cards").innerHTML = `<span class="spinner"></span>`;
  try {
    const d = await api("/search/text?" + p.toString());
    renderSearchResults(d);
  } catch (e) { $("#s_cards").innerHTML = `<span class="muted">${esc(e.message)}</span>`; }
}
async function runImageSearch(ev) {
  const f = ev.target.files[0]; if (!f) return;
  const b64 = await new Promise(res => { const r = new FileReader(); r.onload = () => res(r.result.split(",")[1]); r.readAsDataURL(f); });
  const p = searchFilterQS();
  const body = { image_base64: b64, k: 30 };
  for (const [k, v] of p) body[k] = v;
  $("#s_cards").innerHTML = `<span class="spinner"></span>`;
  try {
    const d = await api("/search/image", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
    toast(`image similarity: ${d.count} hits in ${d.latency_ms} ms`);
    renderSearchResults(d);
  } catch (e) { $("#s_cards").innerHTML = `<span class="muted">${esc(e.message)}</span>`; }
}
function renderSearchResults(d) {
  const res = d.results || [];
  $("#s_count").textContent = `${res.length} hit${res.length === 1 ? "" : "s"} · ${d.latency_ms} ms server`;
  $("#s_cards").innerHTML = res.map(r => `
    <div class="card" data-tile="${esc(r.tile_id)}">
      <img class="thumb" loading="lazy" src="/tile/${encodeURIComponent(r.tile_id)}/thumbnail" alt="">
      <div class="meta">
        <div class="row"><b>score ${num(r.score, 3)}</b><span>${esc(r.acq_date)}</span></div>
        <div class="row"><span>${esc(r.sensor)}</span><span>cloud ${num(r.cloud_fraction * 100, 0)}%</span></div>
        <div class="row"><span>${num(r.lat, 4)}°N</span><span>${num(r.lon, 4)}°E</span></div>
      </div>
    </div>`).join("") || `<span class="muted">no hits</span>`;
  $$("#s_cards .card").forEach(c => c.addEventListener("click", () => {
    $("#disc_seed").value = c.dataset.tile; go("discovery"); setTimeout(runDiscovery, 30);
  }));
  const mx = Math.max(...res.map(r => r.score), 1e-6);
  searchMap.setData(res.map(r => ({
    id: r.tile_id, tile_id: r.tile_id, lon: r.lon, lat: r.lat,
    color: "#1f6feb", r: 3 + 6 * (r.score / mx),
  })), null);
}

/* ---------------------------------------------------------------- QUEUE */
let queueMap, queueInit = false, queueSort = "queue_score";
let queueIdsOverride = null;   // set by "Latest alerts" links -> filters to exactly those candidate_ids
function ensureQueue() {
  if (queueInit) return; queueInit = true;
  queueMap = new CoordMap($("#q_map"), (it) => go("detail/" + it.id));
  legend("#q_legend", TYPES);
  $("#f_go").addEventListener("click", loadQueue);
  $("#f_export").addEventListener("click", exportFiltered);
  $("#f_clear_ids").addEventListener("click", () => { queueIdsOverride = null; loadQueue(); });
  fillRegionSelect($("#f_region"));
  $("#f_region").addEventListener("change", (e) => {
    const r = regionByName(e.target.value);
    $("#f_bbox").value = r ? bboxStr(r.bbox) : "";
    loadQueue();
  });
  $$("#q_table th").forEach(th => th.addEventListener("click", () => {
    const k = th.dataset.k === "earliest" ? "rank" : th.dataset.k;
    queueSort = (queueSort === k) ? "-" + k : k;
    loadQueue();
  }));
  loadQueue();
}
function openQueueForIds(ids, note) {
  queueIdsOverride = { ids, note: note || `${ids.length} candidate(s) from an alert` };
  go("queue"); ensureQueue(); loadQueue();
}
function queueQS(extra) {
  const p = new URLSearchParams();
  if (queueIdsOverride) { p.set("candidate_ids", queueIdsOverride.ids.join(",")); }
  else {
    if ($("#f_type").value) p.set("change_type", $("#f_type").value);
    if ($("#f_conf").value !== "") p.set("min_confidence", $("#f_conf").value);
    if ($("#f_pers").value) p.set("persistence", $("#f_pers").value);
    if ($("#f_sensor").value) p.set("sensor", $("#f_sensor").value);
    if ($("#f_dec").value) p.set("decision", $("#f_dec").value);
    if ($("#f_bbox").value.trim()) p.set("bbox", $("#f_bbox").value.trim());
  }
  Object.entries(extra || {}).forEach(([k, v]) => p.set(k, v));
  return p;
}
async function loadQueue() {
  $("#q_idsbanner").classList.toggle("hidden", !queueIdsOverride);
  $("#f_clear_ids").classList.toggle("hidden", !queueIdsOverride);
  if (queueIdsOverride) $("#q_idsbanner").textContent = `Filtered to ${queueIdsOverride.note}`;
  const p = queueQS({ sort: queueSort, limit: 400 });
  $("#q_table tbody").innerHTML = `<tr><td colspan="10"><span class="spinner"></span></td></tr>`;
  try {
    const d = await api("/candidates?" + p.toString());
    window._queue = d;
    $("#q_summary").innerHTML =
      `<span><b>${d.total}</b> candidates match</span><span>showing <b>${d.count}</b></span>` +
      `<span>sorted by <b>${esc(d.sort)}</b></span>` +
      `<span class="muted">click a row or a footprint &rarr; detail</span>`;
    $("#q_table tbody").innerHTML = d.candidates.map(c => `
      <tr data-id="${esc(c.candidate_id)}">
        <td>${c.rank}</td><td class="mono">${esc(c.candidate_id)}</td>
        <td><span class="badge b-${c.change_type}">${esc(c.change_type)}</span></td>
        <td>${num(c.confidence)}</td><td>${num(c.significance)}</td><td>${num(c.queue_score)}</td>
        <td>${Math.round(c.area_m2).toLocaleString()}</td><td>${esc(c.persistence)}</td>
        <td>${esc((c.earliest_supported || []).join(" → "))}</td>
        <td class="verdict ${c.decision}">${esc(c.decision)}</td>
      </tr>`).join("");
    $$("#q_table tbody tr").forEach(tr => tr.addEventListener("click", () => go("detail/" + tr.dataset.id)));
    queueMap.setData(d.candidates.map(c => ({
      id: c.candidate_id, ring: c.geometry && c.geometry.coordinates[0],
      lon: c.centroid_lonlat[0], lat: c.centroid_lonlat[1],
      color: CTYPE_COLOR[c.change_type] || "#888",
    })), AOI_FALLBACK);
  } catch (e) {
    $("#q_table tbody").innerHTML = `<tr><td colspan="10" class="muted">${esc(e.message)}</td></tr>`;
  }
}
async function exportFiltered() {
  const p = queueQS();
  const filters = {}; for (const [k, v] of p) filters[k] = v;
  toast("building export…");
  try {
    const d = await api("/export", {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ filters, format: "both" }),
    });
    const blob = new Blob([JSON.stringify(d.geojson, null, 1)], { type: "application/geo+json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "geoseek_change_candidates.geojson"; a.click();
    URL.revokeObjectURL(a.href);
    toast(`exported ${d.count} features · server copy: ${d.geojson_path}`);
  } catch (e) { toast("export failed: " + e.message); }
}

/* ---------------------------------------------------------------- DETAIL */
let detailState = { id: null, date: null, after: "2024", data: null };
async function openDetail(id) {
  go("detail/" + id); // keep hash canonical
  $("#d_empty").classList.add("hidden"); $("#d_body").classList.remove("hidden");
  $("#d_head").innerHTML = `<span class="spinner"></span> loading ${esc(id)}…`;
  try {
    const d = await api("/candidates/" + encodeURIComponent(id));
    const before = d.imagery.before_dates || d.imagery.dates || ["2019"];
    detailState = { id, date: before[before.length - 1], after: d.imagery.after_date || "2024", data: d };
    renderDetail(d);
  } catch (e) { $("#d_head").innerHTML = `<span class="muted">${esc(e.message)}</span>`; }
}
function renderDetail(d) {
  const ct = d.change_type;
  const band = confBand(d.confidence);
  const nLater = laterObsCount(d.temporal_trajectory);
  const persHuman = persistenceHuman(d.persistence, nLater);
  $("#d_head").innerHTML = `
    <div class="kv"><span class="k">candidate</span><span class="v mono">${esc(d.candidate_id)}</span></div>
    <div class="kv"><span class="k">type</span><span class="v"><span class="badge b-${ct}">${esc(ct)}</span></span>
      <span class="plain">${esc(CTYPE_HUMAN[ct] || ct)}</span></div>
    <div class="kv" style="min-width:150px"><span class="k">confidence</span>
      <span class="v"><span class="band ${band}">${band[0].toUpperCase() + band.slice(1)}</span></span>
      <span class="under">${num(d.confidence)}</span></div>
    <div class="kv"><span class="k">significance</span><span class="v">${num(d.significance)}</span></div>
    <div class="kv"><span class="k">area</span><span class="v">${Math.round(d.area_m2).toLocaleString()} m²</span></div>
    <div class="kv"><span class="k">location</span><span class="v">${num(d.centroid_lonlat[1], 4)}°N ${num(d.centroid_lonlat[0], 4)}°E</span></div>
    <div class="kv"><span class="k">persistence</span><span class="v">${esc(d.persistence)}</span>
      <span class="plain">${esc(persHuman)}</span></div>
    <div class="kv"><span class="k">earliest supported</span><span class="v">${esc((d.earliest_supported || []).join(" → ") || "–")}</span></div>
    <div class="kv"><span class="k">verdict</span><span class="v verdict ${d.current_decision ? d.current_decision.decision : "undecided"}">${d.current_decision ? d.current_decision.decision : "undecided"}</span></div>`;

  // date selector — BEFORE dates only (AFTER + overlay are pinned to the pair's later obs)
  const beforeDates = d.imagery.before_dates || d.imagery.dates || [detailState.date];
  $("#d_dates").innerHTML =
    `<span class="datelbl">before:</span>` +
    beforeDates.map(x =>
      `<button data-d="${x}" class="${x === detailState.date ? "active" : ""}">${x}</button>`).join("") +
    `<span class="datelbl">after: ${esc(detailState.after)}</span>`;
  $$("#d_dates button").forEach(b => b.addEventListener("click", () => {
    detailState.date = b.dataset.d;
    $$("#d_dates button").forEach(x => x.classList.toggle("active", x === b));
    paintImages();
  }));
  paintImages();
  const es = d.temporal_trajectory.earliest_supported_change || {};
  $("#d_imgnote").textContent = `Change overlay: red outline = this candidate's component, yellow = other ${beforeDates[0]}→${detailState.after} change in view. ${es.caveat || ""}`;

  // trajectory
  $("#d_traj").innerHTML = d.temporal_trajectory.intervals.map(iv => `
    <div class="seg ${iv.changed ? "changed" : "stable"} ${iv.kind === "span" ? "span" : ""}">
      <div class="win">${esc((iv.window || []).join(" → "))}${iv.kind === "span" ? "  (span)" : ""}</div>
      <div class="st">${iv.changed ? "CHANGED" : "stable"}</div>
      <div class="win">p = ${num(iv.probability)}${iv.comparable ? "" : " · not comparable"}</div>
    </div>`).join("");
  $("#d_trajnote").textContent =
    `persistence: ${d.temporal_trajectory.persistence} (confidence ${num(d.temporal_trajectory.persistence_confidence)}) — ` +
    (d.temporal_trajectory.notes || []).join("; ");

  // evidence grid
  const ev = (d.classification && d.classification.evidence) || {};
  const sar = d.provenance.sar_corroboration;
  const bd = Object.fromEntries((d.confidence_breakdown || []).map(l => [l.split(":")[0].trim(), l]));
  const cards = [
    ["model probability", num(d.mean_model_prob), bd["model"] || ""],
    ["persistence", d.persistence, bd["persistence"] || ""],
    ["spectral support", ndeltas(ev), "anomaly = delta minus the scene-wide seasonal delta (drought→green makes raw deltas useless)"],
    ["quality", pct(bd["quality"]), bd["quality"] || ""],
    ["registration residual", regpx(bd["registration"]), bd["registration"] || ""],
    ["radiometric reliability", radv(bd["radiometric"]), bd["radiometric"] || ""],
    ["SAR corroboration", sar ? `${num(sar.vv_median_db, 1)} dB VV · ×${num(sar.confidence_factor)}` : "neutral / unavailable",
      sar ? `${sar.verdict || ""} — speckle: ${sar.speckle_filter || ""}` : "no C-band expectation or outside the S1 swath — weight ×1.0"],
  ];
  const terrain = d.terrain || {};
  if (terrain.elevation_m != null) {
    cards.push(["terrain context", terrain.plain_language || "–",
      "elevation/slope/aspect measured (Copernicus DEM GLO-30); distance to water/built-up derived from the 2024-03-08 reference scene — see full breakdown below"]);
  }
  renderTerrainTable(terrain);
  $("#d_ev").innerHTML = cards.map(([l, v, s]) => `
    <div class="ev"><div class="lbl">${esc(l)}</div><div class="val">${esc(v)}</div><div class="sub">${esc(s)}</div></div>`).join("");

  // suppression trace
  $("#d_trace").innerHTML = (d.suppression.trace || []).map(t => {
    const cls = t.verdict === "pass" ? "" : (t.weight < 1 && t.weight > 0 ? "down" : "sup");
    return `<tr><td>${esc(t.rule)}</td><td><span class="v ${cls}">${esc(t.verdict)}</span></td>
      <td>${num(t.weight)}</td><td class="small">${esc(t.detail)}</td></tr>`;
  }).join("") + `<tr><td colspan="4" class="small muted">combined down-weight ×${num(d.suppression.combined_downweight)} · ${d.suppression.suppressed ? "SUPPRESSED by " + esc(d.suppression.suppressed_by) : "entered the queue"}</td></tr>`;

  // provenance
  $("#d_prov").innerHTML = d.provenance.observations.map(o => {
    const sc = o.scene || {}, co = o.collection || {}, rt = o.representative_tile || {};
    return `<div class="obs"><h4>${o.role.toUpperCase()} &mdash; ${esc(o.observation_id)}</h4>
      <dl>
        <dt>acquired</dt><dd>${esc(o.acquired_at)}</dd>
        <dt>scene id</dt><dd class="mono">${esc(sc.scene_id)}</dd>
        <dt>platform</dt><dd>${esc(sc.platform)} &middot; ${esc(co.sensor)} &middot; ${esc(co.native_gsd_m)} m</dd>
        <dt>source COG</dt><dd class="mono small">${esc(sc.source_url)}</dd>
        <dt>licence</dt><dd>${esc(sc.license)}</dd>
        <dt>CRS</dt><dd>${esc(sc.crs)}</dd>
        <dt>bands</dt><dd>${esc((co.bands || []).join(", "))}</dd>
        <dt>rep. tile</dt><dd class="mono small">${esc(rt.tile_id || "–")}</dd>
        <dt>tile checksum</dt><dd class="mono small">${esc((rt.checksums && (rt.checksums.B04 || Object.values(rt.checksums)[0]) || "").slice(0, 24))}…</dd>
      </dl></div>`;
  }).join("");
  const mp = d.provenance.model, cd = d.provenance.code;
  $("#d_provcode").innerHTML =
    `<span>model <b>${esc(mp.name)}</b></span><span>threshold <b>${esc(mp.threshold)}</b></span>` +
    `<span>weights sha256 <b class="mono">${esc((mp.weights_sha256 || "").slice(0, 24))}…</b></span>` +
    `<span>git <b class="mono">${esc(cd.git_commit)}</b></span><span>pipeline <b>${esc(cd.pipeline_version)}</b></span>` +
    `<span>prob raster <b class="mono small">${esc((d.provenance.probability_raster || "").split(/[\\/]/).pop())}</b></span>`;

  // decision + history
  renderHistory(d.decisions || [], d.current_decision);
  $("#d_confirm").onclick = () => decide("confirm");
  $("#d_reject").onclick = () => decide("reject");

  // similar
  loadSimilar(d.candidate_id);
}
function paintImages() {
  const id = detailState.id, dt = detailState.date, after = detailState.after;
  // "before" follows the date selector; "after" + "overlay" are the pair's later observation
  const cfg = [["before", "rgb", dt], ["after", "rgb", after], ["change overlay", "overlay", after]];
  $("#d_imgs").innerHTML = cfg.map(([cap, view, date]) => `
    <figure><figcaption><span>${cap}</span><span>${date}</span></figcaption>
      <img loading="lazy" src="/candidates/${encodeURIComponent(id)}/imagery?date=${date}&view=${view}&scale=2" alt="${cap} ${date}"></figure>`).join("");
}
function renderTerrainTable(t) {
  const wrap = $("#d_terrainwrap");
  if (!t || t.elevation_m == null) { wrap.classList.add("hidden"); return; }
  wrap.classList.remove("hidden");
  const prov = t.provenance || {};
  const rows = [
    ["elevation", `${num(t.elevation_m, 0)} m`, prov.elevation_m],
    ["slope", t.aspect_compass ? `${num(t.slope_deg, 0)}° (facing ${t.aspect_compass})` : `${num(t.slope_deg, 0)}° (flat)`, prov.slope_deg],
    ["distance to water channel", `${num(t.distance_to_water_m, 0)} m`, prov.distance_to_water_m],
    ["distance to built-up area", `${num(t.distance_to_built_up_m, 0)} m`, prov.distance_to_built_up_m],
  ];
  $("#d_terrain").innerHTML = rows.map(([f, v, basis]) =>
    `<tr><td>${esc(f)}</td><td>${esc(v)}</td><td class="small">${esc(basis || "")}</td></tr>`).join("");
}
function ndeltas(ev) {
  return `NDVI ${sgn(ev.ndvi_anomaly)} · NDBI ${sgn(ev.ndbi_anomaly)} · NDWI ${sgn(ev.ndwi_anomaly)}`;
}
const sgn = (v) => v == null ? "–" : (v >= 0 ? "+" : "") + v.toFixed(2);
function pct(line) { const m = /valid ([\d.]+)%/.exec(line || ""); return m ? m[1] + "% valid" : "–"; }
function regpx(line) { const m = /([\d.]+) px/.exec(line || ""); return m ? m[1] + " px" : "–"; }
function radv(line) { const m = /reliability ([\d.]+)/.exec(line || ""); return m ? m[1] : "–"; }

function renderHistory(hist, cur) {
  $("#d_decstate").innerHTML = cur
    ? `current verdict <b class="verdict ${cur.decision}">${cur.decision}</b> by ${esc(cur.analyst || "?")} at ${esc(cur.created_at)} — <span class="muted">${hist.length} decision(s) on record, append-only</span>`
    : `<span class="muted">no decision recorded yet</span>`;
  $("#d_hist").innerHTML = hist.slice().reverse().map(h => `
    <tr><td class="small">${esc(h.created_at)}</td><td class="verdict ${h.decision}">${esc(h.decision)}</td>
      <td>${esc(h.analyst || "–")}</td><td class="small">${esc(h.analyst_note || "")}</td>
      <td>${num(h.confidence_at_decision)}</td><td class="mono small">${esc((h.weights_sha256 || "").slice(0, 12))}</td></tr>`).join("")
    || `<tr><td colspan="6" class="muted">—</td></tr>`;
}
async function decide(decision) {
  const note = $("#d_note").value.trim();
  try {
    await api(`/candidates/${encodeURIComponent(detailState.id)}/decision`, {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ decision, note, analyst: "analyst" }),
    });
    $("#d_note").value = "";
    toast(`${decision} recorded in the audit trail`);
    const d = await api("/candidates/" + encodeURIComponent(detailState.id));
    detailState.data = d;
    $("#d_head").querySelector(".verdict").className = "v verdict " + decision;
    $("#d_head").querySelector(".verdict").textContent = decision;
    renderHistory(d.decisions || [], d.current_decision);
    if (queueInit) loadQueue();
  } catch (e) { toast("decision failed: " + e.message); }
}
async function loadSimilar(id) {
  $("#d_similar").innerHTML = `<span class="spinner"></span>`;
  try {
    const d = await api(`/candidates/${encodeURIComponent(id)}/similar?k=8`);
    $("#d_similar").innerHTML = (d.results || []).map(r => `
      <div class="card" data-tile="${esc(r.tile_id)}">
        <img class="thumb" loading="lazy" src="/tile/${encodeURIComponent(r.tile_id)}/thumbnail" alt="">
        <div class="meta"><div class="row"><b>sim ${num(r.score, 3)}</b><span>${esc(r.acq_date)}</span></div>
        <div class="row"><span>cluster ${r.cluster == null ? "–" : r.cluster}</span><span>${num(r.centroid_lonlat[1], 3)},${num(r.centroid_lonlat[0], 3)}</span></div></div>
      </div>`).join("") || `<span class="muted">no neighbours</span>`;
    $$("#d_similar .card").forEach(c => c.addEventListener("click", () => {
      $("#disc_seed").value = c.dataset.tile; go("discovery"); setTimeout(runDiscovery, 30);
    }));
  } catch (e) { $("#d_similar").innerHTML = `<span class="muted">${esc(e.message)}</span>`; }
}

/* ---------------------------------------------------------------- DISCOVERY */
let discInit = false;
function ensureDiscovery() {
  if (discInit) return; discInit = true;
  $("#disc_go").addEventListener("click", runDiscovery);
  api("/discovery/clusters").then(d => {
    if (!d.available) { $("#disc_clusters").textContent = "no cluster job run"; return; }
    const dl = d.display_labels || {};
    $("#disc_clusters").innerHTML =
      `<b>${d.n_clusters}</b> clusters over ${d.n_tiles} tiles (${d.noise_count} noise) — ` +
      Object.entries(d.cluster_concepts || {}).map(([k, v]) =>
        `c${k} (n=${d.sizes[k]}): <i>${esc(dl[k] || v[0][0])}</i>`).join(" &nbsp;·&nbsp; ") +
      `<div class="muted small">${esc((d.params || {}).algorithm)} · ${esc((d.params || {}).metric)}</div>`;
    $("#disc_map").src = "/discovery/cluster-map.png";
  }).catch(e => $("#disc_clusters").textContent = e.message);
}
async function runDiscovery() {
  const seed = $("#disc_seed").value.trim();
  const lon = $("#disc_lon").value, lat = $("#disc_lat").value, k = $("#disc_k").value || 12;
  $("#disc_results").innerHTML = `<span class="spinner"></span>`;
  try {
    let d, path;
    if (/^\d{4}_\d{4}_\d+$/.test(seed)) path = `/candidates/${encodeURIComponent(seed)}/similar?k=${k}`;
    else if (seed) path = `/discovery/similar?tile_id=${encodeURIComponent(seed)}&k=${k}`;
    else if (lon && lat) path = `/discovery/similar?lon=${lon}&lat=${lat}&k=${k}`;
    else { toast("enter a seed tile / candidate id or a lon,lat"); return; }
    d = await api(path);
    $("#disc_meta").textContent = `seed ${d.seed_tile_id || seed} · cluster ${d.seed_cluster == null ? "–" : d.seed_cluster} · ${d.latency_ms} ms`;
    $("#disc_results").innerHTML = (d.results || []).map(r => `
      <div class="card" data-tile="${esc(r.tile_id)}">
        <img class="thumb" loading="lazy" src="/tile/${encodeURIComponent(r.tile_id)}/thumbnail" alt="">
        <div class="meta">
          <div class="row"><b>sim ${num(r.score, 3)}</b><span>${esc(r.acq_date)}</span></div>
          <div class="row"><span>cluster ${r.cluster == null ? "–" : r.cluster}</span><span>${num(r.centroid_lonlat[1], 3)}°N</span></div>
        </div></div>`).join("") || `<span class="muted">no neighbours</span>`;
    $$("#disc_results .card").forEach(c => c.addEventListener("click", () => { $("#disc_seed").value = c.dataset.tile; runDiscovery(); }));
  } catch (e) { $("#disc_results").innerHTML = `<span class="muted">${esc(e.message)}</span>`; }
}

/* ---------------------------------------------------------------- OBJECT DETECTION (Phase 8F)
   REAL stored oriented-box detections (scripts/detect_maxar.py's output, read
   through the catalog's DerivedProduct + detections.geojson) drawn on the
   REAL Maxar tile they were found on - nothing here re-runs the detector.
   The backend reprojects each stored EPSG:4326 footprint back to tile pixels;
   this file only draws already-pixel-space polygons on a canvas. */
const DET_CLASS_COLOR = {
  "small-vehicle": "#3fb950", "large-vehicle": "#d98324", ship: "#1f6feb",
  plane: "#c93c37", helicopter: "#8a5a2b", "storage-tank": "#7b3fbf",
  harbor: "#0aa1a3", bridge: "#b5850b",
};
let detInit = false;
const detState = {
  modelInfo: null, observations: [], tiles: [], tileData: null, img: null,
  obsId: null, selectedClasses: new Set(), visible: [],
};

function ensureDetect() {
  if (detInit) return; detInit = true;
  $("#dt_margin").addEventListener("input", () => { updateDetMarginLabel(); drawDetectCanvas(); });
  $("#dt_obs").addEventListener("change", (e) => loadDetectObservation(e.target.value));
  $("#dt_tile").addEventListener("change", (e) => loadDetectTile(e.target.value));
  $("#dt_canvas").addEventListener("mousemove", onDetectHover);
  $("#dt_canvas").addEventListener("mouseleave", () => $("#dt_hover").textContent = " ");
  updateDetMarginLabel();
  loadDetectModelInfo();
  loadDetectObservations();
}

async function loadDetectModelInfo() {
  try {
    detState.modelInfo = await api("/detect/model-info");
    renderDetCaveats(detState.modelInfo.caveats);
    renderDetMetrics(detState.modelInfo.operating_points);
    renderDetClassFilter(detState.modelInfo.classes);
  } catch (e) { toast(e.message); }
}
function renderDetCaveats(caveats) {
  const panel = $("#dt_caveatpanel");
  if (!caveats || !caveats.length) { panel.classList.add("hidden"); return; }
  panel.classList.remove("hidden");
  $("#dt_caveats").innerHTML = caveats.map(c => `<div>&#9888;&nbsp; ${esc(c)}</div>`).join("");
}
function renderDetMetrics(op) {
  const rows = ["small-vehicle", "large-vehicle"].filter(c => op[c]);
  $("#dt_metrics").innerHTML = rows.map(c => {
    const o = op[c];
    return `<div class="ev">
      <div class="lbl">${esc(c)} &mdash; xView TEST-half AP50</div>
      <div class="val">${(o.AP50 * 100).toFixed(1)}%</div>
      <div class="sub">AP50:95 ${(o.AP50_95 * 100).toFixed(1)}% &middot; precision ${(o.precision * 100).toFixed(0)}%
        &middot; recall ${(o.recall * 100).toFixed(0)}% &middot; operating threshold ${num(o.conf, 3)}</div>
    </div>`;
  }).join("") || `<span class="muted">no vehicle operating points on the model card</span>`;
}
function renderDetClassFilter(classes) {
  detState.selectedClasses = new Set(classes);
  $("#dt_classes").innerHTML = classes.map(c => `
    <label><input type="checkbox" data-cls="${esc(c)}" checked>
      <i style="background:${DET_CLASS_COLOR[c] || "#888"}"></i>${esc(c)}</label>`).join("");
  $$("#dt_classes input[type=checkbox]").forEach(cb => cb.addEventListener("change", () => {
    if (cb.checked) detState.selectedClasses.add(cb.dataset.cls); else detState.selectedClasses.delete(cb.dataset.cls);
    drawDetectCanvas();
  }));
}

async function loadDetectObservations() {
  try {
    const { observations } = await api("/detect/observations");
    detState.observations = observations;
    $("#dt_obs").innerHTML = observations.map(o =>
      `<option value="${esc(o.observation_id)}">${esc(o.aoi_name || o.observation_id)} &mdash; ${o.n_detections} detections</option>`).join("")
      || `<option value="">no Maxar detections staged</option>`;
    if (observations.length) await loadDetectObservation(observations[0].observation_id);
  } catch (e) { toast(e.message); }
}
async function loadDetectObservation(obsId) {
  if (!obsId) return;
  detState.obsId = obsId;
  $("#dt_obs").value = obsId;
  const o = detState.observations.find(x => x.observation_id === obsId);
  $("#dt_obstotals").innerHTML = o ? Object.entries(o.by_class || {}).map(([k, v]) =>
    `<span><i style="display:inline-block;width:9px;height:9px;border-radius:2px;background:${DET_CLASS_COLOR[k] || "#888"};margin-right:4px"></i><b>${v}</b> ${esc(k)}</span>`
  ).join("") + `<span class="muted">${o.n_tiles_with_detections}/${o.n_tiles} tiles with a detection</span>` : "";
  const { tiles } = await api(`/detect/observations/${encodeURIComponent(obsId)}/tiles`);
  detState.tiles = tiles;
  $("#dt_tile").innerHTML = tiles.map(t =>
    `<option value="${t.row},${t.col}">r${String(t.row).padStart(3, "0")}_c${String(t.col).padStart(3, "0")} &mdash; ${t.n_detections} detections</option>`).join("")
    || `<option value="">no tile has a detection</option>`;
  if (tiles.length) await loadDetectTile(`${tiles[0].row},${tiles[0].col}`);
  else { detState.tileData = null; detState.img = null; drawDetectCanvas(); renderDetCounts(null); }
}
async function loadDetectTile(rowcol) {
  if (!rowcol) return;
  const [row, col] = rowcol.split(",").map(Number);
  $("#dt_tile").value = rowcol;
  $("#dt_tilelabel").textContent = `r${String(row).padStart(3, "0")}_c${String(col).padStart(3, "0")}`;
  const obsId = detState.obsId;
  const data = await api(`/detect/observations/${encodeURIComponent(obsId)}/tiles/${row}/${col}`);
  if (obsId !== detState.obsId) return;      // observation changed while this was in flight
  detState.tileData = data;
  renderDetCounts(data);
  const img = new Image();
  img.onload = () => { if (detState.tileData === data) { detState.img = img; drawDetectCanvas(); } };
  img.src = `/detect/observations/${encodeURIComponent(obsId)}/tiles/${row}/${col}/image.png`;
}
function detEffectiveThreshold(cls) {
  const op = (detState.modelInfo && detState.modelInfo.operating_points[cls]) || {};
  const base = op.conf != null ? op.conf : 0;
  const frac = (+$("#dt_margin").value || 0) / 100;
  return base + frac * (1 - base);
}
function updateDetMarginLabel() {
  const frac = (+$("#dt_margin").value || 0) / 100;
  $("#dt_marginlabel").textContent = frac === 0
    ? "showing everything stored (each class's own operating point)"
    : `showing only the top ${(100 * (1 - frac)).toFixed(0)}% of each class's confidence range above its floor`;
}
function drawDetectCanvas() {
  const c = $("#dt_canvas"), ctx = c.getContext("2d");
  ctx.clearRect(0, 0, c.width, c.height);
  if (detState.img) ctx.drawImage(detState.img, 0, 0, c.width, c.height);
  const dets = (detState.tileData && detState.tileData.detections) || [];
  detState.visible = dets.filter(d => detState.selectedClasses.has(d.class) && d.score >= detEffectiveThreshold(d.class));
  for (const d of detState.visible) {
    ctx.strokeStyle = DET_CLASS_COLOR[d.class] || "#fff";
    ctx.lineWidth = 2;
    ctx.beginPath();
    d.polygon_px.forEach(([x, y], i) => i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y));
    ctx.closePath(); ctx.stroke();
  }
  $("#dt_legend").innerHTML = Array.from(detState.selectedClasses).map(cl =>
    `<span><i style="background:${DET_CLASS_COLOR[cl] || "#888"}"></i>${esc(cl)}</span>`).join("") +
    `<span class="muted">${detState.visible.length} of ${dets.length} stored detections shown</span>`;
}
function renderDetCounts(data) {
  const counts = {};
  for (const d of (data ? data.detections : [])) counts[d.class] = (counts[d.class] || 0) + 1;
  const op = (detState.modelInfo && detState.modelInfo.operating_points) || {};
  const rows = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  $("#dt_counts tbody").innerHTML = rows.map(([k, v]) =>
    `<tr><td><i style="background:${DET_CLASS_COLOR[k] || "#888"};margin-right:5px"></i>${esc(k)}</td><td>${v}</td>` +
    `<td class="small muted">${op[k] ? num(op[k].conf, 3) : "–"}</td></tr>`
  ).join("") || `<tr><td colspan="3" class="muted">no detections on this tile</td></tr>`;
}
function detPointInPoly(x, y, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if (((yi > y) !== (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi)) inside = !inside;
  }
  return inside;
}
function onDetectHover(e) {
  const c = $("#dt_canvas"), r = c.getBoundingClientRect();
  const x = (e.clientX - r.left) * (c.width / r.width), y = (e.clientY - r.top) * (c.height / r.height);
  const hit = (detState.visible || []).find(d => detPointInPoly(x, y, d.polygon_px));
  $("#dt_hover").textContent = hit
    ? `${hit.class}  conf ${num(hit.score, 3)}${hit.long_side_px ? "  " + hit.long_side_px.toFixed(1) + "px" : ""}`
    : " ";
}

/* ---------------------------------------------------------------- WATCH AREAS
   Phase 8 Step C: define a named AOI + filters once; every change-pipeline
   re-run evaluates it and records newly-matching candidates as notifications.
   Plain CRUD over /watch-areas + /notifications - no map widget, a bbox text
   field (consistent with the Review Queue's own "AOI bbox" filter). */
let watchInit = false, watchEditingId = null;

let watchMap;
function bboxRing([w, s, e, n]) { return [[w, s], [e, s], [e, n], [w, n]]; }
function wRegionItems() {
  return (REGIONS || []).map((r, i) => ({
    id: "region:" + r.name, ring: bboxRing(r.bbox),
    color: REGION_PALETTE[i % REGION_PALETTE.length], regionName: r.name,
  }));
}
function refreshWatchMap() {
  if (!watchMap) return;
  const regionItems = wRegionItems();
  const items = regionItems.slice();
  const bbox = parseBboxInput($("#w_bbox").value);
  if (bbox) items.push({ id: "__selection__", ring: bboxRing(bbox), color: "#17191c" });
  watchMap.setData(items, null);
  watchMap.select(bbox ? "__selection__" : null);
  $("#w_legend").innerHTML = regionItems.map(it =>
    `<span><i style="background:${it.color}"></i>${esc(it.regionName)}</span>`).join("");
}

function ensureWatch() {
  if (watchInit) return; watchInit = true;
  $("#w_types").innerHTML = TYPES.map(t =>
    `<label style="width:auto"><input type="checkbox" value="${t}" style="width:auto;margin-right:4px">${esc(CTYPE_HUMAN[t] || t)}</label>`).join("");
  $("#w_save").addEventListener("click", saveWatchArea);
  $("#w_cancel").addEventListener("click", resetWatchForm);
  watchMap = new CoordMap($("#w_map"), (it) => {
    if (it.regionName) { $("#w_region").value = it.regionName; $("#w_bbox").value = bboxStr(regionByName(it.regionName).bbox); refreshWatchMap(); }
  }, { onDrawRect: (bbox) => { $("#w_region").value = ""; $("#w_bbox").value = bboxStr(bbox); refreshWatchMap(); } });
  fillRegionSelect($("#w_region"));
  loadRegions().then(refreshWatchMap);
  $("#w_region").addEventListener("change", (e) => {
    const r = regionByName(e.target.value);
    $("#w_bbox").value = r ? bboxStr(r.bbox) : "";
    refreshWatchMap();
  });
  $("#w_bbox").addEventListener("input", () => {
    if ($("#w_region").value && $("#w_bbox").value.trim() !== bboxStr(regionByName($("#w_region").value).bbox)) {
      $("#w_region").value = "";   // hand-edited away from the picked region's exact bbox
    }
    refreshWatchMap();
  });
  loadWatch();
}

function resetWatchForm() {
  watchEditingId = null;
  $("#w_name").value = ""; $("#w_bbox").value = ""; $("#w_query").value = ""; $("#w_conf").value = "0.5";
  $("#w_region").value = "";
  $$("#w_types input").forEach(cb => cb.checked = false);
  $("#w_save").textContent = "Create watch area";
  $("#w_cancel").classList.add("hidden");
  refreshWatchMap();
}

function parseBboxInput(s) {
  const parts = (s || "").split(",").map(x => parseFloat(x.trim()));
  return (parts.length === 4 && parts.every(x => !isNaN(x))) ? parts : null;
}

async function saveWatchArea() {
  const name = $("#w_name").value.trim();
  if (!name) { toast("a watch area needs a name"); return; }
  const bbox = parseBboxInput($("#w_bbox").value);
  if ($("#w_bbox").value.trim() && !bbox) { toast("bbox must be west,south,east,north"); return; }
  const change_types = $$("#w_types input:checked").map(cb => cb.value);
  const min_confidence = $("#w_conf").value === "" ? null : Number($("#w_conf").value);
  const body = { name, bbox, text_query: $("#w_query").value.trim(), change_types, min_confidence, created_by: "analyst_ui" };
  try {
    if (watchEditingId) {
      await api(`/watch-areas/${encodeURIComponent(watchEditingId)}`, {
        method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      toast("watch area updated");
    } else {
      await api("/watch-areas", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      toast("watch area created");
    }
    resetWatchForm();
    loadWatch();
  } catch (e) { toast(e.message); }
}

function editWatchArea(w) {
  watchEditingId = w.watch_id;
  $("#w_name").value = w.name || "";
  $("#w_bbox").value = (w.bbox || []).join(",");
  $("#w_query").value = w.text_query || "";
  $("#w_conf").value = w.min_confidence == null ? "" : w.min_confidence;
  $$("#w_types input").forEach(cb => cb.checked = (w.change_types || []).includes(cb.value));
  const match = w.bbox ? (REGIONS || []).find(r => bboxStr(r.bbox) === bboxStr(w.bbox)) : null;
  $("#w_region").value = match ? match.name : "";
  $("#w_save").textContent = "Save changes";
  $("#w_cancel").classList.remove("hidden");
  refreshWatchMap();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

async function deleteWatchArea(id) {
  await api(`/watch-areas/${encodeURIComponent(id)}`, { method: "DELETE" });
  toast("watch area deleted");
  loadWatch();
}

async function loadWatch() {
  try {
    const [wd, nd] = await Promise.all([api("/watch-areas"), api("/notifications")]);
    renderWatchTable(wd.watch_areas || []);
    renderNotifications(nd.notifications || []);
    setWatchBadge((nd.notifications || []).filter(n => !n.seen).length);
  } catch (e) { toast(e.message); }
}

function renderWatchTable(areas) {
  $("#w_count").textContent = `(${areas.length})`;
  $("#w_table tbody").innerHTML = areas.map(w => `
    <tr>
      <td>${esc(w.name)}</td>
      <td class="small">${w.bbox ? w.bbox.map(x => x.toFixed(3)).join(", ") : "unrestricted"}</td>
      <td class="small">${(w.change_types || []).length ? esc(w.change_types.join(", ")) : "any"}</td>
      <td>${w.min_confidence == null ? "–" : w.min_confidence}</td>
      <td>${w.active ? "yes" : "no"}</td>
      <td class="small">${esc((w.created_at || "").slice(0, 10))}</td>
      <td><button class="ghost small" data-edit="${esc(w.watch_id)}">Edit</button>
          <button class="ghost small" data-del="${esc(w.watch_id)}">Delete</button></td>
    </tr>`).join("") || `<tr><td colspan="7" class="muted">no watch areas defined yet</td></tr>`;
  $$("#w_table [data-edit]").forEach(b => b.addEventListener("click", () =>
    editWatchArea(areas.find(w => w.watch_id === b.dataset.edit))));
  $$("#w_table [data-del]").forEach(b => b.addEventListener("click", () => {
    if (confirm("Delete this watch area? Its notification history is deleted too.")) deleteWatchArea(b.dataset.del);
  }));
}

function renderNotifications(list) {
  $("#n_count").textContent = `(${list.filter(n => !n.seen).length} unseen of ${list.length})`;
  $("#n_list").innerHTML = list.map(n => `
    <div class="panel pad" style="${n.seen ? "opacity:.6" : ""}">
      <div class="row"><b>${esc(n.watch_name || n.watch_id)}</b>
        <span class="band ${n.severity || "low"}" title="max(confidence × significance) = ${num(n.severity_score)}">${(n.severity || "low").toUpperCase()}</span>
        <span class="small muted">${esc((n.created_at || "").replace("T", " ").slice(0, 19))}</span></div>
      <div class="small muted">${n.candidates.length} new matching candidate(s) from observation ${esc(n.observation_id)}${n.observation_date ? " · " + esc(n.observation_date) : ""}</div>
      <div class="stack" style="flex-direction:row;flex-wrap:wrap;gap:6px;margin-top:6px">
        ${n.candidates.slice(0, 20).map(c => `<a href="#/detail/${encodeURIComponent(c.candidate_id)}" class="chip"
             style="text-decoration:none">${esc(c.candidate_id)} · ${esc(c.change_type || "?")}</a>`).join("")}
        ${n.candidates.length > 20 ? `<span class="small muted" style="align-self:center">+${n.candidates.length - 20} more — see the Review Queue</span>` : ""}
      </div>
      ${n.seen ? "" : `<button class="ghost small" style="margin-top:6px" data-seen="${esc(n.notification_id)}">Mark seen</button>`}
    </div>`).join("") || `<span class="muted">no notifications yet — new observations fire these once the change pipeline is re-run</span>`;
  $$("#n_list [data-seen]").forEach(b => b.addEventListener("click", async () => {
    await api(`/notifications/${encodeURIComponent(b.dataset.seen)}/seen`, { method: "POST" });
    loadWatch();
  }));
}

/* ---------------------------------------------------------------- GUIDED DEMO
   A scripted 4-step walkthrough for non-specialist viewers. It drives the real
   views by calling the same functions the analyst UI uses and hits the live API
   at every step — no mock data, no hardcoded results. Escapable at any time. */
const sleep = (ms) => new Promise(r => setTimeout(r, ms));
function waitFor(pred, timeout = 8000, interval = 120) {
  return new Promise((resolve) => {
    const t0 = performance.now();
    const tick = () => {
      let ok = false; try { ok = !!pred(); } catch (e) {}
      if (ok || performance.now() - t0 > timeout) return resolve(ok);
      setTimeout(tick, interval);
    };
    tick();
  });
}

const DEMO = { active: false, i: 0, running: false };
const DEMO_STEPS = [
  {
    title: "Ask in plain language",
    body: "No keywords, no coordinates — a sentence. The tile index is searched " +
      "semantically and the best-matching Sentinel-2 tiles come back ranked and " +
      "plotted on the offline canvas.",
    run: async () => {
      go("search"); ensureSearch();
      $("#q").value = (PRES && PRES.demo && PRES.demo.search_query) || "an open water reservoir or pond";
      await runSearch();
      await waitFor(() => $$("#s_cards .card img").length > 0, 9000);
    },
  },
  {
    title: "One place, every date, and the change overlay",
    body: () => "The strongest water-gain candidate. Before / after / overlay imagery " +
      `dominates the view; the ${obsYears().join(" · ")} selector steps through every ` +
      "observation. “Why did the system flag this?” opens the full evidence and suppression trace.",
    run: async () => {
      const id = PRES && PRES.demo && PRES.demo.water_gain_candidate_id;
      if (!id) return;
      // visual-first: the walkthrough always shows the collapsed detail,
      // whatever state a presenter left the disclosures in earlier
      $$("#view-detail details.disclosure").forEach(d => { d.open = false; });
      await openDetail(id);
      await waitFor(() => {
        const im = $$("#d_imgs img");
        return im.length === 3 && im.every(x => x.complete && x.naturalWidth > 0);
      }, 14000);
    },
  },
  {
    title: "Find more places like it",
    body: "One click runs a nearest-neighbour search in the same embedding space — " +
      "other tiles across every indexed region that look like this one, with the " +
      "HDBSCAN cluster map for context.",
    run: async () => {
      const seed = PRES && PRES.demo && PRES.demo.discovery_seed;
      if (seed) $("#disc_seed").value = seed;
      go("discovery"); ensureDiscovery();
      await runDiscovery();
      await waitFor(() => $$("#disc_results .card img").length > 0, 10000);
    },
  },
  {
    title: "All of this ran offline",
    body: "No CDN, no web fonts, no map tiles, no outbound calls — RemoteCLIP, the " +
      "FAISS index, the SQLite catalog and the change rasters are local files. Every " +
      "candidate carries its full provenance chain and every analyst decision goes to " +
      "an append-only audit trail.",
    run: async () => {
      go("overview"); ensureOverview();
      await waitFor(() => $("#ov-counters").children.length > 0, 5000);
      window.scrollTo({ top: 0, behavior: "smooth" });
      for (const el of [$("#ov-offline"), $("#ov-counters")]) {
        if (!el) continue;
        el.classList.remove("demo-pulse"); void el.offsetWidth; el.classList.add("demo-pulse");
      }
    },
  },
];

function obsYears() { return Array.from(new Set(OBS_DATES.map(d => d.slice(0, 4)))); }

async function demoPrewarm() {
  const id = PRES && PRES.demo && PRES.demo.water_gain_candidate_id;
  const u = [];
  const years = obsYears();
  if (id) {
    const b = `/candidates/${encodeURIComponent(id)}/imagery`;
    for (const dt of years) u.push(`${b}?date=${dt}&view=rgb&scale=2`);
    u.push(`${b}?date=${years[years.length - 1] || ""}&view=overlay&scale=2`);
  }
  u.push(...featuredUrls((PRES && PRES.featured) || []));
  prewarm(u);
  await Promise.race([
    Promise.allSettled(u.map(x => fetch(x, { cache: "force-cache" }))),
    sleep(3000),
  ]);
}
async function demoStart() {
  if (DEMO.active) return;
  try {
    if (!PRES) { PRES = await api("/presentation/summary"); OBS_DATES = PRES.observation_dates || []; }
  } catch (e) {}
  DEMO.active = true; DEMO.i = 0;
  $("#demo").hidden = false;
  document.body.classList.add("demo-on");
  toast("pre-warming demo imagery…");
  await demoPrewarm();
  await demoGo(0);
}
function demoExit() {
  DEMO.active = false;
  $("#demo").hidden = true;
  document.body.classList.remove("demo-on");
  $$(".demo-pulse").forEach(el => el.classList.remove("demo-pulse"));
}
async function demoGo(i) {
  if (!DEMO.active || DEMO.running) return;
  DEMO.i = Math.max(0, Math.min(DEMO_STEPS.length - 1, i));
  const s = DEMO_STEPS[DEMO.i];
  $("#demo-progress").innerHTML = DEMO_STEPS.map((_, k) => `<i class="${k <= DEMO.i ? "on" : ""}"></i>`).join("");
  $("#demo-step").textContent = `Step ${DEMO.i + 1} / ${DEMO_STEPS.length}`;
  $("#demo-title").textContent = s.title;
  $("#demo-body").textContent = typeof s.body === "function" ? s.body() : s.body;
  $("#demo-back").disabled = DEMO.i === 0;
  $("#demo-next").textContent = DEMO.i === DEMO_STEPS.length - 1 ? "Finish" : "Next →";
  DEMO.running = true;
  try { await s.run(); } catch (e) { /* never let a step kill the walkthrough */ }
  DEMO.running = false;
}

/* ---------------------------------------------------------------- boot */
function setWatchBadge(n) {
  const el = $("#watch_navbadge");
  el.textContent = n > 99 ? "99+" : String(n);
  el.classList.toggle("hidden", !n);
}
async function refreshWatchBadge() {
  try { setWatchBadge((await api("/notifications?unseen_only=true")).notifications.length); }
  catch (e) { /* nav badge is a nicety */ }
}

async function boot() {
  try {
    const h = await api("/health");
    $("#offlineChip").textContent = `offline · ${h.vectors} vectors · ${h.candidates} candidates`;
  } catch (e) { $("#offlineChip").textContent = "backend unreachable"; $("#offlineChip").classList.remove("ok"); }
  try {
    PRES = await api("/presentation/summary");
    OBS_DATES = PRES.observation_dates || [];
    setWatchBadge((PRES.counters || {}).unseen_notifications || 0);
  } catch (e) { /* overview will retry; other views don't need it */ }

  $("#demoBtn").addEventListener("click", demoStart);
  $("#demo-next").addEventListener("click", () => (DEMO.i >= DEMO_STEPS.length - 1 ? demoExit() : demoGo(DEMO.i + 1)));
  $("#demo-back").addEventListener("click", () => demoGo(DEMO.i - 1));
  $("#demo-exit").addEventListener("click", demoExit);
  document.addEventListener("keydown", e => { if (e.key === "Escape" && DEMO.active) demoExit(); });

  if (!location.hash) location.hash = "#/overview";
  router();
}
boot();
