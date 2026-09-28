/* geoseek analyst - offline single-page UI. Vanilla JS, no dependencies, no
   network beyond this same FastAPI process. Every fetch() is an absolute path
   on the current origin; there is no <script src>, no CDN, no web map tile. */
"use strict";

const TYPES = ["water_gain", "water_loss", "construction", "clearance", "road", "other"];
const AOI_FALLBACK = [82.0124, 26.3613, 82.8459, 27.1098];
const CTYPE_HUMAN = {
  water_gain: "New open water / flooding", water_loss: "Water body shrank or dried",
  construction: "New built-up surface", clearance: "Vegetation or land cleared",
  road: "New road / linear corridor", other: "Surface change (unclassified)",
};
let PRES = null;          // /presentation/summary, fetched once at boot
let OBS_DATES = [];        // e.g. ["2019-03-30","2021-03-04","2024-03-08"]
let REGIONS = null;       // /regions, fetched once and cached: [{name, bbox, n_observations}]
let DISC_CLUSTER_LABELS = {};   // cluster id (string) -> plain "visual type" label, from /discovery/clusters
function clusterLabel(id) {
  if (id == null) return "not grouped";
  return DISC_CLUSTER_LABELS[id] || `visual type ${id}`;
}
const REGION_PALETTE = ["#1f6feb", "#7b3fbf", "#d98324", "#b5850b", "#3f9e6b", "#6b7280",
  "#c93c37", "#0aa1a3", "#8a5a2b"];

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
const esc = (s) => String(s == null ? "" : s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const num = (v, d = 2) => (v == null || isNaN(v)) ? "–" : Number(v).toFixed(d);

// Readable area: hectares once a change is large enough that "196,000 m²"
// would otherwise force an analyst to do the ha conversion in their head
// (1 ha = 10,000 m² - roughly a football pitch), plain m² below that.
function humanArea(m2) {
  if (m2 == null || isNaN(m2)) return "–";
  if (m2 >= 10000) return `${(m2 / 10000).toFixed(1)} ha`;
  return `${Math.round(m2).toLocaleString()} m²`;
}

// Short, plain-language phrase + one-line hover explanation for a raw
// "persistence" code - used everywhere a table cell is too narrow for
// persistenceHuman()'s full sentence (which needs a later-observation count
// this data doesn't always carry). Same six values the backend emits.
const PERSISTENCE_SHORT = {
  persistent: ["Confirmed over time", "Seen consistently across multiple later observations - high trust."],
  progressive: ["Still growing", "The change has grown steadily across later observations."],
  recent: ["Appeared recently", "Only visible in the most recent observation interval."],
  transient: ["Appeared, then reverted", "The candidate appeared then reverted - treat with caution."],
  inconsistent: ["Inconsistent across dates", "Flickers across dates - low trust."],
};
function persistenceShort(p) {
  const hit = PERSISTENCE_SHORT[p];
  return hit ? { label: hit[0], hover: hit[1] }
    : { label: "Single comparison", hover: "Only one before/after pair is available - no additional confirmation yet." };
}

// Nearest staged region containing this point, for the Review Queue's "where"
// column and the Candidate Detail verdict sentence - the closest thing to a
// place name this offline product has (no geocoding service). Falls back to
// null (callers show plain coordinates instead) when no staged region's bbox
// contains the point, which is common for isolated Maxar event AOIs.
function regionForPoint(lon, lat) {
  for (const r of (REGIONS || [])) {
    const [w, s, e, n] = r.bbox;
    if (lon >= w && lon <= e && lat >= s && lat <= n) return friendlyRegionName(r.name);
  }
  return null;
}
function placeLabel(lon, lat) {
  const region = regionForPoint(lon, lat);
  return region || `${num(lat, 3)}°N ${num(lon, 3)}°E`;
}

let LAST_LATENCY = {};
let _inflight = 0;
async function api(path, opts) {
  _inflight++; latencyBadge().start();
  const t0 = performance.now();
  try {
    const r = await fetch(path, opts);
    const ms = performance.now() - t0;
    LAST_LATENCY[path.split("?")[0]] = ms;
    if (_inflight <= 1) latencyBadge().done(`${path.split("?")[0]}  ${ms.toFixed(0)} ms`, ms);
    if (!r.ok) {
      let detail = r.statusText;
      try { detail = (await r.json()).detail || detail; } catch (e) {}
      throw new Error(`${r.status} ${detail}`);
    }
    const ct = r.headers.get("content-type") || "";
    return ct.includes("json") ? r.json() : r;
  } finally {
    _inflight--; if (_inflight <= 0) { _inflight = 0; }
  }
}
// Step 4E: shared error UI for every catch-site that currently just dumped
// e.message (an "HTTP 500 <backend detail>" string) straight into the page.
// The real message is never hidden - it moves into a closed <details> instead
// of being the primary thing shown, and a Retry button re-runs the exact call
// that failed. `container` is a selector, matching every existing call-site
// pattern ($("#id").innerHTML = ...) this replaces.
function renderApiError(container, e, retryFn, opts) {
  const el = $(container);
  if (!el) return;
  const msg = (e && e.message) || String(e);
  const body = `<div class="c-apierror">
    <div class="c-apierror__row">${Components.icon("warning", { size: 20, cls: "c-apierror__icon" })}<span>This didn't load. The system is still running.</span></div>
    <details class="c-apierror__tech"><summary>Technical details</summary><code>${esc(msg)}</code></details>
    ${retryFn ? `<button type="button" class="c-btn c-btn--secondary c-apierror__retry">Retry</button>` : ""}
  </div>`;
  // a <tbody> only accepts <tr> children - a table call site (opts.tdColspan)
  // needs the same content wrapped in one row/cell instead of a bare <div>.
  el.innerHTML = (opts && opts.tdColspan) ? `<tr><td colspan="${opts.tdColspan}">${body}</td></tr>` : body;
  if (retryFn) el.querySelector(".c-apierror__retry").addEventListener("click", retryFn);
}
// One badge instance for the header's #latencyChip. Component owns the
// ~120ms delayed "in-flight" reveal and the ok/over colour-coding against the
// 1s budget - see components.js. While more than one api() call overlaps
// (e.g. loadWatch()'s Promise.all), only the call that observes _inflight
// drop back to 0 reports a finished latency, so the chip shows one settled
// number instead of rewriting itself mid-flight for each concurrent request.
let _latencyBadge = null;
function latencyBadge() {
  if (!_latencyBadge) _latencyBadge = Components.makeLatencyBadge($("#latencyChip"));
  return _latencyBadge;
}
function toast(t) {
  const el = $("#toast"); el.textContent = t; el.classList.add("show");
  clearTimeout(toast._t); toast._t = setTimeout(() => el.classList.remove("show"), 2600);
}

/* ---------------------------------------------------------------- basemap
   Bundled, offline Natural Earth layers (coastline / country+state borders /
   major rivers / cities), simplified and stripped to ~400KB at build time -
   see scripts/build_basemap.py. Fetched once from this same origin, never
   from a tile service, no CDN - shared by every CoordMap instance on the
   page. Alongside it, a real satellite basemap: one true-colour WebP per
   staged Sentinel-2 AOI (scripts/build_mosaic.py), reprojected to plain
   EPSG:4326 so each pixel lines up 1:1 with these same degrees - our own
   imagery, not a tile service. */
let _basemapPromise = null;
function loadBasemap() {
  if (!_basemapPromise) {
    _basemapPromise = fetch("basemap.json").then(r => r.json()).then(d => {
      for (const layer of Object.values(d)) {
        for (const f of layer.features) f._b = geomBounds(f.geometry);
      }
      return d;
    }).catch(() => null);
  }
  return _basemapPromise;
}
let _mosaicPromise = null;
function loadMosaicIndex() {
  if (!_mosaicPromise) {
    _mosaicPromise = fetch("mosaic_index.json").then(r => r.json()).then(d =>
      (d.tiles || []).map(t => {
        const img = new Image();
        img.src = t.file;
        return { ...t, img };
      })
    ).catch(() => []);
  }
  return _mosaicPromise;
}
function geomRings(geom) {
  if (!geom) return [];
  if (geom.type === "Polygon") return geom.coordinates;
  if (geom.type === "MultiPolygon") return geom.coordinates.flat(1);
  if (geom.type === "LineString") return [geom.coordinates];
  if (geom.type === "MultiLineString") return geom.coordinates;
  return [];
}
function geomBounds(geom) {
  if (!geom) return [180, 90, -180, -90];
  if (geom.type === "Point") { const [x, y] = geom.coordinates; return [x, y, x, y]; }
  let w = 180, s = 90, e = -180, n = -90;
  for (const ring of geomRings(geom)) for (const [x, y] of ring) {
    if (x < w) w = x; if (x > e) e = x; if (y < s) s = y; if (y > n) n = y;
  }
  return [w, s, e, n];
}
function bboxIntersects(a, b) { return !(a[2] < b[0] || a[0] > b[2] || a[3] < b[1] || a[1] > b[3]); }
// zoom-dependent detail: how much of the basemap shows at a given bbox span
// (degrees east-west) - full-India view would be unreadable with every state
// border and all 212 bundled cities labelled at once.
function showStatesAt(spanDeg) { return spanDeg < 18; }
function cityPopThresholdAt(spanDeg) {
  if (spanDeg > 25) return 3000000;
  if (spanDeg > 12) return 1000000;
  if (spanDeg > 6) return 300000;
  if (spanDeg > 2) return 100000;
  if (spanDeg > 0.6) return 30000;
  return 0;
}
function niceScale(raw) {
  // same 1/2/5 x10^n "nice number" rule as niceGraticuleStep, for a scale bar
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const nice = norm < 1.5 ? 1 : norm < 3.5 ? 2 : norm < 7.5 ? 5 : 10;
  return nice * mag;
}

// CoordMap items carry .color in one of two shapes: a resolved {r,g,b} triple
// (Tokens.changeType()/.triple() - anything driven by the change-type palette)
// or a plain hex string (REGION_PALETTE below and a couple of fixed chip
// colours - Watch Areas' region/AOI rings, Search's result markers - which
// this redesign phase deliberately leaves untouched, see
// docs/FRONTEND_AUDIT.md). draw() needs r/g/b numbers either way; normalize
// once here rather than assume one shape. This is NOT a reincarnation of the
// deleted hexA() - hexA build an rgba() STRING for one hardcoded alpha at a
// time; this only ever produces the numeric {r,g,b} triple shape Tokens
// itself uses, so every consumer composes its own rgba()/rgb() string the
// same way regardless of where the colour came from.
function _colorTriple(c) {
  if (c && typeof c === "object") return c;
  const m = String(c).replace("#", "");
  const hex = m.length === 3 ? m.split("").map(ch => ch + ch).join("") : m;
  const bi = parseInt(hex, 16);
  return { r: (bi >> 16) & 255, g: (bi >> 8) & 255, b: bi & 255 };
}

/* ---------------------------------------------------------------- map */
class CoordMap {
  constructor(canvas, onPick, opts) {
    this.c = canvas; this.ctx = canvas.getContext("2d");
    this.items = []; this.bbox = AOI_FALLBACK; this.homeBbox = AOI_FALLBACK;
    this.watchAreaItems = []; // Step 4A: independent from `items`/layers.data - see setWatchAreas()
    this.onPick = onPick; this.sel = null;
    this.opts = opts || {};
    this.layers = { sat: true, borders: true, rivers: true, data: true, watchAreas: true };
    const wrap = canvas.parentElement || document.body;
    this.tip = document.createElement("div"); this.tip.className = "maptip";
    wrap.appendChild(this.tip);
    this.scaleEl = document.createElement("div"); this.scaleEl.className = "mapscale";
    this.scaleEl.innerHTML = `<i></i><b></b>`;
    wrap.appendChild(this.scaleEl);
    this.coordEl = document.createElement("div"); this.coordEl.className = "mapcoord";
    wrap.appendChild(this.coordEl);
    if (!this.opts.noControls) {
      this.ctrls = document.createElement("div"); this.ctrls.className = "mapctrls";
      this.ctrls.innerHTML = `
        <button type="button" data-a="in" title="Zoom in">+</button>
        <button type="button" data-a="out" title="Zoom out">&minus;</button>
        <button type="button" data-a="home" title="Reset view">
          <svg class="icon" viewBox="0 0 24 24"><path d="M3 11.5 12 4l9 7.5"/><path d="M5 10v9h14v-9"/></svg>
        </button>
        <button type="button" data-a="layers" title="Layers">
          <svg class="icon" viewBox="0 0 24 24"><path d="m12 3-9 5 9 5 9-5-9-5Z"/><path d="m3 13 9 5 9-5"/></svg>
        </button>`;
      wrap.appendChild(this.ctrls);
      this.ctrls.addEventListener("click", (e) => {
        const b = e.target.closest("button"); if (!b) return;
        if (b.dataset.a === "home") this.resetView();
        else if (b.dataset.a === "layers") this.layersPanel.classList.toggle("show");
        else this._zoomStep(b.dataset.a === "in" ? -1 : 1);
      });
      this.layersPanel = document.createElement("div"); this.layersPanel.className = "maplayers";
      this.layersPanel.innerHTML = `
        <label><input type="checkbox" data-l="sat" checked>Satellite</label>
        <label><input type="checkbox" data-l="borders" checked>Borders &amp; cities</label>
        <label><input type="checkbox" data-l="rivers" checked>Rivers</label>
        <label><input type="checkbox" data-l="data" checked>${esc(this.opts.dataLabel || "Candidates")}</label>
        ${this.opts.showWatchLayer ? `<label><input type="checkbox" data-l="watchAreas" checked>Watch areas</label>` : ""}`;
      wrap.appendChild(this.layersPanel);
      this.layersPanel.addEventListener("change", (e) => {
        const cb = e.target.closest("input[data-l]"); if (!cb) return;
        this.layers[cb.dataset.l] = cb.checked;
        this.draw();
      });
      document.addEventListener("click", (e) => {
        if (!this.layersPanel.classList.contains("show")) return;
        if (e.target === this.layersPanel || this.layersPanel.contains(e.target)) return;
        if (e.target.closest && e.target.closest('[data-a="layers"]') === this.ctrls?.querySelector('[data-a="layers"]')) return;
        this.layersPanel.classList.remove("show");
      });
    }
    if (this.opts.basemap !== false) {
      loadBasemap().then(bm => { this.basemap = bm; this.draw(); });
      loadMosaicIndex().then(tiles => {
        this.mosaicTiles = tiles;
        tiles.forEach(t => t.img.addEventListener("load", () => this.draw()));
      });
    }
    canvas.addEventListener("click", (e) => this._click(e));
    canvas.addEventListener("dblclick", (e) => this._dblclick(e));
    canvas.addEventListener("wheel", (e) => this._wheel(e), { passive: false });
    canvas.addEventListener("mousemove", (e) => this._hover(e));
    canvas.addEventListener("mousemove", (e) => this._updateCoord(e));
    canvas.addEventListener("mouseleave", () => { this._hideTip(); this.coordEl.textContent = ""; });
    canvas.addEventListener("mousedown", (e) => this._mdown(e));
    window.addEventListener("mousemove", (e) => this._mmove(e));
    window.addEventListener("mouseup", (e) => this._mup(e));
    canvas.tabIndex = 0;
    new ResizeObserver(() => this.draw()).observe(canvas);
  }
  _updateCoord(e) {
    if (!this._lastProj) return;
    const r = this.c.getBoundingClientRect();
    const lon = this._lastProj.lon(e.clientX - r.left), lat = this._lastProj.lat(e.clientY - r.top);
    const ns = lat >= 0 ? "N" : "S", ew = lon >= 0 ? "E" : "W";
    this.coordEl.textContent = `${Math.abs(lat).toFixed(4)}°${ns}, ${Math.abs(lon).toFixed(4)}°${ew}`;
  }
  _hideTip() { this.tip.classList.remove("show"); if (!this._pan && !this._drag) this.c.style.cursor = this.opts.onDrawRect ? "crosshair" : "grab"; }
  _hitAt(mx, my) {
    if (this._clusterScreen) {
      for (let i = this._clusterScreen.length - 1; i >= 0; i--) {
        const c = this._clusterScreen[i];
        if (Math.hypot(c.x - mx, c.y - my) < c.r + 2) return { __cluster: c.cluster };
      }
    }
    if (!this._screen) return null;
    for (let i = this._screen.length - 1; i >= 0; i--) {
      const s = this._screen[i];
      if (s.poly) { if (pointInPoly(mx, my, s.scr)) return s.it; }
      else { const [x, y] = s.scr[0]; if (Math.hypot(x - mx, y - my) < Math.max(9, (s.it.r || 0) + 3)) return s.it; }
    }
    return null;
  }
  _hover(e) {
    if (this._pan && this._pan.moved) return;
    if ((!this._screen || !this._screen.length) && (!this._clusterScreen || !this._clusterScreen.length)) return;
    const r = this.c.getBoundingClientRect();
    const mx = e.clientX - r.left, my = e.clientY - r.top;
    const hit = this._hitAt(mx, my);
    if (!hit) { this._hideTip(); return; }
    if (!this._drag && !this._pan) this.c.style.cursor = "pointer";
    if (hit.__cluster) {
      this.tip.style.setProperty("--tip-color", "var(--accent)");
      this.tip.innerHTML = `<b>${hit.__cluster.n.toLocaleString()} items here &middot; click to zoom in</b>`;
    } else {
      const label = hit.label || hit.id || "";
      // --tip-color is a CSS custom property, so it needs a CSS colour
      // STRING - hit.color may be a resolved {r,g,b} triple (change-type
      // items) or a plain hex string (region/AOI chip colours), see
      // _colorTriple() above
      const col = hit.color ? _colorTriple(hit.color) : null;
      this.tip.style.setProperty("--tip-color", col ? `rgb(${col.r},${col.g},${col.b})` : "var(--accent)");
      this.tip.innerHTML = `<b>${esc(label)}</b>`;
    }
    const left = Math.min(mx + 14, r.width - this.tip.offsetWidth - 8);
    const top = Math.max(my - 34, 4);
    this.tip.style.left = Math.max(4, left) + "px";
    this.tip.style.top = top + "px";
    this.tip.classList.add("show");
  }
  /* left-drag: draws an AOI rectangle when onDrawRect is configured (Watch
     Areas), otherwise pans. Shift+drag always pans, even in rect mode, so a
     watch-area AOI can still be drawn after panning to the right spot. */
  _mdown(e) {
    if (e.button !== 0) return;
    const r = this.c.getBoundingClientRect();
    const x = e.clientX - r.left, y = e.clientY - r.top;
    if (this.opts.onDrawRect && !e.shiftKey) {
      this._drag = { x0: x, y0: y, x1: null, y1: null };
    } else {
      this._pan = { x0: e.clientX, y0: e.clientY, bbox0: this.bbox.slice(), moved: false };
      this.c.style.cursor = "grabbing";
    }
  }
  _mmove(e) {
    if (this._drag) {
      const r = this.c.getBoundingClientRect();
      this._drag.x1 = e.clientX - r.left; this._drag.y1 = e.clientY - r.top;
      this.draw();
    } else if (this._pan) {
      const dx = e.clientX - this._pan.x0, dy = e.clientY - this._pan.y0;
      if (Math.abs(dx) > 3 || Math.abs(dy) > 3) this._pan.moved = true;
      if (!this._pan.moved) return;
      const r = this.c.getBoundingClientRect();
      const [w, s, e2, n] = this._pan.bbox0;
      const k = Math.min(r.width / (e2 - w), r.height / (n - s));
      const dlon = -dx / k, dlat = dy / k;
      this._anim = null;
      this.bbox = [w + dlon, s + dlat, e2 + dlon, n + dlat];
      this._hideTip();
      this.draw();
    }
  }
  _mup() {
    if (this._drag) {
      const d = this._drag; this._drag = null;
      const moved = d.x1 != null && (Math.abs(d.x1 - d.x0) > 4 || Math.abs(d.y1 - d.y0) > 4);
      this.draw();
      if (moved && this.opts.onDrawRect) {
        const p = this._lastProj;
        const lon0 = p.lon(Math.min(d.x0, d.x1)), lon1 = p.lon(Math.max(d.x0, d.x1));
        const lat0 = p.lat(Math.max(d.y0, d.y1)), lat1 = p.lat(Math.min(d.y0, d.y1));
        this.opts.onDrawRect([lon0, lat0, lon1, lat1]);
      }
    }
    if (this._pan) {
      this._justPanned = this._pan.moved;
      this._pan = null;
      this.c.style.cursor = this.opts.onDrawRect ? "crosshair" : "grab";
      if (this._justPanned) setTimeout(() => { this._justPanned = false; }, 0);
    }
  }
  _zoomAt(mx, my, factor) {
    const p = this._lastProj || this._proj();
    const lon = p.lon(mx), lat = p.lat(my);
    const [w, s, e, n] = this.bbox;
    let nw = lon - (lon - w) * factor, ne = lon + (e - lon) * factor;
    let ns = lat - (lat - s) * factor, nn = lat + (n - lat) * factor;
    const minSpan = 0.006;
    if (ne - nw < minSpan) { const c = (ne + nw) / 2; nw = c - minSpan / 2; ne = c + minSpan / 2; }
    if (nn - ns < minSpan) { const c = (nn + ns) / 2; ns = c - minSpan / 2; nn = c + minSpan / 2; }
    this._anim = null;
    this.bbox = [nw, ns, ne, nn];
    this.draw();
  }
  _wheel(e) {
    e.preventDefault();
    const r = this.c.getBoundingClientRect();
    this._zoomAt(e.clientX - r.left, e.clientY - r.top, e.deltaY > 0 ? 1.16 : 1 / 1.16);
  }
  _zoomStep(dir) { this._zoomAt(this.c.clientWidth / 2, this.c.clientHeight / 2, dir > 0 ? 1.5 : 1 / 1.5); }
  _dblclick(e) {
    if (!this.opts.onDrawRect) { // rect-draw maps reserve drag for AOI selection; skip dblclick-zoom there
      const r = this.c.getBoundingClientRect();
      this._zoomAt(e.clientX - r.left, e.clientY - r.top, 0.45);
    }
  }
  resetView() { this.flyTo(this.homeBbox, true); }
  _bboxEq(a, b) { return a && b && a.every((v, i) => Math.abs(v - b[i]) < 1e-7); }
  flyTo(bbox, animate = true) {
    if (!bbox) return;
    if (!animate || this._bboxEq(this.bbox, bbox)) { this.bbox = bbox; this._anim = null; this.draw(); return; }
    this._anim = { from: this.bbox.slice(), to: bbox.slice(), t0: performance.now(), dur: 420 };
    const tick = () => {
      if (!this._anim) return;
      const { from, to, t0, dur } = this._anim;
      const t = Math.min(1, (performance.now() - t0) / dur);
      const ease = 1 - Math.pow(1 - t, 3);
      this.bbox = from.map((v, i) => v + (to[i] - v) * ease);
      this.draw();
      if (t < 1) requestAnimationFrame(tick); else this._anim = null;
    };
    requestAnimationFrame(tick);
  }
  setData(items, bbox) {
    this.items = items || [];
    const target = bbox || (this.items.length ? this._autobbox() : null);
    if (target) { this.homeBbox = target; this.flyTo(target, this._everSet); }
    this._everSet = true;
    this.draw();
  }
  /* Like setData, but never touches the camera - for maps whose view is
     driven explicitly (flyTo / resetView) rather than auto-fit to whatever
     items happen to be on screen right now (Watch Areas: redrawing the region
     chips or the AOI selection box must not re-frame the whole map). */
  setItems(items) { this.items = items || []; this._everSet = true; this.draw(); }
  /* Step 4A: watch-area boundaries as their own layer, independent of
     items/layers.data (findings) - never re-frames the camera, same reasoning
     as setItems() above. */
  setWatchAreas(items) { this.watchAreaItems = items || []; this.draw(); }
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
      W, H, k,
      x: (lon) => ox + (lon - w) * k,
      y: (lat) => H - oy - (lat - s) * k,
      lon: (x) => w + (x - ox) / k,
      lat: (y) => s + (H - oy - y) / k,
    };
  }
  _drawLayer(fc, ctx, p, style) {
    if (!fc) return;
    ctx.lineWidth = style.w; ctx.strokeStyle = style.stroke; ctx.fillStyle = style.fill || "transparent";
    if (style.dash) ctx.setLineDash(style.dash); else ctx.setLineDash([]);
    for (const f of fc.features) {
      if (f._b && !bboxIntersects(f._b, this.bbox)) continue;
      ctx.beginPath();
      for (const ring of geomRings(f.geometry)) {
        ring.forEach(([lon, lat], i) => { const x = p.x(lon), y = p.y(lat); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
        if (style.close) ctx.closePath();
      }
      if (style.fill) ctx.fill();
      if (style.stroke) ctx.stroke();
    }
    ctx.setLineDash([]);
  }
  _drawBasemapImages(ctx, p) {
    if (!this.mosaicTiles) return;
    for (const t of this.mosaicTiles) {
      if (!t.img.complete || !t.img.naturalWidth || !bboxIntersects(t.bbox, this.bbox)) continue;
      const [tw, ts, te, tn] = t.bbox;
      const x0 = p.x(tw), x1 = p.x(te), y0 = p.y(tn), y1 = p.y(ts);
      ctx.drawImage(t.img, x0, y0, x1 - x0, y1 - y0);
    }
  }
  _drawCities(ctx, p, span, css) {
    const cities = this.basemap.cities;
    if (!cities) return;
    const threshold = cityPopThresholdAt(span);
    if (!threshold) return;
    ctx.font = "11px system-ui, sans-serif";
    ctx.fillStyle = css.getPropertyValue("--geo-city") || css.getPropertyValue("--ink-dim");
    let drawn = 0;
    for (const f of cities.features) {
      if (drawn >= 40) break;
      const props = f.properties;
      if ((props.pop || 0) < threshold || !bboxIntersects(f._b, this.bbox)) continue;
      const [lon, lat] = f.geometry.coordinates;
      const x = p.x(lon), y = p.y(lat);
      ctx.beginPath(); ctx.arc(x, y, props.capital ? 3 : 2, 0, Math.PI * 2);
      ctx.fillStyle = "#fff"; ctx.fill();
      // halo behind the label - readable over both flat basemap and photo tiles
      ctx.lineWidth = 3; ctx.strokeStyle = "rgba(10,14,22,.75)";
      ctx.strokeText(props.name, x + 5, y + 3);
      ctx.fillStyle = "#eef2fa";
      ctx.fillText(props.name, x + 5, y + 3);
      drawn++;
    }
  }
  _clusterPoints(p) {
    const pts = this.items.filter(it => !it.ring);
    if (!pts.length) return [];
    const cell = 46; // px - tight enough that nearby-but-distinct points stay separate once zoomed in
    const buckets = new Map();
    for (const it of pts) {
      const x = p.x(it.lon), y = p.y(it.lat);
      const key = Math.round(x / cell) + "," + Math.round(y / cell);
      let b = buckets.get(key);
      if (!b) { b = { x: 0, y: 0, n: 0, items: [] }; buckets.set(key, b); }
      b.x += x; b.y += y; b.n++; b.items.push(it);
    }
    return [...buckets.values()].map(b => ({ x: b.x / b.n, y: b.y / b.n, n: b.n, items: b.items }));
  }
  draw() {
    const ctx = this.ctx, p = this._proj();
    this._lastProj = p;
    const css = getComputedStyle(document.body);
    ctx.clearRect(0, 0, p.W, p.H);
    ctx.fillStyle = css.getPropertyValue("--surface-1") || css.getPropertyValue("--panel");
    ctx.fillRect(0, 0, p.W, p.H);

    const [w, s, e, n] = this.bbox;
    const span = e - w;

    // land tint first (shows through wherever we have no imagery), then our
    // own satellite mosaic on top of it, then the vector line-work on top of
    // that again so borders/coastline/rivers/cities stay legible over a photo
    if (this.basemap) {
      this._drawLayer(this.basemap.countries, ctx, p, { fill: css.getPropertyValue("--geo-land"), close: true });
    }
    if (this.layers.sat) this._drawBasemapImages(ctx, p);
    if (this.basemap && this.layers.borders) {
      const bm = this.basemap;
      this._drawLayer(bm.countries, ctx, p, { stroke: css.getPropertyValue("--geo-border"), w: 1, close: true });
      this._drawLayer(bm.coastline, ctx, p, { stroke: css.getPropertyValue("--geo-coast"), w: 1.1 });
      if (showStatesAt(span)) this._drawLayer(bm.states, ctx, p, { stroke: css.getPropertyValue("--geo-state"), w: 0.8, dash: [3, 3], close: true });
    }
    if (this.basemap && this.layers.rivers) {
      this._drawLayer(this.basemap.rivers, ctx, p, { stroke: css.getPropertyValue("--geo-river"), w: 1 });
    }
    if (this.basemap && this.layers.borders) this._drawCities(ctx, p, span, css);

    // graticule - dimmer now that real geography carries most of the context
    ctx.strokeStyle = css.getPropertyValue("--geo-grid") || css.getPropertyValue("--line");
    ctx.fillStyle = css.getPropertyValue("--muted");
    ctx.lineWidth = 1; ctx.font = "10px system-ui, sans-serif";
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
    this._clusterScreen = [];
    if (this.layers.data) {
      // AOI/region polygons - never clustered, there are only ever a handful
      for (const it of this.items) {
        if (!it.ring) continue;
        const selected = it.id != null && it.id === this.sel;
        const scr = it.ring.map(([lo, la]) => [p.x(lo), p.y(la)]);
        this._screen.push({ it, scr, poly: true });
        ctx.beginPath();
        scr.forEach(([x, y], i) => i ? ctx.lineTo(x, y) : ctx.moveTo(x, y));
        ctx.closePath();
        // kept light even when selected - these sit on top of the satellite
        // mosaic (Watch's region/AOI rings), and the old 0.22/0.5 fill was
        // heavy enough to read as a flat colour swatch instead of imagery
        // for every region except the two biggest tiles
        const col = _colorTriple(it.color);
        ctx.fillStyle = `rgba(${col.r},${col.g},${col.b},${selected ? 0.34 : 0.08})`;
        ctx.strokeStyle = `rgb(${col.r},${col.g},${col.b})`; ctx.lineWidth = selected ? 2.5 : 1.2;
        ctx.fill(); ctx.stroke();
      }
      // point items - grid-clustered in screen space, recomputed every frame,
      // so a crowded AOI (e.g. 800+ candidates over Ayodhya) reads as a few
      // count badges instead of an unreadable stack of overlapping dots, and
      // clusters split apart into individual markers on their own as the
      // analyst zooms in (no separate "zoom level" bookkeeping needed)
      for (const cl of this._clusterPoints(p)) {
        if (cl.n === 1) {
          const it = cl.items[0];
          const selected = it.id != null && it.id === this.sel;
          const rad = (it.r || 4.5) + (selected ? 2.5 : 0);
          this._screen.push({ it, scr: [[cl.x, cl.y]], poly: false });
          const col = _colorTriple(it.color);
          ctx.beginPath(); ctx.arc(cl.x, cl.y, rad, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(${col.r},${col.g},${col.b},${selected ? 1 : 0.88})`; ctx.fill();
          ctx.strokeStyle = selected ? css.getPropertyValue("--ink") : `rgba(${col.r},${col.g},${col.b},0.9)`;
          ctx.lineWidth = selected ? 2 : 1;
          ctx.stroke();
        } else {
          const rad = Math.min(22, 11 + Math.sqrt(cl.n) * 2.2);
          const mixed = cl.items.some(x => x.color !== cl.items[0].color);
          const color = mixed ? Tokens.triple("accent") : _colorTriple(cl.items[0].color);
          ctx.beginPath(); ctx.arc(cl.x, cl.y, rad, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(${color.r},${color.g},${color.b},0.86)`; ctx.fill();
          ctx.strokeStyle = "rgba(255,255,255,.85)"; ctx.lineWidth = 1.5; ctx.stroke();
          ctx.fillStyle = "#fff"; ctx.font = "700 " + (cl.n > 99 ? 10.5 : 12) + "px system-ui, sans-serif";
          ctx.textAlign = "center"; ctx.textBaseline = "middle";
          ctx.fillText(cl.n > 999 ? (cl.n / 1000).toFixed(1) + "k" : String(cl.n), cl.x, cl.y + 0.5);
          ctx.textAlign = "start"; ctx.textBaseline = "alphabetic";
          this._clusterScreen.push({ cluster: cl, x: cl.x, y: cl.y, r: rad });
        }
      }
    }
    // Step 4A: watch-area boundaries - a dashed outline only (no fill, not
    // hit-tested), deliberately distinct from the findings styling above so
    // the two layers stay visually separable when both are on.
    if (this.layers.watchAreas && this.watchAreaItems.length) {
      const gold = Tokens.triple("accent-gold");
      ctx.setLineDash([6, 4]); ctx.lineWidth = 1.4;
      ctx.strokeStyle = `rgb(${gold.r},${gold.g},${gold.b})`;
      for (const it of this.watchAreaItems) {
        if (!it.ring) continue;
        const scr = it.ring.map(([lo, la]) => [p.x(lo), p.y(la)]);
        ctx.beginPath();
        scr.forEach(([x, y], i) => i ? ctx.lineTo(x, y) : ctx.moveTo(x, y));
        ctx.closePath(); ctx.stroke();
      }
      ctx.setLineDash([]);
    }
    if (this._drag && this._drag.x1 != null) {
      const x0 = Math.min(this._drag.x0, this._drag.x1), x1 = Math.max(this._drag.x0, this._drag.x1);
      const y0 = Math.min(this._drag.y0, this._drag.y1), y1 = Math.max(this._drag.y0, this._drag.y1);
      const accent = Tokens.triple("accent");
      ctx.fillStyle = `rgba(${accent.r},${accent.g},${accent.b},0.15)`;
      ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
      ctx.strokeStyle = `rgb(${accent.r},${accent.g},${accent.b})`; ctx.lineWidth = 1.5; ctx.setLineDash([5, 3]);
      ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
      ctx.setLineDash([]);
    }
    this._updateScaleBar(p);
    if (!this._drag) this.c.style.cursor = this._pan ? "grabbing" : (this.opts.onDrawRect ? "crosshair" : "grab");
  }
  _updateScaleBar(p) {
    if (!this.scaleEl) return;
    const [, s, , n] = this.bbox;
    const midLat = (s + n) / 2;
    const metersPerDeg = 111320 * Math.cos(midLat * Math.PI / 180);
    if (!(metersPerDeg > 0) || !p.k) { this.scaleEl.style.opacity = 0; return; }
    const metersPerPx = metersPerDeg / p.k;
    const targetPx = Math.min(120, p.W * 0.28);
    const niceM = niceScale(targetPx * metersPerPx);
    const barPx = niceM / metersPerPx;
    const label = niceM >= 1000 ? `${(niceM / 1000).toFixed(niceM % 1000 ? 1 : 0)} km` : `${Math.round(niceM)} m`;
    this.scaleEl.style.opacity = 1;
    this.scaleEl.style.setProperty("--bar-w", barPx.toFixed(1) + "px");
    this.scaleEl.querySelector("b").textContent = label;
  }
  _click(e) {
    if (this._justPanned) return;
    const r = this.c.getBoundingClientRect();
    const mx = e.clientX - r.left, my = e.clientY - r.top;
    const hit = this._hitAt(mx, my);
    if (!hit) return;
    if (hit.__cluster) { this._zoomToCluster(hit.__cluster); return; }
    if (this.onPick) this.onPick(hit);
  }
  // dense areas (e.g. 800+ change candidates over one AOI) collapse into a
  // single count badge at low zoom - clicking one flies in tight around just
  // those items, which shrinks the grid cells enough for the next frame's
  // clustering pass to split them apart into fewer, bigger clusters (or
  // individual markers), rather than a single "zoom in more" no-op.
  _zoomToCluster(cl) {
    let w = 180, s = 90, e = -180, n = -90;
    for (const it of cl.items) { w = Math.min(w, it.lon); e = Math.max(e, it.lon); s = Math.min(s, it.lat); n = Math.max(n, it.lat); }
    const minSpan = 0.02;
    if (e - w < minSpan) { const c = (e + w) / 2; w = c - minSpan / 2; e = c + minSpan / 2; }
    if (n - s < minSpan) { const c = (n + s) / 2; s = c - minSpan / 2; n = c + minSpan / 2; }
    const px = (e - w) * 0.4, py = (n - s) * 0.4;
    this.flyTo([w - px, s - py, e + px, n + py], true);
  }
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
const IMGLOAD = `onload="this.classList.add('loaded');this.parentElement.classList.remove('c-skel')" onerror="this.parentElement.classList.remove('c-skel')"`;
const thumbHTML = (src) => `<div class="thumbwrap c-skel"><img class="thumb" loading="lazy" ${IMGLOAD} src="${esc(src)}" alt=""></div>`;
// emptyState() replaced by Components.emptyState()/.emptyStates.* (components.js)
function legend(elId, types) {
  $(elId).innerHTML = types.map(t => `<span><i style="background:${Tokens.rgb("change-" + t)}"></i>${esc(CTYPE_HUMAN[t] || t)}</span>`).join("");
}

/* ---------------------------------------------------------------- routing */
const views = ["overview", "search", "queue", "detail", "discovery", "detect", "watch"];
// dev-gallery is deliberately NOT in `views`/the sidebar - it's a
// router-reachable diagnostic page (#/dev-gallery), not one of the seven
// analyst screens.
const HIDDEN_VIEWS = ["dev-gallery"];
const ROUTE_TITLES = {
  overview: "Overview", search: "Search", queue: "Review Queue", detail: "Candidate Detail",
  discovery: "Discovery", detect: "Object Detection", watch: "Watch Areas", "dev-gallery": "Component Gallery",
};
function go(route) { location.hash = "#/" + route; }
function router() {
  const parts = (location.hash || "#/overview").slice(2).split("/");
  const route = views.includes(parts[0]) ? parts[0] : (HIDDEN_VIEWS.includes(parts[0]) ? parts[0] : "overview");
  [...views, ...HIDDEN_VIEWS].forEach(v => $("#view-" + v).classList.toggle("active", v === route));
  $$(".nav-item").forEach(b => b.classList.toggle("active", b.dataset.route === route));
  if ($("#topbar_title")) $("#topbar_title").textContent = ROUTE_TITLES[route] || route;
  if (route === "overview") ensureOverview();
  if (route === "search") ensureSearch();
  if (route === "queue") ensureQueue();
  if (route === "discovery") ensureDiscovery();
  if (route === "detect") ensureDetect();
  if (route === "watch") ensureWatch();
  if (route === "dev-gallery") ensureDevGallery();
  if (route === "detail" && parts[1]) openDetail(decodeURIComponent(parts[1]));
  if (route === "detail" || route === "detect") requestAnimationFrame(fitAllImageryLayouts);
}
window.addEventListener("hashchange", router);
$$(".nav-item").forEach(b => b.addEventListener("click", () => go(b.dataset.route)));

/* ---------------------------------------------------------------- SIDEBAR
   Collapse state lives in a module-level variable, not localStorage - "for
   the session" means for the life of this single-page app, same pattern as
   detailState's d_railCollapsed (see docs/REDESIGN_PHASE2B1_REPORT.md). */
let sidebarCollapsed = false;
function setSidebarCollapsed(collapsed) {
  sidebarCollapsed = collapsed;
  $("#sidebar").classList.toggle("is-collapsed", collapsed);
  $("#sidebar_toggle").setAttribute("aria-expanded", String(!collapsed));
}
$("#sidebar_toggle").addEventListener("click", () => setSidebarCollapsed(!sidebarCollapsed));

/* ---------------------------------------------------------------- OVERVIEW
   Demo-facing landing view. Read-only projection of /presentation/summary +
   the same /candidates footprints the Review Queue uses. */
let overviewMap, overviewInit = false;
let OV_REGION_LOOKUP = {};
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
// Step 4B: turns sar_corroboration.verdict (a free-text string from
// geoseek.sar.evidence.sar_factor - "strong agreement" / "weak agreement" /
// "clear disagreement" / "weak disagreement" / "within speckle noise
// (neutral)" / "no usable co-located SAR" / "no C-band expectation for '<type>'")
// into one of five fixed phrases. Substring match, not an exact enum, since
// the backend string is meant to be read directly, not parsed - matching the
// substrings it's actually built from keeps this in sync without a shared enum.
function sarPhrase(sar) {
  const v = sar && sar.verdict;
  if (!v || v.startsWith("no ")) return "no radar coverage";
  if (v.includes("strong agreement")) return "strongly confirmed by radar";
  if (v.includes("weak agreement")) return "weakly confirmed by radar";
  if (v.includes("disagreement")) return "contradicted by radar";
  if (v.includes("neutral")) return "radar neutral";
  return "radar corroboration inconclusive";
}
// One plain-language sentence, built only from fields already computed
// server-side (persistence + SAR) - never a new judgement of its own. Shown
// directly under the confidence badge (Step 4B).
function confidenceReasonSentence(d, band, nLater) {
  const persClause = persistenceHuman(d.persistence, nLater);
  const persLower = persClause.charAt(0).toLowerCase() + persClause.slice(1);
  const radarClause = sarPhrase(d.provenance && d.provenance.sar_corroboration);
  return `${esc(band.toUpperCase())} — ${esc(persLower)}, ${esc(radarClause)}.`;
}
function laterObsCount(traj) {
  const end = ((traj && traj.earliest_supported_change) || {}).window;
  const e = end ? end[1] : null;
  const dates = OBS_DATES.length ? OBS_DATES
    : Array.from(new Set([].concat(...((traj && traj.intervals) || []).map(i => i.window || [])))).sort();
  const n = e ? dates.filter(d => d >= e).length : 0;
  return n || 1;
}
// confBand()/ringHTML()/RING_COL replaced by Components.confBand()/.confRing()
// (components.js) - one copy of the band thresholds instead of two.

// View-mount entrance (fade+rise, ~40ms stagger) fires once per element the
// first time its container is marked mounted - see the .enter-mount/
// .enter-item rules in style.css. Re-renders never replay it because the
// class, once added, is never removed.
function mountOnce(el) { if (el) el.classList.add("is-mounted"); }

function ensureOverview() {
  if (overviewInit) { return; }
  overviewInit = true;
  mountOnce($("#ov-hero")); // hero is static chrome - it enters on view-mount, not on a data fetch
  overviewMap = new CoordMap($("#ov_map"), (it) => { if (it.id) go("detail/" + it.id); },
    { dataLabel: "Findings", showWatchLayer: true });
  // Step 4A: watch-area boundaries as an independent, togglable Overview map
  // layer - read-only GET against the same endpoint the Watch Areas screen
  // already calls; no change to that screen or its own logic.
  api("/watch-areas").then(d => {
    const areas = (d.watch_areas || []).filter(w => w.bbox);
    overviewMap.setWatchAreas(areas.map(w => ({ ring: bboxRing(w.bbox), label: w.name })));
  }).catch(() => {});
  $("#ov_legend").innerHTML = TYPES.map(t =>
    `<span><i style="background:${Tokens.rgb("change-" + t)}"></i>${esc(CTYPE_HUMAN[t] || t)}</span>`).join("");
  $("#ov-go").addEventListener("click", overviewSearch);
  $("#ov-q").addEventListener("keydown", e => { if (e.key === "Enter") overviewSearch(); });
  loadRegions().then(regions => {
    OV_REGION_LOOKUP = {};
    $("#ov_region_dl").innerHTML = regions.map(r => {
      const label = `${friendlyRegionName(r.name)} (${r.n_observations} obs)`;
      OV_REGION_LOOKUP[label.toLowerCase()] = r.name;
      OV_REGION_LOOKUP[friendlyRegionName(r.name).toLowerCase()] = r.name;
      return `<option value="${esc(label)}">`;
    }).join("");
  });
  // a real search box (typing + matching a bundled <datalist>), not a plain
  // dropdown - "fly to a named region" without leaving the keyboard
  $("#ov_region").addEventListener("change", (e) => {
    const raw = OV_REGION_LOOKUP[e.target.value.trim().toLowerCase()];
    if (!raw) {
      if (!e.target.value.trim() && ovViewMode !== "globe") overviewMap.flyTo(overviewMap.homeBbox, true);
      return;
    }
    if (ovViewMode === "globe" && globeInstance) { globeInstance.flyToRegion(raw); return; }
    const r = regionByName(raw);
    overviewMap.flyTo(r ? padBbox(r.bbox) : overviewMap.homeBbox, true);
  });
  $$(".ov-viewtoggle button").forEach(b => b.addEventListener("click", () => setOverviewView(b.dataset.v, true)));
  ensureGlobe();
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
  } catch (e) {
    renderApiError("#ov-featured", e, loadOverview);
  }
  try {
    const n = await api("/notifications");
    renderFeed((n.notifications || []).slice().sort((a, b) => (b.created_at || "").localeCompare(a.created_at || "")));
  } catch (e) {
    renderApiError("#ov-alerts", e, loadOverview);
  }
  try {
    const d = await api("/candidates?limit=400&sort=queue_score");
    const cands = d.candidates || [];
    // bubble-sized by sqrt(area) - true-scale footprints are sub-pixel at this
    // zoomed-out extent, so without this every finding reads as "the same size"
    const sqrts = cands.map(c => Math.sqrt(c.area_m2 || 0));
    const sMin = sqrts.length ? Math.min(...sqrts) : 0, sMax = sqrts.length ? Math.max(...sqrts) : 1;
    overviewMap.setData(cands.map(c => {
      const t = sMax > sMin ? (Math.sqrt(c.area_m2 || 0) - sMin) / (sMax - sMin) : 0.5;
      return {
        id: c.candidate_id, lon: c.centroid_lonlat[0], lat: c.centroid_lonlat[1], r: 3 + t * 11,
        color: Tokens.changeType(c.change_type),
        label: `${CTYPE_HUMAN[c.change_type] || c.change_type} · ${Math.round(c.area_m2).toLocaleString()} m² · conf ${num(c.confidence)}`,
      };
    }), AOI_FALLBACK);
  } catch (e) { /* map is a nicety; counters + cards already rendered */ }
}
// Stat strip: five dense tiles, tabular-figure value + label + (only where
// truthful) a context line derived from the same counters payload - never an
// invented time-over-time delta, since /presentation/summary doesn't expose
// a previous snapshot to diff against (see docs/FRONTEND_AUDIT.md Section 8).
function renderOverviewCounters(p) {
  const c = p.counters || {};
  if (p.aoi) $("#ov-aoi").textContent = p.aoi;
  const dates = p.observation_dates || [];
  const nCand = Number(c.change_candidates) || 0;
  const span = dates.length ? `${dates[0]} → ${dates[dates.length - 1]}` : "unknown span";
  $("#ov-span").textContent =
    `${nf(nCand)} flagged change${nCand === 1 ? "" : "s"} across ${dates.length} observation${dates.length === 1 ? "" : "s"}, ${span}.`;
  $("#ov_caption").textContent = "Marker size shows the affected area, colour shows the change type.";

  const hiConf = Number(c.high_confidence) || 0;
  const hiPct = nCand ? Math.round((hiConf / nCand) * 1000) / 10 : 0;
  const unseen = Number(c.unseen_notifications) || 0;
  const cells = [
    { k: "change_candidates", label: "change candidates", accent: true, context: null,
      hint: "Automatically detected changes waiting for analyst review." },
    { k: "high_confidence", label: "high-confidence", context: nCand ? `${hiPct}% of ${nf(nCand)}` : null,
      hint: "How many candidates the system is at least 85% confident about." },
    { k: "tiles_indexed", label: "tiles indexed", context: c.scenes ? `${nf(c.scenes)} satellite scenes` : null,
      hint: "Image tiles that can be found by a plain-language search." },
    { k: "regions", label: "regions", context: c.vectors ? `${nf(c.vectors)} searchable image tiles` : null,
      hint: "Distinct study areas currently staged for analysis." },
    { k: "analyst_decisions", label: "analyst decisions", context: `${nf(unseen)} unseen notification${unseen === 1 ? "" : "s"}`,
      hint: "Confirm/reject decisions recorded so far, kept in a permanent audit trail." },
  ];
  const el = $("#ov-counters");
  const first = !el.children.length;
  el.innerHTML = cells.map((cell, i) => `
    <div class="ov-stat${cell.accent ? " ov-stat--accent" : ""} enter-item" style="--stagger-i:${i}">
      <span class="ov-stat__value font-numeric" id="ov_stat_${i}" data-target="${Number(c[cell.k]) || 0}">0</span>
      <span class="ov-stat__label">${esc(cell.label)}${Components.hint(cell.hint)}</span>
      ${cell.context ? `<span class="ov-stat__context">${esc(cell.context)}</span>` : ""}
    </div>`).join("");
  // animate from zero on first render only (a re-render with the same data,
  // e.g. a re-run of the pipeline, should not re-trigger the count-up)
  $$("#ov-counters .ov-stat__value").forEach(v => Components.animateStatValue(v, +v.dataset.target, first ? 900 : 0));
  if (first) mountOnce(el);
}

// Narrative feed: real /notifications, severity colour-coded through the
// semantic ramp (kind:"severity" - high severity is danger red, never the
// confidence polarity's green - see components.js's bandPill). This is the
// same data the Watch Areas screen's notification list shows; there is no
// separate "recent activity" endpoint (see docs/FRONTEND_AUDIT.md Section 8),
// so a lone entry here is the backend's true current state, not a layout bug.
function renderFeed(notifications) {
  const el = $("#ov-alerts");
  const panel = $("#ov-alerts-panel");
  if (!notifications.length) {
    el.innerHTML = Components.emptyState({ icon: "alerts", cause: "No alerts yet.", action: "Define a watch area to start monitoring." });
    mountOnce(panel);
    return;
  }
  el.innerHTML = notifications.map((n, i) => `
    <button type="button" class="ov-feed-row enter-item" style="--stagger-i:${i}" data-ids="${esc(n.candidate_ids.join(","))}" data-name="${esc(n.watch_name || n.watch_id)}">
      <span class="ov-feed-row__band">${Components.bandPill(n.severity || "low", { kind: "severity" })}</span>
      <span class="ov-feed-row__title">${esc(n.watch_name || n.watch_id)}</span>
      <span class="ov-feed-row__meta">
        <span>${n.candidate_ids.length} candidate(s)</span>
        <span>obs ${esc(n.observation_id)}${n.observation_date ? " (" + esc(n.observation_date) + ")" : ""}</span>
        <span class="ov-feed-row__time font-numeric">${esc((n.created_at || "").replace("T", " ").slice(0, 16))}</span>
      </span>
    </button>`).join("");
  $$("#ov-alerts .ov-feed-row").forEach(row => row.addEventListener("click", () =>
    openQueueForIds(row.dataset.ids.split(","), `${row.dataset.ids.split(",").length} candidate(s) from alert "${row.dataset.name}"`)));
  mountOnce(panel);
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
  const el = $("#ov-featured");
  if (!featured.length) {
    el.innerHTML = Components.emptyState({ icon: "featured", cause: "No featured findings yet.", action: "Results appear once change candidates are ranked." });
    mountOnce(el);
    return;
  }
  el.innerHTML = featured.map((f, i) => {
    const band = f.confidence_band || Components.confBand(f.confidence);
    return `<figure class="ff-card liftable enter-item" style="--stagger-i:${i}" data-id="${esc(f.candidate_id)}">
      <div class="ff-imgs">
        <div class="ff-im c-skel"><img loading="lazy" ${IMGLOAD} src="${esc(f.imagery.before)}&scale=2" alt="before"><span>before · ${esc(f.before_date)}</span></div>
        <div class="ff-im c-skel"><img loading="lazy" ${IMGLOAD} src="${esc(f.imagery.after)}&scale=2" alt="after"><span>after · ${esc(f.after_date)}</span></div>
      </div>
      <figcaption>
        <div class="ff-badges" style="justify-content:space-between">
          <div class="ff-badges" style="margin:0">
            ${Components.changeBadge(f.change_type, { label: f.change_type_human || CTYPE_HUMAN[f.change_type] || f.change_type })}
            ${Components.bandPill(band)}
          </div>
          ${Components.confRing(f.confidence)}
        </div>
        <p>${esc(f.caption)}</p>
        <div class="ff-sub">${esc(f.persistence_human)}</div>
      </figcaption>
    </figure>`;
  }).join("");
  $$("#ov-featured .ff-card").forEach(card =>
    card.addEventListener("click", () => go("detail/" + card.dataset.id)));
  mountOnce(el);
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
  $("#s_cards").innerHTML = `Loading&hellip;`;
  try {
    const d = await api("/search/text?" + p.toString());
    renderSearchResults(d);
  } catch (e) { renderApiError("#s_cards", e, runSearch); }
}
async function runImageSearch(ev) {
  const f = ev.target.files[0]; if (!f) return;
  const b64 = await new Promise(res => { const r = new FileReader(); r.onload = () => res(r.result.split(",")[1]); r.readAsDataURL(f); });
  const p = searchFilterQS();
  const body = { image_base64: b64, k: 30 };
  for (const [k, v] of p) body[k] = v;
  $("#s_cards").innerHTML = `Loading&hellip;`;
  try {
    const d = await api("/search/image", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
    toast(`image similarity: ${d.count} hits in ${d.latency_ms} ms`);
    renderSearchResults(d);
  } catch (e) { renderApiError("#s_cards", e, () => runImageSearch(ev)); }
}
function renderSearchResults(d) {
  const res = d.results || [];
  $("#s_count").textContent = `${res.length} hit${res.length === 1 ? "" : "s"} · ${d.latency_ms} ms server`;
  $("#s_cards").innerHTML = res.map(r => `
    <div class="card" data-tile="${esc(r.tile_id)}">
      ${thumbHTML(`/tile/${encodeURIComponent(r.tile_id)}/thumbnail`)}
      <div class="meta">
        <div class="row"><b>score ${num(r.score, 3)}</b><span>${esc(r.acq_date)}</span></div>
        <div class="row"><span>${esc(r.sensor)}</span><span>cloud ${num(r.cloud_fraction * 100, 0)}%</span></div>
        <div class="row"><span>${num(r.lat, 4)}°N</span><span>${num(r.lon, 4)}°E</span></div>
      </div>
    </div>`).join("") || Components.emptyStates.noSearchResults();
  $$("#s_cards .card").forEach(c => c.addEventListener("click", () => {
    $("#disc_seed").value = c.dataset.tile; go("discovery"); setTimeout(runDiscovery, 30);
  }));
  const mx = Math.max(...res.map(r => r.score), 1e-6);
  searchMap.setData(res.map(r => ({
    id: r.tile_id, tile_id: r.tile_id, lon: r.lon, lat: r.lat,
    color: "#4c8dff", r: 3 + 6 * (r.score / mx),
    label: `score ${num(r.score, 3)} · ${r.acq_date}`,
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
    if (!th.dataset.k) return;  // "where" is a derived, client-side column - not sortable server-side
    const k = th.dataset.k === "earliest" ? "rank" : th.dataset.k;
    queueSort = (queueSort === k) ? "-" + k : k;
    markSortedHeader();
    loadQueue();
  }));
  markSortedHeader();
  loadQueue();
}
const SORT_FIELD_LABEL = {
  queue_score: "priority", confidence: "confidence", significance: "significance",
  area_m2: "how big", rank: "#", change_type: "what changed", persistence: "seen", decision: "decision",
};
function sortFieldLabel(sort) {
  const asc = String(sort || "").startsWith("-");
  const key = String(sort || "").replace(/^-/, "");
  return (SORT_FIELD_LABEL[key] || key) + (asc ? " (ascending)" : "");
}
function markSortedHeader() {
  // API contract (search/api.py): '-' prefix = ascending, no prefix = descending
  const ascending = queueSort.startsWith("-");
  const k = queueSort.replace(/^-/, "");
  $$("#q_table th").forEach(th => {
    const active = (th.dataset.k === "earliest" ? "rank" : th.dataset.k) === k;
    th.classList.toggle("sorted", active);
    th.classList.toggle("asc", active && ascending); // default (no class) = descending = down chevron
  });
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
  await loadRegions();   // so the "where" column can resolve a place name, not just raw coordinates
  $("#q_idsbanner").classList.toggle("hidden", !queueIdsOverride);
  $("#f_clear_ids").classList.toggle("hidden", !queueIdsOverride);
  if (queueIdsOverride) $("#q_idsbanner").textContent = `Filtered to ${queueIdsOverride.note}`;
  const p = queueQS({ sort: queueSort, limit: 400 });
  $("#q_table tbody").innerHTML = `<tr><td colspan="10">Loading…</td></tr>`;
  try {
    const d = await api("/candidates?" + p.toString());
    window._queue = d;
    $("#q_summary").innerHTML =
      `<span><b>${d.total}</b> candidates match</span><span>showing <b>${d.count}</b></span>` +
      `<span>sorted by <b>${esc(sortFieldLabel(d.sort))}</b></span>` +
      `<span class="muted">click a row or a footprint &rarr; detail</span>`;
    $("#q_table tbody").innerHTML = d.candidates.map(c => {
      const band = Components.confBand(c.confidence);
      const pers = persistenceShort(c.persistence);
      const [lon, lat] = c.centroid_lonlat;
      const region = regionForPoint(lon, lat);
      return `
      <tr data-id="${esc(c.candidate_id)}" tabindex="0" title="candidate ${esc(c.candidate_id)}">
        <td data-numeric>${c.rank}</td>
        <td>${Components.changeBadge(c.change_type, { label: CTYPE_HUMAN[c.change_type] || c.change_type })}</td>
        <td><div>${esc(region || "—")}</div><div class="small muted font-numeric">${num(lat, 3)}°N ${num(lon, 3)}°E</div></td>
        <td data-numeric><div style="display:inline-flex;align-items:center;gap:6px">${Components.confRing(c.confidence, { size: "xs" })}${Components.bandPill(band)}</div></td>
        <td data-numeric>${Components.miniBar(c.significance, { label: "significance" })}</td>
        <td data-numeric>${Components.miniBar(c.queue_score, { label: "priority" })}</td>
        <td data-numeric>${esc(humanArea(c.area_m2))}</td>
        <td title="${esc(pers.hover)}">${esc(pers.label)}</td>
        <td>${esc((c.earliest_supported || []).join(" → ") || "—")}</td>
        <td class="verdict ${c.decision}">${esc(c.decision)}</td>
      </tr>`;
    }).join("") || `<tr><td colspan="10">${Components.emptyStates.noCandidatesFiltered()}</td></tr>`;
    $$("#q_table tbody tr").forEach(tr => {
      tr.addEventListener("click", () => go("detail/" + tr.dataset.id));
      tr.addEventListener("keydown", (e) => { if (e.key === "Enter") go("detail/" + tr.dataset.id); });
    });
    queueMap.setData(d.candidates.map(c => ({
      id: c.candidate_id, ring: c.geometry && c.geometry.coordinates[0],
      lon: c.centroid_lonlat[0], lat: c.centroid_lonlat[1],
      color: Tokens.changeType(c.change_type),
      label: `${CTYPE_HUMAN[c.change_type] || c.change_type} · confidence ${num(c.confidence)} · #${c.rank}`,
    })), AOI_FALLBACK);
  } catch (e) {
    renderApiError("#q_table tbody", e, loadQueue, { tdColspan: 10 });
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

/* ---------------------------------------------------------------- DETAIL
   Imagery-dominant layout (Phase 2B-1): one large frame + a segmented
   before/after/overlay control + a unified 5-date strip, all in the
   `.detail-imagery` column; every numeric panel lives in the collapsible
   `.detail-rail`. See docs/REDESIGN_PHASE2B1_REPORT.md for the design and
   the two named API gaps (per-date sensor/cloud metadata; footprint-outline
   pixel geometry) this section works around rather than fakes. */
let detailState = { id: null, view: "slider", date: null, after: "2024", outline: true, data: null, fullDates: {}, metaByYear: {}, deciding: false, sliderPct: 50 };
let d_railCollapsed = false;   // persists for the session (module scope, not reset per candidate)
let detailUIInit = false;
let _paintToken = 0;           // monotonic guard against out-of-order progressive-load responses

function ensureDetailUI() {
  if (detailUIInit) return; detailUIInit = true;

  $$("#d_segmented button").forEach(b => b.addEventListener("click", () => setDetailSegment(b.dataset.view)));
  $("#d_outline").addEventListener("change", (e) => { detailState.outline = e.target.checked; paintImages(); });
  $("#d_rail_toggle").addEventListener("click", () => {
    d_railCollapsed = !d_railCollapsed;
    $("#d_body").classList.toggle("rail-collapsed", d_railCollapsed);
    $("#d_rail_toggle").setAttribute("aria-expanded", String(!d_railCollapsed));
  });

  // Evidence check (Step 4C): one toggle reveals BOTH the technical evidence
  // grid and the suppression-trace panel - they're the same underlying gates
  // as the plain checklist above, just in raw form, so they show/hide together.
  $("#d_showtech").addEventListener("click", () => {
    const showing = $("#d_showtech").getAttribute("aria-expanded") === "true";
    const next = !showing;
    $("#d_showtech").setAttribute("aria-expanded", String(next));
    $("#d_showtech").textContent = next ? "Hide technical details" : "Show technical details";
    $("#d_techwrap").classList.toggle("hidden", !next);
    $("#d_trace_panel").classList.toggle("hidden", !next);
  });

  // ---- before/after wipe (Slider mode): drag anywhere in the frame (not
  // just the handle graphic), a range input below it, and arrow keys when
  // the frame itself is focused - three inputs, one piece of state
  // (detailState.sliderPct). Pointer capture means one pointerdown/move/up
  // triplet handles both mouse and touch without a document-level listener.
  const sliderEl = $("#d_slider");
  let sliderDragging = false;
  const wipePctFromEvent = (e) => {
    const rect = sliderEl.getBoundingClientRect();
    const x = Math.max(rect.left, Math.min(rect.right, e.clientX));
    return rect.width ? ((x - rect.left) / rect.width) * 100 : 50;
  };
  sliderEl.addEventListener("pointerdown", (e) => {
    if (sliderEl.classList.contains("is-disabled")) return;
    sliderDragging = true;
    sliderEl.setPointerCapture(e.pointerId);
    setWipePct(wipePctFromEvent(e));
    sliderEl.focus();
  });
  sliderEl.addEventListener("pointermove", (e) => { if (sliderDragging) setWipePct(wipePctFromEvent(e)); });
  sliderEl.addEventListener("pointerup", () => { sliderDragging = false; });
  sliderEl.addEventListener("pointercancel", () => { sliderDragging = false; });
  sliderEl.addEventListener("keydown", (e) => {
    if (e.key === "ArrowLeft") { e.preventDefault(); setWipePct(detailState.sliderPct - 5); }
    else if (e.key === "ArrowRight") { e.preventDefault(); setWipePct(detailState.sliderPct + 5); }
  });
  $("#d_slider_range").addEventListener("input", (e) => setWipePct(Number(e.target.value)));

  // C confirm / R reject / Escape clears focus - never while a text input has
  // focus (typing a decision note must not accidentally fire a decision).
  document.addEventListener("keydown", (e) => {
    if (!$("#view-detail").classList.contains("active") || $("#d_body").classList.contains("hidden")) return;
    const tag = (document.activeElement || {}).tagName;
    if (e.key === "Escape") { if (document.activeElement && document.activeElement.blur) document.activeElement.blur(); return; }
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
    const k = e.key.toLowerCase();
    if (k === "c") { e.preventDefault(); decide("confirm"); }
    else if (k === "r") { e.preventDefault(); decide("reject"); }
    else if (e.key === "1") setDetailSegment("slider");
    else if (e.key === "2") setDetailSegment("before");
    else if (e.key === "3") setDetailSegment("after");
    else if (e.key === "4") setDetailSegment("overlay");
  });
}

// Sets the wipe divider (0-100, clamped) and mirrors it to the handle's CSS
// position, the range input, and the ARIA value - the three places that need
// to agree on "where is the divider" regardless of which one moved it.
function setWipePct(pct) {
  pct = Math.max(0, Math.min(100, pct));
  detailState.sliderPct = pct;
  const slider = $("#d_slider");
  slider.style.setProperty("--wipe-pct", pct + "%");
  slider.setAttribute("aria-valuenow", String(Math.round(pct)));
  slider.setAttribute("aria-valuetext", `${Math.round(pct)}% before, ${Math.round(100 - pct)}% after`);
  const range = $("#d_slider_range");
  if (Number(range.value) !== Math.round(pct)) range.value = String(Math.round(pct));
}

// Is the analyst's current before/after pair one the backend's own temporal
// matcher already checked? Reuses the SAME comparability verdict the date
// strip's "low comparability" flag comes from (geoseek.temporal.matcher) -
// never a new client-side heuristic. No verdict exists for the pair (e.g. an
// arbitrary non-adjacent, non-span before/after combination) -> not blocked;
// the images are still pixel-aligned by construction (same tile geometry for
// every date of a given candidate, see geoseek.analyst.imagery), so absence
// of a negative signal is treated as fine, only an explicit comparable:false
// disables the wipe.
function pairComparability(beforeIso, afterIso) {
  const intervals = ((detailState.data.temporal_trajectory || {}).intervals) || [];
  for (const iv of intervals) {
    const w = iv.window || [];
    if (w.length === 2 && ((w[0] === beforeIso && w[1] === afterIso) || (w[0] === afterIso && w[1] === beforeIso))) {
      if (iv.comparable === false) {
        return { ok: false, reason: `Flagged by the ${iv.kind === "span" ? "full-span" : "consecutive"} comparability check.` };
      }
    }
  }
  return { ok: true };
}

async function openDetail(id) {
  ensureDetailUI();
  go("detail/" + id); // keep hash canonical
  $("#d_empty").classList.add("hidden"); $("#d_body").classList.remove("hidden");
  $("#d_summary_body").innerHTML = `Loading ${esc(id)}…`;
  try {
    const d = await api("/candidates/" + encodeURIComponent(id));
    const before = d.imagery.before_dates || d.imagery.dates || ["2019"];
    detailState = {
      id, view: "slider", date: before[before.length - 1], after: d.imagery.after_date || "2024",
      outline: true, data: d, deciding: false, sliderPct: 50,
      fullDates: fullDatesFromTrajectory(d.temporal_trajectory),
      metaByYear: obsMetaByYear(d.provenance),
    };
    renderDetail(d);
  } catch (e) { renderApiError("#d_summary_body", e, () => openDetail(id)); }
}

// Full ISO acquisition dates for all 5 archive dates, derived from the
// interval window boundaries /candidates/{id} already returns (every
// consecutive interval's [start,end] collectively covers every date in the
// stack) - no new endpoint needed for this part.
function fullDatesFromTrajectory(traj) {
  const all = new Set();
  ((traj && traj.intervals) || []).forEach(iv => (iv.window || []).forEach(x => all.add(x)));
  const byYear = {};
  [...all].sort().forEach(iso => { byYear[iso.slice(0, 4)] = iso; });
  return byYear;
}
// Sensor/platform/cloud metadata is ONLY available for this candidate's own
// detection pair (provenance.observations, exactly 2 entries: role
// before/after) - NOT for the other before-years the date strip also lets an
// analyst preview. Named explicitly in the report rather than faked here.
function obsMetaByYear(prov) {
  const map = {};
  ((prov && prov.observations) || []).forEach(o => {
    const year = (o.acquired_at || "").slice(0, 4);
    const rt = o.representative_tile || {};
    map[year] = {
      sensor: (o.collection || {}).sensor, platform: (o.scene || {}).platform,
      cloud: rt.cloud_fraction, acquired_at: o.acquired_at,
    };
  });
  return map;
}
function currentDisplayYear() { return detailState.view === "before" ? detailState.date : detailState.after; }
function truncateMiddle(s, n) {
  s = String(s || ""); if (s.length <= n) return s;
  const half = Math.floor((n - 1) / 2);
  return s.slice(0, half) + "…" + s.slice(-half);
}

function renderDetail(d) {
  const ct = d.change_type;
  const band = Components.confBand(d.confidence);
  const nLater = laterObsCount(d.temporal_trajectory);
  const persHuman = persistenceHuman(d.persistence, nLater);
  const verdict = d.current_decision ? d.current_decision.decision : "undecided";
  renderVerdictChip(verdict);
  renderVerdictSentence(d, band);

  $("#d_summary_body").innerHTML = `
    <div class="kv"><span class="k">candidate</span><span class="v mono" style="font-size:var(--text-sm)">${esc(d.candidate_id)}</span></div>
    <div class="kv"><span class="k">type</span>${Components.changeBadge(ct, { label: CTYPE_HUMAN[ct] || ct })}</div>
    <div class="kv"><span class="k">confidence${Components.hint("How confident the system is that this is a real change - combines model probability, evidence strength and data quality.")}</span>
      <div class="conf-row">${Components.confRing(d.confidence)}${Components.bandPill(band)}</div>
      <p class="confidence-reason">${confidenceReasonSentence(d, band, nLater)}</p>
    </div>
    <div class="kv"><span class="k">significance${Components.hint("How large and how unusual this change is.")}</span><span class="v">${num(d.significance)}</span></div>
    <div class="kv"><span class="k">area</span><span class="v">${esc(humanArea(d.area_m2))}</span></div>
    <div class="kv"><span class="k">location</span><span class="v" style="font-size:var(--text-sm)">${esc(placeLabel(d.centroid_lonlat[0], d.centroid_lonlat[1]))}</span></div>
    <div class="kv"><span class="k">persistence${Components.hint("How consistently this change shows up across multiple satellite passes over time.")}</span><span class="plain">${esc(persHuman)}</span></div>
    <div class="kv"><span class="k">earliest change window</span><span class="v" style="font-size:var(--text-sm)">${esc((d.earliest_supported || []).join(" → ") || "–")}</span></div>`;

  renderDateStrip(d);
  setDetailSegment("slider", /*skipPaint*/ true);
  paintImages();
  // the red/yellow legend and the backend's own earliest-change caveat both
  // existed in the pre-2B-1 layout and must not silently disappear just
  // because the three side-by-side images became one switchable frame - kept
  // as a "?" hint next to the outline toggle (was an always-on two-line
  // paragraph; moving it to a hint gave the frame back that vertical space).
  const es = d.temporal_trajectory.earliest_supported_change || {};
  $("#d_imgnote_hint").title =
    `Footprint outline: red = this candidate's own component, yellow = other change in view (visible when the outline toggle is on, or the Overlay segment is selected). ${es.caveat || ""}`;

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
    <div class="ev"><div class="lbl">${esc(l)}</div><div class="val" style="font-family:var(--font-numeric);font-variant-numeric:tabular-nums">${esc(v)}</div><div class="sub">${esc(s)}</div></div>`).join("");

  renderEvidenceCheck(d.suppression);
  renderTraceStructured(d.suppression);
  renderProvenanceChain(d.provenance);

  // decision + history
  renderHistory(d.decisions || [], d.current_decision);
  $("#d_confirm").onclick = () => decide("confirm");
  $("#d_reject").onclick = () => decide("reject");

  // similar
  loadSimilar(d.candidate_id);
  fitImageryLayoutHeight("#d_body", 22);
}

// The one-sentence plain-English summary at the top of Candidate Detail -
// "New open water / flooding detected near X, first visible between Y and Z.
// Confidence: High." See the redesign brief's requirement 7.
function renderVerdictSentence(d, band) {
  const change = CTYPE_HUMAN[d.change_type] || d.change_type;
  const place = placeLabel(d.centroid_lonlat[0], d.centroid_lonlat[1]);
  const win = d.earliest_supported || [];
  let whenPhrase;
  if (win.length === 2) whenPhrase = `first visible between ${win[0]} and ${win[1]}`;
  else if (win.length === 1) whenPhrase = `first visible in ${win[0]}`;
  else whenPhrase = (persistenceHuman(d.persistence, laterObsCount(d.temporal_trajectory)) || "").toLowerCase();
  $("#d_verdict_sentence").innerHTML =
    `${esc(change)} detected near <b>${esc(place)}</b>, ${esc(whenPhrase)}. <b>Confidence: ${esc(band)}</b>.`;
}

function renderVerdictChip(verdict) {
  const map = { confirm: "success", reject: "danger", undecided: "" };
  const el = $("#d_verdict_chip");
  if (verdict === "undecided") {
    el.innerHTML = `<span class="c-band" data-band="Medium" data-kind="severity" style="opacity:.85">undecided</span>`;
  } else {
    el.innerHTML = `<span class="c-band" data-band="${verdict === "confirm" ? "High" : "Low"}">${esc(verdict)}</span>`;
  }
}

function renderDateStrip(d) {
  const years = [...(d.imagery.before_dates || d.imagery.dates || []), d.imagery.after_date];
  const flagged = new Set();
  ((d.temporal_trajectory && d.temporal_trajectory.intervals) || []).forEach(iv => {
    if (iv.kind === "consecutive" && iv.comparable === false) flagged.add((iv.window || [])[1]);
  });
  $("#d_datestrip").innerHTML = years.map(year => {
    const isAfter = year === d.imagery.after_date;
    const iso = detailState.fullDates[year] || year;
    const meta = detailState.metaByYear[year];
    const metaLine = meta
      ? `${esc(meta.sensor || "")}${meta.cloud != null ? " · " + Math.round(meta.cloud * 100) + "% cloud" : ""}`
      : "";
    const isFlagged = flagged.has(iso);
    const isCloudy = meta && meta.cloud != null && meta.cloud > 0.2; // UI threshold, not backend-defined - see report
    return `<div class="detail-datestrip__item${isAfter ? " is-after" : ""}" data-year="${esc(year)}" tabindex="0" role="button"
        title="${iso}${isAfter ? " (after)" : " (before)"}${isFlagged ? " — not directly comparable to its neighbour" : ""}">
      <span class="detail-datestrip__year">${esc(year)}</span>
      <span class="detail-datestrip__meta">${iso.slice(5)}${metaLine ? " · " + esc(metaLine) : ""}</span>
      ${isFlagged ? `<span class="detail-datestrip__flag" title="flagged not directly comparable to its neighbour">&#9888; low comparability</span>` : ""}
      ${isCloudy ? `<span class="detail-datestrip__flag" title="cloud fraction over 20%">&#9729; cloudy</span>` : ""}
    </div>`;
  }).join("");
  $$("#d_datestrip .detail-datestrip__item").forEach(el => {
    const activate = () => {
      const year = el.dataset.year;
      if (year === detailState.after) setDetailSegment(detailState.view === "before" ? "after" : detailState.view);
      // picking a new "before" date while already comparing in Slider mode
      // should update the wipe in place, not kick the analyst out to Before -
      // only Before/After/Overlay (single-image) modes still jump to Before.
      else { detailState.date = year; setDetailSegment(detailState.view === "slider" ? "slider" : "before"); }
    };
    el.addEventListener("click", activate);
    el.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); activate(); } });
  });
  markDateStripActive();
}
function markDateStripActive() {
  // Slider mode has two dates in play at once - highlight both endpoints,
  // not just one.
  const activeYears = detailState.view === "slider"
    ? new Set([detailState.date, detailState.after])
    : new Set([currentDisplayYear()]);
  $$("#d_datestrip .detail-datestrip__item").forEach(el =>
    el.classList.toggle("is-active", activeYears.has(el.dataset.year)));
}

function setDetailSegment(view, skipPaint) {
  detailState.view = view;
  $$("#d_segmented button").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.view === view)));
  $("#d_outline").disabled = view === "overlay";
  markDateStripActive();
  if (!skipPaint) paintImages();
}

// Progressive load: scale=1 first (small, arrives sooner, ALSO warms the
// backend's native-render LRU cache for this exact candidate/date/view - see
// report), painted immediately; scale=2 fetched right behind it and swapped
// in when ready. The frame's aspect-ratio is fixed by CSS (object-fit:
// contain in a flex-sized box) so the swap causes no layout shift either way.
function paintImages() {
  const isSlider = detailState.view === "slider";
  $("#d_frame_img").classList.toggle("hidden", isSlider);
  $("#d_slider").classList.toggle("hidden", !isSlider);
  $("#d_slider_rangewrap").classList.toggle("hidden", !isSlider);
  $("#d_frame_caption").classList.toggle("hidden", isSlider);
  $("#d_frame_note").classList.toggle("hidden", true); // slider has its own note element; re-shown below for single-image modes
  if (isSlider) { paintSlider(); return; }

  const token = ++_paintToken;
  const id = detailState.id;
  const view = detailState.view;
  const year = currentDisplayYear();
  const iso = detailState.fullDates[year] || year;
  const meta = detailState.metaByYear[year];
  // outline toggle reuses the server's own pixel-accurate overlay render for
  // the SAME date, rather than a client-drawn shape - see report for why
  const effectiveView = view === "overlay" ? "overlay" : (detailState.outline ? "overlay" : "rgb");

  const img = $("#d_frame_img");
  img.classList.add("c-skel");
  const urlFor = (scale) => `/candidates/${encodeURIComponent(id)}/imagery?date=${year}&view=${effectiveView}&scale=${scale}`;

  const cap = view === "before" ? "Before" : view === "after" ? "After" : "Change overlay";
  $("#d_frame_caption").textContent = `${cap} · ${iso}${meta && meta.sensor ? " · " + meta.sensor : ""}${meta && meta.cloud != null ? " · " + Math.round(meta.cloud * 100) + "% cloud" : ""}`;

  const noteEl = $("#d_frame_note");
  const flagged = ((detailState.data.temporal_trajectory || {}).intervals || [])
    .some(iv => iv.kind === "consecutive" && iv.comparable === false && (iv.window || [])[1] === iso);
  if (flagged) { noteEl.textContent = "Flagged: not directly comparable to its neighbouring observation."; noteEl.classList.remove("hidden"); }
  else if (meta && meta.cloud != null && meta.cloud > 0.2) { noteEl.textContent = `Heavily clouded (${Math.round(meta.cloud * 100)}%) — interpret with caution.`; noteEl.classList.remove("hidden"); }
  else { noteEl.classList.add("hidden"); }

  const preview = new Image();
  preview.onload = () => { if (token === _paintToken) { img.src = preview.src; } };
  preview.src = urlFor(1);

  const full = new Image();
  full.onload = () => {
    if (token !== _paintToken) return;
    img.src = full.src;
    img.classList.remove("c-skel");
  };
  full.onerror = () => { if (token === _paintToken) img.classList.remove("c-skel"); };
  full.src = urlFor(2);
}

// Slider mode: both dates render from the same candidate's fixed crop window
// (see geoseek.analyst.imagery - bbox_rc is shared across every date, so the
// two renders are pixel-aligned by construction), so this only fetches two
// images and wires up the wipe. The one thing that can still make the wipe
// misleading is a backend comparability verdict for this exact pair -
// checked before either image is even requested.
function paintSlider() {
  const token = ++_paintToken;
  const id = detailState.id;
  const beforeYear = detailState.date;
  const afterYear = detailState.after;
  const beforeIso = detailState.fullDates[beforeYear] || beforeYear;
  const afterIso = detailState.fullDates[afterYear] || afterYear;
  const effectiveView = detailState.outline ? "overlay" : "rgb";

  $("#d_slider_label_before").textContent = `Before · ${beforeIso}`;
  $("#d_slider_label_after").textContent = `After · ${afterIso}`;

  const slider = $("#d_slider");
  const noteEl = $("#d_slider_note");
  const sameDate = beforeYear === afterYear;
  const compat = sameDate ? { ok: false } : pairComparability(beforeIso, afterIso);
  const usable = compat.ok;
  slider.classList.toggle("is-disabled", !usable);
  $("#d_slider_range").disabled = !usable;
  noteEl.classList.toggle("hidden", usable);
  if (!usable) {
    noteEl.innerHTML = sameDate
      ? `<p class="c-empty__cause">Pick a different "before" date to compare against.</p>`
      : `<p class="c-empty__cause">These two dates aren't co-registered for a reliable wipe.</p><p class="c-empty__action">${esc(compat.reason || "")}</p>`;
    return;
  }

  setWipePct(detailState.sliderPct);

  const urlFor = (year, scale) => `/candidates/${encodeURIComponent(id)}/imagery?date=${year}&view=${effectiveView}&scale=${scale}`;
  [[$("#d_slider_before"), beforeYear], [$("#d_slider_after"), afterYear]].forEach(([imgEl, year]) => {
    imgEl.classList.add("c-skel");
    const preview = new Image();
    preview.onload = () => { if (token === _paintToken) imgEl.src = preview.src; };
    preview.src = urlFor(year, 1);
    const full = new Image();
    full.onload = () => { if (token === _paintToken) { imgEl.src = full.src; imgEl.classList.remove("c-skel"); } };
    full.onerror = () => { if (token === _paintToken) imgEl.classList.remove("c-skel"); };
    full.src = urlFor(year, 2);
  });
}

// threshold vs. actual value, split from the gate's free-text detail string
// where it cleanly parenthesises ("... (<= 5%)") - falls back to showing the
// whole sentence when a gate's wording doesn't split cleanly (phenology's
// multi-axis rule), rather than mis-parsing it into a wrong pair.
function splitTraceDetail(detail) {
  const m = /^(.*?)\s*\(([<>=][^)]*)\)\s*\.?\s*$/.exec((detail || "").split(";")[0].trim());
  return m ? { actual: m[1].trim(), threshold: m[2].trim() } : null;
}
// Plain-language restatement of suppression.trace's gate names ONLY - every
// row here comes 1:1 from a real trace entry, never a rule this map doesn't
// recognise (falls back to the raw rule id rather than inventing a label, so
// a future gate shows up honestly instead of silently vanishing).
const GATE_LABELS = {
  quality: "Image quality",
  registration: "Alignment",
  radiometric: "Brightness normalization",
  phenology: "Seasonal check",
  morphology: "Meaningful size",
};

// suppress.py's real vocabulary is three-valued (pass / downweight / suppress)
// - collapsing "downweight" into a hard fail would misreport a gate that only
// partially reduced confidence as one that rejected the candidate outright.
const GATE_VERDICT_STYLE = {
  pass: { icon: "check", word: "passed" },
  downweight: { icon: "warning", word: "flagged" },
  suppress: { icon: "close", word: "failed" },
};

// Step 4C "Evidence check": the SAME gates as the technical trace below,
// restated as a plain pass/fail list for a non-ML analyst. Never adds a
// check that isn't in sup.trace (no persistence/radar rows here - those
// already live in the Summary panel and the confidence-reason sentence).
function renderEvidenceCheck(sup) {
  const trace = sup.trace || [];
  $("#d_evcheck").innerHTML = trace.map(t => {
    const style = GATE_VERDICT_STYLE[t.verdict] || GATE_VERDICT_STYLE.suppress;
    const label = GATE_LABELS[t.rule] || t.rule;
    return `<div class="evidence-check__row" data-verdict="${esc(t.verdict)}">
      ${Components.icon(style.icon, { size: 20, cls: "evidence-check__icon" })}
      <span class="evidence-check__label">${esc(label)}</span>
      <span class="evidence-check__verdict">${style.word}</span>
    </div>`;
  }).join("") + (sup.suppressed
    ? `<p class="small muted" style="margin:8px 0 0">This candidate was down-weighted by ${esc(sup.suppressed_by || "a gate")} but still reached the queue.</p>`
    : "");
}

function renderTraceStructured(sup) {
  const trace = sup.trace || [];
  $("#d_trace").innerHTML = trace.map(t => {
    const pass = t.verdict === "pass";
    const split = splitTraceDetail(t.detail);
    return `<div class="trace-row">
      <span class="trace-row__gate">${esc(t.rule)}</span>
      <span class="trace-row__verdict" data-pass="${pass}">${esc(t.verdict)}</span>
      ${split
        ? `<div class="trace-row__values"><span>actual <b>${esc(split.actual)}</b></span><span>threshold <b>${esc(split.threshold)}</b></span>${t.weight !== 1 ? `<span>weight <b>×${num(t.weight)}</b></span>` : ""}</div>`
        : `<div class="trace-row__detail">${esc(t.detail)}${t.weight !== 1 ? ` · weight ×${num(t.weight)}` : ""}</div>`}
    </div>`;
  }).join("");
  $("#d_trace").insertAdjacentHTML("beforeend",
    `<div class="trace-summary">combined down-weight ×${num(sup.combined_downweight)} · ${sup.suppressed ? "SUPPRESSED by " + esc(sup.suppressed_by) : "entered the queue"}</div>`);
}

// Rendered as a visible chain (observation -> scene -> collection, then
// model/pipeline as the final link) rather than a flat key-value dump.
function renderProvenanceChain(prov) {
  const nodes = prov.observations.map(o => {
    const sc = o.scene || {}, co = o.collection || {}, rt = o.representative_tile || {};
    const checksum = (rt.checksums && (rt.checksums.B04 || Object.values(rt.checksums)[0])) || "";
    return `<div class="prov-chain__node">
      <div class="prov-chain__role">${esc(o.role)} observation</div>
      <div class="prov-chain__row"><dt>observation</dt><dd>${esc(o.observation_id)} · ${esc(o.acquired_at)}</dd></div>
      <div class="prov-chain__row"><dt>scene</dt><dd class="mono hash-reveal" title="${esc(sc.scene_id || "")}">${esc(truncateMiddle(sc.scene_id || "–", 22))}</dd></div>
      <div class="prov-chain__row"><dt>platform</dt><dd>${esc(sc.platform)} · ${esc(co.sensor)} · ${esc(co.native_gsd_m)} m</dd></div>
      <div class="prov-chain__row"><dt>source</dt><dd class="mono hash-reveal small" title="${esc(sc.source_url || "")}">${esc(truncateMiddle(sc.source_url || "–", 34))}</dd></div>
      <div class="prov-chain__row"><dt>licence</dt><dd>${esc(sc.license)}</dd></div>
      <div class="prov-chain__row"><dt>CRS</dt><dd>${esc(sc.crs)}</dd></div>
      <div class="prov-chain__row"><dt>bands</dt><dd>${esc((co.bands || []).join(", "))}</dd></div>
      <div class="prov-chain__row"><dt>rep. tile</dt><dd class="mono hash-reveal" title="${esc(rt.tile_id || "")}">${esc(truncateMiddle(rt.tile_id || "–", 22))}</dd></div>
      <div class="prov-chain__row"><dt>checksum</dt><dd class="mono hash-reveal" title="${esc(checksum)}">${esc(truncateMiddle(checksum, 16))}</dd></div>
    </div>`;
  });
  const mp = prov.model, cd = prov.code;
  nodes.push(`<div class="prov-chain__node">
    <div class="prov-chain__role">model &amp; pipeline</div>
    <div class="prov-chain__row"><dt>model</dt><dd>${esc(mp.name)} · threshold ${esc(mp.threshold)}</dd></div>
    <div class="prov-chain__row"><dt>weights</dt><dd class="mono hash-reveal" title="${esc(mp.weights_sha256 || "")}">${esc(truncateMiddle(mp.weights_sha256 || "–", 20))}</dd></div>
    <div class="prov-chain__row"><dt>git commit</dt><dd class="mono">${esc(cd.git_commit)}</dd></div>
    <div class="prov-chain__row"><dt>pipeline</dt><dd>${esc(cd.pipeline_version)}</dd></div>
    <div class="prov-chain__row"><dt>prob raster</dt><dd class="mono hash-reveal small" title="${esc(prov.probability_raster || "")}">${esc(truncateMiddle((prov.probability_raster || "").split(/[\\/]/).pop(), 22))}</dd></div>
  </div>`);
  $("#d_prov").innerHTML = nodes.join(`<div class="prov-chain__arrow">&darr;</div>`);
  $("#d_provcode").innerHTML = "";
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
  if (detailState.deciding || !detailState.id) return;
  detailState.deciding = true;
  $("#d_confirm").disabled = true; $("#d_reject").disabled = true;
  const note = $("#d_note").value.trim();
  try {
    await api(`/candidates/${encodeURIComponent(detailState.id)}/decision`, {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ decision, note, analyst: "analyst" }),
    });
    $("#d_note").value = "";
    // permanent, append-only write - the feedback says so plainly rather than
    // reading like a routine UI toast
    toast(`${decision.toUpperCase()} written to the permanent audit record`);
    const d = await api("/candidates/" + encodeURIComponent(detailState.id));
    detailState.data = d;
    renderVerdictChip(d.current_decision ? d.current_decision.decision : "undecided");
    renderHistory(d.decisions || [], d.current_decision);
    if (queueInit) loadQueue();
  } catch (e) { toast("decision failed: " + e.message); }
  finally { detailState.deciding = false; $("#d_confirm").disabled = false; $("#d_reject").disabled = false; }
}
async function loadSimilar(id) {
  $("#d_similar").innerHTML = `Loading&hellip;`;
  try {
    const d = await api(`/candidates/${encodeURIComponent(id)}/similar?k=8`);
    $("#d_similar").innerHTML = (d.results || []).map(r => `
      <div class="card" data-tile="${esc(r.tile_id)}">
        ${thumbHTML(`/tile/${encodeURIComponent(r.tile_id)}/thumbnail`)}
        <div class="meta"><div class="row"><b>match ${num(r.score, 3)}</b><span>${esc(r.acq_date)}</span></div>
        <div class="row"><span>${esc(clusterLabel(r.cluster))}</span><span>${num(r.centroid_lonlat[1], 3)},${num(r.centroid_lonlat[0], 3)}</span></div></div>
      </div>`).join("") || Components.emptyState({ icon: "discovery", cause: "No neighbours found.", action: "Try a different seed tile." });
    $$("#d_similar .card").forEach(c => c.addEventListener("click", () => {
      $("#disc_seed").value = c.dataset.tile; go("discovery"); setTimeout(runDiscovery, 30);
    }));
  } catch (e) { renderApiError("#d_similar", e, () => loadSimilar(id)); }
}

/* ---------------------------------------------------------------- DISCOVERY */
let discInit = false;
function ensureDiscovery() {
  if (discInit) return; discInit = true;
  $("#disc_go").addEventListener("click", runDiscovery);
  api("/discovery/clusters").then(d => {
    if (!d.available) { $("#disc_clusters").textContent = "Visual grouping hasn't been run yet."; return; }
    const dl = d.display_labels || {};
    const concepts = Object.entries(d.cluster_concepts || {});
    DISC_CLUSTER_LABELS = {};
    concepts.forEach(([k, v]) => { DISC_CLUSTER_LABELS[k] = dl[k] || v[0][0]; });
    const names = concepts.map(([k]) => esc(DISC_CLUSTER_LABELS[k]));
    $("#disc_clusters").innerHTML =
      `<p style="margin:0 0 8px">We grouped every indexed location into <b>${d.n_clusters}</b> visual ` +
      `types &mdash; things like ${names.slice(0, 5).join(", ")}${names.length > 5 ? ", and more" : ""}. ` +
      `A seed's own type is shown next to its result below.</p>` +
      `<details class="disclosure inline"><summary>How this works</summary><div class="body small muted">` +
      `${d.n_tiles} indexed tiles were grouped by visual similarity (${esc((d.params || {}).algorithm)} on ` +
      `${esc((d.params || {}).metric)} distance); ${d.noise_count} didn't fit any group cleanly. ` +
      concepts.map(([k, v]) => `type ${esc(k)} (n=${d.sizes[k]}): <i>${esc(dl[k] || v[0][0])}</i>`).join(" &nbsp;·&nbsp; ") +
      `</div></details>`;
    $("#disc_map").src = "/discovery/cluster-map.png";
  }).catch(e => $("#disc_clusters").textContent = e.message);
}
async function runDiscovery() {
  const seed = $("#disc_seed").value.trim();
  const lon = $("#disc_lon").value, lat = $("#disc_lat").value, k = $("#disc_k").value || 12;
  $("#disc_results").innerHTML = `Loading&hellip;`;
  try {
    let d, path;
    if (/^\d{4}_\d{4}_\d+$/.test(seed)) path = `/candidates/${encodeURIComponent(seed)}/similar?k=${k}`;
    else if (seed) path = `/discovery/similar?tile_id=${encodeURIComponent(seed)}&k=${k}`;
    else if (lon && lat) path = `/discovery/similar?lon=${lon}&lat=${lat}&k=${k}`;
    else { toast("enter a seed tile / candidate id or a lon,lat"); return; }
    d = await api(path);
    $("#disc_meta").textContent = `seed ${d.seed_tile_id || seed} · this location's visual type: ${clusterLabel(d.seed_cluster)} · ${d.latency_ms} ms`;
    $("#disc_results").innerHTML = (d.results || []).map(r => `
      <div class="card" data-tile="${esc(r.tile_id)}">
        ${thumbHTML(`/tile/${encodeURIComponent(r.tile_id)}/thumbnail`)}
        <div class="meta">
          <div class="row"><b>match ${num(r.score, 3)}</b><span>${esc(r.acq_date)}</span></div>
          <div class="row"><span>${esc(clusterLabel(r.cluster))}</span><span>${num(r.centroid_lonlat[1], 3)}°N</span></div>
        </div></div>`).join("") || Components.emptyState({ icon: "discovery", cause: "No neighbours found.", action: "Try a different seed tile or a larger k." });
    $$("#disc_results .card").forEach(c => c.addEventListener("click", () => { $("#disc_seed").value = c.dataset.tile; runDiscovery(); }));
  } catch (e) { renderApiError("#disc_results", e, runDiscovery); }
}

/* ---------------------------------------------------------------- OBJECT DETECTION (Phase 8F)
   REAL stored oriented-box detections (scripts/detect_maxar.py's output, read
   through the catalog's DerivedProduct + detections.geojson) drawn on the
   REAL Maxar tile they were found on - nothing here re-runs the detector.
   The backend reprojects each stored EPSG:4326 footprint back to tile pixels;
   this file only draws already-pixel-space polygons on a canvas. */
// per-class colour resolved through Tokens (tokens.css's --detect-* set) -
// replaces the old hand-copied DET_CLASS_COLOR hex object (Phase 2B-1).
function detColor(cls) { const t = Tokens.detectClass(cls); return `rgb(${t.r},${t.g},${t.b})`; }
let detInit = false;
let dt_railCollapsed = false;   // persists for the session, same pattern as Candidate Detail's rail
const detState = {
  modelInfo: null, observations: [], tiles: [], tileData: null, img: null,
  obsId: null, selectedClasses: new Set(), visible: [], boxesOn: true,
};

// Generic trailing debounce - used only for the detection-count/legend text
// below, never for the boxes themselves (see the #dt_margin listener).
function debounce(fn, ms) {
  let t;
  return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}

function ensureDetect() {
  if (detInit) return; detInit = true;
  // 'input' (not 'change') so boxes redraw continuously while dragging, not
  // just on release - filtering detState.tileData.detections client-side is
  // a plain array filter over at most a few hundred items, well under a
  // frame budget, so the boxes themselves never need debouncing. The "N of M
  // stored detections shown" text is debounced on purpose: it is prose, not
  // a rendered box, and updating it on every 'input' tick during a fast drag
  // makes the number flicker faster than it's readable - the boxes still
  // track the pointer live either way.
  // Also re-measures the stage height on every tick, not just on load/resize
  // (see the .dt-marginchip comment in style.css - a chip whose text length
  // varies with slider position can change the toolbar's wrapped row count,
  // which moves the stage's top offset; skipping this re-fit left a stale
  // pixel height that pushed the canvas off the bottom of the viewport).
  $("#dt_margin").addEventListener("input", () => {
    updateDetMarginLabel(); drawDetectCanvas({ liveCount: false }); fitImageryLayoutHeight(".dt-stage", 22);
  });
  $("#dt_obs").addEventListener("change", (e) => loadDetectObservation(e.target.value));
  $("#dt_tile").addEventListener("change", (e) => loadDetectTile(e.target.value));
  $("#dt_boxes_toggle").addEventListener("change", (e) => { detState.boxesOn = e.target.checked; drawDetectCanvas(); });
  $("#dt_rail_toggle").addEventListener("click", () => {
    dt_railCollapsed = !dt_railCollapsed;
    $(".dt-stage").classList.toggle("rail-collapsed", dt_railCollapsed);
    $("#dt_rail_toggle").setAttribute("aria-expanded", String(!dt_railCollapsed));
  });
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
  if (!caveats || !caveats.length) { panel.classList.add("hidden"); fitImageryLayoutHeight(".dt-stage", 22); return; }
  panel.classList.remove("hidden");
  $("#dt_caveats").innerHTML = caveats.map(c => `<span>${esc(c)}</span>`).join("");
  fitImageryLayoutHeight(".dt-stage", 22);
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
    <button type="button" class="dt-classchip" data-cls="${esc(c)}" aria-pressed="true"
      style="--dt-chip-color:${detColor(c)}" title="Show/hide ${esc(c)} detections">
      <i class="dt-classchip__dot" style="background:${detColor(c)}"></i>${esc(c)}</button>`).join("");
  $$("#dt_classes .dt-classchip").forEach(btn => btn.addEventListener("click", () => {
    const on = btn.getAttribute("aria-pressed") !== "true";
    btn.setAttribute("aria-pressed", String(on));
    if (on) detState.selectedClasses.add(btn.dataset.cls); else detState.selectedClasses.delete(btn.dataset.cls);
    drawDetectCanvas();
  }));
  $("#dt_classes_all").onclick = () => setDetClassFilter(classes, true);
  $("#dt_classes_none").onclick = () => setDetClassFilter(classes, false);
}
function setDetClassFilter(classes, on) {
  detState.selectedClasses = new Set(on ? classes : []);
  $$("#dt_classes .dt-classchip").forEach(btn => btn.setAttribute("aria-pressed", String(on)));
  drawDetectCanvas();
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
    `<span><i style="display:inline-block;width:9px;height:9px;border-radius:2px;background:${detColor(k)};margin-right:4px"></i><b>${v}</b> ${esc(k)}</span>`
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
  $(".dt-canvaswrap").classList.add("c-skel");
  const data = await api(`/detect/observations/${encodeURIComponent(obsId)}/tiles/${row}/${col}`);
  if (obsId !== detState.obsId) return;      // observation changed while this was in flight
  detState.tileData = data;
  renderDetCounts(data);
  const img = new Image();
  img.onload = () => {
    $(".dt-canvaswrap").classList.remove("c-skel");
    if (detState.tileData === data) { detState.img = img; drawDetectCanvas(); }
  };
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
function renderDetLegend(totalCount) {
  $("#dt_legend").innerHTML = Array.from(detState.selectedClasses).map(cl =>
    `<span><i style="background:${detColor(cl)}"></i>${esc(cl)}</span>`).join("") +
    `<span class="muted">${detState.boxesOn ? detState.visible.length : 0} of ${totalCount} stored detections shown${detState.boxesOn ? "" : " (boxes hidden)"}</span>`;
}
const renderDetLegendDebounced = debounce(renderDetLegend, 150);

// liveCount=false (only passed from the #dt_margin drag handler) draws the
// boxes on this exact frame but defers the "N of M" text - see ensureDetect.
function drawDetectCanvas({ liveCount = true } = {}) {
  const c = $("#dt_canvas"), ctx = c.getContext("2d");
  ctx.clearRect(0, 0, c.width, c.height);
  if (detState.img) ctx.drawImage(detState.img, 0, 0, c.width, c.height);
  const dets = (detState.tileData && detState.tileData.detections) || [];
  detState.visible = dets.filter(d => detState.selectedClasses.has(d.class) && d.score >= detEffectiveThreshold(d.class));
  // boxes are toggleable (same reasoning as the candidate footprint outline:
  // an analyst needs the raw tile unobstructed to make a trust decision) -
  // the visible-count text still reflects the confidence filter either way,
  // since that's a separate control from "are boxes drawn at all".
  if (detState.boxesOn) {
    for (const d of detState.visible) {
      ctx.strokeStyle = detColor(d.class);
      ctx.lineWidth = 2;
      ctx.beginPath();
      d.polygon_px.forEach(([x, y], i) => i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y));
      ctx.closePath(); ctx.stroke();
    }
  }
  if (liveCount) renderDetLegend(dets.length); else renderDetLegendDebounced(dets.length);
}
function renderDetCounts(data) {
  const counts = {};
  for (const d of (data ? data.detections : [])) counts[d.class] = (counts[d.class] || 0) + 1;
  const op = (detState.modelInfo && detState.modelInfo.operating_points) || {};
  const rows = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  $("#dt_counts tbody").innerHTML = rows.map(([k, v]) =>
    `<tr><td><i style="display:inline-block;width:9px;height:9px;border-radius:2px;background:${detColor(k)};margin-right:5px"></i>${esc(k)}</td>` +
    `<td data-numeric>${v}</td><td data-numeric class="small muted">${op[k] ? num(op[k].conf, 3) : "–"}</td></tr>`
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
let REGION_STATS = null;   // {name: {candidates, observations}}, fetched once from the real /candidates API
function bboxRing([w, s, e, n]) { return [[w, s], [e, s], [e, n], [w, n]]; }
function padBbox([w, s, e, n], frac = 0.15) {
  const px = (e - w) * frac + 1e-3, py = (n - s) * frac + 1e-3;
  return [w - px, s - py, e + px, n + py];
}
// The Maxar disaster-response AOIs (one-off staged events, e.g. the LA wildfires
// quadkey) sit far outside the India study area. Folding them into the map's
// default/auto-fit view is what made it snap to a whole-world extent - and made
// a plain click-drag land somewhere in the Atlantic. They stay selectable as
// their own chips; they just don't drive the default camera.
function isEventAoi(name) { return name.startsWith("maxar_"); }
function coreRegionsBbox() {
  const all = REGIONS || [];
  const core = all.filter(r => !isEventAoi(r.name));
  const list = core.length ? core : all;
  let w = 180, s = 90, e = -180, n = -90;
  for (const r of list) {
    const [rw, rs, re, rn] = r.bbox;
    if (rw < w) w = rw; if (re > e) e = re; if (rs < s) s = rs; if (rn > n) n = rn;
  }
  if (!(e > w)) return AOI_FALLBACK;
  return padBbox([w, s, e, n], 0.1);
}
function wRegionItems() {
  return (REGIONS || []).map((r, i) => ({
    id: "region:" + r.name, ring: bboxRing(r.bbox),
    color: REGION_PALETTE[i % REGION_PALETTE.length], regionName: r.name,
    label: `${r.name} · ${r.n_observations} obs`,
  }));
}
async function loadRegionStats() {
  if (REGION_STATS) return REGION_STATS;
  const regions = await loadRegions();
  REGION_STATS = {};
  await Promise.all(regions.map(async (r) => {
    try {
      const d = await api(`/candidates?bbox=${bboxStr(r.bbox)}&limit=1`);
      REGION_STATS[r.name] = { candidates: d.total || 0, observations: r.n_observations };
    } catch (e) { REGION_STATS[r.name] = { candidates: null, observations: r.n_observations }; }
    renderRegionChips();   // paint each stat in as it arrives instead of blocking on all 12
  }));
  return REGION_STATS;
}
const REGION_WORD_FIX = { losangeles: "Los Angeles" };
function friendlyRegionName(raw) {
  // per-AOI mosaic tile names (mosaic_index.json, from Observation.aoi_name)
  // use hyphens inside the event name ("maxar_india-floods-oct-2023_q...");
  // the coarser /regions grouping uses underscores throughout - normalize so
  // this one parser handles both without garbling the hyphenated form into
  // one long unsplit word.
  raw = raw.replace(/-/g, "_");
  if (!isEventAoi(raw)) return raw.replace(/_/g, " ");
  let s = raw.slice("maxar_".length).replace(/_q\d+/i, "");   // drop the quadkey token
  const parts = s.split("_");
  let place = null;
  if (parts.length > 1 && /^[a-z]+$/i.test(parts[parts.length - 1]) && parts[parts.length - 1].length > 3) {
    place = parts.pop();
  }
  const fix = (w) => REGION_WORD_FIX[w.toLowerCase()] || w;
  const label = parts.map(fix).join(" ");
  return place ? `${label} — ${fix(place)}` : label;
}
function renderRegionChips() {
  const el = $("#w_regionchips");
  if (!el) return;
  const regions = REGIONS || [];
  const selected = $("#w_region").value;
  el.innerHTML = regions.map((r, i) => {
    const color = REGION_PALETTE[i % REGION_PALETTE.length];
    const stat = REGION_STATS && REGION_STATS[r.name];
    const sub = stat
      ? (stat.candidates ? `${nf(stat.candidates)} candidates · ${stat.observations} obs` : `${stat.observations} observation${stat.observations === 1 ? "" : "s"}`)
      : `${r.n_observations} observation${r.n_observations === 1 ? "" : "s"}`;
    const event = isEventAoi(r.name);
    return `<button type="button" class="regioncard${r.name === selected ? " sel" : ""}${event ? " event" : ""}" style="--c:${color}" data-name="${esc(r.name)}">
      ${event ? '<em>Maxar event</em>' : ""}
      <b>${esc(friendlyRegionName(r.name))}</b><span>${esc(sub)}</span>
    </button>`;
  }).join("") || `<span class="muted small">no staged regions</span>`;
  $$("#w_regionchips .regioncard").forEach(b => b.addEventListener("click", () => pickRegion(b.dataset.name)));
}
function pickRegion(name) {
  const r = regionByName(name);
  if (!r) return;
  $("#w_region").value = name;
  $("#w_bbox").value = bboxStr(r.bbox);
  refreshWatchMap();
  watchMap.flyTo(padBbox(r.bbox), true);
}
function refreshWatchMap() {
  if (!watchMap) return;
  const regionItems = wRegionItems();
  const items = regionItems.slice();
  const bbox = parseBboxInput($("#w_bbox").value);
  if (bbox) items.push({ id: "__selection__", ring: bboxRing(bbox), color: "#e9edf5", label: "selected AOI" });
  watchMap.setItems(items);   // never re-fit the camera here - see coreRegionsBbox()
  watchMap.select(bbox ? "__selection__" : null);
  renderRegionChips();
}

function ensureWatch() {
  if (watchInit) return; watchInit = true;
  $("#w_types").innerHTML = TYPES.map(t =>
    `<label style="width:auto"><input type="checkbox" value="${t}" style="width:auto;margin-right:4px">${esc(CTYPE_HUMAN[t] || t)}</label>`).join("");
  $("#w_save").addEventListener("click", saveWatchArea);
  $("#w_cancel").addEventListener("click", resetWatchForm);
  $("#w_clearaoi").addEventListener("click", () => {
    $("#w_region").value = ""; $("#w_bbox").value = "";
    refreshWatchMap();
    watchMap.flyTo(coreRegionsBbox(), true);
  });
  watchMap = new CoordMap($("#w_map"), (it) => {
    if (it.regionName) pickRegion(it.regionName);
  }, {
    dataLabel: "Regions",
    onDrawRect: (bbox) => { $("#w_region").value = ""; $("#w_bbox").value = bboxStr(bbox); refreshWatchMap(); },
  });
  loadRegions().then(() => {
    refreshWatchMap();
    // fit India's staged regions, not the Maxar outliers, and remember this as
    // "home" so the reset-view control returns here rather than to the
    // pre-regions AOI_FALLBACK the map is constructed with
    watchMap.homeBbox = coreRegionsBbox();
    watchMap.flyTo(watchMap.homeBbox, false);
  });
  loadRegionStats();
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
  if (w.bbox) watchMap.flyTo(padBbox(w.bbox), true);
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
      <td><button class="c-btn c-btn--ghost" data-edit="${esc(w.watch_id)}">Edit</button>
          <button class="c-btn c-btn--danger" data-del="${esc(w.watch_id)}">Delete</button></td>
    </tr>`).join("") || `<tr><td colspan="7">${Components.emptyStates.noWatchAreas()}</td></tr>`;
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
        ${Components.bandPill(n.severity || "low", { kind: "severity", title: `max(confidence × significance) = ${num(n.severity_score)}` })}
        <span class="small muted">${esc((n.created_at || "").replace("T", " ").slice(0, 19))}</span></div>
      <div class="small muted">${n.candidates.length} new matching candidate(s) from observation ${esc(n.observation_id)}${n.observation_date ? " · " + esc(n.observation_date) : ""}</div>
      <div class="stack" style="flex-direction:row;flex-wrap:wrap;gap:6px;margin-top:6px">
        ${n.candidates.slice(0, 20).map(c => `<a href="#/detail/${encodeURIComponent(c.candidate_id)}" class="chip"
             style="text-decoration:none">${esc(c.candidate_id)} · ${esc(c.change_type || "?")}</a>`).join("")}
        ${n.candidates.length > 20 ? `<span class="small muted" style="align-self:center">+${n.candidates.length - 20} more — see the Review Queue</span>` : ""}
      </div>
      ${n.seen ? "" : `<button class="c-btn c-btn--ghost" style="margin-top:6px" data-seen="${esc(n.notification_id)}">Mark seen</button>`}
    </div>`).join("") || Components.emptyStates.noNotifications();
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
      // whatever state a presenter left the disclosure panels in earlier
      Components.setPanelCollapsed("d_why", true);
      Components.setPanelCollapsed("d_provwrap", true);
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

/* ---------------------------------------------------------------- FIRST-RUN WALKTHROUGH
   A separate, much simpler overlay from the Guided Tour above - four plain
   steps aimed at someone who has never opened this product before, not a
   presenter's demo script. Shown automatically once (a localStorage flag),
   skippable, re-openable any time from the "Help" link. Points at the real
   running app - the same real API calls the rest of the app makes, nothing
   scripted or faked - never at a screenshot. */
const ONBOARD_SEEN_KEY = "geoseek_onboarding_seen_v1";
const ONBOARD = { active: false, i: 0 };
const ONBOARD_STEPS = [
  {
    title: "Your change queue",
    body: "Every detected change lands here, ranked by priority — the most important ones first. This is usually where you start.",
    spotlight: "#q_table",
    run: async () => { go("queue"); ensureQueue(); await waitFor(() => $$("#q_table tbody tr[data-id]").length > 0, 6000); },
  },
  {
    title: "Open one to see the evidence",
    body: "Click any row. You'll get the before/after imagery, a confidence score, and the reasoning behind the flag.",
    spotlight: "#q_table tbody tr:first-child",
  },
  {
    title: "Confirm or reject it",
    body: "Once you're satisfied a change is real (or isn't), record your decision here — it's kept in a permanent, append-only record.",
    spotlight: "#d_decision",
    run: async () => {
      const tr = $("#q_table tbody tr[data-id]");
      if (tr) { await openDetail(tr.dataset.id); await waitFor(() => !$("#d_body").classList.contains("hidden"), 6000); }
    },
  },
  {
    title: "Or just ask",
    body: "Describe what you're looking for in plain English — e.g. “a new bridge over a river” — and get matching locations.",
    spotlight: "#q",
    run: async () => { go("search"); ensureSearch(); },
  },
];
function onboardSpotlight(sel) {
  $$(".is-onboard-spot").forEach(el => el.classList.remove("is-onboard-spot"));
  if (!sel) return;
  const el = $(sel);
  if (el) el.classList.add("is-onboard-spot");
}
async function onboardGo(i) {
  ONBOARD.i = Math.max(0, Math.min(ONBOARD_STEPS.length - 1, i));
  const s = ONBOARD_STEPS[ONBOARD.i];
  $("#onboard-dots").innerHTML = ONBOARD_STEPS.map((_, k) => `<i class="${k <= ONBOARD.i ? "on" : ""}"></i>`).join("");
  $("#onboard-title").textContent = `${ONBOARD.i + 1}. ${s.title}`;
  $("#onboard-body").textContent = s.body;
  $("#onboard-next").textContent = ONBOARD.i === ONBOARD_STEPS.length - 1 ? "Got it" : "Next →";
  $("#onboard-back").disabled = ONBOARD.i === 0;
  try { if (s.run) await s.run(); } catch (e) { /* never let a step kill the walkthrough */ }
  onboardSpotlight(s.spotlight);
}
function onboardStart() {
  ONBOARD.active = true;
  $("#onboard").hidden = false;
  onboardGo(0);
}
function onboardExit(markSeen) {
  ONBOARD.active = false;
  $("#onboard").hidden = true;
  onboardSpotlight(null);
  if (markSeen) { try { localStorage.setItem(ONBOARD_SEEN_KEY, "1"); } catch (e) { /* private mode etc. - just won't remember */ } }
}

/* ---------------------------------------------------------------- globe
   Lazy-imports globe.js (and, through it, the ~685KB vendor/three.module.min.js
   + OrbitControls.js staged offline by scripts/stage_threejs.py) the first
   time the Overview view mounts - never on any other view - so it never
   delays the ordinary map render path. The 2D CoordMap is what Overview
   shows in the meantime; once the globe is ready it becomes the default view
   (unless the analyst already switched to 2D themselves), per "primary way
   to pick a location" - the 2D map stays one click away for scanning. */
let globeInstance = null, globeLoading = false, ovViewMode = "2d", ovUserPickedView = false;
function ensureGlobe() {
  if (globeInstance || globeLoading) return;
  globeLoading = true;
  const status = $("#ov_globe_status");
  status.textContent = "Loading interactive globe…";
  status.classList.remove("hidden");
  import("./globe.js").then(async (mod) => {
    const g = new mod.Globe($("#ov_globe"), {
      friendlyName: friendlyRegionName,
      onOpenRegion: (bbox, name) => openQueueForBbox(bbox, friendlyRegionName(name)),
      onOpenCandidate: (id) => go("detail/" + id),
    });
    await g.ready;
    globeInstance = g;
    status.classList.add("hidden");
    // Step 4A: open already framed on the AOI, not the whole Earth - the
    // same AOI_FALLBACK bbox every other Overview/Search/Queue map already
    // treats as "the" AOI, so this introduces no new notion of location.
    g.flyToBbox(AOI_FALLBACK, 0);
    if (!ovUserPickedView) setOverviewView("globe");
    else if (ovViewMode === "globe") g.setActive(true);
    new ResizeObserver(() => globeInstance && globeInstance.resize()).observe($("#ov_globe"));
  }).catch(() => { status.textContent = "Globe unavailable on this device — showing the 2D map."; })
    .finally(() => { globeLoading = false; });
}
function setOverviewView(mode, fromUser) {
  ovViewMode = mode;
  if (fromUser) ovUserPickedView = true;
  $("#ov_globewrap").classList.toggle("hidden", mode !== "globe");
  $("#ov_mapwrap").classList.toggle("hidden", mode !== "2d");
  $$(".ov-viewtoggle button").forEach(b => b.classList.toggle("active", b.dataset.v === mode));
  if (globeInstance) {
    globeInstance.setActive(mode === "globe");
    if (mode === "globe") globeInstance.resize();
  }
}
function openQueueForBbox(bbox, label) {
  go("queue"); ensureQueue();
  $("#f_region").value = ""; $("#f_bbox").value = bboxStr(bbox);
  loadQueue();
  if (label) toast(`queue filtered to ${label}`);
}

/* ---------------------------------------------------------------- DEV GALLERY
   Phase 2A: every shared component in every state, in one screenshot. Not in
   <nav>, reachable only at #/dev-gallery. Renders once, no data fetching. */
let devGalleryInit = false;
function ensureDevGallery() {
  if (devGalleryInit) return; devGalleryInit = true;
  const C = window.Components;
  const row = (label, html) => `<div class="c-gallery__cell"><span class="c-gallery__cell-label">${esc(label)}</span>${html}</div>`;

  const badges = TYPES.map((t) => row(t, C.changeBadge(t))).join("");

  const confRows = [0.97, 0.72, 0.41].map((v) =>
    row(`${Math.round(v * 100)}%`, C.confRing(v))).join("") +
    row("large", C.confRing(0.9, { size: "lg" })) +
    row("bar (detail-panel variant)", `<div style="width:220px">${C.confBar(0.72)}</div>`);

  const galleryTableRows = [
    { id: "a", cls: "", type: "water_gain", conf: 0.94, area: 196000, mono: "2019_2026_004510" },
    { id: "b", cls: "is-demo-hover", type: "construction", conf: 0.81, area: 42000, mono: "2019_2026_005124" },
    { id: "c", cls: "is-demo-focus", type: "clearance", conf: 0.63, area: 15000, mono: "2019_2026_005218" },
  ];
  const tableHTML = `<div class="c-table-wrap"><table class="c-table">
    <thead><tr>
      <th class="sorted asc">rank (ascending) ${C.icon("chevron", { cls: "c-sort-icon" })}</th>
      <th>type</th><th data-numeric>confidence</th><th data-numeric>area m²</th><th>candidate</th>
    </tr></thead>
    <tbody>${galleryTableRows.map((r, i) => `
      <tr class="${r.cls}" ${r.cls === "is-demo-focus" ? 'aria-selected="true"' : ""} tabindex="0">
        <td data-numeric>${i + 1}</td><td>${C.changeBadge(r.type)}</td>
        <td data-numeric>${(r.conf * 100).toFixed(0)}%</td>
        <td data-numeric>${r.area.toLocaleString()}</td>
        <td data-mono>${r.mono}</td>
      </tr>`).join("")}</tbody>
  </table></div>`;

  const panelDefault = C.panel({ id: "gal_panel_1", iconName: "detail", title: "Panel — static", bodyHTML: "<p style=\"margin:0;color:var(--text-secondary);font-size:var(--text-sm)\">Body content sits here.</p>" });
  const panelOpen = C.panel({ id: "gal_panel_2", iconName: "watch", title: "Panel — collapsible, open", collapsible: true, bodyHTML: "<p style=\"margin:0;color:var(--text-secondary);font-size:var(--text-sm)\">Click the header to collapse. State is remembered in memory for the rest of this session.</p>", footerHTML: `<span class="muted small">footer slot</span>` });
  const panelClosed = C.panel({ id: "gal_panel_3", iconName: "watch", title: "Panel — collapsible, closed", collapsible: true, bodyHTML: "<p>never seen while collapsed</p>" });

  const buttonStates = (cls, label) => `
    <button class="c-btn ${cls}">${label}</button>
    <button class="c-btn ${cls} is-demo-hover">${label}</button>
    <button class="c-btn ${cls} is-demo-focus">${label}</button>
    <button class="c-btn ${cls}" disabled>${label}</button>`;
  const buttonsHTML = `
    <div class="c-gallery__row">${row("primary (default/hover/focus/disabled)", buttonStates("c-btn--primary", "Apply"))}</div>
    <div class="c-gallery__row">${row("secondary", buttonStates("c-btn--secondary", "Cancel"))}</div>
    <div class="c-gallery__row">${row("ghost", buttonStates("c-btn--ghost", "Export"))}</div>
    <div class="c-gallery__row">${row("danger", buttonStates("c-btn--danger", "Delete"))}</div>
    <div class="c-gallery__row">${row("decision (deliberate weight)", `
      <button class="c-btn c-btn--decide c-btn--confirm">${C.icon("check")} Confirm</button>
      <button class="c-btn c-btn--decide c-btn--reject">${C.icon("close")} Reject</button>`)}</div>
    <div class="c-gallery__row">${row("segmented control", `
      <div class="c-segmented">
        <button aria-pressed="true">Before</button><button class="is-demo-hover">After</button><button>Overlay</button>
      </div>`)}</div>
    <div class="c-gallery__row">
      ${row("toggle (off/on)", `<label class="c-toggle"><input type="checkbox"><span class="c-toggle__track"></span><span class="c-toggle__thumb"></span></label>
        <label class="c-toggle" style="margin-left:12px"><input type="checkbox" checked><span class="c-toggle__track"></span><span class="c-toggle__thumb"></span></label>`)}
      ${row("checkbox", `<label class="c-checkbox"><input type="checkbox" checked>label</label>`)}
    </div>
    <div class="c-gallery__row">
      ${row("select", `<select class="c-select"><option>any</option></select>`)}
      ${row("text input", `<input class="c-input" placeholder="placeholder">`)}
      ${row("range", `<input class="c-range" type="range" style="width:160px">`)}
    </div>`;

  const skeletonHTML = row("imagery placeholder", `<div class="c-skel" style="width:160px;height:120px;border-radius:var(--radius-panel)"></div>`);

  const emptyHTML = `<div class="c-gallery__row">
    ${row("no search results", `<div style="width:260px">${C.emptyStates.noSearchResults()}</div>`)}
    ${row("no candidates filtered", `<div style="width:260px">${C.emptyStates.noCandidatesFiltered()}</div>`)}
    ${row("no notifications", `<div style="width:260px">${C.emptyStates.noNotifications()}</div>`)}
    ${row("no watch areas", `<div style="width:260px">${C.emptyStates.noWatchAreas()}</div>`)}
  </div>`;

  const statHTML = `<div class="c-gallery__row" style="max-width:640px">
    ${C.statCard({ id: "gal_stat_1", label: "change candidates", value: 841, accent: true })}
    ${C.statCard({ id: "gal_stat_2", label: "regions", value: 12, delta: 2 })}
    ${C.statCard({ id: "gal_stat_3", label: "rejected", value: 4, delta: -1 })}
  </div>`;

  const latencyHTML = row("states (ok / over-budget / in-flight)", `
    <span class="c-latency" data-state="ok"><i class="c-latency__dot"></i>/candidates 82 ms</span>
    <span class="c-latency" data-state="over" style="margin-left:8px"><i class="c-latency__dot"></i>/discovery/similar 1425 ms</span>
    <span class="c-latency" data-state="inflight" style="margin-left:8px"><i class="c-latency__dot"></i>fetching…</span>`);

  const iconNames = ["overview", "search", "queue", "detail", "discovery", "detect", "watch", "play", "alerts", "layers", "featured", "check", "grid", "home", "warning", "chevron", "close"];
  const iconsHTML = `<div class="c-gallery__row">` + iconNames.map((n) =>
    row(n, `${C.icon(n)} ${C.icon(n, { size: 20 })} ${C.icon(n, { size: 24 })}`)).join("") + `</div>`;

  $("#dev_gallery_root").innerHTML = `<div class="c-gallery">
    <h2>1. Change-type badge</h2><div class="c-gallery__row">${badges}</div>
    <h2>2. Confidence display</h2><div class="c-gallery__row">${confRows}</div>
    <h2>3. Data table</h2>${tableHTML}
    <h2>4. Panel / card</h2><div class="c-gallery__row">${panelDefault}${panelOpen}${panelClosed}</div>
    <h2>5. Buttons and controls</h2>${buttonsHTML}
    <h2>6. Skeleton (imagery only)</h2><div class="c-gallery__row">${skeletonHTML}</div>
    <h2>7. Empty state</h2>${emptyHTML}
    <h2>8. Stat / counter</h2>${statHTML}
    <h2>9. Latency badge</h2><div class="c-gallery__row">${latencyHTML}</div>
    <h2>10. Icon set</h2>${iconsHTML}
  </div>`;

  // force the third panel closed for the screenshot (collapse memory starts
  // empty each fresh load; this just seeds one demo panel into the "closed"
  // state so both states are visible at once without a click)
  document.getElementById("gal_panel_3").classList.add("is-collapsed");
  $$("#dev_gallery_root .c-stat__value").forEach((el) => C.animateStatValue(el, +el.dataset.target, 900));
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

// Candidate Detail / Object Detection both want their imagery-dominant grid
// to fill "the rest of the viewport" below whatever chrome precedes it
// (header, filters, the caveat banner when visible). A fixed CSS px offset
// was tried first and was WRONG in practice - verified live, not assumed:
// Object Detection's filter rows wrap differently at narrower viewport
// widths, and the caveat banner's visibility is data-dependent, so a
// hardcoded constant either overflowed the viewport or left an inconsistent
// gap depending on what else was on screen. Measuring the element's own
// offsetTop and setting height from that is the robust fix - re-run on
// resize AND whenever content above the layout could have changed height.
function fitImageryLayoutHeight(selector, bottomMarginPx) {
  const el = $(selector);
  if (!el || el.closest(".view.active") == null) return;
  const top = el.getBoundingClientRect().top;
  el.style.height = Math.max(320, window.innerHeight - top - bottomMarginPx) + "px";
}
function fitAllImageryLayouts() {
  fitImageryLayoutHeight("#d_body", 22);
  fitImageryLayoutHeight(".dt-stage", 22);
}
window.addEventListener("resize", fitAllImageryLayouts);

async function boot() {
  Components.initPanelCollapse(document);
  try {
    const h = await api("/health");
    $("#sb_vectors").textContent = nf(h.vectors);
    $("#sb_candidates").textContent = nf(h.candidates);
  } catch (e) {
    $("#sidebar_status").classList.remove("ok");
    $("#sidebar_status .sidebar__status-text").textContent = "backend unreachable";
  }
  try {
    // build/commit line - the closest truthful analogue to a "host" line this
    // backend exposes; there is no GPU/host telemetry endpoint (see the phase
    // report's "cannot truthfully populate" list) so this names the running
    // code instead of inventing hardware info.
    const s = await api("/stats");
    const b = (s && s.build) || {};
    if (b.git_commit) $("#sb_build").textContent = `build ${b.git_commit.slice(0, 7)} · pipeline ${b.pipeline_version || "?"}`;
  } catch (e) { /* sidebar footer is a nicety */ }
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

  $("#helpBtn").addEventListener("click", () => onboardStart());
  $("#onboard-next").addEventListener("click", () => (ONBOARD.i >= ONBOARD_STEPS.length - 1 ? onboardExit(true) : onboardGo(ONBOARD.i + 1)));
  $("#onboard-back").addEventListener("click", () => onboardGo(ONBOARD.i - 1));
  $("#onboard-skip").addEventListener("click", () => onboardExit(true));
  document.addEventListener("keydown", e => { if (e.key === "Escape" && ONBOARD.active) onboardExit(true); });

  if (!location.hash) location.hash = "#/overview";
  router();

  // first-run only - a returning analyst (or one who skipped/finished before)
  // never sees this again, but can always bring it back via "Help".
  let seenOnboarding = true;
  try { seenOnboarding = localStorage.getItem(ONBOARD_SEEN_KEY) === "1"; } catch (e) { /* private mode etc. */ }
  if (!seenOnboarding) setTimeout(onboardStart, 700);
}
boot();
