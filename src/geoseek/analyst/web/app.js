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
  constructor(canvas, onPick) {
    this.c = canvas; this.ctx = canvas.getContext("2d");
    this.items = []; this.bbox = AOI_FALLBACK; this.onPick = onPick; this.sel = null;
    canvas.addEventListener("click", (e) => this._click(e));
    new ResizeObserver(() => this.draw()).observe(canvas);
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
    const [w, s, e, n] = this.bbox, step = 0.1;
    for (let lon = Math.ceil(w / step) * step; lon < e; lon += step) {
      const x = p.x(lon); ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, p.H); ctx.stroke();
      ctx.fillText(lon.toFixed(1) + "°E", x + 2, p.H - 3);
    }
    for (let lat = Math.ceil(s / step) * step; lat < n; lat += step) {
      const y = p.y(lat); ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(p.W, y); ctx.stroke();
      ctx.fillText(lat.toFixed(1) + "°N", 3, y - 3);
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
function pointInPoly(x, y, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if (((yi > y) !== (yj > y)) && (x < ((xj - xi) * (y - yi)) / (yj - yi) + xi)) inside = !inside;
  }
  return inside;
}
function legend(elId, types) {
  $(elId).innerHTML = types.map(t => `<span><i style="background:${CTYPE_COLOR[t] || "#888"}"></i>${t}</span>`).join("");
}

/* ---------------------------------------------------------------- routing */
const views = ["search", "queue", "detail", "discovery"];
function go(route) { location.hash = "#/" + route; }
function router() {
  const parts = (location.hash || "#/queue").slice(2).split("/");
  const route = views.includes(parts[0]) ? parts[0] : "queue";
  views.forEach(v => $("#view-" + v).classList.toggle("active", v === route));
  $$("nav button").forEach(b => b.classList.toggle("active", b.dataset.route === route));
  if (route === "search") ensureSearch();
  if (route === "queue") ensureQueue();
  if (route === "discovery") ensureDiscovery();
  if (route === "detail" && parts[1]) openDetail(decodeURIComponent(parts[1]));
}
window.addEventListener("hashchange", router);
$$("nav button").forEach(b => b.addEventListener("click", () => go(b.dataset.route)));

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
function ensureQueue() {
  if (queueInit) return; queueInit = true;
  queueMap = new CoordMap($("#q_map"), (it) => go("detail/" + it.id));
  legend("#q_legend", TYPES);
  $("#f_go").addEventListener("click", loadQueue);
  $("#f_export").addEventListener("click", exportFiltered);
  $$("#q_table th").forEach(th => th.addEventListener("click", () => {
    const k = th.dataset.k === "earliest" ? "rank" : th.dataset.k;
    queueSort = (queueSort === k) ? "-" + k : k;
    loadQueue();
  }));
  loadQueue();
}
function queueQS(extra) {
  const p = new URLSearchParams();
  if ($("#f_type").value) p.set("change_type", $("#f_type").value);
  if ($("#f_conf").value !== "") p.set("min_confidence", $("#f_conf").value);
  if ($("#f_pers").value) p.set("persistence", $("#f_pers").value);
  if ($("#f_sensor").value) p.set("sensor", $("#f_sensor").value);
  if ($("#f_dec").value) p.set("decision", $("#f_dec").value);
  if ($("#f_bbox").value.trim()) p.set("bbox", $("#f_bbox").value.trim());
  Object.entries(extra || {}).forEach(([k, v]) => p.set(k, v));
  return p;
}
async function loadQueue() {
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
  $("#d_head").innerHTML = `
    <div class="kv"><span class="k">candidate</span><span class="v mono">${esc(d.candidate_id)}</span></div>
    <div class="kv"><span class="k">type</span><span class="v"><span class="badge b-${ct}">${esc(ct)}</span></span></div>
    <div class="kv" style="min-width:160px"><span class="k">confidence ${num(d.confidence)}</span>
      <span class="v"><span class="meter"><i style="width:${Math.round((d.confidence || 0) * 100)}%"></i></span></span></div>
    <div class="kv"><span class="k">significance</span><span class="v">${num(d.significance)}</span></div>
    <div class="kv"><span class="k">area</span><span class="v">${Math.round(d.area_m2).toLocaleString()} m²</span></div>
    <div class="kv"><span class="k">location</span><span class="v">${num(d.centroid_lonlat[1], 4)}°N ${num(d.centroid_lonlat[0], 4)}°E</span></div>
    <div class="kv"><span class="k">persistence</span><span class="v">${esc(d.persistence)}</span></div>
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
      <img loading="lazy" src="/candidates/${encodeURIComponent(id)}/imagery?date=${date}&view=${view}" alt="${cap} ${date}"></figure>`).join("");
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
    $("#disc_clusters").innerHTML =
      `<b>${d.n_clusters}</b> clusters over ${d.n_tiles} tiles (${d.noise_count} noise) — ` +
      Object.entries(d.cluster_concepts || {}).map(([k, v]) =>
        `c${k} (n=${d.sizes[k]}): <i>${esc(v[0][0])}</i>`).join(" &nbsp;·&nbsp; ") +
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

/* ---------------------------------------------------------------- boot */
async function boot() {
  try {
    const h = await api("/health");
    $("#offlineChip").textContent = `offline · ${h.vectors} vectors · ${h.candidates} candidates`;
  } catch (e) { $("#offlineChip").textContent = "backend unreachable"; $("#offlineChip").classList.remove("ok"); }
  if (!location.hash) location.hash = "#/queue";
  router();
}
boot();
