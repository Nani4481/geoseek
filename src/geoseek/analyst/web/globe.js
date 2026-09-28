/* Interactive 3D globe - the primary way an analyst picks a location on
   Overview. Pure ES module, imported dynamically from app.js only once the
   Overview view actually mounts, so the ~685KB three.js + OrbitControls
   payload (src/geoseek/analyst/web/vendor/, staged offline by
   scripts/stage_threejs.py - licence/SHA256 in the provenance manifest)
   never delays any other view. No CDN, no network call at runtime beyond
   this same FastAPI origin.

   The base sphere is a real Earth, not a drawn one: NASA Blue Marble day /
   night-lights / ocean-specular textures (day+night both full 2048x1024,
   ~0.62MB total, staged offline by scripts/stage_earth_textures.py -
   source/licence/SHA256 in the provenance manifest), composited in a small
   custom shader that mixes day->night across a soft terminator driven by a
   fixed sun direction and adds an ocean-only specular highlight, plus a
   BackSide atmospheric rim-glow shell just above the surface. Country/state
   borders + an India outline (a highlighted stroke, not a filled tint - see
   buildBordersTexture's own comment for why) are a second, separate
   transparent texture (baked from basemap.json's own vectors, same as
   before) alpha-blended on top as a true overlay layer, not mixed into the
   photographic texture itself.

   One small curved patch per staged AOI (scripts/build_mosaic.py's own
   ~900px WebP tiles, used as-is, no re-encoding) drapes our real Sentinel-2 /
   Maxar imagery where we have it, geometrically clipped to that AOI's own
   lon/lat bbox via SphereGeometry's phi/theta start+length params and laid a
   hair above the base sphere - each tile's own alpha channel now feathers
   towards its edges (see build_mosaic.py) so it blends into the Blue Marble
   base instead of reading as a hard rectangle. */
import * as THREE from "three";
import { OrbitControls } from "./vendor/OrbitControls.js";

const DEG2RAD = Math.PI / 180;
const BASE_TEX_W = 2048, BASE_TEX_H = 1024;
const MIN_DIST = 1.12, MAX_DIST = 6.2, START_DIST = 3.4;
const POINTS_VISIBLE_DIST = 1.75;   // candidates only appear once zoomed in this far
const PATCH_RADIUS = 1.004;         // just above the base sphere - avoids z-fighting
const PIN_RADIUS = 1.02;
// idle auto-rotate: slow enough to read as "alive," never as a spin. speed
// 0.4 against OrbitControls' own "30s/orbit at speed 2.0, 60fps" baseline is
// one full revolution every ~150s - restrained on purpose. Paused the moment
// the analyst touches the globe (OrbitControls' own start/end events, which
// already fire for drag, wheel-zoom and touch alike) and only resumed after
// a real idle gap, not the instant a drag ends.
const AUTO_ROTATE_SPEED = 0.4;
const AUTO_ROTATE_IDLE_MS = 3000;
const REDUCE_MOTION = !!(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches);

function esc(s) { return String(s == null ? "" : s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }
function geomPolys(geom) {
  if (!geom) return [];
  if (geom.type === "Polygon") return [geom.coordinates];
  if (geom.type === "MultiPolygon") return geom.coordinates;
  if (geom.type === "LineString") return [[geom.coordinates]];
  if (geom.type === "MultiLineString") return geom.coordinates.map(r => [r]);
  return [];
}
function latLonToVec3(lat, lon, r) {
  const phi = (90 - lat) * DEG2RAD, theta = (lon + 180) * DEG2RAD;
  return new THREE.Vector3(-r * Math.sin(phi) * Math.cos(theta), r * Math.cos(phi), r * Math.sin(phi) * Math.sin(theta));
}
// Exact inverse of latLonToVec3 - used only for the cursor lat/lon readout
// (Step 4A), which needs to go from a raycast hit point back to lat/lon.
const EARTH_RADIUS_KM = 6371;
function vec3ToLatLon(v) {
  const r = v.length();
  const phi = Math.acos(Math.max(-1, Math.min(1, v.y / r)));
  const theta = Math.atan2(v.z, -v.x);
  let lon = theta / DEG2RAD - 180;
  if (lon < -180) lon += 360; else if (lon > 180) lon -= 360;
  return { lat: 90 - phi / DEG2RAD, lon };
}
// spherical linear interpolation between two unit directions - lerping the
// raw camera positions instead (as a first cut of this globe did) draws a
// straight chord between them, which dips *inside* the sphere for anything
// but a short hop; slerping the direction and lerping distance separately
// keeps the whole flight arcing just above the surface, which is what makes
// a globe->region->AOI dive read as one continuous swoop instead of a cut.
function slerpDir(a, b, t) {
  const dot = THREE.MathUtils.clamp(a.dot(b), -1, 1);
  const theta = Math.acos(dot) * t;
  if (theta < 1e-6) return a.clone();
  const rel = b.clone().addScaledVector(a, -dot).normalize();
  return a.clone().multiplyScalar(Math.cos(theta)).add(rel.multiplyScalar(Math.sin(theta)));
}
function bboxContains([w, s, e, n], lon, lat) { return lon >= w && lon <= e && lat >= s && lat <= n; }

// borders/coastline/India highlight - a transparent overlay texture (drawn on
// top of the real Blue Marble photography in the shader below), not part of
// the photographic base itself. India is called out with a brighter, thicker
// OUTLINE rather than a filled tint: an early version filled the interior
// with a flat translucent colour, which flattened every bit of underlying
// photographic/night-light detail into a solid cutout - it read as a pasted-
// on decal, not "a real textured Earth" (see docs/REDESIGN_PHASE2B2_REPORT.md).
// An outline calls out the same region without touching a single interior
// pixel of the actual Earth photography.
async function buildBordersTexture(basemap) {
  const W = BASE_TEX_W, H = BASE_TEX_H;
  const c = document.createElement("canvas"); c.width = W; c.height = H;
  const ctx = c.getContext("2d"); // transparent by default - no fillRect base
  const X = (lon) => (lon + 180) / 360 * W, Y = (lat) => (90 - lat) / 180 * H;

  function drawFC(fc, style) {
    if (!fc) return;
    for (const f of fc.features) {
      const highlighted = style.highlightName && f.properties && f.properties.name === style.highlightName;
      ctx.beginPath();
      for (const poly of geomPolys(f.geometry)) for (const ring of poly) {
        ring.forEach(([lon, lat], i) => { const x = X(lon), y = Y(lat); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
        if (style.close) ctx.closePath();
      }
      if (style.stroke) { ctx.strokeStyle = style.stroke; ctx.lineWidth = style.w || 1; ctx.stroke(); }
      // redrawn on top, thicker + brighter - never a fill, so the country's
      // own interior imagery (day photography or night lights) stays intact
      if (highlighted && style.highlightStroke) { ctx.strokeStyle = style.highlightStroke; ctx.lineWidth = style.highlightWidth || 2; ctx.stroke(); }
    }
  }
  if (basemap) {
    drawFC(basemap.countries, {
      stroke: "rgba(230,238,252,.5)", w: 1, close: true,
      highlightName: "India", highlightStroke: "rgba(150,205,110,.95)", highlightWidth: 2.2,
    });
    drawFC(basemap.states, { stroke: "rgba(210,222,245,.3)", w: 0.55, close: true });
    drawFC(basemap.coastline, { stroke: "rgba(240,246,255,.4)", w: 0.9 });
  }
  const tex = new THREE.CanvasTexture(c);
  tex.anisotropy = 4;
  return tex;
}

const EARTH_VERT_SHADER = `
varying vec2 vUv;
varying vec3 vNormalW;
void main() {
  vUv = uv;
  vNormalW = normalize(normalMatrix * normal);
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}`;
const EARTH_FRAG_SHADER = `
uniform sampler2D dayMap;
uniform sampler2D nightMap;
uniform sampler2D specMap;
uniform sampler2D bordersMap;
uniform vec3 sunDirection;
varying vec2 vUv;
varying vec3 vNormalW;
void main() {
  vec3 day = texture2D(dayMap, vUv).rgb;
  vec3 night = texture2D(nightMap, vUv).rgb;
  float spec = texture2D(specMap, vUv).r;
  float ndotl = dot(normalize(vNormalW), normalize(sunDirection));
  // soft terminator - a smooth day/night blend rather than a hard line
  float dayMix = smoothstep(-0.16, 0.16, ndotl);
  vec3 color = mix(night * 1.25, day, dayMix);
  // ocean-only specular glint, day side only (the source mask is bright over
  // water, ~0 over land, so this naturally excludes land automatically)
  float highlight = pow(clamp(ndotl, 0.0, 1.0), 22.0) * spec * dayMix;
  color += vec3(0.9, 0.95, 1.0) * highlight * 0.7;
  vec4 borders = texture2D(bordersMap, vUv);
  color = mix(color, borders.rgb, borders.a);
  gl_FragColor = vec4(color, 1.0);
}`;

// brighter than a flat radial gradient: a white hot core (reads against any
// background, day-lit land or dark ocean alike) inside the coloured glow.
// Takes a resolved {r,g,b} triple (from Tokens.triple/.changeType), not a
// hex string - the alpha-faded stops below are built directly from r/g/b
// instead of through a hex-parsing helper (see docs/FRONTEND_AUDIT.md
// Section 3/10 for why a hex-string-parsing helper was the wrong shape here).
function pinTexture(rgb) {
  const s = 160, cx = s / 2, cy = s / 2;
  const c = document.createElement("canvas"); c.width = s; c.height = s;
  const ctx = c.getContext("2d");
  const solid = `rgb(${rgb.r},${rgb.g},${rgb.b})`;
  const glow = ctx.createRadialGradient(cx, cy, 0, cx, cy, s / 2);
  glow.addColorStop(0, "#ffffff");
  glow.addColorStop(0.14, solid);
  glow.addColorStop(0.42, `rgba(${rgb.r},${rgb.g},${rgb.b},0.5)`);
  glow.addColorStop(1, `rgba(${rgb.r},${rgb.g},${rgb.b},0)`);
  ctx.fillStyle = glow; ctx.fillRect(0, 0, s, s);
  ctx.beginPath(); ctx.arc(cx, cy, s * 0.085, 0, Math.PI * 2);
  ctx.fillStyle = "#ffffff"; ctx.fill();
  return new THREE.CanvasTexture(c);
}
// a visible halo ring around the glow - Sprites always face the camera, so a
// plain stroked circle reads as a clean ring from any viewing angle
function ringTexture(rgb) {
  const s = 160;
  const c = document.createElement("canvas"); c.width = s; c.height = s;
  const ctx = c.getContext("2d");
  ctx.strokeStyle = `rgb(${rgb.r},${rgb.g},${rgb.b})`; ctx.globalAlpha = 0.8; ctx.lineWidth = s * 0.05;
  ctx.beginPath(); ctx.arc(s / 2, s / 2, s * 0.36, 0, Math.PI * 2); ctx.stroke();
  return new THREE.CanvasTexture(c);
}
// a small rounded name-badge, rendered once per region and faded in once the
// camera is close enough to read it (see _tick's zoom-dependent label fade)
function labelTexture(text) {
  const scale = 2, padX = 16, h = 40;
  const mctx = document.createElement("canvas").getContext("2d");
  mctx.font = "600 25px system-ui, sans-serif";
  const w = Math.ceil(mctx.measureText(text).width) + padX * 2;
  const c = document.createElement("canvas"); c.width = w * scale; c.height = h * scale;
  const ctx = c.getContext("2d"); ctx.scale(scale, scale);
  const r = h / 2;
  ctx.beginPath();
  ctx.moveTo(r, 1); ctx.arcTo(w, 1, w, h, r); ctx.arcTo(w, h, 0, h, r); ctx.arcTo(0, h, 0, 1, r); ctx.arcTo(0, 1, w, 1, r);
  ctx.closePath();
  ctx.fillStyle = "rgba(8,13,22,.8)"; ctx.fill();
  ctx.strokeStyle = "rgba(255,255,255,.18)"; ctx.lineWidth = 1; ctx.stroke();
  ctx.font = "600 25px system-ui, sans-serif"; ctx.textBaseline = "middle"; ctx.fillStyle = "#eef2fa";
  ctx.fillText(text, padX, h / 2 + 1);
  return { tex: new THREE.CanvasTexture(c), aspect: w / h };
}

export class Globe {
  constructor(canvas, opts = {}) {
    this.canvas = canvas;
    this.opts = opts;
    this.active = false;
    this._raf = null;
    this._disposed = false;
    this.ready = this._init();
  }

  async _init() {
    const [basemap, mosaicIdx, regionsResp, candResp] = await Promise.all([
      fetch("basemap.json").then(r => r.json()).catch(() => null),
      fetch("mosaic_index.json").then(r => r.json()).catch(() => ({ tiles: [] })),
      fetch("/regions").then(r => r.json()).catch(() => ({ regions: [] })),
      fetch("/candidates?limit=900").then(r => r.json()).catch(() => ({ candidates: [] })),
    ]);
    if (this._disposed) return;
    this.tiles = mosaicIdx.tiles || [];
    this.regions = regionsResp.regions || [];
    this.candidates = (candResp.candidates || []).filter(c => Array.isArray(c.centroid_lonlat));

    const renderer = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: true, alpha: false, powerPreference: "high-performance" });
    renderer.setClearColor(0x040608, 1);
    this.renderer = renderer;

    const scene = new THREE.Scene();
    this.scene = scene;
    const camera = new THREE.PerspectiveCamera(42, 1, 0.05, 100);
    camera.position.set(0, 0, START_DIST);
    this.camera = camera;

    scene.add(new THREE.AmbientLight(0xffffff, 0.62));
    const SUN_DIR = new THREE.Vector3(3, 2, 2.4).normalize();
    const sun = new THREE.DirectionalLight(0xffffff, 1.05);
    sun.position.copy(SUN_DIR);
    scene.add(sun);

    // star field - a cheap static point cloud, no per-frame cost beyond one draw call
    const starCount = 2600;
    const starPos = new Float32Array(starCount * 3);
    for (let i = 0; i < starCount; i++) {
      const r = 40 + Math.random() * 30;
      const u = Math.random(), v = Math.random();
      const th = 2 * Math.PI * u, ph = Math.acos(2 * v - 1);
      starPos[i * 3] = r * Math.sin(ph) * Math.cos(th);
      starPos[i * 3 + 1] = r * Math.cos(ph);
      starPos[i * 3 + 2] = r * Math.sin(ph) * Math.sin(th);
    }
    const starGeo = new THREE.BufferGeometry();
    starGeo.setAttribute("position", new THREE.BufferAttribute(starPos, 3));
    const stars = new THREE.Points(starGeo, new THREE.PointsMaterial({ color: 0xcfd9ee, size: 0.09, sizeAttenuation: true }));
    scene.add(stars);

    // base sphere - real NASA Blue Marble day/night/specular photography,
    // composited by a small custom shader (day<->night terminator + an
    // ocean-only specular glint), with borders/India-tint as a transparent
    // overlay texture on top - see the file header for the full breakdown
    const texLoader = new THREE.TextureLoader();
    const [dayTex, nightTex, specTex, bordersTex] = await Promise.all([
      texLoader.loadAsync("vendor/earth/day.jpg"),
      texLoader.loadAsync("vendor/earth/night.jpg"),
      texLoader.loadAsync("vendor/earth/specular.jpg"),
      buildBordersTexture(basemap),
    ]);
    if (this._disposed) return;
    dayTex.colorSpace = THREE.SRGBColorSpace;
    nightTex.colorSpace = THREE.SRGBColorSpace;
    const globeMesh = new THREE.Mesh(
      new THREE.SphereGeometry(1, 96, 96),
      new THREE.ShaderMaterial({
        uniforms: {
          dayMap: { value: dayTex }, nightMap: { value: nightTex },
          specMap: { value: specTex }, bordersMap: { value: bordersTex },
          sunDirection: { value: SUN_DIR },
        },
        vertexShader: EARTH_VERT_SHADER,
        fragmentShader: EARTH_FRAG_SHADER,
      })
    );
    scene.add(globeMesh);
    this.globeMesh = globeMesh;

    // atmosphere glow
    const atmo = new THREE.Mesh(
      new THREE.SphereGeometry(1.03, 64, 64),
      new THREE.MeshBasicMaterial({ color: 0x7cae4c, transparent: true, opacity: 0.13, side: THREE.BackSide })
    );
    scene.add(atmo);

    // real imagery patches - one small curved SphereGeometry slice per staged
    // AOI, clipped to its exact bbox and textured with its own full-res tile
    // (scripts/build_mosaic.py) instead of being downsampled into the shared
    // base texture, so a close-up on e.g. Kanha stays sharp.
    this.patchTiles = [];
    let patchBytes = 0;
    const loader = new THREE.TextureLoader();
    for (const t of this.tiles) {
      const [w, s, e, n] = t.bbox;
      const thetaStart = (90 - n) * DEG2RAD, thetaLength = Math.max(0.0001, (n - s) * DEG2RAD);
      const phiStart = (w + 180) * DEG2RAD, phiLength = Math.max(0.0001, (e - w) * DEG2RAD);
      const geo = new THREE.SphereGeometry(PATCH_RADIUS, 24, 24, phiStart, phiLength, thetaStart, thetaLength);
      // transparent: true is what actually makes each tile's feathered edge
      // (see build_mosaic.py) blend into the Blue Marble base instead of
      // reading as a hard rectangle - without it the alpha channel is simply
      // ignored and every patch draws as an opaque box
      const mat = new THREE.MeshLambertMaterial({ color: 0x333333, transparent: true, depthWrite: false });
      const mesh = new THREE.Mesh(geo, mat);
      scene.add(mesh);
      this.patchTiles.push(mesh);
      loader.load(t.file, (tex) => {
        tex.colorSpace = THREE.SRGBColorSpace;
        mat.map = tex; mat.color.set(0xffffff); mat.needsUpdate = true;
      });
    }
    this._reportPatchBytes = () => patchBytes;

    // region pins - one glowing sprite per staged REGION (the same coarser
    // grouping Watch Areas' region chips use - a region like "dehradun" can
    // span 2-3 of the finer imagery tiles above, and giving each of those a
    // separate overlapping pin made nearby ones melt into an additively-
    // blended white blob), scaled/coloured by how many change candidates
    // fall inside that region's bbox (computed client side from the
    // candidates we already fetched, instead of N more API calls)
    this.pins = []; this.rings = []; this.labels = [];
    const pinGroup = new THREE.Group();
    scene.add(pinGroup);
    const counts = this.regions.map(r => this.candidates.filter(c => bboxContains(r.bbox, c.centroid_lonlat[0], c.centroid_lonlat[1])).length);
    const maxCount = Math.max(1, ...counts);
    this.topPinIndex = counts.every(n => n === 0) ? -1 : counts.indexOf(Math.max(...counts));
    this.regions.forEach((r, i) => {
      const [w, s, e, n] = r.bbox;
      const lon = (w + e) / 2, lat = (s + n) / 2;
      const n_ = counts[i];
      const scale = 0.03 + Math.min(0.05, (n_ / maxCount) * 0.05);
      // "has candidates" reads as an attention/warning cue, an empty region as
      // neutral/informational - resolved through Tokens like every other
      // themed colour on this page, so it inherits a future theme switch too
      const color = n_ > 0 ? window.Tokens.triple("warning") : window.Tokens.triple("info");

      const ring = new THREE.Sprite(new THREE.SpriteMaterial({ map: ringTexture(color), transparent: true, depthWrite: false, opacity: 0.75 }));
      ring.position.copy(latLonToVec3(lat, lon, PIN_RADIUS - 0.002));
      ring.scale.set(scale * 1.9, scale * 1.9, 1);
      pinGroup.add(ring);
      this.rings.push(ring);

      const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: pinTexture(color), transparent: true, depthWrite: false, opacity: 0.95 }));
      sprite.position.copy(latLonToVec3(lat, lon, PIN_RADIUS));
      sprite.scale.set(scale, scale, 1);
      sprite.userData = { kind: "pin", name: r.name, bbox: r.bbox, lat, lon, count: n_, obs: r.n_observations, baseScale: scale };
      pinGroup.add(sprite);
      this.pins.push(sprite);

      const name = (this.opts.friendlyName && this.opts.friendlyName(r.name)) || r.name;
      const { tex: labelTex, aspect } = labelTexture(name);
      const label = new THREE.Sprite(new THREE.SpriteMaterial({ map: labelTex, transparent: true, depthWrite: false, opacity: 0 }));
      const labelH = 0.02; // distance-scaled every frame in _tick (see pinScale) - this is just the base size
      label.scale.set(labelH * aspect, labelH, 1);
      label.userData = { baseAspect: aspect, baseH: labelH };
      label.position.copy(latLonToVec3(lat, lon, PIN_RADIUS + 0.075));
      pinGroup.add(label);
      this.labels.push(label);
    });
    this.pinGroup = pinGroup;

    // change-candidate points - built once, only shown once zoomed in close
    // (see _tick), so the point cloud never competes with the pins/labels at
    // whole-globe zoom
    const cPos = new Float32Array(this.candidates.length * 3);
    const cCol = new Float32Array(this.candidates.length * 3);
    // resolved through the SAME Tokens.changeType() the 2D canvas map and every
    // badge/legend use - this is what fixes the water_loss colour drift the
    // audit found (globe candidate points used to be a hand-copied float-triple
    // palette, independently wrong for water_loss vs. the CSS/canvas colour)
    this.candidates.forEach((c, i) => {
      const v = latLonToVec3(c.centroid_lonlat[1], c.centroid_lonlat[0], 1.006);
      cPos[i * 3] = v.x; cPos[i * 3 + 1] = v.y; cPos[i * 3 + 2] = v.z;
      const rgb = window.Tokens.changeType(c.change_type);
      cCol[i * 3] = rgb.r / 255; cCol[i * 3 + 1] = rgb.g / 255; cCol[i * 3 + 2] = rgb.b / 255;
    });
    const cGeo = new THREE.BufferGeometry();
    cGeo.setAttribute("position", new THREE.BufferAttribute(cPos, 3));
    cGeo.setAttribute("color", new THREE.BufferAttribute(cCol, 3));
    const points = new THREE.Points(cGeo, new THREE.PointsMaterial({ size: 0.012, vertexColors: true, sizeAttenuation: true, transparent: true, opacity: 0.92 }));
    points.visible = false;
    scene.add(points);
    this.points = points;

    // controls - drag to orbit, scroll to zoom, inertia via damping
    const controls = new OrbitControls(camera, this.canvas);
    controls.enableDamping = true; controls.dampingFactor = 0.08;
    controls.rotateSpeed = 0.55; controls.zoomSpeed = 0.85;
    controls.minDistance = MIN_DIST; controls.maxDistance = MAX_DIST;
    controls.enablePan = false;
    // idle auto-rotate (see AUTO_ROTATE_* above) - starts once the globe has
    // settled on its default framing below, pauses on any drag/zoom/touch,
    // resumes only after AUTO_ROTATE_IDLE_MS of real idle. `update()` applies
    // autoRotate unconditionally (it isn't gated by controls.enabled), so
    // _diveToRegion explicitly pauses it too - otherwise it would fight the
    // programmatic camera path of a region dive.
    controls.autoRotateSpeed = AUTO_ROTATE_SPEED;
    controls.autoRotate = false;
    this.controls = controls;
    if (!REDUCE_MOTION) {
      this._pauseAutoRotate = () => {
        controls.autoRotate = false;
        if (this._resumeAutoRotateTimer) { clearTimeout(this._resumeAutoRotateTimer); this._resumeAutoRotateTimer = null; }
      };
      this._scheduleAutoRotateResume = () => {
        if (this._resumeAutoRotateTimer) clearTimeout(this._resumeAutoRotateTimer);
        this._resumeAutoRotateTimer = setTimeout(() => {
          this._resumeAutoRotateTimer = null;
          if (!this._diving && !this._disposed) controls.autoRotate = true;
        }, AUTO_ROTATE_IDLE_MS);
      };
      controls.addEventListener("start", this._pauseAutoRotate);
      controls.addEventListener("end", this._scheduleAutoRotateResume);
    }

    this._raycaster = new THREE.Raycaster();
    this._raycaster.params.Points.threshold = 0.02;
    this._pointer = new THREE.Vector2();

    this.canvas.addEventListener("pointermove", (e) => this._onPointerMove(e));
    this.canvas.addEventListener("pointerleave", () => { this._hideTip(); if (this.coordEl) this.coordEl.textContent = ""; });
    this.canvas.addEventListener("click", (e) => this._onClick(e));
    this._keyHandler = (e) => { if (e.key === "Escape" && this._diving) this._flightSkip = true; };
    window.addEventListener("keydown", this._keyHandler);

    this.tip = document.createElement("div"); this.tip.className = "maptip globe-tip";
    (this.canvas.parentElement || document.body).appendChild(this.tip);

    // Step 4A: coordinate + approximate-scale readouts, matching the 2D map's
    // .mapcoord/.mapscale so the globe is never missing them just because
    // it's the default view. Cursor lat/lon updates in _onPointerMove()
    // (raycast against the globe sphere); scale updates once per render tick
    // from camera distance (see _updateScaleLabel()).
    // .mapwrap is taller than the canvas alone (the long "drag to spin..."
    // caption sits below it in normal flow), so .mapcoord/.mapscale's own
    // bottom:12px - meant to hug the CANVAS's bottom edge, as it does for the
    // 2D map's shorter caption - would land on top of that caption text
    // instead. This HUD layer is sized to exactly the canvas's own height
    // (canvas.map's fixed 520px, see style.css) so children positioned
    // "bottom:12px" inside IT land on the canvas edge regardless of how long
    // the caption below turns out to be.
    const wrap = this.canvas.parentElement || document.body;
    this._hud = document.createElement("div"); this._hud.className = "globe-hud";
    wrap.appendChild(this._hud);
    this.coordEl = document.createElement("div"); this.coordEl.className = "mapcoord";
    this._hud.appendChild(this.coordEl);
    this.scaleEl = document.createElement("div"); this.scaleEl.className = "mapscale globe-scale";
    this.scaleEl.innerHTML = "<b></b>";
    this._hud.appendChild(this.scaleEl);

    this._flyTo(22.5, 79.0, START_DIST, 0); // frame India by default
    if (!REDUCE_MOTION) controls.autoRotate = true; // idle motion starts from here
    this.resize();
    return true;
  }

  _ndcFromEvent(e) {
    const r = this.canvas.getBoundingClientRect();
    return new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
  }
  _pickPin(ndc) {
    this._raycaster.setFromCamera(ndc, this.camera);
    const hits = this._raycaster.intersectObjects(this.pins, false);
    return hits.length ? hits[0].object : null;
  }
  _pickPoint(ndc) {
    if (!this.points.visible) return null;
    this._raycaster.setFromCamera(ndc, this.camera);
    const hits = this._raycaster.intersectObject(this.points, false);
    return hits.length ? hits[0].index : null;
  }
  _hideTip() { if (this.tip) this.tip.classList.remove("show"); }
  // Step 4A: lat/lon under the cursor, independent of pin/point hit-testing -
  // raycasts the globe sphere itself so the readout works anywhere on the
  // visible disc, not just over a pin.
  _updateCoord(e) {
    if (!this.coordEl || !this.globeMesh) return;
    const ndc = this._ndcFromEvent(e);
    this._raycaster.setFromCamera(ndc, this.camera);
    const hit = this._raycaster.intersectObject(this.globeMesh, false)[0];
    if (!hit) { this.coordEl.textContent = ""; return; }
    const { lat, lon } = vec3ToLatLon(hit.point);
    const ns = lat >= 0 ? "N" : "S", ew = lon >= 0 ? "E" : "W";
    this.coordEl.textContent = `${Math.abs(lat).toFixed(2)}°${ns}, ${Math.abs(lon).toFixed(2)}°${ew}`;
  }
  // Approximate only (a perspective view has no single scale) - horizontal
  // ground distance spanned by the camera's field of view at the point
  // closest to it, using Earth's real radius against the globe's unit-sphere
  // model (see latLonToVec3 - the sphere IS radius 1 = 6371 km throughout).
  _updateScaleLabel() {
    if (!this.scaleEl || !this.camera) return;
    const camDist = this.camera.position.length();
    const heightAboveSurface = Math.max(0.001, camDist - 1);
    const halfFovRad = (this.camera.fov / 2) * DEG2RAD;
    const spanKm = 2 * heightAboveSurface * Math.tan(halfFovRad) * EARTH_RADIUS_KM;
    const label = spanKm >= 1000 ? `${(spanKm / 1000).toFixed(1)}k km` : `${Math.round(spanKm)} km`;
    const text = `≈ ${label} across view`;
    if (this._lastScaleText !== text) { this._lastScaleText = text; this.scaleEl.querySelector("b").textContent = text; }
  }
  _onPointerMove(e) {
    this._updateCoord(e);
    const ndc = this._ndcFromEvent(e);
    const pin = this._pickPin(ndc);
    const pIdx = pin ? null : this._pickPoint(ndc);
    if (!pin && pIdx == null) { this._hideTip(); this.canvas.style.cursor = "grab"; return; }
    this.canvas.style.cursor = "pointer";
    const r = this.canvas.getBoundingClientRect();
    const left = Math.min(e.clientX - r.left + 14, r.width - 220);
    const top = Math.max(e.clientY - r.top - 34, 4);
    this.tip.style.left = left + "px"; this.tip.style.top = top + "px";
    if (pin) {
      const d = pin.userData;
      this.tip.innerHTML = `<b>${esc((this.opts.friendlyName && this.opts.friendlyName(d.name)) || d.name)}</b><br><span style="opacity:.75">${d.count} candidate${d.count === 1 ? "" : "s"} &middot; ${d.obs} observation${d.obs === 1 ? "" : "s"}</span>`;
    } else {
      const c = this.candidates[pIdx];
      this.tip.innerHTML = `<b>${esc(c.change_type || "change")}</b><br><span style="opacity:.75">conf ${(+c.confidence || 0).toFixed(2)}</span>`;
    }
    this.tip.classList.add("show");
  }
  _onClick(e) {
    // a dive in progress absorbs the next click as "skip to the end" rather
    // than as a fresh pick - otherwise a trigger-happy analyst re-clicking
    // mid-flight would just restart the animation at a new target
    if (this._diving) { this._flightSkip = true; return; }
    const ndc = this._ndcFromEvent(e);
    const pin = this._pickPin(ndc);
    if (pin) { this._diveToRegion(pin.userData); return; }
    const pIdx = this._pickPoint(ndc);
    if (pIdx != null && this.opts.onOpenCandidate) {
      const c = this.candidates[pIdx];
      this.opts.onOpenCandidate(c.candidate_id);
    }
  }
  // one continuous globe -> region -> AOI dive, ending right over the
  // candidate imagery, then a short fade before handing off into the review
  // queue filtered to that region - "the analyst should feel they flew from
  // space to the candidate," not hop to a country-level pin and cut away
  _diveToRegion(d) {
    const camDist = this.camera.position.length();
    const finalDist = MIN_DIST + 0.02;
    const duration = Math.max(1500, Math.min(3200, 1300 + (camDist - finalDist) * 480));
    this._diving = true;
    if (this._pauseAutoRotate) this._pauseAutoRotate(); // a dive owns the camera path outright - idle rotation would fight the slerp
    if (this.controls) this.controls.enabled = false; // don't fight the programmatic dive with a drag mid-flight
    // ---- ZOOM ENDPOINT --------------------------------------------------
    // This callback is the exact handoff point for a future continuous
    // descent (not built here - see file header). At the moment it fires:
    //   - this.camera sits at `finalDist` (≈ MIN_DIST, the zoom-in limit)
    //     pointed at (d.lat, d.lon) - the region's centroid
    //   - d.bbox / d.name identify which AOI/imagery to descend into
    //   - this.globeMesh, this.pinGroup etc. are all still live and rendered
    // A later phase can replace the fade-to-black + cut-to-Queue below with
    // an actual camera descent past this point (e.g. crossing into a
    // logarithmic-depth-buffer scene, or draping/zooming the matching
    // mosaic-tile patch) without changing anything upstream of this line.
    this._flyTo(d.lat, d.lon, finalDist, duration, () => {
      this._diving = false;
      if (this.controls) this.controls.enabled = true;
      if (!this.opts.onOpenRegion) return;
      const canvas = this.canvas;
      canvas.style.transition = "opacity .45s ease";
      canvas.style.opacity = "0";
      setTimeout(() => {
        this.opts.onOpenRegion(d.bbox, d.name);
        setTimeout(() => { canvas.style.transition = ""; canvas.style.opacity = ""; }, 60);
      }, 460);
    });
    // ---- end ZOOM ENDPOINT ------------------------------------------------
  }
  _flyTo(lat, lon, distance, durationMs = 1200, onComplete) {
    if (!this.camera) return;
    const dirTo = latLonToVec3(lat, lon, 1).normalize();
    const dirFrom = this.camera.position.clone().normalize();
    const distFrom = this.camera.position.length();
    if (!durationMs) {
      this.camera.position.copy(dirTo.clone().multiplyScalar(distance));
      this.controls?.update();
      if (onComplete) onComplete();
      return;
    }
    if (this._flightRaf) cancelAnimationFrame(this._flightRaf);
    const t0 = performance.now();
    this._flightSkip = false;
    const step = (now) => {
      if (this._disposed) return;
      const p = this._flightSkip ? 1 : Math.min(1, (now - t0) / durationMs);
      const ease = 1 - Math.pow(1 - p, 3);
      const dir = slerpDir(dirFrom, dirTo, ease);
      const dist = distFrom + (distance - distFrom) * ease;
      this.camera.position.copy(dir.multiplyScalar(dist));
      this.controls.update();
      if (p < 1) { this._flightRaf = requestAnimationFrame(step); }
      else { this._flightRaf = null; if (onComplete) onComplete(); }
    };
    this._flightRaf = requestAnimationFrame(step);
  }
  flyToRegion(name) {
    const t = this.regions.find(x => x.name === name);
    if (!t) return;
    const [w, s, e, n] = t.bbox;
    this._flyTo((s + n) / 2, (w + e) / 2, 1.3, 1700);
  }
  // Step 4A: same framing as flyToRegion, but from a raw [w,s,e,n] bbox
  // (Overview's own AOI_FALLBACK) rather than a named /regions lookup - used
  // once on first mount so the globe opens already centred on the AOI
  // instead of the whole-Earth default view.
  flyToBbox(bbox, durationMs = 1700) {
    const [w, s, e, n] = bbox;
    this._flyTo((s + n) / 2, (w + e) / 2, 1.3, durationMs);
  }

  resize() {
    if (!this.renderer || !this.canvas.clientWidth) return;
    const w = this.canvas.clientWidth, h = this.canvas.clientHeight || w;
    this.renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }
  setActive(on) {
    this.active = on;
    if (on && !this._raf) this._tick();
    if (!on && this._raf) { cancelAnimationFrame(this._raf); this._raf = null; }
  }
  _tick() {
    if (!this.active || this._disposed) { this._raf = null; return; }
    this._raf = requestAnimationFrame(() => this._tick());
    // real elapsed seconds, not an assumed frame count - OrbitControls' own
    // autoRotate falls back to a fixed "assume 60fps" angle per call when no
    // deltaTime is given, which would make AUTO_ROTATE_SPEED run at half the
    // intended rate on a display refreshing at ~30fps (as measured under
    // software-rendered/headless GL) and faster on a high-refresh display -
    // passing real dt keeps the "restrained" rotation speed the same
    // regardless of frame rate. Clamped so a backgrounded-tab pause doesn't
    // produce one giant jump in rotation when the tab regains focus.
    const now = performance.now();
    const dt = this._lastTickTime ? Math.min(0.25, (now - this._lastTickTime) / 1000) : 0;
    this._lastTickTime = now;
    if (this.controls) this.controls.update(dt);
    const camDist = this.camera.position.length();
    if (this.points) this.points.visible = camDist < POINTS_VISIBLE_DIST;
    if (this.pins) {
      // pins are sprites (world-space size), so left alone they'd balloon to
      // fill the screen up close like any other object - shrink them as the
      // camera nears the surface so a pin never blocks the imagery/points
      // it is marking, roughly like a map marker holding its screen size
      const pinScale = Math.max(0.32, Math.min(1, camDist / START_DIST));
      // once close enough for individual candidates to show, the region pin
      // has done its job (got the analyst here) - fade it back so its glow
      // doesn't swallow the very points it was marking
      const pinOpacity = camDist < POINTS_VISIBLE_DIST ? 0.3 : 0.95;
      // the single busiest region gets a slow pulse - the one visual cue that
      // says "look here" before the analyst has clicked anything
      const pulse = 1 + Math.sin(performance.now() / 420) * 0.16;
      // labels are unreadable clutter at whole-globe *and* region-overview
      // zoom (several neighbouring AOIs are visible at once there) - fade
      // them in only once the camera has committed to one specific AOI
      const labelT = 1 - Math.max(0, Math.min(1, (camDist - 1.16) / (1.55 - 1.16)));
      // sprites are billboards (world-space size), so left unscaled they'd
      // grow as the camera gets closer, the opposite of what the pin-scale
      // shrink above does - counter that so a label stays a roughly steady,
      // readable screen size across the whole zoom-in range it's visible for
      const labelScale = Math.max(0.45, Math.min(1.3, camDist / 1.8));
      this.pins.forEach((pin, i) => {
        const isTop = i === this.topPinIndex;
        const p = isTop ? pulse : 1;
        pin.scale.setScalar(pin.userData.baseScale * pinScale * p);
        pin.material.opacity = pinOpacity;
        const ring = this.rings[i];
        ring.scale.setScalar(pin.userData.baseScale * 1.9 * pinScale * (isTop ? pulse * 1.05 : 1));
        ring.material.opacity = (isTop ? 0.55 + 0.35 * (pulse - 1) / 0.16 : 0.75) * (pinOpacity / 0.95);
        const label = this.labels[i];
        label.scale.set(label.userData.baseH * label.userData.baseAspect * labelScale, label.userData.baseH * labelScale, 1);
        label.material.opacity = labelT * 0.95;
        label.visible = labelT > 0.02;
      });
    }
    this._updateScaleLabel();
    if (this.renderer) this.renderer.render(this.scene, this.camera);
  }
  dispose() {
    this._disposed = true;
    this.setActive(false);
    if (this._flightRaf) cancelAnimationFrame(this._flightRaf);
    if (this._resumeAutoRotateTimer) clearTimeout(this._resumeAutoRotateTimer);
    if (this.controls && this._pauseAutoRotate) {
      this.controls.removeEventListener("start", this._pauseAutoRotate);
      this.controls.removeEventListener("end", this._scheduleAutoRotateResume);
    }
    if (this._keyHandler) window.removeEventListener("keydown", this._keyHandler);
    if (this.tip) this.tip.remove();
    if (this.renderer) this.renderer.dispose();
  }
}
