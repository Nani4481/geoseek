// terrain-3d.js - isometric 3D terrain grid, pure Canvas 2D, no dependencies.
// Renders a before/after heightmap for a change-detection candidate so an
// analyst can see the shape of the change, not just read a percentage. This
// is explicitly a stylized model (procedural noise seeded by change type,
// shaped by the candidate's real magnitude) - not a claim of measured
// elevation - so it's lit and colored for legibility and impact rather than
// literal terrain accuracy.

const CHANGE_RAMPS = {
  // [low-height color, high-height color] per class, plus the "ghost"
  // wireframe tint used for the faint before-state outline.
  water_gain: { low: "#0C2E4A", high: "#3D8BE0", ghost: "#5FC8E8" },
  water_loss: { low: "#4A3A28", high: "#B08A5A", ghost: "#5FC8E8" },
  construction: { low: "#3A3C3F", high: "#9AA0A6", ghost: "#4FC58B" },
  clearance: { low: "#3D2A12", high: "#B4703A", ghost: "#4FC58B" },
  road: { low: "#4A3A28", high: "#B08A5A", ghost: "#C7CBCD" },
  other: { low: "#4A4030", high: "#B0A078", ghost: "#C7CBCD" },
};

// Light direction for the flat-shaded facets (pointing toward the light),
// normalized. Coming from upper-left-front reads as "sunlit" in the same
// isometric convention the projection already uses.
const LIGHT_DIR = normalize3([-0.45, -0.65, 0.62]);
const AMBIENT = 0.38;   // never fully black even facing away from the light
const LIGHT_GAIN = 0.85;

function normalize3(v) {
  const len = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / len, v[1] / len, v[2] / len];
}

function colorsFor(changeType) {
  return CHANGE_RAMPS[changeType] || CHANGE_RAMPS.other;
}

function hexToRgb(hex) {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function lerpColor(c0, c1, t) {
  return [c0[0] + (c1[0] - c0[0]) * t, c0[1] + (c1[1] - c0[1]) * t, c0[2] + (c1[2] - c0[2]) * t];
}

function rgbString([r, g, b], shade = 1) {
  return `rgb(${Math.round(Math.max(0, Math.min(255, r * shade)))},${Math.round(Math.max(0, Math.min(255, g * shade)))},${Math.round(Math.max(0, Math.min(255, b * shade)))})`;
}

// Simple 2D value noise: random values on an integer lattice, bilinear
// interpolation in between, smoothstepped. ~20 lines, no library.
function makeValueNoise(seed) {
  let s = seed >>> 0 || 1;
  function rand() {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  }
  const latticeSize = 9;
  const lattice = Array.from({ length: latticeSize }, () =>
    Array.from({ length: latticeSize }, () => rand()));
  function smooth(t) { return t * t * (3 - 2 * t); }
  return function noise(x, y) {
    // Wrapped into [0, 1) so a caller sampling at a coarser frequency (the
    // "construction" block pattern below calls noise(gx/6, gy/6), which is
    // well outside [0,1) for any gx/gy past 6) can't index the lattice out
    // of bounds - that used to throw and abort the whole candidate render.
    x -= Math.floor(x);
    y -= Math.floor(y);
    const gx = x * (latticeSize - 1), gy = y * (latticeSize - 1);
    const x0 = Math.floor(gx), y0 = Math.floor(gy);
    const x1 = Math.min(x0 + 1, latticeSize - 1), y1 = Math.min(y0 + 1, latticeSize - 1);
    const tx = smooth(gx - x0), ty = smooth(gy - y0);
    const v00 = lattice[y0][x0], v10 = lattice[y0][x1];
    const v01 = lattice[y1][x0], v11 = lattice[y1][x1];
    const a = v00 + (v10 - v00) * tx;
    const b = v01 + (v11 - v01) * tx;
    return a + (b - a) * ty;
  };
}

export class Terrain3D {
  constructor(canvas, options = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.gridSize = options.gridSize || 34;
    this.cellSize = options.cellSize || 13;
    this.heightScale = options.heightScale || 42;
    this.rotation = options.rotation || 0;
    this.before = null;   // Float32Array, height 0..1 per cell
    this.after = null;    // Float32Array, height 0..1 per cell
    this.colors = colorsFor("other");
    this.summary = "";
    this.animationProgress = 0;
    this.isAnimating = false;
    this._rafId = null;
    this._autoRotate = false;
  }

  // Derive a heightmap pair from the candidate's real change signal.
  // magnitude in [0,1] drives how far "after" departs from "before";
  // changeType picks the terrain shape and palette.
  setData(changeType, magnitude, options = {}) {
    const n = this.gridSize;
    const mag = Math.max(0, Math.min(1, magnitude || 0));
    const noise = makeValueNoise(options.seed || hashString(changeType || "other"));
    const before = new Float32Array(n * n);
    const after = new Float32Array(n * n);
    const cx = (n - 1) / 2, cy = (n - 1) / 2;
    const maxR = Math.sqrt(cx * cx + cy * cy);

    for (let gy = 0; gy < n; gy++) {
      for (let gx = 0; gx < n; gx++) {
        const i = gy * n + gx;
        const nBase = noise(gx / n, gy / n);
        const r = Math.sqrt((gx - cx) ** 2 + (gy - cy) ** 2) / maxR;
        const falloff = Math.max(0, 1 - r); // change is strongest near center
        before[i] = 0.12 + nBase * 0.18;

        if (changeType === "water_gain") {
          after[i] = before[i] * (1 - falloff * mag) + 0.02 * falloff * mag;
        } else if (changeType === "water_loss") {
          before[i] = 0.05 + nBase * 0.05;
          after[i] = before[i] + falloff * mag * (0.2 + nBase * 0.15);
        } else if (changeType === "construction") {
          const blockNoise = noise(gx / 6, gy / 6);
          const block = blockNoise > 0.55 ? 1 : 0;
          after[i] = before[i] + falloff * mag * block * (0.45 + blockNoise * 0.3);
        } else if (changeType === "clearance") {
          before[i] = 0.15 + nBase * 0.4; // vegetation: rougher, taller
          after[i] = before[i] * (1 - falloff * mag * 0.85);
        } else if (changeType === "road") {
          const onLine = Math.abs((gy - cy) - (gx - cx) * 0.15) < 1.4 ? 1 : 0;
          after[i] = Math.max(0.04, before[i] - onLine * mag * 0.12);
        } else {
          after[i] = before[i] + (nBase - 0.5) * falloff * mag * 0.4;
        }
        after[i] = Math.max(0, Math.min(1, after[i]));
      }
    }
    this.before = before;
    this.after = after;
    this.colors = colorsFor(changeType);
    this.summary = summaryFor(changeType, mag, options);
    this.animationProgress = 0;
    return this;
  }

  // World-space (pre-projection) position for a grid vertex - grid units in
  // X/Y, real height units in Z - used both for the screen projection and
  // for lighting normals, so the two stay consistent as the model rotates.
  _world(gridX, gridY, height) {
    const cos = Math.cos(this.rotation), sin = Math.sin(this.rotation);
    const cx0 = (this.gridSize - 1) / 2, cy0 = (this.gridSize - 1) / 2;
    const rx = gridX - cx0, ry = gridY - cy0;
    return [rx * cos - ry * sin, rx * sin + ry * cos, height * (this.heightScale / this.cellSize)];
  }

  _projectWorld([x, y, z]) {
    const logicalW = this.canvas.clientWidth || this.canvas.width;
    const logicalH = this.canvas.clientHeight || this.canvas.height;
    const canvasCx = logicalW / 2, canvasCy = logicalH / 2 + this.heightScale * 0.25;
    const screenX = (x - y) * this.cellSize * 0.5 + canvasCx;
    const screenY = (x + y) * this.cellSize * 0.25 - z * this.cellSize + canvasCy;
    return [screenX, screenY];
  }

  _project(gridX, gridY, height) {
    return this._projectWorld(this._world(gridX, gridY, height));
  }

  _drawBackdrop() {
    const ctx = this.ctx;
    const w = this.canvas.clientWidth || this.canvas.width;
    const h = this.canvas.clientHeight || this.canvas.height;
    const g = ctx.createRadialGradient(w / 2, h * 0.38, 0, w / 2, h * 0.38, Math.max(w, h) * 0.62);
    g.addColorStop(0, "rgba(95,200,232,0.07)");
    g.addColorStop(1, "rgba(95,200,232,0)");
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, w, h);
  }

  _drawGroundShadow(heights) {
    const ctx = this.ctx;
    const n = this.gridSize;
    // Project the footprint at height 0 to get the shape's screen silhouette,
    // then draw a soft dark ellipse under it for grounding.
    const c0 = this._project(0, 0, 0);
    const c1 = this._project(n - 1, n - 1, 0);
    const cx = (c0[0] + c1[0]) / 2, cy = (c0[1] + c1[1]) / 2;
    const rx = Math.abs(c1[0] - c0[0]) * 0.62, ry = Math.abs(c1[1] - c0[1]) * 0.4 + 10;
    const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, Math.max(rx, ry));
    g.addColorStop(0, "rgba(0,0,0,0.38)");
    g.addColorStop(1, "rgba(0,0,0,0)");
    ctx.save();
    ctx.translate(cx, cy);
    ctx.scale(rx / Math.max(rx, ry), ry / Math.max(rx, ry));
    ctx.translate(-cx, -cy);
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(cx, cy, Math.max(rx, ry), 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }

  // Flat-shaded lighting for one triangle (as world-space points): normal
  // from the edge cross product, clamped dot with the light direction, plus
  // a constant ambient floor so shadowed facets stay readable, not black.
  _lightFor(pA, pB, pC) {
    const u = [pB[0] - pA[0], pB[1] - pA[1], pB[2] - pA[2]];
    const v = [pC[0] - pA[0], pC[1] - pA[1], pC[2] - pA[2]];
    let nx = u[1] * v[2] - u[2] * v[1];
    let ny = u[2] * v[0] - u[0] * v[2];
    let nz = u[0] * v[1] - u[1] * v[0];
    const len = Math.hypot(nx, ny, nz) || 1;
    nx /= len; ny /= len; nz /= len;
    if (nz < 0) { nx = -nx; ny = -ny; nz = -nz; } // keep normals facing the camera
    const dot = Math.max(0, nx * LIGHT_DIR[0] + ny * LIGHT_DIR[1] + nz * LIGHT_DIR[2]);
    return AMBIENT + dot * LIGHT_GAIN;
  }

  _drawCell(gx, gy, heights, opacity, wireframeOnly) {
    const ctx = this.ctx;
    const n = this.gridSize;
    const h00 = heights[gy * n + gx];
    const h10 = gx + 1 < n ? heights[gy * n + gx + 1] : h00;
    const h01 = gy + 1 < n ? heights[(gy + 1) * n + gx] : h00;
    const h11 = (gx + 1 < n && gy + 1 < n) ? heights[(gy + 1) * n + gx + 1] : h00;

    const w00 = this._world(gx, gy, h00), w10 = this._world(gx + 1, gy, h10);
    const w01 = this._world(gx, gy + 1, h01), w11 = this._world(gx + 1, gy + 1, h11);
    const p00 = this._projectWorld(w00), p10 = this._projectWorld(w10);
    const p01 = this._projectWorld(w01), p11 = this._projectWorld(w11);

    if (wireframeOnly) {
      ctx.globalAlpha = opacity;
      ctx.strokeStyle = this.colors.ghost;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(p00[0], p00[1]); ctx.lineTo(p10[0], p10[1]);
      ctx.lineTo(p11[0], p11[1]); ctx.lineTo(p01[0], p01[1]); ctx.closePath();
      ctx.stroke();
      ctx.globalAlpha = 1;
      return;
    }

    const avgHeight = (h00 + h10 + h01 + h11) / 4;
    const base = lerpColor(hexToRgb(this.colors.low), hexToRgb(this.colors.high), Math.min(1, avgHeight * 2.2));

    ctx.globalAlpha = opacity;

    // right side face: drop straight down from the gx+1 edge to its own base
    const b10 = this._world(gx + 1, gy, 0), b11 = this._world(gx + 1, gy + 1, 0);
    const sideRightLight = this._lightFor(w10, b10, w11) * 0.85;
    ctx.fillStyle = rgbString(base, sideRightLight);
    ctx.beginPath();
    const pb10 = this._projectWorld(b10), pb11 = this._projectWorld(b11);
    ctx.moveTo(p10[0], p10[1]); ctx.lineTo(p11[0], p11[1]);
    ctx.lineTo(pb11[0], pb11[1]); ctx.lineTo(pb10[0], pb10[1]);
    ctx.closePath();
    ctx.fill();

    // left/front side face (down from the gy+1 edge)
    const b01 = this._world(gx, gy + 1, 0);
    const sideFrontLight = this._lightFor(w01, b01, w11) * 0.7;
    ctx.fillStyle = rgbString(base, sideFrontLight);
    ctx.beginPath();
    const pb01 = this._projectWorld(b01);
    ctx.moveTo(p01[0], p01[1]); ctx.lineTo(p11[0], p11[1]);
    ctx.lineTo(pb11[0], pb11[1]); ctx.lineTo(pb01[0], pb01[1]);
    ctx.closePath();
    ctx.fill();

    // top face - lit per-facet so the whole undulating surface actually
    // reads as lit 3D relief instead of one flat color.
    const topLight = this._lightFor(w00, w10, w01);
    ctx.fillStyle = rgbString(base, topLight);
    ctx.beginPath();
    ctx.moveTo(p00[0], p00[1]); ctx.lineTo(p10[0], p10[1]);
    ctx.lineTo(p11[0], p11[1]); ctx.lineTo(p01[0], p01[1]);
    ctx.closePath();
    ctx.fill();

    ctx.strokeStyle = "rgba(255,255,255,0.08)";
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  render() {
    const ctx = this.ctx;
    const dpr = window.devicePixelRatio || 1;
    if (this.canvas.width !== Math.round(this.canvas.clientWidth * dpr) && this.canvas.clientWidth) {
      this.canvas.width = Math.round(this.canvas.clientWidth * dpr);
      this.canvas.height = Math.round(this.canvas.clientHeight * dpr);
    }
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    if (dpr !== 1) ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this._drawBackdrop();
    if (!this.before || !this.after) return;

    const n = this.gridSize;
    const t = this.animationProgress;
    const current = new Float32Array(n * n);
    for (let i = 0; i < n * n; i++) current[i] = this.before[i] + (this.after[i] - this.before[i]) * t;

    this._drawGroundShadow(current);

    // painter's algorithm: back-to-front by (gx + gy)
    const order = [];
    for (let gy = 0; gy < n - 1; gy++) for (let gx = 0; gx < n - 1; gx++) order.push([gx, gy]);
    order.sort((a, b) => (a[0] + a[1]) - (b[0] + b[1]));

    // "before" ghost wireframe underneath, at low opacity
    for (const [gx, gy] of order) this._drawCell(gx, gy, this.before, 0.2, true);
    // animated "after" state, solid
    for (const [gx, gy] of order) this._drawCell(gx, gy, current, 1, false);
  }

  animate() {
    const reduceMotion = window.matchMedia &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduceMotion) {
      this.animationProgress = 1;
      this.render();
      if (this._autoRotate) this._loopAutoRotate();
      return;
    }
    if (this._rafId) cancelAnimationFrame(this._rafId);
    const duration = 1500;
    const start = performance.now();
    this.isAnimating = true;
    const step = (now) => {
      const t = Math.min(1, (now - start) / duration);
      this.animationProgress = 1 - Math.pow(1 - t, 3); // ease-out cubic
      this.render();
      if (t < 1) {
        this._rafId = requestAnimationFrame(step);
      } else {
        this.isAnimating = false;
        if (this._autoRotate) this._loopAutoRotate();
      }
    };
    this._rafId = requestAnimationFrame(step);
  }

  enableAutoRotate() {
    this._autoRotate = true;
    if (!this.isAnimating) this._loopAutoRotate();
  }

  _loopAutoRotate() {
    const reduceMotion = window.matchMedia &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduceMotion || !this._autoRotate) return;
    const step = () => {
      if (!this._autoRotate) return;
      this.rotation += 0.0035;
      this.render();
      this._rafId = requestAnimationFrame(step);
    };
    this._rafId = requestAnimationFrame(step);
  }

  destroy() {
    this._autoRotate = false;
    if (this._rafId) cancelAnimationFrame(this._rafId);
  }
}

function hashString(str) {
  let h = 0;
  for (let i = 0; i < str.length; i++) h = (h * 31 + str.charCodeAt(i)) >>> 0;
  return h;
}

function summaryFor(changeType, mag, options) {
  const pct = Math.round(mag * 100);
  const structures = options.structureEstimate;
  if (changeType === "water_gain") return `Water surface expanded by an estimated ${pct}% across the monitored area.`;
  if (changeType === "water_loss") return `Water surface retreated by an estimated ${pct}% across the monitored area.`;
  if (changeType === "clearance") return `Vegetation cover reduced by an estimated ${pct}% — possible clearing activity.`;
  if (changeType === "construction") return `New construction detected — surface disturbance of ${pct}% relative to the surrounding footprint${structures ? `, roughly ${structures} structures estimated` : ""}.`;
  if (changeType === "road") return `A new linear cut is visible across the footprint, an estimated ${pct}% change in surface cover.`;
  return `Ground cover changed by an estimated ${pct}% across the monitored area.`;
}
