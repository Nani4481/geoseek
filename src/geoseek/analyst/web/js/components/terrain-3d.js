// terrain-3d.js - isometric 3D terrain grid, pure Canvas 2D, no dependencies.
// Renders a before/after heightmap for a change-detection candidate so an
// analyst can see the shape of the change, not just read a percentage.

const CHANGE_COLORS = {
  water_gain: { top: "#1E5A8C", side: "#123C5C", ghost: "#3D7BE0" },
  water_loss: { top: "#8B7355", side: "#5E4C39", ghost: "#3D7BE0" },
  construction: { top: "#6B6E72", side: "#45474A", ghost: "#4FC58B" },
  clearance: { top: "#8B4513", side: "#5C2D0C", ghost: "#33B45A" },
  road: { top: "#8B7355", side: "#5E4C39", ghost: "#8E9699" },
  other: { top: "#9A8A6E", side: "#665C48", ghost: "#A7AEB1" },
};

function colorsFor(changeType) {
  return CHANGE_COLORS[changeType] || CHANGE_COLORS.other;
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
    // x, y in [0, 1)
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
    this.gridSize = options.gridSize || 32;
    this.cellSize = options.cellSize || 14;
    this.heightScale = options.heightScale || 40;
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

  _project(gridX, gridY, height) {
    const cos = Math.cos(this.rotation), sin = Math.sin(this.rotation);
    const cx0 = (this.gridSize - 1) / 2, cy0 = (this.gridSize - 1) / 2;
    const rx = gridX - cx0, ry = gridY - cy0;
    const x = rx * cos - ry * sin;
    const y = rx * sin + ry * cos;
    // logical (CSS-pixel) coordinate space: the canvas context is already
    // scaled by devicePixelRatio in render(), so projecting against the raw
    // (DPR-multiplied) canvas.width/height here would double-scale on
    // high-DPI screens.
    const logicalW = this.canvas.clientWidth || this.canvas.width;
    const logicalH = this.canvas.clientHeight || this.canvas.height;
    const canvasCx = logicalW / 2, canvasCy = logicalH / 2 + this.heightScale * 0.25;
    const screenX = (x - y) * this.cellSize * 0.5 + canvasCx;
    const screenY = (x + y) * this.cellSize * 0.25 - height * this.heightScale + canvasCy;
    return [screenX, screenY];
  }

  _shade(hex, factor) {
    const n = parseInt(hex.slice(1), 16);
    const r = Math.max(0, Math.min(255, Math.round(((n >> 16) & 255) * factor)));
    const g = Math.max(0, Math.min(255, Math.round(((n >> 8) & 255) * factor)));
    const b = Math.max(0, Math.min(255, Math.round((n & 255) * factor)));
    return `rgb(${r},${g},${b})`;
  }

  _drawCell(gx, gy, heights, opacity, wireframeOnly) {
    const ctx = this.ctx;
    const n = this.gridSize;
    const h00 = heights[gy * n + gx];
    const h10 = gx + 1 < n ? heights[gy * n + gx + 1] : h00;
    const h01 = gy + 1 < n ? heights[(gy + 1) * n + gx] : h00;
    const h11 = (gx + 1 < n && gy + 1 < n) ? heights[(gy + 1) * n + gx + 1] : h00;

    const p00 = this._project(gx, gy, h00);
    const p10 = this._project(gx + 1, gy, h10);
    const p01 = this._project(gx, gy + 1, h01);
    const p11 = this._project(gx + 1, gy + 1, h11);

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

    ctx.globalAlpha = opacity;
    // right side face: drop straight down from the gx+1 edge to its own base
    ctx.fillStyle = this._shade(this.colors.side, 0.75);
    ctx.beginPath();
    const b10 = this._project(gx + 1, gy, 0);
    const b11 = this._project(gx + 1, gy + 1, 0);
    ctx.moveTo(p10[0], p10[1]); ctx.lineTo(p11[0], p11[1]);
    ctx.lineTo(b11[0], b11[1]); ctx.lineTo(b10[0], b10[1]);
    ctx.closePath();
    ctx.fill();

    // left/front side face (down from the gy+1 edge)
    ctx.fillStyle = this._shade(this.colors.side, 0.6);
    ctx.beginPath();
    const b01 = this._project(gx, gy + 1, 0);
    ctx.moveTo(p01[0], p01[1]); ctx.lineTo(p11[0], p11[1]);
    ctx.lineTo(b11[0], b11[1]); ctx.lineTo(b01[0], b01[1]);
    ctx.closePath();
    ctx.fill();

    // top face
    ctx.fillStyle = this.colors.top;
    ctx.beginPath();
    ctx.moveTo(p00[0], p00[1]); ctx.lineTo(p10[0], p10[1]);
    ctx.lineTo(p11[0], p11[1]); ctx.lineTo(p01[0], p01[1]);
    ctx.closePath();
    ctx.fill();

    ctx.strokeStyle = "rgba(255,255,255,0.06)";
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
    if (!this.before || !this.after) return;

    const n = this.gridSize;
    const t = this.animationProgress;
    const current = new Float32Array(n * n);
    for (let i = 0; i < n * n; i++) current[i] = this.before[i] + (this.after[i] - this.before[i]) * t;

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
      this.rotation += 0.002;
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
