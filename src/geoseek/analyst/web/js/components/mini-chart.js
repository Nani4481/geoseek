// components/mini-chart.js - small Chart.js wrappers over the app's own
// real, already-computed numbers (the 5 evidence-gate pass/fail booleans,
// per-class detection counts). No chart here plots anything the rest of the
// UI doesn't already state in text; the chart is a second, faster-to-scan
// view of the same real values.
//
// Global `Chart` comes from vendor/chart.umd.js, loaded as a classic script
// in index.html before this module runs. Colors are read from the page's
// own CSS custom properties at chart-creation time, since a <canvas> draw
// call can't resolve var(--x) itself the way an element's style can.

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

// A canvas 2D context can't resolve var(--x) itself the way an element's
// CSS can - group colors elsewhere in the app are handed around as literal
// "var(--detect-box)" strings (see viewmodels/detection.js's GROUP_COLOR),
// so any of those passed in here need resolving to a real color value
// before Chart.js hands them to the canvas, or every bar silently renders
// in the context's default black.
function resolveColor(value) {
  const m = /^var\((--[\w-]+)\)$/.exec((value || "").trim());
  return m ? cssVar(m[1]) || value : value;
}

const commonOptions = {
  responsive: true,
  maintainAspectRatio: false,
  animation: { duration: 200 },
  plugins: { legend: { display: false } },
};

export function mountGateChart(canvas, gates) {
  const pass = cssVar("--gate-pass") || "#4FC58B";
  const fail = cssVar("--gate-fail") || "#E86A4A";
  const gridColor = cssVar("--hairline-2") || "rgba(255,255,255,0.07)";
  const tickColor = cssVar("--ink-dim") || "#8E9699";
  const monoFont = cssVar("--font-mono") || "monospace";

  const chart = new Chart(canvas, {
    type: "bar",
    data: {
      labels: gates.map((g) => g.label),
      datasets: [{
        data: gates.map((g) => (g.pass ? 1 : 0)),
        backgroundColor: gates.map((g) => (g.pass ? pass : fail)),
        borderRadius: 0,
        barThickness: 14,
      }],
    },
    options: {
      ...commonOptions,
      indexAxis: "y",
      scales: {
        x: { min: 0, max: 1, ticks: { display: false }, grid: { color: gridColor } },
        y: { ticks: { color: tickColor, font: { size: 10, family: monoFont } }, grid: { display: false } },
      },
      plugins: {
        ...commonOptions.plugins,
        tooltip: {
          callbacks: { label: (ctx) => (ctx.raw ? "Pass" : "Fail") },
        },
      },
    },
  });
  return { destroy: () => chart.destroy() };
}

export function mountCountsChart(canvas, groups) {
  const gridColor = cssVar("--hairline-2") || "rgba(255,255,255,0.07)";
  const tickColor = cssVar("--ink-dim") || "#8E9699";
  const monoFont = cssVar("--font-mono") || "monospace";

  const chart = new Chart(canvas, {
    type: "bar",
    data: {
      labels: groups.map((g) => g.name),
      datasets: [{
        data: groups.map((g) => g.count),
        backgroundColor: groups.map((g) => resolveColor(g.color)),
        borderRadius: 0,
        barThickness: 18,
      }],
    },
    options: {
      ...commonOptions,
      scales: {
        x: { ticks: { color: tickColor, font: { size: 10, family: monoFont } }, grid: { display: false } },
        y: { beginAtZero: true, ticks: { color: tickColor, precision: 0 }, grid: { color: gridColor } },
      },
      plugins: {
        ...commonOptions.plugins,
        tooltip: { callbacks: { label: (ctx) => `${ctx.parsed.y} detections` } },
      },
    },
  });
  return { destroy: () => chart.destroy() };
}
