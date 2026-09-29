// intro.js - plays the one-time boot sequence in #intro. The overlay itself
// is already painted from pure HTML/CSS (css/intro.css) before this module
// even loads, so there's no flash of blank page while JS boots; this module
// only ticks the status line and decides when to fade the overlay out.
const STATUS_LINES = [
  "MOUNTING LOCAL ARCHIVE",
  "LOADING RETRIEVAL INDEX",
  "PREPARING WORKSPACE",
  "READY",
];
const MIN_DISPLAY_MS = 1150;
const STATUS_INTERVAL_MS = 260;

export function runIntro() {
  const el = document.getElementById("intro");
  if (!el) return;

  const reducedMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
  if (reducedMotion) {
    el.classList.add("intro-hide");
    return;
  }

  const statusEl = document.getElementById("intro-status");
  let i = 0;
  if (statusEl) statusEl.textContent = STATUS_LINES[0];
  const tick = setInterval(() => {
    i = Math.min(i + 1, STATUS_LINES.length - 1);
    if (statusEl) statusEl.textContent = STATUS_LINES[i];
    if (i === STATUS_LINES.length - 1) clearInterval(tick);
  }, STATUS_INTERVAL_MS);

  let done = false;
  function finish() {
    if (done) return;
    done = true;
    clearInterval(tick);
    el.classList.add("intro-hide");
  }

  el.addEventListener("click", finish, { once: true });
  document.addEventListener("keydown", finish, { once: true });
  setTimeout(finish, MIN_DISPLAY_MS);
}
