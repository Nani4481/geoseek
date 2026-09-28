/* components.js — Phase 2A shared component layer (behaviour half; markup/
   colour rules are in components.css). Classic script, loaded after tokens.js
   and before app.js, exposing window.Components the same way tokens.js
   exposes window.Tokens.

   Every component here is presentational + the minimum behaviour it needs
   (collapse memory, sort icon, delayed in-flight reveal, one-time count-up).
   None of it owns application state or does its own fetching - callers
   (app.js) still decide what data to show; these functions only decide how
   to show it consistently. */
"use strict";

(function () {
  const esc = (s) => String(s == null ? "" : s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  // ============================================================ ICONS
  // Inline SVG only, one shared 24x24 viewBox, single stroke weight - see
  // docs/REDESIGN_PHASE2A_REPORT.md for the full audit of which of the seven
  // screens use which icon and which existing glyphs were reused vs. new.
  // Every path below already existed somewhere in index.html/app.js before
  // this phase EXCEPT "chevron" and "close", the two new ones the component
  // set actually needed (panel collapse / sort direction, and the reject
  // button / dismiss affordance).
  const ICONS = {
    overview: '<rect x="3" y="3" width="7" height="9" rx="1.5"/><rect x="14" y="3" width="7" height="5" rx="1.5"/><rect x="14" y="12" width="7" height="9" rx="1.5"/><rect x="3" y="16" width="7" height="5" rx="1.5"/>',
    search: '<circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/>',
    queue: '<path d="M4 6h16M4 12h16M4 18h10"/>',
    detail: '<rect x="3" y="4" width="18" height="14" rx="2"/><path d="M3 15l4.5-4.5L11 14l3-3L21 15"/>',
    discovery: '<circle cx="7" cy="7" r="2.5"/><circle cx="17" cy="6" r="2"/><circle cx="16" cy="16" r="2.5"/><circle cx="7" cy="16" r="1.7"/><path d="M9 8l5.5-1M8.5 9l5.7 5.5M8.3 15l-.6-6.5"/>',
    detect: '<path d="M3 8V5a2 2 0 0 1 2-2h3M21 8V5a2 2 0 0 0-2-2h-3M3 16v3a2 2 0 0 0 2 2h3M21 16v3a2 2 0 0 1-2 2h-3"/><rect x="8" y="8" width="8" height="8" rx="1"/>',
    watch: '<path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7-10-7-10-7Z"/><circle cx="12" cy="12" r="3"/>',
    play: '<path d="M6 4l14 8-14 8V4Z"/>',
    alerts: '<path d="M12 3v4M12 21v-4M3 12h4M21 12h-4M5.6 5.6l2.8 2.8M18.4 5.6l-2.8 2.8M5.6 18.4l2.8-2.8M18.4 18.4l-2.8-2.8"/><circle cx="12" cy="12" r="3"/>',
    layers: '<path d="m12 3-9 5 9 5 9-5-9-5Z"/><path d="m3 13 9 5 9-5"/>',       // consolidated: replaces two different "layers" glyphs (panel header + map control) with one
    featured: '<path d="M11 2 3 14h7l-1 8 10-14h-7l1-6Z"/>',
    check: '<path d="M20 6 9 17l-5-5"/>',
    grid: '<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18M9 21V9"/>',
    home: '<path d="M3 11.5 12 4l9 7.5"/><path d="M5 10v9h14v-9"/>',
    warning: '<path d="M12 3 2 20h20L12 3Z"/><path d="M12 10v4M12 17.5v.01"/>',
    chevron: '<path d="m6 9 6 6 6-6"/>',           // new: panel collapse, sort direction
    close: '<path d="M6 6l12 12M18 6 6 18"/>',      // new: reject button, dismiss
  };

  function icon(name, opts) {
    opts = opts || {};
    const body = ICONS[name];
    if (!body) throw new Error(`[components.js] unknown icon "${name}"`);
    const sizeClass = opts.size === 20 ? " sz20" : opts.size === 24 ? " sz24" : "";
    return `<svg class="c-icon${sizeClass}${opts.cls ? " " + opts.cls : ""}" viewBox="0 0 24 24">${body}</svg>`;
  }

  // ============================================================ CONFIDENCE
  // Thresholds are NOT invented here - they are copied from the backend's
  // own geoseek/analyst/service.py:63-65 (_confidence_band): High >= 0.85,
  // Medium >= 0.60, else Low. The backend already ships a precomputed
  // "confidence_band" string on /presentation/summary's featured cards;
  // everywhere else in the product (the Review Queue table, Candidate
  // Detail's headline figure) only ships the raw float, so the frontend has
  // to recompute the same band client-side. This is the ONE place that
  // happens now - app.js's old, separate confBand()/RING_COL was deleted and
  // every caller redirected here, closing a second (harmless, since the
  // numbers already matched) copy of the same thresholds down to one.
  function confBand(v) {
    v = +v || 0;
    return v >= 0.85 ? "High" : v >= 0.60 ? "Medium" : "Low";
  }
  const BAND_TOKEN = { High: "success", Medium: "warning", Low: "danger" };

  // Accepts any casing ("low", "LOW", "Low") - severity fields elsewhere in
  // the product (watch-area notifications) are lowercase, confidence bands
  // are already "High"/"Medium"/"Low"; normalizing here means every caller
  // gets the same three CSS states regardless of which one it's showing.
  //
  // "High" means opposite things for the two things this pill shows: high
  // CONFIDENCE is good (green); high notification SEVERITY is bad (red).
  // opts.kind picks which polarity applies - "confidence" (default) or
  // "severity". Getting this wrong silently shows a "High severity" alert in
  // the same green as a "High confidence" candidate, which is worse than a
  // cosmetic mismatch: it tells an analyst the opposite of what it means.
  function bandPill(band, opts) {
    opts = opts || {};
    const norm = String(band || "").toLowerCase();
    const label = norm.charAt(0).toUpperCase() + norm.slice(1);
    const title = opts.title ? ` title="${esc(opts.title)}"` : "";
    const kindAttr = opts.kind === "severity" ? ' data-kind="severity"' : "";
    return `<span class="c-band" data-band="${label}"${kindAttr}${title}>${esc(label)}</span>`;
  }

  function confRing(value, opts) {
    opts = opts || {};
    const pct = Math.round((+value || 0) * 100);
    const band = confBand(value);
    const sizeClass = opts.size === "lg" ? " lg" : opts.size === "xs" ? " xs" : "";
    return `<div class="c-conf-ring${sizeClass}" style="--c-pct:${pct};--c-band-color:var(--${BAND_TOKEN[band]})" role="img" aria-label="confidence ${pct}% (${band})">
      <div class="c-conf-ring__arc"></div>
      <div class="c-conf-ring__hole"><span class="c-conf-ring__num">${pct}</span></div>
    </div>`;
  }

  function confBar(value) {
    const pct = Math.round((+value || 0) * 100);
    const band = confBand(value);
    return `<div class="c-conf-bar" role="img" aria-label="confidence ${pct}% (${band})">
      <div class="c-conf-bar__row"><span class="c-conf-bar__num">${pct}%</span>${bandPill(band)}</div>
      <div class="c-conf-bar__track"><div class="c-conf-bar__fill" style="--c-pct-width:${pct}%;--c-band-color:var(--${BAND_TOKEN[band]})"></div></div>
    </div>`;
  }

  // ============================================================ CHANGE-TYPE BADGE
  // Defaults to the raw type string (e.g. "water_gain"), matching exactly
  // what the markup this replaces showed at every mechanical-swap call site
  // this phase - callers that want a humanised label (app.js already has its
  // own CTYPE_HUMAN for that) pass opts.label explicitly. Not defaulting to a
  // humanised label here is deliberate: a "mechanical swap" must not quietly
  // change what data is shown.
  function changeBadge(type, opts) {
    opts = opts || {};
    const label = opts.label || type;
    return `<span class="c-badge" data-type="${esc(type)}"><i class="c-badge__dot"></i>${esc(label)}</span>`;
  }

  // ============================================================ EMPTY STATE
  // Copy for the four cases the brief names - terse, operational, no filler.
  function emptyState({ icon: iconName, cause, action }) {
    return `<div class="c-empty">${icon(iconName, { size: 24 })}<div class="c-empty__cause">${esc(cause)}</div><div class="c-empty__action">${esc(action)}</div></div>`;
  }
  const emptyStates = {
    noSearchResults: () => emptyState({
      icon: "search", cause: "No tiles matched that query.",
      action: "Broaden the query, widen the AOI, or clear a filter.",
    }),
    noCandidatesFiltered: () => emptyState({
      icon: "queue", cause: "No candidates match these filters.",
      action: "Lower the confidence threshold or clear the AOI.",
    }),
    noNotifications: () => emptyState({
      icon: "alerts", cause: "No notifications yet.",
      action: "Notifications fire when a re-run of the pipeline matches a watch area.",
    }),
    noWatchAreas: () => emptyState({
      icon: "watch", cause: "No watch areas defined.",
      action: "Define one on the left to start monitoring an AOI.",
    }),
  };

  // ============================================================ PANEL / CARD
  // Collapse state lives in memory for the session only (a plain Map, not
  // localStorage) - lost on reload, which is what "for the session" means
  // for a single-page app that never persists UI state across reloads today.
  const PANEL_COLLAPSE = new Map(); // panel id -> bool

  function panel({ id, iconName, title, collapsible, bodyHTML, footerHTML }) {
    const collapsed = collapsible && PANEL_COLLAPSE.get(id) === true;
    return `<div class="c-panel${collapsed ? " is-collapsed" : ""}" id="${esc(id)}">
      <div class="c-panel__header${collapsible ? " is-collapsible" : ""}" ${collapsible ? 'data-panel-toggle="' + esc(id) + '"' : ""}>
        ${iconName ? icon(iconName) : ""}<span class="c-panel__title">${esc(title)}</span>
        ${collapsible ? `<span class="c-panel__collapse">${icon("chevron")}</span>` : ""}
      </div>
      <div class="c-panel__body">${bodyHTML || ""}</div>
      ${footerHTML ? `<div class="c-panel__footer">${footerHTML}</div>` : ""}
    </div>`;
  }
  // Event delegation, wired once - so panels created/replaced later (innerHTML
  // re-renders, which this codebase does constantly) don't need their own
  // listener re-attached every time, unlike CoordMap's one-time-only pattern.
  function initPanelCollapse(root) {
    (root || document).addEventListener("click", (e) => {
      const header = e.target.closest("[data-panel-toggle]");
      if (!header) return;
      const id = header.dataset.panelToggle;
      const el = document.getElementById(id);
      if (!el) return;
      const next = !el.classList.contains("is-collapsed");
      el.classList.toggle("is-collapsed", next);
      PANEL_COLLAPSE.set(id, next);
    });
  }
  // Programmatic override - e.g. the guided demo resets disclosure panels to
  // a known (open) state before each walkthrough step, regardless of
  // whatever an analyst left them at earlier (was previously done by setting
  // a native <details>.open = false directly).
  function setPanelCollapsed(id, collapsed) {
    const el = document.getElementById(id);
    if (!el) return;
    el.classList.toggle("is-collapsed", collapsed);
    PANEL_COLLAPSE.set(id, collapsed);
  }

  // ============================================================ STAT / COUNTER
  // Ports app.js's existing animateCount() (same ease-out cubic, same
  // "only animate on first render" contract) - the exact function, not a
  // reimplementation, so Overview's counters keep behaving identically.
  const nf = (n) => (n == null ? "–" : Number(n).toLocaleString());
  function animateStatValue(el, target, ms) {
    if (!ms || !target || (window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches)) {
      el.textContent = nf(target);
      return;
    }
    const t0 = performance.now();
    const ease = (x) => 1 - Math.pow(1 - x, 3);
    const tick = (now) => {
      const p = Math.min(1, (now - t0) / ms);
      el.textContent = nf(Math.round(target * ease(p)));
      if (p < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }
  function statCard({ id, label, value, accent, delta }) {
    const deltaHTML = delta == null ? "" :
      `<span class="c-stat__delta" data-sign="${delta >= 0 ? "up" : "down"}">${delta >= 0 ? "+" : ""}${nf(delta)}</span>`;
    return `<div class="c-stat${accent ? " c-stat--accent" : ""}">
      <div class="c-stat__value" id="${esc(id)}" data-target="${Number(value) || 0}">0</div>
      <div class="c-stat__label">${esc(label)}${deltaHTML}</div>
    </div>`;
  }

  // ============================================================ HINT ("?" tooltip)
  // Plain-language help, everywhere a column header / metric / filter label
  // could otherwise leave a non-expert guessing what a word means. Offline
  // only - the browser's own native `title` attribute, no tooltip library.
  // A tiny focusable glyph (not just an invisible title on the label text
  // itself) so the affordance is visible without hovering first.
  function hint(text) {
    return `<button type="button" class="c-hint" tabindex="0" title="${esc(text)}" aria-label="${esc(text)}">?</button>`;
  }

  // ============================================================ MINI BAR
  // A plain, non-confidence score shown as a small bar instead of a bare
  // decimal (e.g. Significance, Priority in the Review Queue) - neutral
  // "info" colour, never the confidence semantic ramp, so it's never
  // mistaken for a good/bad judgement the way the confidence ring is.
  function miniBar(value01, opts) {
    opts = opts || {};
    const pct = Math.round(Math.max(0, Math.min(1, +value01 || 0)) * 100);
    return `<div class="c-minibar" role="img" aria-label="${esc(opts.label || "score")} ${pct}%">
      <span class="c-minibar__num">${num2(value01)}</span>
      <div class="c-minibar__track"><div class="c-minibar__fill" style="--c-pct-width:${pct}%"></div></div>
    </div>`;
  }
  const num2 = (v) => (v == null || isNaN(v)) ? "–" : Number(v).toFixed(2);

  // ============================================================ LATENCY BADGE
  // In-flight state is deliberately delayed ~120ms before it appears -
  // requests on this stack complete in tens of ms (see
  // docs/REDESIGN_PHASE1_REPORT.md's verify_offline_perf.py numbers), so an
  // undelayed badge would flash on essentially every request. 120ms matches
  // --duration-base, the token for "a state change that should read as
  // deliberate, not instant."
  function makeLatencyBadge(el) {
    let showTimer = null;
    return {
      start() {
        clearTimeout(showTimer);
        showTimer = setTimeout(() => {
          el.dataset.state = "inflight";
          el.innerHTML = `<i class="c-latency__dot"></i>fetching…`;
        }, 120);
      },
      done(label, ms) {
        clearTimeout(showTimer);
        const state = ms == null ? "ok" : ms < 1000 ? "ok" : "over";
        el.dataset.state = state;
        el.innerHTML = `<i class="c-latency__dot"></i>${esc(label)}`;
      },
    };
  }

  window.Components = {
    icon, confBand, bandPill, confRing, confBar, changeBadge,
    emptyState, emptyStates, panel, initPanelCollapse, setPanelCollapsed, animateStatValue, statCard,
    makeLatencyBadge, hint, miniBar,
  };
})();
