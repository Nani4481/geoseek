import { api } from "../api-client.js";
import { registerScreen, go } from "../router.js";
import { currentAoi } from "../shell.js";

let root;

function mount() {
  root = document.getElementById("screen-watch");
  root.classList.add("wa-root");
}

async function cardData(w) {
  const { notifications } = await api.listNotifications({ watch_id: w.watch_id });
  const unseen = notifications.filter((n) => !n.seen);
  let largestArea = 0, candidateCount = 0;
  notifications.forEach((n) => {
    candidateCount += n.candidates.length;
    n.candidates.forEach((c) => { if (c.confidence) largestArea = Math.max(largestArea, c.confidence); });
  });
  return { watch: w, notifications, unseenCount: unseen.length, candidateCount };
}

function cardHtml(d) {
  const needsLook = d.unseenCount >= 1;
  const summary = d.unseenCount > 0
    ? `${d.unseenCount} new notification${d.unseenCount === 1 ? "" : "s"} since you last looked, matching ${d.candidateCount} candidate${d.candidateCount === 1 ? "" : "s"} in total.`
    : `Nothing new since your last review. ${d.notifications.length} notification${d.notifications.length === 1 ? "" : "s"} logged in total.`;
  // the design's fixed "12 passes" activity bar chart has no backing
  // endpoint (no per-pass cadence is recorded) - this lists the real
  // notification history instead of fabricating a bar-per-pass chart.
  const historyHtml = d.notifications.slice(-6).reverse().map((n) =>
    `<div class="wa-hist-row" title="${n.observation_date || n.created_at} · ${n.candidates.length} candidate(s)">
      <span class="t-meta">${n.observation_date || n.created_at.slice(0, 10)}</span>
      <span class="pill band-${n.severity === "high" ? "low" : n.severity === "medium" ? "medium" : "high"}">${n.severity.toUpperCase()}</span>
      <span class="t-small" style="color:var(--ink-dim);">${n.candidates.length} candidate(s)</span>
    </div>`).join("") || `<div class="t-small" style="color:var(--ink-ghost);">No notifications logged yet.</div>`;

  return `
    <div class="wa-card panel" style="border-color:${needsLook ? "rgba(240,160,48,0.3)" : "rgba(255,255,255,0.1)"};" data-watch="${d.watch.watch_id}">
      <div class="wa-card-top">
        <div class="wa-card-thumb"></div>
        <div class="min0" style="flex:1;">
          <div style="display:flex; justify-content:space-between; align-items:center; gap:8px;">
            <span class="t-card-title" style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${d.watch.name}</span>
            <span class="pill ${needsLook ? "band-medium" : "band-high"}">${needsLook ? "NEEDS A LOOK" : "STEADY"}</span>
          </div>
          <div class="t-meta">${d.watch.bbox ? d.watch.bbox.map((v) => v.toFixed(2)).join(", ") : "no bbox set"}</div>
          <div class="t-small" style="color:var(--ink-3); margin-top:4px;">${summary}</div>
        </div>
      </div>
      <div class="wa-history">
        <div class="t-eyebrow" style="margin-bottom:4px;">Notification history</div>
        ${historyHtml}
      </div>
      <div class="wa-card-footer">
        <button class="btn flex1" data-action="review" data-watch="${d.watch.watch_id}">Review ${d.unseenCount > 0 ? d.unseenCount + " new" : "queue"}</button>
        <button class="btn tone-red" data-action="delete" data-watch="${d.watch.watch_id}">Delete</button>
      </div>
    </div>`;
}

function formHtml() {
  const aoi = currentAoi();
  return `
    <div class="wa-form panel">
      <div class="panel-header t-panel-title">Add area from map</div>
      <div style="padding:12px;">
        <input id="wa-name" class="wa-input" placeholder="Watch area name" value="${aoi ? aoi.name + " watch" : ""}">
        <input id="wa-minconf" class="wa-input" placeholder="Minimum confidence (0-1, optional)">
        <div class="t-small" style="color:var(--ink-dim); margin:6px 0;">
          Uses the current top-bar area of interest as the bbox: ${aoi ? aoi.bbox.map((v) => v.toFixed(3)).join(", ") : "none selected"}.
        </div>
        <button class="btn tone-cyan" id="wa-create">Create watch area</button>
      </div>
    </div>`;
}

async function refresh() {
  const list = document.getElementById("wa-list");
  list.innerHTML = `<div class="loading-state">Loading watch areas…</div>`;
  try {
    const body = await api.listWatchAreas();
    if (!body.watch_areas.length) {
      list.innerHTML = `<div class="empty-state"><div class="empty-state-title">No watch areas yet</div>
        Add one from the form above to start standing observation over an area.</div>`;
      return;
    }
    const datas = await Promise.all(body.watch_areas.map(cardData));
    list.innerHTML = datas.map(cardHtml).join("");
    list.querySelectorAll('[data-action="review"]').forEach((b) =>
      b.addEventListener("click", async () => {
        const { notifications: notifs } = await api.listNotifications({ watch_id: b.dataset.watch, unseen_only: true });
        await Promise.all(notifs.map((n) => api.markNotificationSeen(n.notification_id)));
        go("queue");
      }));
    list.querySelectorAll('[data-action="delete"]').forEach((b) =>
      b.addEventListener("click", async () => {
        await api.deleteWatchArea(b.dataset.watch);
        refresh();
      }));
  } catch (e) {
    list.innerHTML = `<div class="error-state"><div class="error-state-msg">Could not load watch areas.</div>
      <button class="btn" id="wa-retry">Retry</button>
      <details style="margin-top:10px;"><summary class="t-small">Details</summary><div class="error-detail">${e.message}</div></details></div>`;
    document.getElementById("wa-retry")?.addEventListener("click", refresh);
  }
}

async function show() {
  root.innerHTML = `
    <div class="wa-head fixed">
      <span class="t-screen-title">Watch Areas</span>
      <div class="t-body" style="color:var(--ink-3); margin-top:4px;">Places kept under standing observation. Each card answers one question: what changed since you last looked.</div>
    </div>
    <div class="wa-body scroll-pane">
      ${formHtml()}
      <div class="wa-list" id="wa-list"></div>
    </div>`;
  document.getElementById("wa-create").addEventListener("click", async () => {
    const aoi = currentAoi();
    const name = document.getElementById("wa-name").value.trim();
    if (!name || !aoi) return;
    const minConf = parseFloat(document.getElementById("wa-minconf").value);
    try {
      await api.createWatchArea({ name, bbox: aoi.bbox, min_confidence: isNaN(minConf) ? null : minConf });
      refresh();
    } catch (e) { /* surfaced via the list's own error state on next refresh */ }
  });
  refresh();
}

registerScreen("watch", { mount, show });
