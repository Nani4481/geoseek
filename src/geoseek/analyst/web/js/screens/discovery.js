import { api } from "../api-client.js";
import { registerScreen, go } from "../router.js";
import { loadRegions } from "../regions.js";
import { toDiscoveryViewModel } from "../viewmodels/discovery.js";
import { setBrowseOrder } from "./candidate.js";

let root;
let seedCandidateId = null;

function mount() {
  root = document.getElementById("screen-discovery");
  root.classList.add("ds-root");
}

function tileCardHtml(r) {
  return `
    <div class="ds-tile" data-tile="${r.tileId}">
      <div class="ds-tile-img"><img src="${api.tileThumbnailUrl(r.tileId)}" alt="" loading="lazy"></div>
      <div class="t-meta" style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${r.region}</div>
      <div class="t-small" style="color:var(--ink-dim);">${r.date}</div>
    </div>`;
}

function tierHtml(tier) {
  return `
    <div class="ds-tier">
      <div class="ds-tier-head">
        <span class="t-section-title" style="font-size:14px;">${tier.title}</span>
        <span class="t-micro">${tier.results.length} LOCATIONS · ${tier.spread}</span>
      </div>
      <hr class="hairline-rule">
      <div class="t-small" style="color:var(--ink-3); margin:6px 0 10px;">${tier.rationale}</div>
      <div class="ds-tile-grid">
        ${tier.results.length ? tier.results.map(tileCardHtml).join("") :
          `<div class="t-small" style="color:var(--ink-ghost);">None found.</div>`}
      </div>
    </div>`;
}

async function loadForSeed(candidateId) {
  seedCandidateId = candidateId;
  root.innerHTML = `<div class="loading-state">Finding look-alike locations…</div>`;
  try {
    const [detail, regions] = await Promise.all([api.getCandidate(candidateId), loadRegions(api)]);
    const [lon, lat] = detail.centroid_lonlat;
    const similarBody = await api.discoverySimilar({ lon, lat, k: 24 });
    const seedResult = { lon, lat, centroid_lonlat: detail.centroid_lonlat };
    const vm = toDiscoveryViewModel(seedResult, similarBody, regions);
    const seedRegion = vm.seedRegion;
    const areaHa = ((detail.area_m2 || 0) / 10000).toFixed(1);

    root.innerHTML = `
      <div class="ds-head fixed">
        <span class="t-screen-title">Look-alike locations</span>
        <div class="t-body" style="color:var(--ink-3); margin-top:4px;">
          Pick a place you care about and the system pulls everywhere else in the archive that looks like it.
        </div>
      </div>
      <div class="ds-body scroll-pane">
        <div class="ds-layout">
          <div class="panel ds-start">
            <div class="panel-header t-eyebrow" style="color:var(--cyan);">Starting point</div>
            <div class="ds-start-img"><img src="${api.candidateImageryUrl(candidateId, { date: detail.imagery.after_date, view: "rgb" })}" alt=""></div>
            <div style="padding:10px 12px;">
              <div class="t-row-title">${seedRegion}</div>
              <div class="t-meta">${candidateId} · ${detail.imagery.after_date}</div>
              <div class="t-small" style="color:var(--ink-3); margin-top:8px;">
                You are matching on a ${areaHa} hectare footprint of this change type.
              </div>
              <button class="btn tone-cyan" id="ds-change-start" style="margin-top:10px; width:100%;">Use a different starting point</button>
            </div>
          </div>
          <div class="ds-tiers">${vm.tiers.map(tierHtml).join("")}</div>
        </div>
      </div>`;

    root.querySelector("#ds-change-start").addEventListener("click", pickRandomSeed);
    root.querySelectorAll(".ds-tile").forEach((el) => {
      // discovery tiles are search-index tile records, not change candidates -
      // there is no candidate to open, so this just cross-links back to Search.
      el.addEventListener("click", () => go("search"));
    });
  } catch (e) {
    root.innerHTML = `<div class="error-state"><div class="error-state-msg">Could not load look-alike locations.</div>
      <button class="btn" id="ds-retry">Retry</button>
      <details style="margin-top:10px;"><summary class="t-small">Details</summary><div class="error-detail">${e.message}</div></details></div>`;
    root.querySelector("#ds-retry")?.addEventListener("click", () => loadForSeed(candidateId));
  }
}

async function pickRandomSeed() {
  try {
    const body = await api.listCandidates({ sort: "queue_score", limit: 20 });
    const pool = body.candidates.filter((c) => c.candidate_id !== seedCandidateId);
    const pick = pool[Math.floor(Math.random() * pool.length)] || body.candidates[0];
    if (pick) loadForSeed(pick.candidate_id);
  } catch (e) { /* keep current seed on failure */ }
}

async function show() {
  if (seedCandidateId) return loadForSeed(seedCandidateId);
  try {
    const pres = await api.presentationSummary();
    const seed = pres.demo && pres.demo.discovery_seed;
    if (seed) return loadForSeed(seed);
    const body = await api.listCandidates({ sort: "queue_score", limit: 1 });
    if (body.candidates[0]) return loadForSeed(body.candidates[0].candidate_id);
    root.innerHTML = `<div class="empty-state">No candidates available to start from.</div>`;
  } catch (e) {
    root.innerHTML = `<div class="error-state"><div class="error-state-msg">Could not load Discovery.</div>
      <details><summary class="t-small">Details</summary><div class="error-detail">${e.message}</div></details></div>`;
  }
}

registerScreen("discovery", { mount, show });
