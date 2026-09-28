// api-client.js - the ONLY module in this app that calls fetch().
// Every screen and every view-model goes through the functions exported
// here. Nothing else may reach the network. Raw backend field names and
// values (gate keys, decision kinds, similarity scores) pass through
// unchanged here; renaming/hiding/deriving happens one layer up, in the
// view-models under js/viewmodels/.

const BASE = "";

class ApiError extends Error {
  constructor(status, detail, path) {
    super(`${status} ${detail || ""}`.trim());
    this.status = status;
    this.detail = detail;
    this.path = path;
  }
}

async function request(method, path, { params, body } = {}) {
  let url = BASE + path;
  if (params) {
    const qs = new URLSearchParams();
    for (const [k, v] of Object.entries(params)) {
      if (v === undefined || v === null || v === "") continue;
      qs.set(k, v);
    }
    const s = qs.toString();
    if (s) url += (url.includes("?") ? "&" : "?") + s;
  }
  let res;
  try {
    res = await fetch(url, {
      method,
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    });
  } catch (e) {
    throw new ApiError(0, e.message, path);
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const j = await res.json();
      detail = j.detail || detail;
    } catch (_) { /* body wasn't JSON */ }
    throw new ApiError(res.status, detail, path);
  }
  const ct = res.headers.get("content-type") || "";
  if (ct.includes("application/json")) return res.json();
  return res; // caller wants the raw Response (image endpoints -> use .url instead)
}

const get = (path, params) => request("GET", path, { params });
const post = (path, body) => request("POST", path, { body: body || {} });
const put = (path, body) => request("PUT", path, { body: body || {} });
const del = (path) => request("DELETE", path);

export const api = {
  ApiError,

  // -- health / stats / regions -----------------------------------------
  health: () => get("/health"),
  stats: () => get("/stats"),
  regions: () => get("/regions"),
  presentationSummary: () => get("/presentation/summary"),

  // -- candidates ----------------------------------------------------------
  listCandidates: (params) => get("/candidates", params),
  getCandidate: (id) => get(`/candidates/${encodeURIComponent(id)}`),
  candidateImageryUrl: (id, { date, view = "rgb", scale } = {}) => {
    const qs = new URLSearchParams();
    if (date) qs.set("date", date);
    qs.set("view", view);
    if (scale) qs.set("scale", scale);
    return `${BASE}/candidates/${encodeURIComponent(id)}/imagery?${qs.toString()}`;
  },
  postDecision: (id, { decision, note, analyst }) =>
    post(`/candidates/${encodeURIComponent(id)}/decision`, { decision, note, analyst }),
  similarToCandidate: (id, k) => get(`/candidates/${encodeURIComponent(id)}/similar`, { k }),

  // -- audit -----------------------------------------------------------
  audit: (params) => get("/audit", params),

  // -- search -----------------------------------------------------------
  searchText: (q, params) => get("/search/text", { q, ...params }),
  searchImage: (body) => post("/search/image", body),
  tileThumbnailUrl: (tileId) => `${BASE}/tile/${encodeURIComponent(tileId)}/thumbnail`,

  // -- discovery -----------------------------------------------------------
  discoveryClusters: () => get("/discovery/clusters"),
  discoverySimilar: (params) => get("/discovery/similar", params),

  // -- object detection -----------------------------------------------------------
  detectModelInfo: () => get("/detect/model-info"),
  detectObservations: () => get("/detect/observations"),
  detectTiles: (obsId) => get(`/detect/observations/${encodeURIComponent(obsId)}/tiles`),
  detectTileDetections: (obsId, row, col) =>
    get(`/detect/observations/${encodeURIComponent(obsId)}/tiles/${row}/${col}`),
  detectTileImageUrl: (obsId, row, col) =>
    `${BASE}/detect/observations/${encodeURIComponent(obsId)}/tiles/${row}/${col}/image.png`,

  // -- watch areas -----------------------------------------------------------
  listWatchAreas: (activeOnly) => get("/watch-areas", { active_only: activeOnly }),
  createWatchArea: (body) => post("/watch-areas", body),
  updateWatchArea: (id, body) => put(`/watch-areas/${encodeURIComponent(id)}`, body),
  deleteWatchArea: (id) => del(`/watch-areas/${encodeURIComponent(id)}`),
  listNotifications: (params) => get("/notifications", params),
  markNotificationSeen: (id) => post(`/notifications/${encodeURIComponent(id)}/seen`),

  // -- sector brief -----------------------------------------------------------
  sectorBrief: (params) => get("/sector-brief", params),
};
