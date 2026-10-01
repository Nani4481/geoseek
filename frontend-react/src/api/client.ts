// Thin same-origin fetch wrappers. The console only ever talks to the backend that served it: relative paths, no
// base URL, no credentials. (The page CSP also refuses anything cross-origin.)
import type {
  BBox, CandidateDetail, CandidateList, ClusterInfo, ConsoleMetrics, DetectModelInfo, DetectObservation, DetectTile,
  HealthInfo, LatencyProbe, Notification, PresentationSummary, RegionListItem, SearchResponse, SimilarResponse,
  TileDetections, TileFootprint, Timeline, Decision,
} from './types';

export class ApiError extends Error {
  constructor(public status: number, message: string) { super(message); }
}

type Params = Record<string, string | number | boolean | null | undefined>;

function qs(params?: Params): string {
  if (!params) return '';
  const u = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) if (v !== undefined && v !== null && v !== '') u.set(k, String(v));
  const s = u.toString();
  return s ? `?${s}` : '';
}

async function parse<T>(r: Response): Promise<T> {
  if (!r.ok) {
    let detail = r.statusText;
    try { const j = await r.json(); detail = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail ?? j); } catch { /* keep statusText */ }
    throw new ApiError(r.status, detail || `HTTP ${r.status}`);
  }
  return r.json() as Promise<T>;
}

export const getJSON = <T,>(path: string, params?: Params, signal?: AbortSignal): Promise<T> =>
  fetch(path + qs(params), { signal, credentials: 'omit' }).then((r) => parse<T>(r));

export const postJSON = <T,>(path: string, body: unknown, signal?: AbortSignal): Promise<T> =>
  fetch(path, {
    method: 'POST', credentials: 'omit', signal,
    headers: { 'content-type': 'application/json' }, body: JSON.stringify(body),
  }).then((r) => parse<T>(r));

export const bboxParam = (b: BBox | null | undefined): string | undefined => (b ? b.map((x) => x.toFixed(5)).join(',') : undefined);

export interface CandidateQuery {
  bbox?: BBox | null; change_type?: string; min_confidence?: number; persistence?: string; decision?: string;
  sort?: string; limit?: number; offset?: number; year?: string; sensor?: string;
}

export const api = {
  health: (s?: AbortSignal) => getJSON<HealthInfo>('/health', undefined, s),
  metrics: (s?: AbortSignal) => getJSON<ConsoleMetrics>('/ui/metrics', undefined, s),
  latency: (s?: AbortSignal) => getJSON<LatencyProbe>('/ui/latency', undefined, s),
  presentation: (s?: AbortSignal) => getJSON<PresentationSummary>('/presentation/summary', undefined, s),
  regions: (s?: AbortSignal) => getJSON<{ regions: RegionListItem[] }>('/regions', undefined, s),
  restrictedZones: (s?: AbortSignal) =>
    getJSON<{ zones: { name: string; level: string; min_lon: number; max_lon: number; min_lat: number; max_lat: number }[] }>('/restricted-zones', undefined, s),
  notifications: (s?: AbortSignal) => getJSON<{ notifications: Notification[] }>('/notifications', undefined, s),

  candidates: (q: CandidateQuery, s?: AbortSignal) =>
    getJSON<CandidateList>('/candidates', { ...q, bbox: bboxParam(q.bbox) }, s),
  candidate: (id: string, s?: AbortSignal) => getJSON<CandidateDetail>(`/candidates/${encodeURIComponent(id)}`, undefined, s),
  timeline: (id: string, s?: AbortSignal) => getJSON<Timeline>(`/ui/candidates/${encodeURIComponent(id)}/timeline`, undefined, s),
  similarToCandidate: (id: string, k: number, s?: AbortSignal) =>
    getJSON<SimilarResponse>(`/candidates/${encodeURIComponent(id)}/similar`, { k }, s),
  decide: (id: string, decision: 'confirm' | 'reject' | 'reopen', note: string, analyst: string) =>
    postJSON<Decision>(`/candidates/${encodeURIComponent(id)}/decision`, { decision, note, analyst }),
  audit: (limit: number, s?: AbortSignal) => getJSON<{ count: number; append_only: boolean; decisions: Decision[] }>('/audit', { limit }, s),
  exportGeoJSON: (candidate_ids?: string[], filters?: Record<string, unknown>) =>
    postJSON<{ count: number; geojson: unknown; geojson_path?: string }>('/export', { candidate_ids, filters, format: 'geojson' }),

  searchText: (p: { q: string; k: number; bbox?: BBox | null; date_start?: string; date_end?: string; max_cloud_fraction?: number }, s?: AbortSignal) =>
    getJSON<SearchResponse>('/search/text', { ...p, bbox: bboxParam(p.bbox) }, s),
  searchImage: (p: { tile_id: string; k: number; bbox?: BBox | null }) =>
    postJSON<SearchResponse>('/search/image', { tile_id: p.tile_id, k: p.k, bbox: bboxParam(p.bbox) }),
  similarAt: (lon: number, lat: number, k: number, s?: AbortSignal) =>
    getJSON<SimilarResponse>('/discovery/similar', { lon, lat, k }, s),
  similarTile: (tile_id: string, k: number, s?: AbortSignal) =>
    getJSON<SimilarResponse>('/discovery/similar', { tile_id, k }, s),
  tiles: (ids: string[], s?: AbortSignal) => getJSON<{ tiles: TileFootprint[] }>('/ui/tiles', { ids: ids.join(',') }, s),

  clusters: (s?: AbortSignal) => getJSON<ClusterInfo>('/discovery/clusters', undefined, s),

  detectModel: (s?: AbortSignal) => getJSON<DetectModelInfo>('/detect/model-info', undefined, s),
  detectObservations: (s?: AbortSignal) => getJSON<{ observations: DetectObservation[] }>('/detect/observations', undefined, s),
  detectTiles: (obs: string, s?: AbortSignal) => getJSON<{ tiles: DetectTile[] }>(`/detect/observations/${encodeURIComponent(obs)}/tiles`, undefined, s),
  detectTile: (obs: string, row: number, col: number, s?: AbortSignal) =>
    getJSON<TileDetections>(`/detect/observations/${encodeURIComponent(obs)}/tiles/${row}/${col}`, undefined, s),
};

// ---- URL builders for images the backend renders ----
export const tileThumb = (tileId: string) => `/tile/${encodeURIComponent(tileId)}/thumbnail`;
export const candidateImage = (id: string, year: string, view: 'rgb' | 'overlay', scale = 3) =>
  `/candidates/${encodeURIComponent(id)}/imagery?date=${encodeURIComponent(year)}&view=${view}&scale=${scale}`;
export const detectTileImage = (obs: string, row: number, col: number) =>
  `/detect/observations/${encodeURIComponent(obs)}/tiles/${row}/${col}/image.png`;
export const CLUSTER_MAP_URL = '/discovery/cluster-map.png';
