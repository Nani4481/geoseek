// Response shapes of the GeoSeek backend, as observed from the live API. Nothing here is a default or a fixture.

export type LonLat = [number, number];
export type BBox = [number, number, number, number]; // west, south, east, north

export interface Polygon { type: 'Polygon'; coordinates: number[][][] }

export type Verdict = 'confirm' | 'reject' | 'undecided';

export interface RestrictedZone { name: string; alert_level: 'critical' | 'warning' | string }

export interface SarSummary {
  available: boolean;
  vv_median_db?: number | null;
  verdict?: string | null;
  factor?: number | null;
}

export interface Candidate {
  rank: number;
  candidate_id: string;
  pair: string;
  centroid_lonlat: LonLat;
  area_px: number;
  area_m2: number;
  change_type: string;
  confidence: number;
  significance: number;
  queue_score: number;
  persistence: string;
  earliest_supported: [string, string];
  mean_model_prob: number;
  geometry: Polygon;
  decision: Verdict;
  gates: Record<string, boolean>;
  spectral_anomaly_max: number;
  sar: SarSummary;
  restricted_zone: RestrictedZone | null;
}

export interface CandidateList {
  total: number; count: number; offset: number; limit: number; sort: string;
  candidates: Candidate[];
}

export interface TraceItem { rule: string; verdict: 'pass' | 'fail' | string; weight: number; detail: string; values: Record<string, number | boolean | null> }

export interface Decision {
  decision_id: string; candidate_id: string; decision: 'confirm' | 'reject' | 'reopen';
  analyst_note: string; analyst: string; created_at: string;
  model_version: string; weights_sha256: string; git_commit: string; pipeline_version: string;
  confidence_at_decision: number | null;
}

export interface ProvenanceObservation {
  role: 'before' | 'after'; observation_id: string; acquired_at: string | null;
  scene: { scene_id: string; platform: string; source_url: string; license: string; processing_baseline: string; crs: string } | null;
  collection: { collection_id: string; sensor: string; platform: string; bands: string[]; native_gsd_m: number } | null;
}

export interface Terrain {
  elevation_m: number; slope_deg: number; aspect_compass: string;
  distance_to_water_m: number; distance_to_built_up_m: number; plain_language: string;
}

export interface CandidateDetail extends Candidate {
  classification: { change_type: string; rule: string; detail: string; evidence: Record<string, number> };
  confidence_breakdown: string[];
  suppression: { suppressed: boolean; suppressed_by: string | null; combined_downweight: number; trace: TraceItem[] };
  sar: SarSummary & Record<string, unknown>;
  terrain: Terrain | null;
  provenance: {
    observations: ProvenanceObservation[];
    model: { name: string; threshold: number; weights_sha256: string };
    code: { git_commit: string; pipeline_version: string };
    sar_corroboration: Record<string, unknown> | null;
    report_generated_at: string;
  };
  decisions: Decision[];
  effective_decision: Verdict;
  imagery: { before_dates: string[]; after_date: string; views: string[]; url_template: string };
}

export interface TimelinePoint {
  date: string; year: string;
  role: 'baseline' | 'before_detection' | 'first_detected' | 'supported' | 'not_supported' | 'no_change_seen';
  interval: { from: string; to: string; changed: boolean; probability: number; comparable: boolean } | null;
}
export interface Timeline {
  candidate_id: string; dates: string[]; points: TimelinePoint[];
  persistence: string; persistence_confidence: number | null;
  first_detected: { date: string; bracket: [string, string]; caveat: string | null; quality_note: string | null } | null;
  supporting_dates: string[]; n_supporting: number; n_total: number;
  notes: string[]; definition: string;
}

export interface SensorInfo { collection_id: string; sensor: string; platform: string; native_gsd_m: number; n_scenes: number }
export interface RegionInfo { name: string; bbox: BBox; n_observations: number; candidates: number; center: LonLat }

export interface ChangeModelMetrics {
  available: boolean; name?: string; operating_threshold?: number;
  f1?: number; precision?: number; recall?: number; iou?: number; f1_at_0_50?: number; validation_f1?: number;
  dataset?: string; caveat?: string; source?: string;
}
export interface DetectorMetrics {
  available: boolean;
  dota_val?: {
    small_vehicle_ap50: number; small_vehicle_ap50_ci: [number, number] | null;
    ground_vehicles_ap50: number; ground_vehicles_ap50_ci: [number, number] | null;
    all_classes_ap50: number; all_classes_ap50_ci: [number, number] | null;
    n_images: number; dataset: string; source: string;
  };
  xview_test?: { small_vehicle_ap50: number; large_vehicle_ap50: number; dataset: string; source: string };
  caveat: string;
}
export interface ConsoleMetrics {
  counters: {
    tiles_indexed: number; vectors_searchable: number | null; scenes: number; regions: number; sensors: number;
    collections: number; change_candidates: number; analyst_decisions: number;
  };
  sensors: SensorInfo[];
  findings_by_region: RegionInfo[];
  findings_by_type: Record<string, number>;
  change_pipeline_aoi: string | null;
  observation_dates: string[];
  change_model: ChangeModelMetrics;
  detector: DetectorMetrics;
  generated_at: string;
}
export interface LatencyProbe {
  median_ms: number; p95_ms: number; min_ms: number; max_ms: number; n_queries: number; k: number;
  vectors: number; measured_at: string; source: string;
}

export interface PresentationSummary {
  observation_dates: string[];
  change_type_labels: Record<string, string>;
  featured: { candidate_id: string; change_type: string; change_type_human: string; confidence: number; caption: string; centroid_lonlat: LonLat }[];
  counters: Record<string, number>;
  demo: { search_query: string; water_gain_candidate_id: string | null };
}

export interface SearchHit {
  tile_id: string; score: number; lon: number; lat: number; acq_date: string; sensor: string;
  cloud_fraction: number; scene_id: string;
}
export interface SearchResponse { query?: string; tile_id?: string | null; k: number; latency_ms: number; count: number; results: SearchHit[] }

export interface SimilarHit {
  tile_id: string; score: number; observation_id: string; acq_date: string; centroid_lonlat: LonLat; cluster: number | null;
}
export interface SimilarResponse { seed_tile_id: string; seed_candidate_id?: string; k: number; results: SimilarHit[]; latency_ms: number; seed_cluster: number | null }

export interface TileFootprint {
  tile_id: string; bbox: BBox; width_km: number; height_km: number; area_km2: number; cloud_fraction: number; observation_id: string;
}

export interface Notification {
  notification_id?: string; watch_id: string; watch_name: string | null; observation_date: string | null;
  severity: 'high' | 'medium' | 'low'; severity_score: number; seen?: boolean;
  candidates: { candidate_id: string; change_type: string | null; confidence: number | null }[];
  created_at?: string;
}

export interface DetectObservation {
  observation_id: string; aoi_name: string; role: string; n_tiles: number; n_tiles_with_detections: number;
  n_detections: number; by_class: Record<string, number>;
}
export interface DetectTile { tile_id: string; row: number; col: number; n_detections: number; by_class: Record<string, number>; mean_score: number; cloud_fraction: number }
export interface Detection { class: string; score: number; polygon_px: number[][]; long_side_px: number; heading_deg: number }
export interface TileDetections { tile_id: string; width: number; height: number; detections: Detection[] }
export interface DetectModelInfo { classes: string[]; caveats: string[]; weights_sha256: string; architecture: { family: string; n_params: number; input: string } }

export interface ClusterInfo {
  available: boolean; n_clusters?: number; noise_count?: number; n_tiles?: number;
  sizes?: Record<string, number>; display_labels?: Record<string, string>;
  cluster_concepts?: Record<string, [string, number][]>;
  params?: Record<string, unknown>;
}
export interface RegionListItem { name: string; bbox: BBox; n_observations: number }
export interface HealthInfo { status: string; vectors: number; analyst: string; candidates?: number }
