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

export interface RepresentativeTile { tile_id: string; cloud_fraction: number | null; acquired_at: string | null; row: number; col: number }
export interface ProvenanceObservation {
  role: 'before' | 'after'; observation_id: string; acquired_at: string | null;
  representative_tile?: RepresentativeTile | null;
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
  observation_id: string; aoi_name: string; role: string; acquired_at: string | null; platform: string | null; sensor: string | null; native_gsd_m: number | null; n_tiles: number; n_tiles_with_detections: number;
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
  /** per cluster: size, dominant region, its share, and the tile count in every region (measured when the clustering ran) */
  region_purity?: Record<string, ClusterRegions>;
  params?: Record<string, unknown>;
}
export interface ClusterRegions { size: number; dominant_region: string; purity: number; regions: Record<string, number> }
export interface RegionListItem { name: string; bbox: BBox; n_observations: number }
export interface HealthInfo { status: string; vectors: number; analyst: string; candidates?: number }

export type RingKind = 'restricted_zone' | 'change_candidate' | 'detection' | 'watch_area';
export interface RingFeature {
  kind: RingKind; id: string; name: string; distance_m: number; ring_index: number; ring_m: number; contains_centre: boolean;
  lon: number; lat: number; nearest: [number, number]; detail: Record<string, string | number | boolean | null>;
}
export interface ThreatRings {
  center: { lon: number; lat: number }; radii_m: number[];
  rings: { radius_m: number; counts: Partial<Record<RingKind, number>>; total: number }[];
  features: RingFeature[]; truncated: boolean; n_features: number;
  sources: Partial<Record<RingKind, { features_searched: number; within_largest_ring: number }>>;
  excluded: string | null; notes: string[];
}
export interface DetectionPoint {
  id: string; lon: number; lat: number; class: string; score: number; tile_id: string;
  length_m: number | null; width_m: number | null; heading_deg: number | null; long_side_px: number | null;
}

export interface BasemapScene { observation_id: string; date: string; platform: string; sensor: string; bbox: BBox; mean_cloud?: number }
export interface BasemapCoverage {
  layer: string; available: boolean; fraction: number; scenes: BasemapScene[]; dates: string[]; native_max_zoom: number; gsd_m: number;
  source: string; selection_rule?: string; year?: string | null; overview_max_zoom?: number; coarse_max_zoom?: number;
  /** every archive acquisition whose granule touches the view, regardless of the year filter (Sentinel-2 layer only) */
  acquisitions?: { date: string; observation_id: string; platform: string; sensor: string; mean_cloud: number }[];
}
export interface ClusterExample { tile_id: string; region: string; acq_date: string; cloud_fraction: number; centroid_lonlat: [number, number] }
export interface ClusterGeoEntry { n_tiles: number; n_cells: number; bbox: BBox; cells: number[][]; examples?: ClusterExample[] }
export interface ClusterGeo { available: boolean; cell_deg: number; n_clustered_tiles: number; n_unplaced: number; clusters: Record<string, ClusterGeoEntry>; source: string }

export interface DossierObservation {
  role: 'before' | 'after'; requested: string; observation_id: string; acquired_at: string | null; acquired_at_precision?: string | null;
  scene_id: string | null; platform: string | null; sensor: string | null; collection_id: string | null; native_gsd_m: number | null;
  processing_baseline: string | null; crs: string | null; license: string | null; tile_id: string | null; cloud_fraction: number | null;
  // present ONLY when the catalog holds them - never a placeholder
  sun_elevation_deg?: number; sun_azimuth_deg?: number; off_nadir_deg?: number;
}
export interface DossierProvenance {
  candidate_id: string; centroid_lonlat: LonLat; observations: DossierObservation[]; view_fields_not_catalogued: string[]; source: string;
}

export interface ProjectionSample {
  available: boolean; reason?: string;
  n_total: number; n_shown: number; sampled: boolean; seed: number;
  regions: string[]; clusters: number[];
  tile_ids: string[]; xyz: number[]; lon: number[]; lat: number[]; region: number[]; cluster: number[];
  stale: boolean; current_vectors: number | null;
  meta: { method: string; umap: Record<string, number | string>; pca_components: number; libraries: Record<string, string>;
    wall_seconds: { total: number; umap: number }; power_source: string; created_at: string; caveat: string; n_points: number; partial: boolean };
}
export interface ProjectionLookup {
  available: boolean; missing: string[];
  points: { tile_id: string; xyz: [number, number, number]; lon: number; lat: number; region: string; cluster: number }[];
}
export interface SpectralStats { mean: number; std: number; p10: number; p50: number; p90: number }
export interface SpectralLayer {
  label: string; meaning: string; formula: string; bands: string[]; domain: [number, number]; stops: [number, string][];
  high: string; low: string; stats: SpectralStats | null; image_url: string;
}
export interface TileSpectral {
  tile_id: string; width_px: number; height_px: number; gsd_m: number; scene_id: string; acquired_at: string;
  valid_fraction: number; usable: boolean; unusable_reason: string | null;
  layers: Record<'ndvi' | 'ndwi' | 'ndbi', SpectralLayer>;
  classes: Record<'water_frac' | 'veg_frac' | 'dense_veg_frac' | 'bare_frac' | 'built_frac', number> | null;
  class_rules: Record<'water_frac' | 'veg_frac' | 'dense_veg_frac' | 'bare_frac' | 'built_frac', string>;
  relevance: { query: string | null; matches: { index: 'ndvi' | 'ndwi' | 'ndbi'; terms: string[]; why: string }[]; note: string | null };
  embedding_patch_m: number | null; source: string; caveat: string;
}

// ---- explainability + suppression pipeline (geoseek.analyst.explain) ----
export interface ExplainTerm {
  name: string; label: string; plain: string; value: number; weight: number; weight_share: number; factor: number;
  effect_points: number | null; strength: 'supports' | 'weak'; raw: string;
}
export interface ExplainMultiplier { name: string; label: string; factor: number; effect_points: number | null; plain: string }
export interface SpectralTest { index: string; type: string; test: string; op: string; threshold: number; value: number; met: boolean; decisive: boolean; text: string }
export interface SpectralAnomaly { index: 'ndvi' | 'ndbi' | 'ndwi'; label: string; delta: number | null; seasonal: number | null; anomaly: number | null; season_band: number; outside_season_band: boolean }
export interface TraceStage {
  id: string; kind: 'gate' | 'typing' | 'score'; verdict: string; detail: string; values?: Record<string, unknown>;
  gate_weight?: number | null; rule?: string; change_type?: string; persistence?: string;
  effect: { factor: number | null; term: string | null; text: string; points: number | null } | null;
}
export interface CandidateExplain {
  available: boolean; candidate_id: string; change_type: string;
  lead: { headline: string; persistence: string; text: string; rule: string; source: string };
  evidence: {
    terms: ExplainTerm[]; multipliers: ExplainMultiplier[]; method: string; stored_confidence: number; recomputed_confidence: number;
    max_abs_error: number; reproduces: boolean; breakdown_lines: string[]; significance: number; queue_score: number;
  };
  spectral: { rule: string; rule_detail: string; anomalies: SpectralAnomaly[]; tests: SpectralTest[]; note: string };
  weak_or_absent: { key: string; level: 'weak' | 'absent'; text: string }[];
  sar: { available: boolean; verdict: string | null; factor: number | null; vv_median_db: number | null; run_note: string | null };
  terrain: { plain_language: string; used_in_confidence: boolean } | null;
  trace: TraceStage[];
  ranking: { queue_score: string | null; significance: string | null };
  overlay: { tile_id: string; observation_id: string; acquired_at: string | null; focus_indices: string[]; geometry: Polygon | null; note: string } | null;
  scope: string;
}

export interface FunnelStage { rule: string; removed: number; remaining: number; input: number; share_of_raw: number | null; share_of_input: number | null }
export interface FunnelPair {
  name: string; earlier: string; later: string; comparable: boolean | null; raw: number; survivors: number; suppressed: number; consistent: boolean;
  stages: FunnelStage[]; downweighted_survivors: number | null; class_distribution: Record<string, number> | null;
  context: Record<string, number> | null;
}
export interface FunnelGate { rule: string; kind: 'reject' | 'downweight'; thresholds: Record<string, number>; what: string }
export interface LabelledStage { stage: string; precision: number; recall: number; f1: number; delta_f1: number | null; delta_precision: number | null; delta_recall: number | null }
export interface SuppressionFunnel {
  available: boolean; aoi: string | null; span_pair: string; scope: string; model_threshold: number | null;
  source: { report: string; candidates: string; report_generated_at: string };
  attribution_note: string; morphology_note: string; gates: FunnelGate[]; pairs: FunnelPair[];
  post_gate: {
    pair: string; survivors: number; matches_report: boolean;
    typing: { by_type: Record<string, number>; unclassified: number };
    persistence: { by_class: Record<string, number>; supported: number; contradicted: number; no_support: number; penalty: Record<string, number> };
    downweighted_by_gate: Record<string, number>;
    confidence_bands: Record<string, number>;
    sar: { available: boolean; note: string | null; moved_up: number; moved_down: number; neutral: number };
  };
  demoted_sample: { candidate_id: string; change_type: string | null; persistence: string; mean_model_prob: number | null; confidence: number | null; penalty: number; confidence_without_penalty: number | null; reason: string }[];
  rejected: { retained: boolean; text: string; demoted_instead: string; to_emit: string };
  labelled_benchmark: {
    available: boolean; reason?: string; threshold?: number; stages?: LabelledStage[]; generated_at?: string; dataset?: string;
    config_matches_current?: boolean; order_note?: string; caveat?: string; source?: string;
  };
  generated_at: string;
}

// ---- temporal view of the pipeline's own output (geoseek.analyst.temporal) ----
export interface TemporalBucket {
  n: number; area_m2: number; by_type: Record<string, number>; area_m2_by_type: Record<string, number>; persistence: Record<string, number>;
  held: number; demoted: number; sar_available: number; sar_coverage: number | null;
}
export interface TemporalInterval {
  id: string; from: string; to: string; days: number; stored: TemporalBucket;
  pair_run: { name: string; raw: number; survivors: number; comparable: boolean | null; by_type: Record<string, number> } | null;
}
export interface TemporalArchive {
  available: boolean; aoi: string | null; bbox: BBox | null; dates: string[]; types: string[];
  intervals: TemporalInterval[];
  multi_interval: (TemporalBucket & { id: string; from: string; to: string; days: number })[];
  no_interval: TemporalBucket;
  span_run: { name: string; from: string; to: string; raw: number; survivors: number } | null;
  cumulative: { date: string | null; n_by_type: Record<string, number>; area_m2_by_type: Record<string, number>; n: number; area_m2: number }[];
  totals: { stored_in_scope: number };
  sar: { available_candidates: number; note: string | null };
  held_classes: string[]; demoted_classes: string[];
  pair_run_note: string; scope: string;
  source: { report: string; candidates: string; report_generated_at: string };
}
