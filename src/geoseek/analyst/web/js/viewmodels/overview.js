import { confidenceBand, confidencePct, CHANGE_TYPE_LABEL } from "./candidate.js";
import { regionNameFor } from "../regions.js";

export function toOverviewViewModel(presentation, candidatesBody, regions, stats, api) {
  const featured = (presentation.featured || []).map((f) => {
    const [lon, lat] = f.centroid_lonlat || [null, null];
    return {
      id: f.candidate_id,
      changeTypeCode: f.change_type,
      changeTypeLabel: f.change_type_human,
      band: confidenceBand(f.confidence),
      confidencePct: confidencePct(f.confidence),
      locationName: regionNameFor(regions, lon, lat),
      lon, lat,
      areaHectares: (f.area_m2 || 0) / 10000,
      caption: f.caption,
      window: [f.before_date, f.after_date],
      imagery: f.imagery,
    };
  });

  const findings = (candidatesBody.candidates || []).slice(0, 9).map((c) => {
    const [lon, lat] = c.centroid_lonlat || [null, null];
    const window = c.earliest_supported || [];
    // imagery endpoint's `date` is a bare ingested-observation year, see
    // the matching note in viewmodels/candidate.js's toCandidateSummaryViewModel.
    const thumbYear = window[1] ? window[1].slice(0, 4) : null;
    const thumbUrl = api && thumbYear ? api.candidateImageryUrl(c.candidate_id, { date: thumbYear, view: "rgb" }) : null;
    return {
      id: c.candidate_id,
      changeTypeCode: c.change_type,
      changeTypeLabel: CHANGE_TYPE_LABEL[c.change_type] || "Surface change",
      band: confidenceBand(c.confidence),
      confidencePct: confidencePct(c.confidence),
      locationName: regionNameFor(regions, lon, lat),
      lon, lat,
      areaHectares: (c.area_m2 || 0) / 10000,
      window,
      gates: c.gates || {},
      sarAvailable: !!(c.sar && c.sar.available),
      sarVerdict: c.sar ? c.sar.verdict : null,
      thumbUrl,
      restrictedZone: c.restricted_zone || null,
    };
  });

  const years = Array.from(new Set((presentation.observation_dates || []).map((d) => d.slice(0, 4)))).sort();

  return {
    aoiName: presentation.aoi,
    counters: {
      candidates: presentation.counters.change_candidates,
      tiles: presentation.counters.tiles_indexed,
      regions: presentation.counters.regions,
      decided: presentation.counters.analyst_decisions,
      highConfidence: presentation.counters.high_confidence,
      watchAreas: presentation.counters.watch_areas ?? 0,
      restrictedZoneAlerts: presentation.counters.restricted_zone_alerts ?? 0,
      reviewRatePct: presentation.counters.review_rate_pct ?? 0,
    },
    tilesIndexed: stats?.index?.tiles,
    years,
    featured,
    findings,
  };
}
