import { confidenceBand, confidencePct } from "./candidate.js";
import { regionNameFor } from "../regions.js";

export function toOverviewViewModel(presentation, candidatesBody, regions, stats) {
  const featured = (presentation.featured || []).map((f) => {
    const [lon, lat] = f.centroid_lonlat || [null, null];
    return {
      id: f.candidate_id,
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
    return {
      id: c.candidate_id,
      changeTypeLabel: c.change_type,
      band: confidenceBand(c.confidence),
      confidencePct: confidencePct(c.confidence),
      locationName: regionNameFor(regions, lon, lat),
      lon, lat,
      areaHectares: (c.area_m2 || 0) / 10000,
      window: c.earliest_supported || [],
      gates: c.gates || {},
      sarAvailable: !!(c.sar && c.sar.available),
      sarVerdict: c.sar ? c.sar.verdict : null,
    };
  });

  return {
    aoiName: presentation.aoi,
    counters: {
      candidates: presentation.counters.change_candidates,
      tiles: presentation.counters.tiles_indexed,
      regions: presentation.counters.regions,
      decided: presentation.counters.analyst_decisions,
      highConfidence: presentation.counters.high_confidence,
    },
    tilesIndexed: stats?.index?.tiles,
    featured,
    findings,
  };
}
