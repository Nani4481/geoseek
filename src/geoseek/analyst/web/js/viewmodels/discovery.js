// viewmodels/discovery.js - the backend returns one flat ranked list
// {tile_id, score, observation_id, acq_date, centroid_lonlat}, with no
// tiering, no spread label, and no rationale text. The three tiers here are
// derived honestly from real fields already returned - region membership
// (same point-in-bbox join used everywhere else) and real great-circle
// distance from the seed - rather than invented groupings. The lowest-third
// of real rank order becomes "probably unrelated": a genuine statement
// about where a result sits in the real ranking, not a fabricated
// similarity judgement. The raw score itself is never surfaced.
import { regionNameFor } from "../regions.js";

function haversineKm(a, b) {
  if (!a || !b) return Infinity;
  const [lon1, lat1] = a, [lon2, lat2] = b;
  const R = 6371, toRad = (d) => (d * Math.PI) / 180;
  const dLat = toRad(lat2 - lat1), dLon = toRad(lon2 - lon1);
  const h = Math.sin(dLat / 2) ** 2 + Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(h));
}

export function toDiscoveryViewModel(seedResult, similarBody, regions) {
  const seedRegion = regionNameFor(regions, seedResult.lon, seedResult.lat);
  const results = similarBody.results.map((r, i) => ({
    tileId: r.tile_id,
    rank: i + 1,
    date: r.acq_date,
    centroid: r.centroid_lonlat,
    region: regionNameFor(regions, r.centroid_lonlat[0], r.centroid_lonlat[1]),
    distanceKm: haversineKm(seedResult.centroid_lonlat || [seedResult.lon, seedResult.lat], r.centroid_lonlat),
  }));

  const n = results.length;
  const lowerThirdStart = Math.ceil((n * 2) / 3);
  const tier1 = [], tier2 = [], tier3 = [];
  results.forEach((r, i) => {
    if (i >= lowerThirdStart) tier3.push(r);
    else if (r.region === seedRegion) tier1.push(r);
    else tier2.push(r);
  });

  const maxDist1 = tier1.length ? Math.max(...tier1.map((r) => r.distanceKm)) : 0;
  const nRegions2 = new Set(tier2.map((r) => r.region)).size;

  return {
    seedRegion,
    tiers: [
      { title: "Same signature, elsewhere in this region", spread: `WITHIN ${Math.ceil(maxDist1)} KM`,
        rationale: "Ranked closest to the starting point among results in the same region.", results: tier1 },
      { title: "Same signature, other regions", spread: `${nRegions2} REGION${nRegions2 === 1 ? "" : "S"}`,
        rationale: "Comparable ranking, but the starting point's own region does not contain these.", results: tier2 },
      { title: "Looks similar, but probably unrelated", spread: "MIXED",
        rationale: "Ranked in the lower third of this search. Shown so you can rule them out rather than wonder.", results: tier3 },
    ],
  };
}
