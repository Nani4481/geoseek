// viewmodels/search.js - the backend's SearchResult carries only
// {tile_id, score, lon, lat, acq_date, sensor, cloud_fraction, scene_id} -
// no location name and, per the no-ML-vocabulary rule, the raw similarity
// score must never reach a component. Score is dropped entirely here; the
// "reason" line is derived from the other real fields the backend already
// returns (rank position, sensor, acquisition date, region) rather than the
// mockup's fabricated visual-description sentences, which have no backend
// source.
import { regionNameFor } from "../regions.js";

export function toSearchResultViewModel(result, rank, regions, api) {
  const region = regionNameFor(regions, result.lon, result.lat);
  const reason = `Rank ${rank} for this search · ${result.sensor} imagery from ${result.acq_date} in ${region}.`;
  return {
    tileId: result.tile_id,
    rank,
    date: result.acq_date,
    sensor: result.sensor,
    locationName: region,
    reason,
    thumbnailUrl: api.tileThumbnailUrl(result.tile_id),
  };
}
