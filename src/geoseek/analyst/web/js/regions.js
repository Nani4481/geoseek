// regions.js - derives a human place name for a lon/lat point.
//
// The backend has no location-name field anywhere (candidates, search
// results and discovery results only ever carry centroid_lonlat/tile_id).
// This is a real, honest derivation from real data: /regions returns the
// actual staged AOI names with a bbox spanning their observation
// footprints, and we do a point-in-bbox test against it. It is not a
// fabrication - it is the same data the top bar's own AOI stepper is built
// from - but a point that falls in more than one AOI's bbox (they can
// overlap slightly) resolves to the first match, and a point outside every
// known bbox falls back to a generic label instead of guessing.

let _cache = null;

export async function loadRegions(api) {
  if (_cache) return _cache;
  const { regions } = await api.regions();
  _cache = regions || [];
  return _cache;
}

export function regionNameFor(regions, lon, lat) {
  if (lon == null || lat == null) return "Unplaced";
  for (const r of regions) {
    const [w, s, e, n] = r.bbox;
    if (lon >= w && lon <= e && lat >= s && lat <= n) {
      return humanizeRegionName(r.name);
    }
  }
  return "Outside known regions";
}

export function humanizeRegionName(raw) {
  if (!raw) return "Unknown region";
  return raw
    .split(/[_-]/)
    .filter(Boolean)
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}
