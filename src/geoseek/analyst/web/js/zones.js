// zones.js - cached lookup from a restricted zone's name (all a candidate's
// own restricted_zone field carries) back to its real bbox, so the
// candidate map can draw the actual zone rectangle instead of just naming it.
let _cache = null;

export async function loadZones(api) {
  if (_cache) return _cache;
  const { zones } = await api.restrictedZones();
  _cache = zones || [];
  return _cache;
}

export function zoneBboxByName(zones, name) {
  return zones.find((z) => z.name === name) || null;
}
