// components/candidate-map.js - an interactive Leaflet map for one
// candidate's real location: its actual detection polygon, its actual
// "after" evidence imagery draped at that polygon's true geographic
// bounds, and (when applicable) the restricted-zone rectangle it falls
// inside. There is no offline tile-pyramid basemap to lay under this (see
// docs/ARCHITECTURE.md "what is deliberately not here"), so the map has no
// base layer - the void background plus these real, correctly-georeferenced
// layers is genuinely what there is to show, rather than a fabricated
// street/satellite backdrop.
//
// Global `L` comes from vendor/leaflet/leaflet.js, loaded as a classic
// script in index.html before this module runs.

// Bounds of the polygon's outer ring, read directly from the GeoJSON
// coordinates - computed before the map exists, since Leaflet's Path
// renderer (used by geoJSON/rectangle layers) needs the map to already
// have a defined center/zoom the moment a layer is added; asking it to
// project points against a not-yet-positioned map throws inside its own
// internal clipping code.
function ringBounds(geometry) {
  const ring = geometry?.coordinates?.[0];
  if (!ring || !ring.length) return null;
  let minLon = Infinity, minLat = Infinity, maxLon = -Infinity, maxLat = -Infinity;
  for (const [lon, lat] of ring) {
    minLon = Math.min(minLon, lon); maxLon = Math.max(maxLon, lon);
    minLat = Math.min(minLat, lat); maxLat = Math.max(maxLat, lat);
  }
  return L.latLngBounds([[minLat, minLon], [maxLat, maxLon]]);
}

export function mountCandidateMap(container, { geometry, centroidLonLat, imageUrl, restrictedZone }) {
  let fitTarget = (geometry && ringBounds(geometry)) || null;
  if (!fitTarget && centroidLonLat && centroidLonLat[0] != null) {
    fitTarget = L.latLngBounds([[centroidLonLat[1], centroidLonLat[0]]]);
  }

  const map = L.map(container, {
    attributionControl: false,
    zoomControl: true,
    scrollWheelZoom: false,
    maxZoom: 19,
    minZoom: 3,
  });
  // Give the map a real view *before* any Path layer is added to it - see
  // ringBounds() above for why the ordering matters.
  if (fitTarget && fitTarget.isValid()) map.fitBounds(fitTarget.pad(1.2), { maxZoom: 17 });
  else map.setView([0, 0], 3);

  L.control.attribution({ prefix: false })
    .addAttribution("GeoSeek &middot; local catalog, no basemap tiles")
    .addTo(map);

  const layers = [];

  if (geometry) {
    layers.push(L.geoJSON(geometry, { className: "geo-map-candidate-poly" }).addTo(map));
  }

  if (imageUrl && fitTarget) {
    // Drape the real evidence image over the candidate's own polygon
    // bounds - a small pad so the imagery reads as context around the
    // detection, not a tight crop to it.
    L.imageOverlay(imageUrl, fitTarget.pad(0.8), { opacity: 0.92 }).addTo(map);
  }

  if (restrictedZone) {
    layers.push(L.rectangle(
      [[restrictedZone.min_lat, restrictedZone.min_lon], [restrictedZone.max_lat, restrictedZone.max_lon]],
      { className: "geo-map-zone" },
    ).addTo(map));
  }

  let marker = null;
  if (centroidLonLat && centroidLonLat[0] != null) {
    marker = L.marker([centroidLonLat[1], centroidLonLat[0]]).addTo(map);
    layers.push(marker);
  }

  return {
    marker,
    destroy() { map.remove(); },
  };
}
