// viewmodels/detection.js - the real detector has 8 classes
// (small-vehicle, large-vehicle, ship, plane, helicopter, storage-tank,
// harbor, bridge). The design calls for 4 analyst-facing groups; this
// regroups the real per-class precision/recall into them (found% = recall,
// false% = 1-precision, both real numbers from /detect/model-info) rather
// than inventing new figures. "Median size" has no backend source anywhere
// (not in model-info, not aggregated from stored detections) so it is
// dropped rather than fabricated. Per-class caveats are the model's own
// real caveat strings (data-driven low-precision warnings), attributed to
// whichever group contains that class; a group with no real caveat text
// shows none rather than invented copy.

const GROUPS = {
  Vehicles: ["small-vehicle", "large-vehicle"],
  Aircraft: ["plane", "helicopter"],
  Ships: ["ship"],
  "Fixed assets": ["storage-tank", "harbor", "bridge"],
};

// box stroke colour, not the semantic --green (gate-pass/confirmed) token -
// kept distinct so a green detection box is never read as "confirmed"
const GROUP_COLOR = { Vehicles: "var(--detect-box)", Aircraft: "var(--cyan)", Ships: "var(--green)", "Fixed assets": "var(--violet)" };

export function toModelInfoViewModel(modelInfo) {
  const op = modelInfo.operating_points || {};
  const groups = Object.entries(GROUPS).map(([name, classes]) => {
    const present = classes.filter((c) => op[c]);
    if (!present.length) return null;
    const recalls = present.map((c) => op[c].recall).filter((v) => v != null);
    const precisions = present.map((c) => op[c].precision).filter((v) => v != null);
    const avgRecall = recalls.length ? recalls.reduce((a, b) => a + b, 0) / recalls.length : null;
    const avgPrecision = precisions.length ? precisions.reduce((a, b) => a + b, 0) / precisions.length : null;
    const caveats = (modelInfo.caveats || []).filter((c) => present.some((cls) => c.includes(cls)));
    return {
      name, color: GROUP_COLOR[name], classes: present,
      foundPct: avgRecall != null ? Math.round(avgRecall * 100) : null,
      falsePct: avgPrecision != null ? Math.round((1 - avgPrecision) * 100) : null,
      caveats,
    };
  }).filter(Boolean);
  return { groups, weightsSha256: modelInfo.weights_sha256, architecture: modelInfo.architecture };
}

export function classToGroupName(cls) {
  for (const [name, classes] of Object.entries(GROUPS)) if (classes.includes(cls)) return name;
  return "Other";
}
export function groupColor(name) { return GROUP_COLOR[name] || "var(--ink-dim)"; }

// -- contextual one-liners --------------------------------------------------
// Hardcoded logic rules over the real per-tile detection geometry (pixel
// centroids, real class labels) - no invented counts or positions. A "convoy"
// is a plain single-link clustering of small-vehicle centroids within 120px
// of each other in the tile's native resolution; the threshold is a fixed
// rule of thumb, not a measured/tuned parameter.
const CONVOY_PX_RADIUS = 120;
const CONVOY_MIN_SIZE = 3;

function boxCentroidPx(d) {
  const xs = d.polygon_px.map((p) => p[0]);
  const ys = d.polygon_px.map((p) => p[1]);
  return [(Math.min(...xs) + Math.max(...xs)) / 2, (Math.min(...ys) + Math.max(...ys)) / 2];
}

function pxDist(a, b) { return Math.hypot(a[0] - b[0], a[1] - b[1]); }

export function contextualNotes(detections) {
  const notes = [];
  const smallVehicles = detections.filter((d) => d.class === "small-vehicle");
  const centroids = smallVehicles.map(boxCentroidPx);
  const used = new Array(smallVehicles.length).fill(false);
  for (let i = 0; i < smallVehicles.length; i++) {
    if (used[i]) continue;
    const group = [i];
    for (let j = i + 1; j < smallVehicles.length; j++) {
      if (!used[j] && pxDist(centroids[i], centroids[j]) < CONVOY_PX_RADIUS) { group.push(j); used[j] = true; }
    }
    if (group.length >= CONVOY_MIN_SIZE) {
      notes.push(`${group.length} small vehicles clustered — possible convoy`);
    }
    used[i] = true;
  }
  const largeVehicles = detections.filter((d) => d.class === "large-vehicle");
  if (largeVehicles.length) {
    notes.push(`${largeVehicles.length} large vehicle${largeVehicles.length > 1 ? "s" : ""} detected — possible logistics movement`);
  }
  return notes;
}
