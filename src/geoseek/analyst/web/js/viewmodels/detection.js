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

const GROUP_COLOR = { Vehicles: "var(--yellow)", Aircraft: "var(--cyan)", Ships: "var(--green)", "Fixed assets": "var(--violet)" };

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
