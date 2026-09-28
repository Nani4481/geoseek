// viewmodels/candidate.js - maps raw /candidates and /candidates/{id}
// backend records to the analyst-facing shapes the Candidate Detail and
// Review Queue screens render. Components never see backend field/gate
// names or raw confidence past this file.
import { regionNameFor } from "../regions.js";

// Analyst-facing gate labels, in the fixed order the design specifies.
// registration -> Alignment, radiometric -> Brightness match,
// phenology -> Seasonal check, morphology -> Shape & size.
export const GATE_LABELS = [
  ["quality", "Image quality"],
  ["registration", "Alignment"],
  ["radiometric", "Brightness match"],
  ["phenology", "Seasonal check"],
  ["morphology", "Shape & size"],
];

const CHANGE_TYPE_PHRASE = {
  water_gain: "a new body of standing water",
  water_loss: "the retreat of a water body",
  construction: "new hard-surfaced construction",
  clearance: "a loss of vegetation cover",
  road: "a new linear cut across open ground",
  other: "a surface change",
};

export const CHANGE_TYPE_LABEL = {
  water_gain: "New open water",
  water_loss: "Water loss",
  construction: "New construction",
  clearance: "Vegetation cleared",
  road: "New road / track",
  other: "Surface change",
};

export function confidenceBand(confidence0to1) {
  const pct = Math.round((confidence0to1 || 0) * 100);
  if (pct >= 75) return "high";
  if (pct >= 52) return "medium";
  return "low";
}

export function confidencePct(confidence0to1) {
  return Math.round((confidence0to1 || 0) * 100);
}

const BAND_MEANING = {
  high: "The evidence lines up. Treat this as a real ground change unless you see something the checks missed.",
  medium: "Probably real, but at least one check is unhappy. Look at the imagery yourself before you report it.",
  low: "Weak. Enough is wrong with the evidence that this should not leave your desk without a second look.",
};

function gateFromTrace(trace, rule) {
  return (trace || []).find((t) => t.rule === rule) || null;
}

function buildGates(suppression) {
  const trace = (suppression || {}).trace || [];
  return GATE_LABELS.map(([rule, label]) => {
    const t = gateFromTrace(trace, rule);
    return {
      rule,
      label,
      pass: t ? t.verdict === "pass" : true,
      // the note is the pipeline's own real per-candidate explanation string
      // (technical but true), not invented boilerplate copy.
      note: t ? t.detail : "No evidence recorded for this check.",
    };
  });
}

function buildReasons(detail) {
  const reasons = [];
  const gates = buildGates(detail.suppression);
  const morphologyGate = gates.find((g) => g.rule === "morphology");
  if (morphologyGate && morphologyGate.pass) {
    reasons.push({ sign: "plus",
      text: "The footprint passed the shape-and-size check, which is what this kind of change looks like." });
  }
  const traj = detail.temporal_trajectory || {};
  const laterCount = (traj.intervals || []).filter((iv) => iv.kind === "consecutive" && iv.changed).length;
  if (laterCount > 0) {
    reasons.push({ sign: "plus",
      text: `It is still there on ${laterCount} later pass${laterCount === 1 ? "" : "es"}, so it is not a one-frame artefact.` });
  }
  const sar = detail.sar;
  if (sar && sar.available) {
    if (sar.verdict === "agree" || sar.verdict === "agrees") {
      reasons.push({ sign: "plus", text: "A radar pass across the same window shows the same change, which is independent support." });
    } else if (sar.verdict) {
      reasons.push({ sign: "minus", text: "A radar pass across the same window does not show this change." });
    }
  } else {
    reasons.push({ sign: "neutral", text: "No radar pass covers these dates for this area, so there is nothing to cross-check against." });
  }
  gates.filter((g) => !g.pass).forEach((g) => {
    reasons.push({ sign: "minus", text: `${g.label}: ${g.note}` });
  });
  const areaHa = (detail.area_m2 || 0) / 10000;
  if (areaHa > 0 && areaHa < 6) {
    reasons.push({ sign: "neutral",
      text: `At ${areaHa.toFixed(1)} hectares this is a small footprint; small changes are the easiest to get wrong.` });
  }
  return reasons;
}

function verdictSentence(detail, before, after, locationName) {
  const phrase = CHANGE_TYPE_PHRASE[detail.change_type] || CHANGE_TYPE_PHRASE.other;
  const ha = ((detail.area_m2 || 0) / 10000).toFixed(1);
  return `Between ${before} and ${after}, ${phrase} appeared at ${locationName}, covering about ${ha} hectares.`;
}

function sarBlock(sar) {
  if (!sar || !sar.available) {
    return { state: "unavailable",
      title: "No radar coverage",
      text: "No radar pass covers these dates for this area, so there is nothing to cross-check against." };
  }
  const agrees = sar.verdict === "agree" || sar.verdict === "agrees";
  return agrees
    ? { state: "agree", title: "Radar agrees",
        text: "A radar pass across the same window shows the same change in the same place. Radar sees through cloud and works at night, so this is independent support." }
    : { state: "disagree", title: "Radar does not agree",
        text: "A radar pass across the same window does not show this change. That does not rule it out, but treat the optical finding as unconfirmed." };
}

export function toCandidateDetailViewModel(detail, regions, api) {
  const [lon, lat] = detail.centroid_lonlat || [null, null];
  const locationName = regionNameFor(regions, lon, lat);
  const prov = detail.provenance || {};
  const beforeObs = (prov.observations || []).find((o) => o.role === "before");
  const afterObs = (prov.observations || []).find((o) => o.role === "after");
  const before = beforeObs ? beforeObs.acquired_at : (detail.imagery || {}).before_dates?.slice(-1)[0];
  const after = afterObs ? afterObs.acquired_at : (detail.imagery || {}).after_date;
  // /candidates/{id}/imagery takes a YEAR string (detail.imagery.before_dates /
  // .after_date, e.g. "2019"/"2026"), not the ISO acquired_at date shown above -
  // these are the two real but differently-shaped date fields the detail
  // payload carries, and the imagery endpoint only accepts the former.
  const imgAfterYear = (detail.imagery || {}).after_date;
  const imgBeforeYear = ((detail.imagery || {}).before_dates || []).slice(-1)[0] || imgAfterYear;
  const gates = buildGates(detail.suppression);
  const band = confidenceBand(detail.confidence);

  return {
    id: detail.candidate_id,
    changeTypeLabel: CHANGE_TYPE_LABEL[detail.change_type] || "Surface change",
    locationName,
    areaHectares: (detail.area_m2 || 0) / 10000,
    confidencePct: confidencePct(detail.confidence),
    band,
    bandMeaning: BAND_MEANING[band],
    verdictSentence: verdictSentence(detail, before, after, locationName),
    beforeDate: before,
    afterDate: after,
    earliestSupported: detail.earliest_supported || [before, after],
    gates,
    gatesPassCount: gates.filter((g) => g.pass).length,
    reasons: buildReasons(detail),
    sar: sarBlock(detail.sar),
    provenance: {
      beforeDate: before, afterDate: after,
      beforeGsd: beforeObs?.collection?.native_gsd_m, afterGsd: afterObs?.collection?.native_gsd_m,
      sceneId: afterObs?.scene?.scene_id, observationId: afterObs?.observation_id,
      tileId: afterObs?.representative_tile?.tile_id,
    },
    imagery: {
      beforeUrl: api.candidateImageryUrl(detail.candidate_id, { date: imgBeforeYear, view: "rgb" }),
      afterUrl: api.candidateImageryUrl(detail.candidate_id, { date: imgAfterYear, view: "overlay" }),
      dates: (detail.imagery || {}).dates || [],
    },
    timeline: buildTimeline(detail),
    effectiveDecision: detail.effective_decision || "undecided",
    currentDecision: detail.current_decision || null,
  };
}

function buildTimeline(detail) {
  const dates = (detail.imagery || {}).dates || [];
  const traj = detail.temporal_trajectory || {};
  const win = (traj.earliest_supported_change || {}).window || detail.earliest_supported || [];
  return dates.map((d) => {
    let state = "before";
    if (win.length === 2 && d >= win[0] && d < win[1]) state = "entering";
    else if (win.length === 2 && d >= win[1]) state = "present";
    return { date: d, state };
  });
}

export function toCandidateSummaryViewModel(row, regions, api) {
  const [lon, lat] = row.centroid_lonlat || [null, null];
  const gates = GATE_LABELS.map(([rule, label]) => ({
    rule, label, pass: row.gates ? row.gates[rule] !== false : true,
  }));
  const window = row.earliest_supported || [];
  // real cropped tile imagery for the row's "after" date, not a fabricated
  // preview - candidateImageryUrl is a pure URL builder (no fetch happens
  // here), see the same pattern in toCandidateDetailViewModel. The imagery
  // endpoint's `date` is a bare ingested-observation year ("2024"), not the
  // full earliest_supported date ("2024-03-08"), so slice to the year.
  const thumbYear = window[1] ? window[1].slice(0, 4) : null;
  const thumbUrl = api && thumbYear ? api.candidateImageryUrl(row.candidate_id, { date: thumbYear, view: "rgb" }) : null;
  return {
    id: row.candidate_id,
    changeTypeLabel: CHANGE_TYPE_LABEL[row.change_type] || "Surface change",
    locationName: regionNameFor(regions, lon, lat),
    areaHectares: (row.area_m2 || 0) / 10000,
    confidencePct: confidencePct(row.confidence),
    band: confidenceBand(row.confidence),
    decision: row.decision || "undecided",
    window,
    gates,
    queueScore: row.queue_score,
    sarAvailable: !!(row.sar && row.sar.available),
    sarVerdict: row.sar ? row.sar.verdict : null,
    thumbUrl,
  };
}
