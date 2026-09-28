// viewmodels/activity.js - the Overview activity feed. Every row is derived
// from data the analyst service already stores (the append-only decision
// audit trail + the ranked candidate list, including its restricted-zone
// cross-reference) - nothing here is invented or randomly generated.
import { CHANGE_TYPE_LABEL, confidenceBand } from "./candidate.js";
import { regionNameFor } from "../regions.js";

function fmtTimestamp(ts) {
  if (!ts) return "";
  return String(ts).replace("T", " ").slice(0, 16);
}

function decisionItem(d) {
  if (d.decision === "reopen") return null; // reopen just resets a verdict; not its own event
  const verb = d.decision === "confirm" ? "confirmed" : "reviewed — no threat";
  return {
    tone: "normal",
    icon: "\u{1F7E2}",
    text: `Candidate ${d.candidate_id} ${verb} by analyst`,
    ts: d.created_at,
  };
}

function candidateItem(c, regions) {
  const [lon, lat] = c.centroid_lonlat || [null, null];
  const loc = regionNameFor(regions, lon, lat);
  const ts = (c.earliest_supported || [])[1] || "";
  const typeLabel = CHANGE_TYPE_LABEL[c.change_type] || "Surface change";
  if (c.restricted_zone) {
    return {
      tone: "critical",
      icon: "\u{1F534}",
      text: `CRITICAL: ${typeLabel} detected in ${c.restricted_zone.name}`,
      ts,
    };
  }
  return {
    tone: "attention",
    icon: "\u{1F7E1}",
    text: `New ${typeLabel.toLowerCase()} detected — ${loc} · Confidence: ${confidenceBand(c.confidence).toUpperCase()}`,
    ts,
  };
}

export function buildActivityFeed({ candidates = [], auditRows = [], regions = [] } = {}, limit = 20) {
  const items = [];
  for (const d of auditRows) {
    const it = decisionItem(d);
    if (it) items.push(it);
  }
  for (const c of candidates) items.push(candidateItem(c, regions));

  return items
    .filter((it) => it.ts)
    .sort((a, b) => (a.ts < b.ts ? 1 : a.ts > b.ts ? -1 : 0))
    .slice(0, limit)
    .map((it) => ({ ...it, tsLabel: fmtTimestamp(it.ts) }));
}
