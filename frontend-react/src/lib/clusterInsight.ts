// What the stored clustering run says about itself, derived ONLY from /discovery/clusters (+ the bbox from /ui/clusters/geo).
// Pure functions, no DOM: tools/selftest-lib.mjs runs them under Node. Every number is counted from the API payload; the only
// constants are the two thresholds below, and the interpretation line prints them so the reader can see what "concentrated" meant.

import type { ClusterInfo } from '@/api/types';

/** a cluster is "concentrated" when one region holds at least this share of its tiles */
export const CONCENTRATED_AT = 0.8;
/** a region counts as "spanned" by a cluster when it holds at least this share of the cluster's tiles (a handful of strays do not) */
export const SPANS_AT = 0.05;

export interface RegionShare { region: string; n: number; share: number }
export interface ClusterStat {
  id: string; n: number; share: number; name: string; concept: string; conceptScore: number | null;
  regions: RegionShare[];            // every region with at least one tile, largest first
  dominant: string | null; purity: number;
  spans: string[];                   // regions holding >= SPANS_AT of the cluster
  concentrated: boolean;
}

export function clusterStats(d: ClusterInfo | null | undefined): ClusterStat[] {
  if (!d?.available || !d.sizes) return [];
  const total = Object.values(d.sizes).reduce((a, b) => a + b, 0) || 1;
  return Object.entries(d.sizes).map(([id, n]) => {
    const top = d.cluster_concepts?.[id]?.[0];
    const rp = d.region_purity?.[id];
    const regions = Object.entries(rp?.regions ?? {}).map(([region, c]) => ({ region, n: c, share: c / (n || 1) }))
      .sort((a, b) => b.n - a.n || a.region.localeCompare(b.region));
    const lead = regions[0];
    return {
      id, n, share: n / total,
      name: d.display_labels?.[id] ?? top?.[0] ?? `cluster ${id}`,
      concept: top?.[0] ?? '', conceptScore: top?.[1] ?? null,
      regions, dominant: lead?.region ?? null, purity: lead?.share ?? 0,
      spans: regions.filter((r) => r.share >= SPANS_AT).map((r) => r.region),
      concentrated: !!lead && lead.share >= CONCENTRATED_AT,
    };
  }).sort((a, b) => b.n - a.n || Number(a.id) - Number(b.id));
}

/** clusters grouped by their closest text concept (biggest group first, members by size) - the order the region matrix is read in */
export function conceptGroups(stats: ClusterStat[]): { concept: string; members: ClusterStat[]; tiles: number }[] {
  const by = new Map<string, ClusterStat[]>();
  for (const s of stats) by.set(s.concept || '—', [...(by.get(s.concept || '—') ?? []), s]);
  return [...by.entries()].map(([concept, members]) => ({ concept, members, tiles: members.reduce((a, m) => a + m.n, 0) }))
    .sort((a, b) => b.tiles - a.tiles || a.concept.localeCompare(b.concept));
}

/** regions that occur in the run, biggest first, with the tiles they contribute */
export function regionColumns(stats: ClusterStat[]): { region: string; tiles: number }[] {
  const t = new Map<string, number>();
  for (const s of stats) for (const r of s.regions) t.set(r.region, (t.get(r.region) ?? 0) + r.n);
  return [...t.entries()].map(([region, tiles]) => ({ region, tiles })).sort((a, b) => b.tiles - a.tiles || a.region.localeCompare(b.region));
}

/** width x height of a lon/lat box in km (equirectangular at the box's mid latitude - a label, not a survey) */
export function extentKm(b: [number, number, number, number]): { w: number; h: number } {
  const midLat = ((b[1] + b[3]) / 2) * (Math.PI / 180);
  return { w: Math.max(0, b[2] - b[0]) * 111.32 * Math.cos(midLat), h: Math.max(0, b[3] - b[1]) * 110.57 };
}

/** The box holding the middle `1 - 2q` of a cluster's tiles on each axis (cells are [lon, lat, tileCount]; counts weight the quantiles), so a
 *  handful of stray tiles far away do not stretch the cluster's extent across the whole archive. */
export function coreBox(cells: number[][], q = 0.025): [number, number, number, number] | null {
  const total = cells.reduce((a, c) => a + c[2], 0);
  if (!cells.length || total <= 0) return null;
  const at = (axis: 0 | 1, frac: number) => {
    const s = [...cells].sort((a, b) => a[axis] - b[axis]);
    let acc = 0;
    for (const c of s) { acc += c[2]; if (acc >= frac * total) return c[axis]; }
    return s[s.length - 1][axis];
  };
  return [at(0, q), at(1, q), at(0, 1 - q), at(1, 1 - q)];
}

export interface Seg { t: string; b?: boolean }
export interface Insight {
  nClusters: number; nTiles: number; nConcentrated: number; nSpread: number;
  counts: Seg[];                      // line 1: how many clusters, concentrated vs spread
  fact: Seg[];                        // line 2: the most notable thing, computed
  factKind: 'concept-splits-by-region' | 'most-spread' | 'none';
  /** cluster ids the fact is about (the UI can select / mark them) */
  factClusters: string[];
}

const plural = (n: number, one: string, many = one + 's') => `${n.toLocaleString('en-US')} ${n === 1 ? one : many}`;
const pct = (x: number) => `${Math.round(x * 100)}%`;
function joinList(xs: string[]): string {
  if (xs.length <= 1) return xs.join('');
  return `${xs.slice(0, -1).join(', ')} and ${xs[xs.length - 1]}`;
}

/**
 * Two lines for the top of the Discovery screen, computed from the run.
 *  1. how many clusters, how many are concentrated in one region vs spread over several;
 *  2. the single most notable fact: the closest-concept label that the most clusters share while those clusters each sit in a
 *     *different* region (the embedding separating what one text label cannot); if no label does that, the most spread cluster.
 * `label` renders a region key for display (e.g. 'delhi_ncr' -> 'Delhi NCR').
 */
export function interpret(stats: ClusterStat[], label: (region: string) => string = (r) => r): Insight | null {
  if (!stats.length) return null;
  const nTiles = stats.reduce((a, s) => a + s.n, 0);
  const conc = stats.filter((s) => s.concentrated), spread = stats.filter((s) => !s.concentrated);
  const counts: Seg[] = [
    { t: plural(stats.length, 'cluster'), b: true }, { t: ` over ` }, { t: plural(nTiles, 'tile'), b: true }, { t: ': ' },
    { t: `${conc.length}`, b: true }, { t: ` sit ${pct(CONCENTRATED_AT)} or more inside a single region (concentrated); ` },
    { t: `${spread.length}`, b: true }, { t: ` are spread across several regions, none holding ${pct(CONCENTRATED_AT)}.` },
  ];

  // a concept label shared by several clusters that nevertheless resolve to different regions
  let best: { concept: string; all: ClusterStat[]; conc: ClusterStat[]; regions: string[] } | null = null;
  for (const g of conceptGroups(stats)) {
    if (g.members.length < 2) continue;
    const c = g.members.filter((m) => m.concentrated);
    const regions = [...new Set(c.map((m) => m.dominant!))];
    if (regions.length < 2) continue;
    if (!best || regions.length > best.regions.length || (regions.length === best.regions.length && c.length > best.conc.length)) {
      best = { concept: g.concept, all: g.members, conc: c, regions };
    }
  }
  if (best) {
    const names = best.regions.map(label);
    return {
      nClusters: stats.length, nTiles, nConcentrated: conc.length, nSpread: spread.length, counts,
      fact: [
        { t: `${best.conc.length} of ${best.all.length}`, b: true }, { t: ` clusters whose closest concept is “${best.concept}” each sit ${pct(CONCENTRATED_AT)}+ inside one region, and those regions differ (` },
        { t: joinList(names), b: true }, { t: `): the embedding separates terrain that a single text label cannot tell apart. Clustering used only the image embeddings, not the region names.` },
      ],
      factKind: 'concept-splits-by-region', factClusters: best.conc.map((m) => m.id),
    };
  }
  const widest = [...stats].sort((a, b) => b.spans.length - a.spans.length || b.n - a.n)[0];
  if (widest && widest.spans.length >= 2) {
    return {
      nClusters: stats.length, nTiles, nConcentrated: conc.length, nSpread: spread.length, counts,
      fact: [
        { t: `Cluster ${widest.id}` , b: true }, { t: ` (“${widest.name}”, ${pct(widest.share)} of all tiles) is the most widely spread: ` },
        { t: plural(widest.spans.length, 'region'), b: true }, { t: ` each hold at least ${pct(SPANS_AT)} of it` + (widest.dominant ? `, the largest (${label(widest.dominant)}) only ${pct(widest.purity)}.` : '.') },
      ],
      factKind: 'most-spread', factClusters: [widest.id],
    };
  }
  return { nClusters: stats.length, nTiles, nConcentrated: conc.length, nSpread: spread.length, counts, fact: [], factKind: 'none', factClusters: [] };
}
