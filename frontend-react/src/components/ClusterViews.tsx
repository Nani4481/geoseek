import { Fragment, useMemo } from 'react';
import { tileThumb } from '@/api/client';
import type { ClusterGeo } from '@/api/types';
import { TileImg } from '@/components/Widgets';
import { DASH, fmtInt, fmtPct, regionLabel } from '@/fmt';
import { clusterColor } from '@/lib/clusterColors';
import { CONCENTRATED_AT, SPANS_AT, conceptGroups, coreBox, extentKm, regionColumns, type ClusterStat, type Insight, type Seg } from '@/lib/clusterInsight';

const Segs = ({ s }: { s: Seg[] }) => <>{s.map((x, i) => (x.b ? <b key={i}>{x.t}</b> : <Fragment key={i}>{x.t}</Fragment>))}</>;

/** the row label inside a concept group: the part of the name after the concept ("(kutch) (c13)"), or the whole name when nothing is left */
const inGroup = (s: ClusterStat, concept: string) => { const r = s.name.startsWith(concept) ? s.name.slice(concept.length).trim() : ''; return r || s.name; };

export const Swatch = ({ id, size = 9 }: { id: string; size?: number }) => (
  <i data-swatch={id} style={{ display: 'inline-block', width: size, height: size, borderRadius: 2, background: clusterColor(id), marginRight: 6, flex: 'none' }} />
);

/** Two computed lines above the map: what the run contains, and the single most notable thing in it. */
export function ClusterInterpretation({ insight, source, onPreview, onSelect, selected }: {
  insight: Insight; source: string; onPreview: (id: string | null) => void; onSelect: (id: string) => void; selected: string | null;
}) {
  return (
    <div className="interp" data-testid="cluster-insight">
      <p className="interp-counts" data-testid="insight-counts"><Segs s={insight.counts} /></p>
      {insight.fact.length > 0 && <p className="interp-fact" data-testid="insight-fact" data-kind={insight.factKind}><Segs s={insight.fact} /></p>}
      {insight.factClusters.length > 0 && (
        <div className="interp-chips" data-testid="insight-clusters" aria-label="Clusters this finding is about">
          {insight.factClusters.map((id) => (
            <button key={id} type="button" className={`chip mono ${selected === id ? 'on' : ''}`} data-insight-cluster={id}
              onMouseEnter={() => onPreview(id)} onMouseLeave={() => onPreview(null)} onFocus={() => onPreview(id)} onBlur={() => onPreview(null)} onClick={() => onSelect(id)}
              title="Hover to preview this cluster on the map; click to select it">
              <Swatch id={id} size={8} />#{id}
            </button>
          ))}
        </div>
      )}
      <p className="faint interp-src">Counted from {source}. “Concentrated” = one region holds at least {Math.round(CONCENTRATED_AT * 100)}% of a cluster’s tiles; a region is “spanned” when it holds at least {Math.round(SPANS_AT * 100)}%. Regions are the archive’s staging areas.</p>
    </div>
  );
}

/** What one cluster is: size, closest concept, regional spread, geographic extent and a few real tiles from it. */
export function ClusterSummary({ stat, geo, total, pinned, nRegions, nClusters }: {
  stat: ClusterStat | null; geo: ClusterGeo | null; total: number; pinned: boolean; nRegions: number; nClusters: number;
}) {
  if (!stat) {
    return (
      <div className="csum empty-state" data-testid="cluster-summary" data-cluster="">
        <p><b>{nClusters} clusters</b> are drawn on the map, all at full strength.</p>
        <p className="dim">Hover a cluster in the list, the map or the region matrix to preview what it is; click to select it, zoom to it and dim the rest. Esc clears the selection.</p>
        <p className="faint">Each cluster shows its size, closest concept, the regions it spans, its extent and a few of its tiles.</p>
      </div>
    );
  }
  const g = geo?.clusters[stat.id];
  const ext = g ? extentKm(g.bbox) : null;
  const coreB = g ? coreBox(g.cells) : null, core = coreB ? extentKm(coreB) : null;
  const km = (v: number) => (v >= 10 ? Math.round(v).toLocaleString('en-US') : v.toFixed(1));
  return (
    <div className="csum" data-testid="cluster-summary" data-cluster={stat.id} data-pinned={pinned ? '1' : '0'}>
      <div className="csum-head">
        <Swatch id={stat.id} size={12} />
        <b className="mono">#{stat.id}</b><span className="csum-name">{stat.name}</span>
        <span className={`chip ${pinned ? 'cyan' : ''}`}>{pinned ? 'selected' : 'preview'}</span>
      </div>
      <div className="csum-body">
        <dl className="kv csum-kv">
          <dt>Tiles</dt><dd data-field="tiles">{fmtInt(stat.n)}</dd>
          <dt>Share of archive</dt><dd data-field="share">{fmtPct(stat.n / (total || 1))}</dd>
          <dt>Closest concept</dt><dd data-field="concept" title={stat.conceptScore !== null ? `cosine similarity ${stat.conceptScore.toFixed(3)} to the concept's text embedding` : undefined}>{stat.concept || DASH}{stat.conceptScore !== null && <span className="faint"> · {stat.conceptScore.toFixed(2)}</span>}</dd>
          <dt>Regions spanned</dt>
          <dd data-field="spans" title={`regions holding at least ${Math.round(SPANS_AT * 100)}% of this cluster; ${stat.regions.length} of ${nRegions} have at least one tile`}>
            {stat.spans.length} of {nRegions}<span className="faint"> · {stat.regions.length} touched</span>
          </dd>
          <dt>Extent</dt>
          <dd data-field="extent" title={g ? `lon ${g.bbox[0].toFixed(3)} … ${g.bbox[2].toFixed(3)}, lat ${g.bbox[1].toFixed(3)} … ${g.bbox[3].toFixed(3)} · ${g.n_cells} map cells` : undefined}>
            {ext ? `${km(ext.w)} × ${km(ext.h)} km` : DASH}{g && <span className="faint"> · {fmtInt(g.n_cells)} cells</span>}
          </dd>
          {core && ext && core.w < ext.w * 0.8 && <><dt>Core (95% of tiles)</dt><dd data-field="core" title="the box holding the middle 95% of this cluster's tiles on each axis; the map zooms to it">{km(core.w)} × {km(core.h)} km</dd></>}
        </dl>
        <div className="csum-mix" aria-label="Where the tiles are">
          {stat.regions.slice(0, 4).map((r) => (
            <div key={r.region} className="mixrow" data-region={r.region} title={`${fmtInt(r.n)} tiles`}>
              <span>{regionLabel(r.region)}</span>
              <div className="bar"><i style={{ width: `${r.share * 100}%`, background: 'var(--cyan)' }} /></div>
              <span className="mono">{fmtPct(r.share, r.share < 0.1 ? 1 : 0)}</span>
            </div>
          ))}
          {stat.regions.length > 4 && <div className="faint" style={{ fontSize: 10.5 }}>+ {stat.regions.length - 4} more, {fmtPct(stat.regions.slice(4).reduce((a, r) => a + r.share, 0))} together</div>}
        </div>
        <div className="csum-ex">
          <div className="csum-tiles" data-testid="cluster-examples">
            {(g?.examples ?? []).map((e) => (
              <figure key={e.tile_id} className="csum-tile" data-tile={e.tile_id}>
                <TileImg src={tileThumb(e.tile_id)} alt={`Example tile of cluster ${stat.id} in ${regionLabel(e.region)}`} />
                <figcaption><span>{regionLabel(e.region)}</span><span className="faint">{e.acq_date}</span></figcaption>
              </figure>
            ))}
            {!g?.examples?.length && <div className="faint" style={{ fontSize: 11 }}>No example tiles available for this cluster.</div>}
          </div>
          <div className="faint csum-note">Examples are the clearest tiles (lowest cloud fraction), taken in turn from the regions holding ≥{Math.round(SPANS_AT * 100)}% of the cluster — chosen by a fixed rule, not hand-picked, and not necessarily the most typical.</div>
        </div>
      </div>
    </div>
  );
}

/** Region × cluster: share of each cluster's tiles that lie in each region. Rows are grouped by closest concept, so a concept that
 *  recurs in different regions shows up as several rows lighting different columns. */
export function RegionMatrix({ stats, active, selected, onPreview, onSelect }: {
  stats: ClusterStat[]; active: string | null; selected: string | null; onPreview: (id: string | null) => void; onSelect: (id: string) => void;
}) {
  const cols = useMemo(() => regionColumns(stats), [stats]);
  const groups = useMemo(() => conceptGroups(stats), [stats]);
  return (
    <div className="mx-wrap">
      <table className="mx" aria-label="Region by cluster" data-testid="region-matrix">
        <thead>
          <tr>
            <th className="mx-corner" scope="col">cluster</th>
            {cols.map((c) => <th key={c.region} scope="col" className="mx-col" data-region={c.region} title={`${regionLabel(c.region)} · ${fmtInt(c.tiles)} clustered tiles`}><span>{regionLabel(c.region)}</span></th>)}
          </tr>
        </thead>
        <tbody>
          {groups.map((g) => (
            <Fragment key={g.concept}>
              <tr className="mx-group"><th colSpan={cols.length + 1} scope="colgroup">{g.concept} <span className="faint">· {g.members.length} cluster{g.members.length === 1 ? '' : 's'} · {fmtInt(g.tiles)} tiles</span></th></tr>
              {g.members.map((s) => (
                <tr key={s.id} data-mx-cluster={s.id} className={`${active === s.id ? 'hl' : ''} ${selected === s.id ? 'sel' : ''}`.trim()} tabIndex={0}
                  onMouseEnter={() => onPreview(s.id)} onMouseLeave={() => onPreview(null)} onFocus={() => onPreview(s.id)} onBlur={() => onPreview(null)}
                  onClick={() => onSelect(s.id)} onKeyDown={(e) => { if (e.key === 'Enter') onSelect(s.id); }}>
                  <th scope="row" className="mx-row" title={s.name}><Swatch id={s.id} /><span className="mono">{s.id}</span><span className="mx-name">{inGroup(s, g.concept)}</span></th>
                  {cols.map((c) => {
                    const r = s.regions.find((x) => x.region === c.region);
                    const share = r?.share ?? 0;
                    return (
                      <td key={c.region} data-region={c.region} data-n={r?.n ?? 0} data-share={share.toFixed(4)} className={r && s.dominant === c.region ? 'lead' : ''}
                        style={r ? { background: `rgba(76, 201, 240, ${(0.07 + 0.86 * Math.pow(share, 0.7)).toFixed(3)})` } : undefined}
                        title={`Cluster ${s.id} · ${regionLabel(c.region)}: ${fmtInt(r?.n ?? 0)} of ${fmtInt(s.n)} tiles (${fmtPct(share)})`}>
                        {share >= SPANS_AT ? Math.round(share * 100) : ''}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </Fragment>
          ))}
        </tbody>
      </table>
      <div className="faint mx-note">Each row is one cluster; each cell is the % of that cluster’s tiles lying in that region (rows sum to 100%; cells under {Math.round(SPANS_AT * 100)}% are left blank, hover for the exact count). Rows are grouped by the text concept the cluster is closest to.</div>
    </div>
  );
}
