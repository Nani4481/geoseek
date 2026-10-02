import { useEffect, useMemo, useState } from 'react';
import { api, tileThumb } from '@/api/client';
import type { BBox } from '@/api/types';
import { ClusterInterpretation, ClusterSummary, RegionMatrix, Swatch } from '@/components/ClusterViews';
import { GeoMap, type MapCells } from '@/components/GeoMap';
import { Panel } from '@/components/Panel';
import { ErrorNote, Loading, TileImg } from '@/components/Widgets';
import { fmtInt, fmtLonLat, fmtMs, fmtPct, regionLabel } from '@/fmt';
import { useApi } from '@/hooks/useApi';
import { clusterColor } from '@/lib/clusterColors';
import { clusterStats, coreBox, interpret, regionColumns } from '@/lib/clusterInsight';
import { useStore } from '@/state/store';

export function Discovery() {
  const { label } = useStore();
  const cl = useApi((s) => api.clusters(s), []);
  const top = useApi((s) => api.candidates({ sort: 'queue_score', limit: 30 }, s), []);
  const [seed, setSeed] = useState('');
  const geo = useApi((s) => api.clusterGeo(s), []);
  const [hover, setHover] = useState<string | null>(null);       // cluster id under the pointer / focus (list, matrix or map): a preview
  const [sel, setSel] = useState<string | null>(null);           // the selected cluster: map zoomed to it, the others dimmed
  const [fit, setFit] = useState<BBox | null>(null);
  useEffect(() => { if (!seed && top.data?.candidates[0]) setSeed(top.data.candidates[0].candidate_id); }, [top.data, seed]);
  const sim = useApi(seed ? (s) => api.similarToCandidate(seed, 12, s) : null, [seed]);

  const stats = useMemo(() => clusterStats(cl.data), [cl.data]);
  const insight = useMemo(() => interpret(stats, regionLabel), [stats]);
  const nRegions = useMemo(() => regionColumns(stats).length, [stats]);
  const total = useMemo(() => stats.reduce((a, r) => a + r.n, 0), [stats]);
  const byId = useMemo(() => new Map(stats.map((s) => [s.id, s])), [stats]);
  const max = stats[0]?.n ?? 1;
  const active = hover ?? sel;                                    // what the summary shows and what the map keeps lit
  const cells = useMemo<MapCells[]>(() => Object.entries(geo.data?.clusters ?? {}).map(([id, c]) => ({ id, color: clusterColor(id), cells: c.cells })), [geo.data]);
  const archiveBox = useMemo<BBox | null>(() => {
    const bs = Object.values(geo.data?.clusters ?? {}).map((c) => c.bbox);
    return bs.length ? [Math.min(...bs.map((b) => b[0])), Math.min(...bs.map((b) => b[1])), Math.max(...bs.map((b) => b[2])), Math.max(...bs.map((b) => b[3]))] : null;
  }, [geo.data]);
  useEffect(() => { if (archiveBox) setFit([...archiveBox] as BBox); }, [archiveBox]);
  const showAll = () => { setSel(null); if (archiveBox) setFit([...archiveBox] as BBox); };
  /** select a cluster (zoom to it, dim the rest); selecting the selected one again goes back to the whole archive */
  const choose = (id: string) => {
    if (sel === id) { showAll(); return; }
    setSel(id);
    const g = geo.data?.clusters[id]; const b = g ? coreBox(g.cells) ?? g.bbox : null;   // where 95% of its tiles lie, not the stray ones
    if (b) setFit([...b] as BBox);
  };
  useEffect(() => {
    const k = (e: KeyboardEvent) => { if (e.key === 'Escape' && sel) { setSel(null); if (archiveBox) setFit([...archiveBox] as BBox); } };
    window.addEventListener('keydown', k);
    return () => window.removeEventListener('keydown', k);
  }, [sel, archiveBox]);
  const placed = geo.data ? Object.values(geo.data.clusters).reduce((a, c) => a + c.n_tiles, 0) : 0;
  const nameOf = (id: number | null) => (id === null ? 'unclustered' : byId.get(String(id))?.name ?? `cluster ${id}`);
  const rowEvents = (id: string) => ({
    onMouseEnter: () => setHover(id), onMouseLeave: () => setHover(null), onFocus: () => setHover(id), onBlur: () => setHover(null),
    onClick: () => choose(id), onKeyDown: (e: React.KeyboardEvent) => { if (e.key === 'Enter') choose(id); },
  });
  const algo = typeof cl.data?.params?.algorithm === 'string' ? ` (${cl.data.params.algorithm})` : '';

  return (
    <div className="disc-page">
      <Panel title="What the clustering found" actions={cl.data?.available && <span className="chip mono">{cl.data.n_clusters} clusters · {fmtInt(cl.data.n_tiles)} tiles · {cl.data.noise_count} unclustered</span>}>
        {cl.error ? <ErrorNote error={cl.error} onRetry={cl.reload} /> : !cl.data ? <Loading rows={3} /> : !cl.data.available ? (
          <div className="empty">Clustering has not been run for this archive.</div>
        ) : insight ? (
          <ClusterInterpretation insight={insight} source={`/discovery/clusters, the stored clustering run${algo}`} selected={sel} onPreview={setHover} onSelect={choose} />
        ) : <div className="empty">The clustering run carries no cluster sizes.</div>}
      </Panel>

      <div className="disc-grid">
        <Panel title="Archive clusters" className="disc-list">
          {cl.error ? <ErrorNote error={cl.error} onRetry={cl.reload} /> : !cl.data ? <Loading rows={8} /> : !cl.data.available ? (
            <div className="empty">Clustering has not been run for this archive.</div>
          ) : (
            <table className="tbl cl-list" style={{ cursor: 'default' }} aria-label="Archive clusters">
              <thead><tr><th style={{ width: 50 }}>#</th><th>Cluster</th><th style={{ width: 128 }}>Where</th><th className="num" style={{ width: 66 }}>Tiles</th><th style={{ width: 84 }}>Share</th></tr></thead>
              <tbody>
                {stats.map((r) => (
                  <tr key={r.id} data-cluster={r.id} className={`${hover === r.id ? 'hl' : ''} ${sel === r.id ? 'sel' : ''}`.trim()} style={{ cursor: 'pointer' }} tabIndex={0} {...rowEvents(r.id)}
                    title="Hover to preview this cluster on the map · click to select it and zoom to where it lies">
                    <td className="mono"><Swatch id={r.id} />{r.id}</td>
                    <td className="cl-name" title={r.concept ? `${r.name} — closest concept: ${r.concept}` : r.name}>{r.name}</td>
                    <td className="dim cl-where" data-concentrated={r.concentrated ? '1' : '0'}
                      title={r.regions.slice(0, 4).map((x) => `${regionLabel(x.region)} ${fmtPct(x.share)}`).join(' · ')}>
                      {r.concentrated && r.dominant ? `${regionLabel(r.dominant)} ${fmtPct(r.purity, 0)}` : `${r.spans.length} regions`}
                    </td>
                    <td className="num">{fmtInt(r.n)}</td>
                    <td><div className="bar" style={{ width: '100%' }}><i style={{ width: `${(r.n / max) * 100}%`, background: clusterColor(r.id) }} /></div></td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </Panel>

        <div className="col">
          <Panel title="Cluster map" flush
            actions={(
              <>
                {sel && <button className="btn sm" data-testid="show-all" onClick={showAll} title="Clear the selection (Esc)">Show all clusters</button>}
                {geo.data?.available && <span className="chip mono" title={geo.data.source}>{fmtInt(placed)} tiles placed</span>}
              </>
            )}>
            {geo.error ? <div style={{ padding: 12 }}><ErrorNote error={geo.error} onRetry={geo.reload} /></div> : (
              <GeoMap ariaLabel="Geographic spread of the tile clusters" cells={cells} cellDeg={geo.data?.cell_deg} activeCell={active} onCellHover={setHover} onCellClick={choose} basemap={{}} height={400} fit={fit} fitMaxZoom={11} />
            )}
            <div className="faint" style={{ padding: '7px 12px', fontSize: 11 }}>
              Each square is a ~{((geo.data?.cell_deg ?? 0.03) * 111).toFixed(0)} km cell holding tiles of one cluster, coloured as the swatches in the list; where one place falls in different clusters on different dates the later-drawn one is on top.
              The imagery beneath is one representative acquisition per granule, not the date of the clusters; areas with no staged imagery are dark.
            </div>
          </Panel>
          <Panel title={active ? `Cluster ${active} · what it is` : 'Cluster summary'} grow>
            {!cl.data ? <Loading rows={4} /> : (
              <ClusterSummary stat={active ? byId.get(active) ?? null : null} geo={geo.data ?? null} total={total} pinned={hover === null ? sel !== null : hover === sel} nRegions={nRegions} nClusters={stats.length} />
            )}
          </Panel>
        </div>
      </div>

      <div className="disc-grid">
        <Panel title="Region × cluster · where each cluster occurs">
          {!cl.data ? <Loading rows={6} /> : stats.length === 0 ? <div className="empty">No cluster sizes to tabulate.</div> : !cl.data.region_purity ? (
            <div className="empty">This clustering run does not record per-region counts; re-run scripts/cluster_at_scale.py to produce them.</div>
          ) : <RegionMatrix stats={stats} active={active} selected={sel} onPreview={setHover} onSelect={choose} />}
        </Panel>
        <Panel title="Seed explorer · what else looks like this?"
          actions={sim.data && <span className="chip green mono">{fmtMs(sim.data.latency_ms)} ms</span>}>
          <label className="field" style={{ marginBottom: 10 }}>Seed candidate
            <select className="input" value={seed} onChange={(e) => setSeed(e.target.value)}>
              {top.data?.candidates.map((c) => <option key={c.candidate_id} value={c.candidate_id}>#{c.rank} · {label(c.change_type)} · {c.candidate_id}</option>)}
            </select>
          </label>
          {sim.error ? <ErrorNote error={sim.error} /> : sim.loading || !sim.data ? <Loading rows={4} /> : (
            <>
              <div className="dim" style={{ fontSize: 11.5, marginBottom: 8 }}>Seed tile sits in <b style={{ color: clusterColor(sim.data.seed_cluster) }}>{nameOf(sim.data.seed_cluster)}</b>.</div>
              <div className="fp-tiles">
                {sim.data.results.map((r) => (
                  <div key={r.tile_id} className="fp-tile" style={{ cursor: 'default' }}>
                    <TileImg src={tileThumb(r.tile_id)} alt={`Tile ${r.tile_id}`} />
                    <div className="cap"><span style={{ color: clusterColor(r.cluster) }}>{nameOf(r.cluster)}</span><span>{r.acq_date} · sim {r.score.toFixed(3)}</span><span>{fmtLonLat(r.centroid_lonlat)}</span></div>
                  </div>
                ))}
              </div>
            </>
          )}
          <div className="faint" style={{ fontSize: 10.5, marginTop: 8 }}>Cluster membership comes from the stored clustering run; similarity is an exact nearest-neighbour search over the tile vectors.</div>
        </Panel>
      </div>
    </div>
  );
}
