import { useEffect, useMemo, useState } from 'react';
import { api, tileThumb } from '@/api/client';
import type { BBox } from '@/api/types';
import { GeoMap, type MapCells } from '@/components/GeoMap';
import { Panel } from '@/components/Panel';
import { ErrorNote, Loading, TileImg } from '@/components/Widgets';
import { fmtInt, fmtLonLat, fmtMs } from '@/fmt';
import { useApi } from '@/hooks/useApi';
import { clusterColor } from '@/lib/clusterColors';
import { useStore } from '@/state/store';

export function Discovery() {
  const { label } = useStore();
  const cl = useApi((s) => api.clusters(s), []);
  const top = useApi((s) => api.candidates({ sort: 'queue_score', limit: 30 }, s), []);
  const [seed, setSeed] = useState('');
  const geo = useApi((s) => api.clusterGeo(s), []);
  const [hover, setHover] = useState<string | null>(null);       // cluster id under the pointer / focus in the list
  const [fit, setFit] = useState<BBox | null>(null);
  const [zoomed, setZoomed] = useState<string | null>(null);
  useEffect(() => { if (!seed && top.data?.candidates[0]) setSeed(top.data.candidates[0].candidate_id); }, [top.data, seed]);
  const sim = useApi(seed ? (s) => api.similarToCandidate(seed, 12, s) : null, [seed]);

  const rows = useMemo(() => {
    const d = cl.data;
    if (!d?.available || !d.sizes) return [];
    const total = Object.values(d.sizes).reduce((a, b) => a + b, 0) || 1;
    return Object.entries(d.sizes).map(([id, n]) => ({
      id, n, share: n / total,
      name: d.display_labels?.[id] ?? d.cluster_concepts?.[id]?.[0]?.[0] ?? `cluster ${id}`,
      concept: d.cluster_concepts?.[id]?.[0]?.[0] ?? '',
    })).sort((a, b) => b.n - a.n);
  }, [cl.data]);
  const max = rows[0]?.n ?? 1;
  const cells = useMemo<MapCells[]>(() => Object.entries(geo.data?.clusters ?? {}).map(([id, c]) => ({ id, color: clusterColor(id), cells: c.cells })), [geo.data]);
  const archiveBox = useMemo<BBox | null>(() => {
    const bs = Object.values(geo.data?.clusters ?? {}).map((c) => c.bbox);
    return bs.length ? [Math.min(...bs.map((b) => b[0])), Math.min(...bs.map((b) => b[1])), Math.max(...bs.map((b) => b[2])), Math.max(...bs.map((b) => b[3]))] : null;
  }, [geo.data]);
  useEffect(() => { if (archiveBox) setFit([...archiveBox] as BBox); }, [archiveBox]);
  const zoomTo = (id: string) => { const b = geo.data?.clusters[id]?.bbox; if (b) { setZoomed(id); setFit([...b] as BBox); } };
  const placed = geo.data ? Object.values(geo.data.clusters).reduce((a, c) => a + c.n_tiles, 0) : 0;
  const nameOf = (id: number | null) => (id === null ? 'unclustered' : rows.find((r) => r.id === String(id))?.name ?? `cluster ${id}`);

  return (
    <div className="disc-grid">
      <div className="col">
        <Panel title="Archive clusters"
          actions={cl.data?.available && <span className="chip mono">{cl.data.n_clusters} clusters · {fmtInt(cl.data.n_tiles)} tiles · {cl.data.noise_count} unclustered</span>}>
          {cl.error ? <ErrorNote error={cl.error} onRetry={cl.reload} /> : !cl.data ? <Loading rows={8} /> : !cl.data.available ? (
            <div className="empty">Clustering has not been run for this archive.</div>
          ) : (
            <table className="tbl" style={{ cursor: 'default' }}>
              <thead><tr><th>#</th><th>Cluster</th><th>Closest concept</th><th className="num">Tiles</th><th style={{ width: '28%' }}>Share</th></tr></thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={r.id} data-cluster={r.id} className={hover === r.id ? 'hl' : zoomed === r.id ? 'sel' : ''} style={{ cursor: 'pointer' }} tabIndex={0}
                    onMouseEnter={() => setHover(r.id)} onMouseLeave={() => setHover(null)} onFocus={() => setHover(r.id)} onBlur={() => setHover(null)}
                    onClick={() => zoomTo(r.id)} onKeyDown={(e) => { if (e.key === 'Enter') zoomTo(r.id); }}
                    title="Hover to highlight this cluster on the map · click to zoom to where it lies">
                    <td className="mono"><i data-swatch={r.id} style={{ display: 'inline-block', width: 9, height: 9, borderRadius: 2, background: clusterColor(r.id), marginRight: 6 }} />{r.id}</td>
                    <td>{r.name}</td><td className="dim">{r.concept}</td>
                    <td className="num">{fmtInt(r.n)}</td>
                    <td><div className="bar" style={{ width: '100%' }}><i style={{ width: `${(r.n / max) * 100}%`, background: clusterColor(r.id) }} /></div></td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </Panel>
      </div>

      <div className="col">
        <Panel title="Cluster map" flush
          actions={(
            <>
              {zoomed && <button className="btn sm" onClick={() => { setZoomed(null); if (archiveBox) setFit([...archiveBox] as BBox); }}>Show the whole archive</button>}
              {geo.data?.available && <span className="chip mono" title={geo.data.source}>{fmtInt(placed)} tiles placed</span>}
            </>
          )}>
          {geo.error ? <div style={{ padding: 12 }}><ErrorNote error={geo.error} onRetry={geo.reload} /></div> : (
            <GeoMap ariaLabel="Geographic spread of the tile clusters" cells={cells} cellDeg={geo.data?.cell_deg} activeCell={hover ?? zoomed} basemap={{}} height={500} fit={fit} fitMaxZoom={11} />
          )}
          <div className="faint" style={{ padding: '7px 12px', fontSize: 11 }}>
            Each square is a ~{((geo.data?.cell_deg ?? 0.03) * 111).toFixed(0)} km cell holding tiles of one cluster, coloured as in the list. Where one place belongs to different clusters on different dates, the later-drawn cluster is on top.
            Hover a row to highlight that cluster; click it to zoom to it.
          </div>
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
