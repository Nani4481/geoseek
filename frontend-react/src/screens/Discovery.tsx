import { useEffect, useMemo, useState } from 'react';
import { CLUSTER_MAP_URL, api, tileThumb } from '@/api/client';
import { Panel } from '@/components/Panel';
import { ErrorNote, Loading, TileImg } from '@/components/Widgets';
import { fmtInt, fmtLonLat, fmtMs } from '@/fmt';
import { useApi } from '@/hooks/useApi';
import { useStore } from '@/state/store';

const PALETTE = ['#4cc9f0', '#f5a524', '#34d399', '#a78bfa', '#ff6b7a', '#5b8cff', '#ff9f6b', '#f472b6', '#9be15d', '#e8c24a'];
const clusterColor = (id: number | string | null | undefined) => (id === null || id === undefined ? '#62729f' : PALETTE[Number(id) % PALETTE.length]);

export function Discovery() {
  const { label } = useStore();
  const cl = useApi((s) => api.clusters(s), []);
  const top = useApi((s) => api.candidates({ sort: 'queue_score', limit: 30 }, s), []);
  const [seed, setSeed] = useState('');
  const [mapBad, setMapBad] = useState(false);
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
  const nameOf = (id: number | null) => (id === null ? 'unclustered' : rows.find((r) => r.id === String(id))?.name ?? `cluster ${id}`);

  return (
    <div className="disc-grid">
      <div className="col">
        <Panel title="Archive clusters" tier="live"
          actions={cl.data?.available && <span className="chip mono">{cl.data.n_clusters} clusters · {fmtInt(cl.data.n_tiles)} tiles · {cl.data.noise_count} unclustered</span>}>
          {cl.error ? <ErrorNote error={cl.error} onRetry={cl.reload} /> : !cl.data ? <Loading rows={8} /> : !cl.data.available ? (
            <div className="empty">Clustering has not been run for this archive.</div>
          ) : (
            <table className="tbl" style={{ cursor: 'default' }}>
              <thead><tr><th>#</th><th>Cluster</th><th>Closest concept</th><th className="num">Tiles</th><th style={{ width: '28%' }}>Share</th></tr></thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={r.id} style={{ cursor: 'default' }}>
                    <td className="mono"><i style={{ display: 'inline-block', width: 9, height: 9, borderRadius: 2, background: clusterColor(r.id), marginRight: 6 }} />{r.id}</td>
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
        <Panel title="Cluster map" tier="live">
          {mapBad ? <div className="empty">Cluster map image has not been generated.</div> : <img className="cluster-img" src={CLUSTER_MAP_URL} alt="Geographic scatter of tile clusters" onError={() => setMapBad(true)} />}
        </Panel>
        <Panel title="Seed explorer · what else looks like this?" tier="live"
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
