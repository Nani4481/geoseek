import { useMemo, useState } from 'react';
import { api, tileThumb } from '@/api/client';
import type { BBox } from '@/api/types';
import { GeoMap, type MapPoint } from '@/components/GeoMap';
import { Panel } from '@/components/Panel';
import { ErrorNote, Loading, TileImg } from '@/components/Widgets';
import { fmtLonLat, fmtMs, regionLabel } from '@/fmt';
import { useApi } from '@/hooks/useApi';
import { href } from '@/router';
import { useStore } from '@/state/store';

interface Hit { tile_id: string; score: number; lon: number; lat: number; acq_date: string; sensor?: string; cloud?: number; cluster?: number | null }
interface Result { mode: 'text' | 'tile' | 'point'; title: string; latency: number; hits: Hit[] }

export function Search() {
  const { summary } = useStore();
  const regions = useApi((s) => api.regions(s), []);
  const clusters = useApi((s) => api.clusters(s), []);
  const [q, setQ] = useState('');
  const [k, setK] = useState(20);
  const [dateStart, setDateStart] = useState('');
  const [dateEnd, setDateEnd] = useState('');
  const [cloud, setCloud] = useState(1);
  const [bbox, setBbox] = useState<BBox | null>(null);
  const [fit, setFit] = useState<BBox | null>(null);
  const [draw, setDraw] = useState(false);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [res, setRes] = useState<Result | null>(null);

  const suggestions = useMemo(() => {
    const out = new Set<string>();
    if (summary?.demo.search_query) out.add(summary.demo.search_query);
    for (const v of Object.values(clusters.data?.cluster_concepts ?? {})) { if (v[0]?.[0]) out.add(v[0][0]); if (out.size >= 7) break; }
    return [...out];
  }, [summary, clusters.data]);

  const run = async <T,>(fn: () => Promise<T>, done: (r: T) => Result) => {
    setBusy(true); setErr(null);
    try { setRes(done(await fn())); } catch (e) { setRes(null); setErr(e instanceof Error ? e.message : String(e)); } finally { setBusy(false); }
  };

  const searchText = (text = q) => {
    if (!text.trim()) return;
    setQ(text);
    run(() => api.searchText({ q: text.trim(), k, bbox, date_start: dateStart || undefined, date_end: dateEnd || undefined, max_cloud_fraction: cloud < 1 ? cloud : undefined }),
      (r) => ({ mode: 'text', title: `“${r.query}”`, latency: r.latency_ms, hits: r.results.map((h) => ({ tile_id: h.tile_id, score: h.score, lon: h.lon, lat: h.lat, acq_date: h.acq_date, sensor: h.sensor, cloud: h.cloud_fraction })) }));
  };
  const moreLikeTile = (tile: string) =>
    run(() => api.searchImage({ tile_id: tile, k, bbox }),
      (r) => ({ mode: 'tile', title: `similar to tile ${tile}`, latency: r.latency_ms, hits: r.results.map((h) => ({ tile_id: h.tile_id, score: h.score, lon: h.lon, lat: h.lat, acq_date: h.acq_date, sensor: h.sensor, cloud: h.cloud_fraction })) }));
  const moreLikePoint = (lon: number, lat: number) =>
    run(() => api.similarAt(lon, lat, Math.min(k, 50)),
      (r) => ({ mode: 'point', title: `similar to the map point ${fmtLonLat([lon, lat])}`, latency: r.latency_ms, hits: r.results.map((h) => ({ tile_id: h.tile_id, score: h.score, lon: h.centroid_lonlat[0], lat: h.centroid_lonlat[1], acq_date: h.acq_date, cluster: h.cluster })) }));

  const regionBoxes = useMemo(() => (regions.data?.regions ?? []).map((r) => ({ name: r.name, bbox: r.bbox, label: regionLabel(r.name) })), [regions.data]);
  const points: MapPoint[] = useMemo(() => (res?.hits ?? []).map((h, i) => ({ id: h.tile_id, lon: h.lon, lat: h.lat, label: `#${i + 1} ${h.acq_date}`, color: i === 0 ? '#f5a524' : '#4cc9f0', radius: i === 0 ? 7 : 5 })), [res]);

  return (
    <div className="search-grid">
      <div className="col">
        <Panel title="Semantic search" tier="live">
          <form className="col" style={{ gap: 10 }} onSubmit={(e) => { e.preventDefault(); searchText(); }}>
            <div className="row" style={{ gap: 8 }}>
              <input className="input" style={{ flex: 1, fontSize: 14, padding: '9px 12px' }} value={q} onChange={(e) => setQ(e.target.value)}
                placeholder="Describe what you are looking for — e.g. an open water reservoir or pond" aria-label="Search query" />
              <button className="btn primary" type="submit" disabled={busy || !q.trim()}>Search</button>
            </div>
            <div className="row" style={{ flexWrap: 'wrap', gap: 6 }}>
              <span className="dim" style={{ fontSize: 11 }}>Try:</span>
              {suggestions.map((s) => <button type="button" key={s} className="chip cyan" style={{ cursor: 'pointer' }} onClick={() => searchText(s)}>{s}</button>)}
            </div>
            <div className="filters">
              <label className="field">Results
                <select className="input" value={k} onChange={(e) => setK(Number(e.target.value))}>{[10, 20, 30, 50].map((n) => <option key={n} value={n}>{n}</option>)}</select>
              </label>
              <label className="field">From<input className="input" type="date" value={dateStart} onChange={(e) => setDateStart(e.target.value)} /></label>
              <label className="field">To<input className="input" type="date" value={dateEnd} onChange={(e) => setDateEnd(e.target.value)} /></label>
              <label className="field" style={{ width: 150 }}>Max cloud {cloud < 1 ? `${Math.round(cloud * 100)}%` : 'any'}
                <input type="range" min={0.05} max={1} step={0.05} value={cloud} onChange={(e) => setCloud(Number(e.target.value))} aria-label="Maximum cloud fraction" /></label>
              <label className="field">Region
                <select className="input" value="" onChange={(e) => { const r = regions.data?.regions.find((x) => x.name === e.target.value); if (r) { setBbox(r.bbox); setFit(r.bbox); } }}>
                  <option value="">Set box from region…</option>{regions.data?.regions.map((r) => <option key={r.name} value={r.name}>{regionLabel(r.name)}</option>)}
                </select>
              </label>
            </div>
          </form>
          <div className="row" style={{ alignItems: 'center', gap: 8, marginTop: 10, flexWrap: 'wrap' }}>
            <span className="dim" style={{ fontSize: 11 }}>Spatial filter:</span>
            {bbox ? <span className="chip cyan mono">{bbox.map((v) => v.toFixed(2)).join(', ')}</span> : <span className="chip">whole archive</span>}
            {bbox && <button className="btn sm" onClick={() => setBbox(null)}>Clear box</button>}
          </div>
        </Panel>

        <Panel title={res ? `Results · ${res.title}` : 'Results'} tier="live"
          actions={res && <span className="chip green mono" title="latency reported by the search engine for this request">{fmtMs(res.latency)} ms · {res.hits.length} hits</span>}>
          {err && <ErrorNote error={err} />}
          {busy ? <Loading rows={5} /> : !res ? (
            <div className="empty">Run a search, or click anywhere on the map to find the places that look most like that spot.</div>
          ) : res.hits.length === 0 ? <div className="empty">No tiles matched the query and filters.</div> : (
            <div className="results">
              {res.hits.map((h, i) => (
                <article key={h.tile_id} className="rcard">
                  <div className="thumbwrap"><TileImg src={tileThumb(h.tile_id)} alt={`Tile ${h.tile_id}`} /><span className="rk">#{i + 1}</span></div>
                  <div className="meta">
                    <span className="mono">{h.acq_date}{h.sensor ? ` · ${h.sensor}` : ''}</span>
                    <span className="dim mono" style={{ fontSize: 10 }}>{fmtLonLat([h.lon, h.lat])}</span>
                    <span className="dim mono" style={{ fontSize: 10 }}>similarity {h.score.toFixed(3)}{h.cloud !== undefined ? ` · cloud ${Math.round(h.cloud * 100)}%` : ''}{h.cluster !== undefined && h.cluster !== null ? ` · cluster ${h.cluster}` : ''}</span>
                    <div className="row" style={{ gap: 5, marginTop: 4 }}>
                      <button className="btn sm" onClick={() => moreLikeTile(h.tile_id)} title="Image-to-image search seeded with this tile">More like this</button>
                      <a className="btn sm" href={href('fingerprints', 'tile:' + h.tile_id)} title="Open the structural fingerprint gallery for this tile">Compare</a>
                    </div>
                  </div>
                </article>
              ))}
            </div>
          )}
        </Panel>
      </div>

      <Panel title="Map · spatial filter" tier="live" flush style={{ position: 'sticky', top: 0 }}
        actions={<button className={`btn sm ${draw ? 'on' : ''}`} onClick={() => setDraw((d) => !d)} aria-pressed={draw}>{draw ? 'Drag on map…' : 'Draw box'}</button>}>
        <GeoMap ariaLabel="Search map" regions={regionBoxes} points={points} bbox={bbox} fit={fit} drawMode={draw} height={520}
          onBBox={(b) => { setBbox(b); setDraw(false); }}
          onMapClick={(lon, lat) => moreLikePoint(lon, lat)} onPointClick={(id) => moreLikeTile(id)} />
        <div className="faint" style={{ padding: '7px 12px', fontSize: 11 }}>Click the map → find places that look like that point · click a result dot → more like that tile · “Draw box” → restrict the search area.</div>
      </Panel>
    </div>
  );
}
