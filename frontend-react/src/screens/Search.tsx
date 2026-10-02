import { useMemo, useState } from 'react';
import { api, tileThumb } from '@/api/client';
import type { BBox } from '@/api/types';
import { GeoMap, type MapPoint } from '@/components/GeoMap';
import { Panel } from '@/components/Panel';
import { SpectralEvidence } from '@/components/SpectralEvidence';
import { VectorSpace, type PickedTile } from '@/components/VectorSpace';
import { ErrorNote, Loading, TileImg } from '@/components/Widgets';
import { fmtLonLat, fmtMs, regionLabel } from '@/fmt';
import { fitPoints } from '@/lib/mapfit';
import { useApi } from '@/hooks/useApi';
import { href } from '@/router';
import { useStore } from '@/state/store';

interface Hit { tile_id: string; score: number; lon: number; lat: number; acq_date: string; sensor?: string; cloud?: number; cluster?: number | null }
interface Result { mode: 'text' | 'tile' | 'point'; title: string; latency: number; hits: Hit[]; query: string | null }

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
  const [evidenceTile, setEvidenceTile] = useState<string | null>(null);
  const [focus, setFocus] = useState<PickedTile | null>(null);
  const [hover, setHover] = useState<string | null>(null);   // tile id hovered on a card or on its map pin
  const [outside, setOutside] = useState<number[]>([]);      // result numbers left out of the map frame (far from the rest)

  const suggestions = useMemo(() => {
    const out = new Set<string>();
    if (summary?.demo.search_query) out.add(summary.demo.search_query);
    for (const v of Object.values(clusters.data?.cluster_concepts ?? {})) { if (v[0]?.[0]) out.add(v[0][0]); if (out.size >= 7) break; }
    return [...out];
  }, [summary, clusters.data]);

  const run = async <T,>(fn: () => Promise<T>, done: (r: T) => Result) => {
    setBusy(true); setErr(null);
    try {
      const r = done(await fn());
      setRes(r); setHover(null);
      const f = fitPoints(r.hits);                               // frame the results on EVERY search, not just the first
      setOutside(f.outside.map((i) => i + 1));
      if (f.bbox) setFit(f.bbox);
    } catch (e) { setRes(null); setErr(e instanceof Error ? e.message : String(e)); } finally { setBusy(false); }
  };

  const searchText = (text = q) => {
    if (!text.trim()) return;
    setQ(text);
    run(() => api.searchText({ q: text.trim(), k, bbox, date_start: dateStart || undefined, date_end: dateEnd || undefined, max_cloud_fraction: cloud < 1 ? cloud : undefined }),
      (r) => ({ mode: 'text', query: r.query ?? text.trim(), title: `“${r.query}”`, latency: r.latency_ms, hits: r.results.map((h) => ({ tile_id: h.tile_id, score: h.score, lon: h.lon, lat: h.lat, acq_date: h.acq_date, sensor: h.sensor, cloud: h.cloud_fraction })) }));
  };
  const moreLikeTile = (tile: string) =>
    run(() => api.searchImage({ tile_id: tile, k, bbox }),
      (r) => ({ mode: 'tile', query: null, title: `similar to tile ${tile}`, latency: r.latency_ms, hits: r.results.map((h) => ({ tile_id: h.tile_id, score: h.score, lon: h.lon, lat: h.lat, acq_date: h.acq_date, sensor: h.sensor, cloud: h.cloud_fraction })) }));
  const moreLikePoint = (lon: number, lat: number) =>
    run(() => api.similarAt(lon, lat, Math.min(k, 50)),
      (r) => ({ mode: 'point', query: null, title: `similar to the map point ${fmtLonLat([lon, lat])}`, latency: r.latency_ms, hits: r.results.map((h) => ({ tile_id: h.tile_id, score: h.score, lon: h.centroid_lonlat[0], lat: h.centroid_lonlat[1], acq_date: h.acq_date, cluster: h.cluster })) }));

  const regionBoxes = useMemo(() => (regions.data?.regions ?? []).map((r) => ({ name: r.name, bbox: r.bbox, label: regionLabel(r.name) })), [regions.data]);
  const points: MapPoint[] = useMemo(() => {
    const out: MapPoint[] = (res?.hits ?? []).map((h, i) => ({ id: h.tile_id, lon: h.lon, lat: h.lat, pin: { text: String(i + 1) }, label: `#${i + 1} · ${h.acq_date} · similarity ${h.score.toFixed(3)} · click for more like this` }));
    if (focus) out.push({ id: 'focus:' + focus.tile_id, lon: focus.lon, lat: focus.lat, label: `tile ${focus.tile_id} (picked in the embedding space)`, color: '#ffffff', radius: 9 });
    return out;
  }, [res, focus]);
  const hitIds = useMemo(() => (res?.hits ?? []).map((h) => h.tile_id), [res]);
  const pickTile = (t: PickedTile) => { setFocus(t); setFit([t.lon - 0.02, t.lat - 0.02, t.lon + 0.02, t.lat + 0.02]); };

  return (
    <div className="search-grid">
      <div className="col">
        <Panel title="Semantic search">
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

        {evidenceTile && <SpectralEvidence tileId={evidenceTile} query={res?.mode === 'text' ? res.query : null} onClose={() => setEvidenceTile(null)} />}
        <Panel title={res ? `Results · ${res.title}` : 'Results'}
          actions={res && <span className="chip green mono" title="latency reported by the search engine for this request">{fmtMs(res.latency)} ms · {res.hits.length} hits</span>}>
          {err && <ErrorNote error={err} />}
          {busy ? <Loading rows={5} /> : !res ? (
            <div className="empty">Run a search, or click anywhere on the map to find the places that look most like that spot.</div>
          ) : res.hits.length === 0 ? <div className="empty">No tiles matched the query and filters.</div> : (
            <div className="results">
              {res.hits.map((h, i) => (
                <article key={h.tile_id} className={`rcard ${hover === h.tile_id ? 'hl' : ''}`.trim()} data-tile={h.tile_id} data-n={i + 1}
                  onMouseEnter={() => setHover(h.tile_id)} onMouseLeave={() => setHover(null)}>
                  <div className="thumbwrap"><TileImg src={tileThumb(h.tile_id)} alt={`Tile ${h.tile_id}`} /><span className="rk">#{i + 1}</span></div>
                  <div className="meta">
                    <span className="mono">{h.acq_date}{h.sensor ? ` · ${h.sensor}` : ''}</span>
                    <span className="dim mono" style={{ fontSize: 10 }}>{fmtLonLat([h.lon, h.lat])}</span>
                    <span className="dim mono" style={{ fontSize: 10 }}>similarity {h.score.toFixed(3)}{h.cloud !== undefined ? ` · cloud ${Math.round(h.cloud * 100)}%` : ''}{h.cluster !== undefined && h.cluster !== null ? ` · cluster ${h.cluster}` : ''}</span>
                    <div className="actions">
                      <button className="btn sm" onClick={() => moreLikeTile(h.tile_id)} title="More like this: image-to-image search seeded with this tile">Similar</button>
                      <a className="btn sm" href={href('fingerprints', 'tile:' + h.tile_id)} title="Open the structural fingerprint gallery for this tile">Compare</a>
                      <button className="btn sm" data-evidence={h.tile_id} onClick={() => setEvidenceTile(h.tile_id)} title="Per-pixel NDWI / NDBI / NDVI of this tile: the measured reasons it looks like your query">Evidence</button>
                    </div>
                  </div>
                </article>
              ))}
            </div>
          )}
        </Panel>
      </div>

      <div className="col">
      <Panel title="Map · spatial filter" flush
        actions={(
          <>
            {bbox && (
              <span className="chip amber mono" data-testid="map-bbox-chip" title="The search is restricted to this box (west, south, east, north)">
                box {bbox.map((v) => v.toFixed(2)).join(', ')}
                <button className="chip-x" onClick={() => setBbox(null)} aria-label="Clear the search box" title="Clear the search box">✕</button>
              </span>
            )}
            <button className={`btn sm ${draw ? 'on drawing' : ''}`} onClick={() => setDraw((d) => !d)} aria-pressed={draw}
              title="Drag a rectangle on the map to restrict the search to it">{draw ? '✎ Drawing… (Esc)' : '▭ Draw box'}</button>
          </>
        )}>
        <GeoMap ariaLabel="Search map" regions={regionBoxes} points={points} bbox={bbox} fit={fit} drawMode={draw} height={540} basemap={{}}
          hoverId={hover} onHover={setHover} onCancelDraw={() => setDraw(false)}
          onBBox={(b) => { setBbox(b); setDraw(false); }}
          onMapClick={(lon, lat) => moreLikePoint(lon, lat)} onPointClick={(id) => moreLikeTile(id)} />
        {outside.length > 0 && (
          <div className="warnbox" style={{ margin: '8px 12px 0' }} data-testid="map-outside">
            {outside.length === 1 ? `Result #${outside[0]} lies` : `Results ${outside.map((n) => '#' + n).join(', ')} lie`} far from the others and outside this map view; the cards below still show {outside.length === 1 ? 'it' : 'them'}.
          </div>
        )}
        <div className="faint" style={{ padding: '7px 12px', fontSize: 11 }}>Numbered pins match the result cards (hover either to see the other). Click the map → find places that look like that point · click a pin → more like that tile · “Draw box” → restrict the search area.</div>
      </Panel>
        <VectorSpace hits={hitIds} selected={focus?.tile_id ?? null} onPick={pickTile} />
      </div>
    </div>
  );
}
