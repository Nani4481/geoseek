import { useEffect, useMemo, useState } from 'react';
import { api, detectTileImage } from '@/api/client';
import type { Detection } from '@/api/types';
import { Panel } from '@/components/Panel';
import { ErrorNote, Loading } from '@/components/Widgets';
import { DASH, fmtFixed, fmtInt, isNum, regionLabel } from '@/fmt';
import { useApi } from '@/hooks/useApi';

// Styling colours per DOTA class (presentation only).
const CLASS_COLOR: Record<string, string> = {
  'small-vehicle': '#e8c24a', 'large-vehicle': '#ff9f6b', ship: '#34d399', plane: '#4cc9f0', helicopter: '#a78bfa',
  'storage-tank': '#f472b6', harbor: '#5b8cff', bridge: '#ffffff',
};
const color = (c: string) => CLASS_COLOR[c] ?? '#ffffff';

export function Detect() {
  const obs = useApi((s) => api.detectObservations(s), []);
  const model = useApi((s) => api.detectModel(s), []);
  const metrics = useApi((s) => api.metrics(s), []);
  const [obsId, setObsId] = useState<string | null>(null);
  const [tileKey, setTileKey] = useState<{ row: number; col: number } | null>(null);
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const [minScore, setMinScore] = useState(0);
  const [hover, setHover] = useState<number | null>(null);

  useEffect(() => { if (!obsId && obs.data?.observations.length) setObsId([...obs.data.observations].sort((a, b) => b.n_detections - a.n_detections)[0].observation_id); }, [obs.data, obsId]);
  const tiles = useApi(obsId ? (s) => api.detectTiles(obsId, s) : null, [obsId]);
  const sortedTiles = useMemo(() => [...(tiles.data?.tiles ?? [])].sort((a, b) => b.n_detections - a.n_detections), [tiles.data]);
  useEffect(() => { setTileKey(sortedTiles[0] ? { row: sortedTiles[0].row, col: sortedTiles[0].col } : null); }, [sortedTiles]);
  const det = useApi(obsId && tileKey ? (s) => api.detectTile(obsId, tileKey.row, tileKey.col, s) : null, [obsId, tileKey?.row, tileKey?.col]);

  const shown: Detection[] = useMemo(() => (det.data?.detections ?? []).filter((d) => !hidden.has(d.class) && d.score >= minScore), [det.data, hidden, minScore]);
  const counts = useMemo(() => { const c: Record<string, number> = {}; for (const d of det.data?.detections ?? []) c[d.class] = (c[d.class] ?? 0) + 1; return c; }, [det.data]);
  const cur = obs.data?.observations.find((o) => o.observation_id === obsId);
  const dota = metrics.data?.detector.dota_val, xv = metrics.data?.detector.xview_test;
  const hov = hover !== null ? shown[hover] : null;

  return (
    <div className="detect-grid">
      <div className="col">
        <Panel title="Scenes with detections" tier="live">
          {obs.error ? <ErrorNote error={obs.error} onRetry={obs.reload} /> : !obs.data ? <Loading rows={4} /> : (
            <ul className="pick-list" style={{ maxHeight: 260 }}>
              {obs.data.observations.map((o) => (
                <li key={o.observation_id}>
                  <button className={o.observation_id === obsId ? 'sel' : ''} onClick={() => setObsId(o.observation_id)}>
                    <b style={{ fontSize: 12 }}>{regionLabel(o.aoi_name.replace(/-/g, '_'))}</b>
                    <span className="dim mono" style={{ fontSize: 10.5 }}>{fmtInt(o.n_detections)} detections · {o.n_tiles_with_detections}/{o.n_tiles} tiles</span>
                    <span className="faint" style={{ fontSize: 10.5 }}>{o.role.split('(')[0]}</span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </Panel>
        <Panel title="Tiles · most objects first" tier="live">
          {tiles.error ? <ErrorNote error={tiles.error} /> : tiles.loading ? <Loading rows={5} /> : sortedTiles.length === 0 ? <div className="empty">No tile has detections in this scene.</div> : (
            <ul className="pick-list">
              {sortedTiles.slice(0, 80).map((t) => (
                <li key={t.tile_id}>
                  <button className={tileKey?.row === t.row && tileKey?.col === t.col ? 'sel' : ''} onClick={() => setTileKey({ row: t.row, col: t.col })}>
                    <span className="mono" style={{ fontSize: 11.5 }}>r{t.row} · c{t.col}</span>
                    <span className="dim mono" style={{ fontSize: 10.5 }}>{t.n_detections} objects · mean score {t.mean_score.toFixed(2)}</span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </Panel>
      </div>

      <Panel title={`Detections${cur ? ' · ' + regionLabel(cur.aoi_name.replace(/-/g, '_')) : ''}`} tier="live"
        actions={det.data && <span className="chip mono">{shown.length} of {det.data.detections.length} shown</span>}>
        {det.error ? <ErrorNote error={det.error} /> : !obsId || !tileKey ? <Loading rows={4} /> : (
          <div className="detect-stage">
            <img src={detectTileImage(obsId, tileKey.row, tileKey.col)} alt="Scene tile" />
            {det.data && (
              <svg viewBox={`0 0 ${det.data.width} ${det.data.height}`} preserveAspectRatio="none" role="img" aria-label="Oriented bounding boxes">
                {shown.map((d, i) => (
                  <polygon key={i} points={d.polygon_px.map((p) => p.join(',')).join(' ')} fill={color(d.class)} fillOpacity={hover === i ? 0.45 : 0.12}
                    stroke={color(d.class)} strokeWidth={hover === i ? 4 : 2} vectorEffect="non-scaling-stroke"
                    onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)} />
                ))}
              </svg>
            )}
          </div>
        )}
        <div className="legend-row" style={{ marginTop: 10 }}>
          {Object.keys(counts).length === 0 ? <span className="faint">no objects in this tile</span> : Object.entries(counts).map(([c, n]) => (
            <label key={c} style={{ display: 'inline-flex', alignItems: 'center', gap: 5, cursor: 'pointer' }}>
              <input type="checkbox" checked={!hidden.has(c)} onChange={() => setHidden((h) => { const x = new Set(h); if (x.has(c)) x.delete(c); else x.add(c); return x; })} />
              <i style={{ background: color(c), width: 9, height: 9, borderRadius: 2, display: 'inline-block' }} />{c} <span className="mono">{n}</span>
            </label>
          ))}
          <label className="field" style={{ flexDirection: 'row', alignItems: 'center', gap: 8, marginLeft: 'auto' }}>Min score {minScore.toFixed(2)}
            <input type="range" min={0} max={0.95} step={0.05} value={minScore} onChange={(e) => setMinScore(Number(e.target.value))} aria-label="Minimum detection score" /></label>
        </div>
        <div className="mono dim" style={{ fontSize: 11.5, marginTop: 8, minHeight: 18 }}>
          {hov ? `${hov.class} · score ${hov.score.toFixed(3)} · long side ${hov.long_side_px.toFixed(0)} px · heading ${hov.heading_deg.toFixed(0)}°` : 'Hover a box for details.'}
        </div>
      </Panel>

      <div className="col">
        <Panel title="Measured accuracy" tier="live">
          {metrics.error ? <ErrorNote error={metrics.error} /> : !metrics.data ? <Loading rows={5} /> : (
            <div className="col" style={{ gap: 10 }}>
              <div>
                <div className="dim" style={{ fontSize: 10, letterSpacing: '.1em', textTransform: 'uppercase' }}>DOTA official val · {fmtInt(dota?.n_images)} images</div>
                <dl className="kv" style={{ marginTop: 5 }}>
                  <dt>small-vehicle AP50</dt><dd>{fmtFixed(dota?.small_vehicle_ap50)}</dd>
                  <dt>ground vehicles AP50</dt><dd>{fmtFixed(dota?.ground_vehicles_ap50)} <span className="faint">[{dota?.ground_vehicles_ap50_ci?.map((v) => v.toFixed(3)).join('–') ?? DASH}]</span></dd>
                  <dt>all 8 classes AP50</dt><dd>{fmtFixed(dota?.all_classes_ap50)}</dd>
                </dl>
              </div>
              <div>
                <div className="dim" style={{ fontSize: 10, letterSpacing: '.1em', textTransform: 'uppercase' }}>xView independent test (transfer)</div>
                <dl className="kv" style={{ marginTop: 5 }}>
                  <dt>small-vehicle AP50</dt><dd>{fmtFixed(xv?.small_vehicle_ap50)}</dd>
                  <dt>large-vehicle AP50</dt><dd>{fmtFixed(xv?.large_vehicle_ap50)}</dd>
                </dl>
              </div>
              <div className="warnbox">{metrics.data.detector.caveat}</div>
            </div>
          )}
        </Panel>
        <Panel title="Detector & caveats" tier="live">
          {model.error ? <ErrorNote error={model.error} /> : !model.data ? <Loading rows={4} /> : (
            <div className="col" style={{ gap: 8 }}>
              <dl className="kv">
                <dt>Architecture</dt><dd>{model.data.architecture.family}</dd>
                <dt>Parameters</dt><dd>{isNum(model.data.architecture.n_params) ? (model.data.architecture.n_params / 1e6).toFixed(1) + ' M' : DASH}</dd>
                <dt>Weights SHA-256</dt><dd title={model.data.weights_sha256}>{model.data.weights_sha256.slice(0, 16)}…</dd>
              </dl>
              {model.data.caveats.map((c, i) => <div className="warnbox" key={i}>{c}</div>)}
            </div>
          )}
        </Panel>
      </div>
    </div>
  );
}
