import { useEffect, useMemo, useState } from 'react';
import { api, detectTileImage } from '@/api/client';
import type { BBox, Detection, DetectObservation } from '@/api/types';
import { GeoMap, type MapPoint } from '@/components/GeoMap';
import { Panel } from '@/components/Panel';
import { ThreatRingsResults, useThreatRings } from '@/components/ThreatRings';
import { ErrorNote, Loading } from '@/components/Widgets';
import { DASH, fmtFixed, fmtInt, fmtLonLat, isNum, regionLabel } from '@/fmt';
import { useApi } from '@/hooks/useApi';

// Styling colours per DOTA class (presentation only).
const CLASS_COLOR: Record<string, string> = {
  'small-vehicle': '#e8c24a', 'large-vehicle': '#ff9f6b', ship: '#34d399', plane: '#4cc9f0', helicopter: '#a78bfa',
  'storage-tank': '#f472b6', harbor: '#5b8cff', bridge: '#ffffff',
};
const color = (c: string) => CLASS_COLOR[c] ?? '#ffffff';

const tileRC = (tileId: string) => { const m = tileId.match(/_r(\d+)_c(\d+)$/); return m ? { row: Number(m[1]), col: Number(m[2]) } : null; };
// marker size / opacity follow the detector's confidence score: t is the score's position in this scene's own range
const MARKER_R = [1.3, 3.4], MARKER_OP = [0.3, 0.95];
const lerp = (r: number[], t: number) => r[0] + (r[1] - r[0]) * t;

/** Every stored detection of the scene as an oriented-box marker on the scene's own imagery. Right-click one to draw threat rings. */
function DetectionMap({ obs, onOpenTile }: { obs: DetectObservation | undefined; onOpenTile: (row: number, col: number) => void }) {
  const obsId = obs?.observation_id ?? null;
  const pts = useApi(obsId ? (s) => api.detectionPoints(obsId, s) : null, [obsId]);
  const tr = useThreatRings();
  const setCenter = tr.setCenter;
  const [minScore, setMinScore] = useState(0);
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const [hover, setHover] = useState<string | null>(null);
  const [picked, setPicked] = useState<string | null>(null);
  useEffect(() => { setCenter(null); setPicked(null); setHidden(new Set()); setMinScore(0); }, [obsId, setCenter]);

  const all = pts.data?.points ?? [];
  const byId = useMemo(() => new Map(all.map((p) => [p.id, p])), [all]);
  const range = useMemo(() => {
    const sc = all.map((p) => p.score);
    return sc.length ? { lo: Math.min(...sc), hi: Math.max(...sc), mid: [...sc].sort((x, y) => x - y)[Math.floor(sc.length / 2)] } : null;
  }, [all]);
  const totals = useMemo(() => { const c: Record<string, number> = {}; for (const p of all) c[p.class] = (c[p.class] ?? 0) + 1; return c; }, [all]);
  const classes = useMemo(() => Object.entries(totals).sort((x, y) => y[1] - x[1]), [totals]);
  const visible = useMemo(() => all.filter((p) => !hidden.has(p.class) && p.score >= minScore), [all, hidden, minScore]);
  const shown = useMemo(() => { const c: Record<string, number> = {}; for (const p of visible) c[p.class] = (c[p.class] ?? 0) + 1; return c; }, [visible]);

  const base = useMemo<MapPoint[]>(() => visible.map((p) => {
    const t = range && range.hi > range.lo ? (p.score - range.lo) / (range.hi - range.lo) : 1;
    const size = isNum(p.length_m) && isNum(p.width_m) ? ` · box ${p.length_m.toFixed(1)} × ${p.width_m.toFixed(1)} m` : '';
    return { id: p.id, lon: p.lon, lat: p.lat, color: color(p.class), radius: lerp(MARKER_R, t), opacity: lerp(MARKER_OP, t),
      label: `${p.class} · confidence ${p.score.toFixed(3)}${size} · click for details, right-click for threat rings` };
  }), [visible, range]);
  const points = useMemo(() => [...base, ...tr.overlay], [base, tr.overlay]);
  const fit = useMemo<BBox | null>(() => {
    if (!all.length) return null;
    const xs = all.map((p) => p.lon), ys = all.map((p) => p.lat);
    return [Math.min(...xs) - 0.0008, Math.min(...ys) - 0.0008, Math.max(...xs) + 0.0008, Math.max(...ys) + 0.0008];
  }, [all]);
  const centreOn = (id: string) => {
    const real = id.startsWith('ring:') ? id.slice(5) : id;
    const p = byId.get(real);
    if (p) tr.setCenter({ id: real, lon: p.lon, lat: p.lat, label: `${p.class} · ${p.tile_id}` });
  };
  const sel = picked ? byId.get(picked) : undefined;
  const rc = sel ? tileRC(sel.tile_id) : null;
  const toggle = (c: string) => setHidden((h) => { const x = new Set(h); if (x.has(c)) x.delete(c); else x.add(c); return x; });

  return (
    <Panel title="Detection map · threat rings"
      actions={pts.data && <span className="chip mono" data-testid="map-shown">{visible.length === all.length ? fmtInt(all.length) : `${fmtInt(visible.length)} of ${fmtInt(all.length)}`} detections</span>}>
      {pts.error ? <ErrorNote error={pts.error} onRetry={pts.reload} /> : (
        <div className="col" style={{ gap: 10 }}>
          {obs && (
            <p className="dim" style={{ fontSize: 11.5, lineHeight: 1.5 }} data-testid="map-subtitle">
              <b style={{ color: 'var(--ink-2)' }}>{regionLabel(obs.aoi_name.replace(/-/g, '_'))}</b>
              {' · '}acquired <b className="mono" style={{ color: 'var(--ink-2)' }}>{obs.acquired_at ?? DASH}</b>
              {' · '}{[obs.platform, obs.sensor].filter(Boolean).join(' · ') || DASH}{isNum(obs.native_gsd_m) ? ` · ${obs.native_gsd_m.toFixed(2)} m pixels` : ''}.
              {' '}Each marker is <b>one oriented detection</b> (the centre of one detected box) found by the detector in this scene's imagery.
            </p>
          )}
          <div style={{ borderRadius: 8, overflow: 'hidden' }}>
            <GeoMap ariaLabel="Detection map" points={points} circles={tr.circles} fit={tr.fit ?? fit} height={380} fitMaxZoom={19}
              basemap={obsId ? { scene: obsId } : undefined} hoverId={hover ?? picked} onHover={setHover}
              onPointClick={(id) => { if (byId.has(id)) setPicked(id); }} onPointContext={centreOn} />
          </div>
          <div className="map-legend" data-testid="map-legend">
            <div className="legend-classes">
              {classes.map(([c, n]) => (
                <label key={c} className={hidden.has(c) ? 'off' : ''} data-class={c} data-shown={shown[c] ?? 0} data-total={n}>
                  <input type="checkbox" checked={!hidden.has(c)} onChange={() => toggle(c)} aria-label={`Show ${c}`} />
                  <i style={{ background: color(c) }} />{c}
                  <span className="mono">{fmtInt(shown[c] ?? 0)}<span className="faint"> / {fmtInt(n)}</span></span>
                </label>
              ))}
            </div>
            <label className="field legend-slider">Min confidence {minScore.toFixed(2)}
              <input type="range" min={0} max={0.95} step={0.05} value={minScore} onChange={(e) => setMinScore(Number(e.target.value))} aria-label="Minimum confidence on the map" />
            </label>
            {range && (
              <div className="legend-size" data-testid="map-size-legend" title="Marker size and opacity follow each detection's confidence score, scaled to this scene's own score range">
                <span className="dim">Marker size · opacity = confidence</span>
                {[['lowest', range.lo, 0], ['median', range.mid, range.hi > range.lo ? (range.mid - range.lo) / (range.hi - range.lo) : 1], ['highest', range.hi, 1]].map(([name, v, t]) => (
                  <span key={String(name)} className="sample">
                    <i style={{ width: lerp(MARKER_R, Number(t)) * 2, height: lerp(MARKER_R, Number(t)) * 2, opacity: lerp(MARKER_OP, Number(t)) }} />
                    <span className="mono">{Number(v).toFixed(2)}</span><span className="faint">{String(name)}</span>
                  </span>
                ))}
              </div>
            )}
          </div>
          {sel && (
            <div className="detection-chip" role="dialog" aria-label="Detection details" data-testid="detection-chip" data-id={sel.id}>
              <button className="btn sm" style={{ float: 'right' }} onClick={() => setPicked(null)} aria-label="Close detection details">✕</button>
              <div className="row" style={{ alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                <i className="ring-dot" style={{ background: color(sel.class) }} /><b>{sel.class}</b>
                <span className="chip mono">confidence {sel.score.toFixed(3)}</span>
                {isNum(sel.length_m) && isNum(sel.width_m) && <span className="chip mono">box {sel.length_m.toFixed(1)} × {sel.width_m.toFixed(1)} m</span>}
                {isNum(sel.heading_deg) && <span className="chip mono">heading {sel.heading_deg.toFixed(0)}°</span>}
                <span className="dim mono" style={{ fontSize: 11 }}>{fmtLonLat([sel.lon, sel.lat])}</span>
              </div>
              <div className="row" style={{ gap: 8, marginTop: 8 }}>
                {rc && <button className="btn sm" onClick={() => onOpenTile(rc.row, rc.col)} title="Show the oriented boxes of this detection's 256-px tile over the scene">Open tile r{rc.row} · c{rc.col}</button>}
                <button className="btn sm" onClick={() => centreOn(sel.id)}>Draw threat rings here</button>
              </div>
            </div>
          )}
          <ThreatRingsResults tr={tr} hint="Click a marker for its details, or right-click one to draw threat rings around it and list the restricted zones, change candidates, other detections and watch areas inside each ring." />
        </div>
      )}
    </Panel>
  );
}

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
        <Panel title="Scenes with detections">
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
        <Panel title="Tiles · most objects first">
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

      <div className="col">
      <Panel title={`Detections${cur ? ' · ' + regionLabel(cur.aoi_name.replace(/-/g, '_')) : ''}`}
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

        <DetectionMap obs={cur} onOpenTile={(row, col) => setTileKey({ row, col })} />
      </div>

      <div className="col">
        <Panel title="Measured accuracy">
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
        <Panel title="Detector & caveats">
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
