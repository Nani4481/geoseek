import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { candidateImage } from '@/api/client';
import type { LonLat } from '@/api/types';
import { useCandidate } from '@/components/CandidatePanels';
import { Globe, type Focus, type Pin } from '@/components/Globe';
import { OfflinePill } from '@/components/Shell';
import { TierBadge } from '@/components/Panel';
import { bandOf, fmtHa, fmtLonLat, fmtPct } from '@/fmt';
import { go } from '@/router';
import { useStore } from '@/state/store';

type Tool = 'orbit' | 'pen' | 'arrow' | 'rect';
interface Stroke { tool: Exclude<Tool, 'orbit'>; color: string; pts: [number, number][] } // pts normalised 0..1
interface Site { id: string; label: string; lonlat: LonLat }

const COLORS = ['#ffc400', '#00e5ff', '#ff4d5e', '#ffffff'];
const STEP_MS = 7500;

function drawStroke(ctx: CanvasRenderingContext2D, s: Stroke, w: number, h: number) {
  ctx.strokeStyle = s.color; ctx.fillStyle = s.color; ctx.lineWidth = 4; ctx.lineCap = 'round'; ctx.lineJoin = 'round';
  const P = s.pts.map(([x, y]) => [x * w, y * h] as const);
  if (s.tool === 'pen') { ctx.beginPath(); P.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y))); ctx.stroke(); return; }
  if (P.length < 2) return;
  const [a, b] = [P[0], P[P.length - 1]];
  if (s.tool === 'rect') { ctx.strokeRect(a[0], a[1], b[0] - a[0], b[1] - a[1]); return; }
  ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
  const ang = Math.atan2(b[1] - a[1], b[0] - a[0]), L = 18;
  ctx.beginPath(); ctx.moveTo(b[0], b[1]);
  ctx.lineTo(b[0] - L * Math.cos(ang - 0.45), b[1] - L * Math.sin(ang - 0.45));
  ctx.lineTo(b[0] - L * Math.cos(ang + 0.45), b[1] - L * Math.sin(ang + 0.45));
  ctx.closePath(); ctx.fill();
}

export function Briefing() {
  const { bookmarks, summary, label } = useStore();
  // Tour = the analyst's bookmarked sites; with none, the featured findings the backend itself highlights.
  const sites: Site[] = useMemo(() => {
    if (bookmarks.length) return bookmarks.map((b) => ({ id: b.id, label: b.label, lonlat: b.lonlat }));
    return (summary?.featured ?? []).map((f) => ({ id: f.candidate_id, label: `${f.change_type_human} · ${f.candidate_id}`, lonlat: f.centroid_lonlat }));
  }, [bookmarks, summary]);
  const defaultTour = bookmarks.length === 0;

  const [idx, setIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [focus, setFocus] = useState<Focus | null>(null);
  const [tool, setTool] = useState<Tool>('orbit');
  const [color, setColor] = useState(COLORS[0]);
  const [spot, setSpot] = useState(false);
  const [hc, setHc] = useState(true);
  const [chrome, setChrome] = useState(true);
  const [fs, setFs] = useState(!!document.fullscreenElement);
  const [strokes, setStrokes] = useState<Record<string, Stroke[]>>({});
  const [tick, setTick] = useState(0);
  const site = sites[Math.min(idx, Math.max(0, sites.length - 1))];
  const bundle = useCandidate(site?.id ?? null);

  const flyTo = useCallback((s: Site, n: number) => {
    setFocus({ lon: s.lonlat[0], lat: s.lonlat[1], key: `${s.id}:${n}`, distance: 1.75, lift: 0.9, ms: 2400 });
  }, []);
  const go2 = useCallback((i: number) => {
    if (!sites.length) return;
    const j = (i + sites.length) % sites.length;
    setIdx(j); flyTo(sites[j], Date.now());
  }, [sites, flyTo]);

  // keyframe tour: advance on a timer while playing
  useEffect(() => {
    if (!playing || sites.length < 2) return;
    const t = window.setTimeout(() => go2(idx + 1), STEP_MS);
    return () => window.clearTimeout(t);
  }, [playing, idx, sites.length, go2]);
  useEffect(() => { if (sites.length && !focus) flyTo(sites[0], 0); }, [sites]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const onFs = () => setFs(!!document.fullscreenElement);
    document.addEventListener('fullscreenchange', onFs);
    return () => document.removeEventListener('fullscreenchange', onFs);
  }, []);
  const exit = useCallback(() => { if (document.fullscreenElement) void document.exitFullscreen(); go('dashboard'); }, []);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.tagName === 'INPUT') return;
      if (e.key === 'Escape') exit();
      else if (e.key === 'ArrowRight') go2(idx + 1);
      else if (e.key === 'ArrowLeft') go2(idx - 1);
      else if (e.key === ' ') { setPlaying((p) => !p); e.preventDefault(); }
      else if (e.key.toLowerCase() === 'h') setHc((v) => !v);
      else if (e.key.toLowerCase() === 't') setChrome((v) => !v);
      else if (e.key.toLowerCase() === 'p') setTool('pen');
      else if (e.key.toLowerCase() === 'o') setTool('orbit');
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [exit, go2, idx]);

  const toggleFs = () => { if (document.fullscreenElement) void document.exitFullscreen(); else void document.documentElement.requestFullscreen().catch(() => { /* denied: stays windowed */ }); };

  // ---- annotation canvas ----
  const stage = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const draft = useRef<Stroke | null>(null);
  const clearGen = useRef(0);
  const key = site?.id ?? '_';
  const redraw = useCallback(() => {
    const c = canvas.current, st = stage.current;
    if (!c || !st) return;
    const dpr = window.devicePixelRatio || 1, w = st.clientWidth, h = st.clientHeight;
    if (c.width !== w * dpr || c.height !== h * dpr) { c.width = w * dpr; c.height = h * dpr; }
    const gen = ++clearGen.current;
    // Nothing left to show (Undo / Clear / site change): reset the bitmap outright, and once more a frame later. A lone
    // clearRect is occasionally dropped by Chrome's software-rendered 2D canvas when it follows draw ops within the same
    // frame - reproduced in a bare page with no app code, and the stale stroke is genuinely displayed, not just read
    // back. The generation token stops the deferred clear from ever erasing a stroke drawn in between.
    if (!(strokes[key]?.length) && !draft.current) {
      c.width = c.width;
      requestAnimationFrame(() => { if (clearGen.current === gen && canvas.current) canvas.current.width = canvas.current.width; });
      return;
    }
    const ctx = c.getContext('2d')!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, w, h);
    for (const s of strokes[key] ?? []) drawStroke(ctx, s, w, h);
    if (draft.current) drawStroke(ctx, draft.current, w, h);
  }, [strokes, key]);
  useEffect(() => { redraw(); }, [redraw, tick]);
  useEffect(() => { const on = () => setTick((t) => t + 1); window.addEventListener('resize', on); return () => window.removeEventListener('resize', on); }, []);
  const norm = (e: React.PointerEvent): [number, number] => {
    const r = stage.current!.getBoundingClientRect();
    return [(e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height];
  };
  const onDown = (e: React.PointerEvent) => {
    if (tool === 'orbit') return;
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    draft.current = { tool, color, pts: [norm(e)] };
  };
  const onMove = (e: React.PointerEvent) => {
    if (spot) setSpotPos({ x: e.clientX, y: e.clientY });
    if (!draft.current) return;
    if (draft.current.tool === 'pen') draft.current.pts.push(norm(e)); else draft.current.pts = [draft.current.pts[0], norm(e)];
    redraw();
  };
  const onUp = () => {
    const d = draft.current; draft.current = null;
    if (d && d.pts.length > 1) setStrokes((s) => ({ ...s, [key]: [...(s[key] ?? []), d] }));
  };
  const [spotPos, setSpotPos] = useState({ x: -999, y: -999 });

  const d = bundle.detail, tl = bundle.timeline;
  const band = bandOf(d?.confidence);
  const before = tl?.dates[0], after = tl?.dates[tl.dates.length - 1];
  const pins: Pin[] = useMemo(() => sites.map((s, i) => ({ id: s.id, lon: s.lonlat[0], lat: s.lonlat[1], label: `${i + 1}. ${s.label}`, color: '#ffc400', emphasis: i === idx, size: i === idx ? 1.8 : 1.2 })), [sites, idx]);

  return (
    <div className={`brief ${hc ? 'hc' : ''}`} role="region" aria-label="Briefing mode">
      <div className="stage" ref={stage} onPointerMove={(e) => { if (spot) setSpotPos({ x: e.clientX, y: e.clientY }); }}>
        <Globe pins={pins} focus={focus} selectedPin={site?.id ?? null} autoRotate={false} onPinClick={(id) => { const i = sites.findIndex((s) => s.id === id); if (i >= 0) go2(i); }} />
        {spot && <div className="spot" style={{ left: spotPos.x, top: spotPos.y }} aria-hidden="true" />}
        <canvas ref={canvas} className="anno" style={{ pointerEvents: tool === 'orbit' ? 'none' : 'auto', cursor: tool === 'orbit' ? 'default' : 'crosshair' }}
          onPointerDown={onDown} onPointerMove={onMove} onPointerUp={onUp} onPointerCancel={onUp} aria-label="Annotation layer" />

        {chrome && (
          <>
            <div className="hud" style={{ left: 16, top: 16, maxWidth: 'min(420px, 46vw)' }}>
              {!site ? (
                <div>No sites to present. Bookmark candidates in the Changes view (☆) to build a tour.</div>
              ) : (
                <>
                  <div className="mono" style={{ fontSize: 12, opacity: 0.85 }}>SITE {idx + 1} / {sites.length}{defaultTour ? ' · featured findings (bookmark sites to build your own tour)' : ' · bookmarked'}</div>
                  <div style={{ fontSize: 20, fontWeight: 700, margin: '3px 0' }}>{d ? label(d.change_type) : site.label}</div>
                  <div className="mono" style={{ fontSize: 12.5 }}>{fmtLonLat(site.lonlat)}</div>
                  {d && <div style={{ marginTop: 6, display: 'flex', gap: 14, flexWrap: 'wrap', fontSize: 14 }}>
                    <span>confidence <b style={{ color: band.color }}>{fmtPct(d.confidence, 0)}</b></span>
                    <span>{fmtHa(d.area_m2)}</span>
                    {tl?.first_detected && <span>first seen <b className="mono">{tl.first_detected.date}</b></span>}
                    {tl?.first_detected && <span>{tl.n_supporting} of {tl.n_total} observations</span>}
                  </div>}
                  {before && after && before !== after && (
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginTop: 10 }}>
                      {[before, after].map((dt) => (
                        <figure key={dt} style={{ margin: 0 }}>
                          <img src={candidateImage(site.id, dt.slice(0, 4), 'rgb', 2)} alt={`Site imagery ${dt}`} style={{ width: '100%', aspectRatio: '1', objectFit: 'cover', border: '2px solid #fff', borderRadius: 6, background: '#000' }} />
                          <figcaption className="mono" style={{ fontSize: 11.5, textAlign: 'center' }}>{dt}</figcaption>
                        </figure>
                      ))}
                    </div>
                  )}
                </>
              )}
            </div>
            <div style={{ position: 'absolute', right: 16, top: 16, zIndex: 7, display: 'flex', gap: 10, alignItems: 'center' }}>
              <OfflinePill />
              <TierBadge tier="surfaced" />
              <button className="btn" onClick={exit} style={{ borderColor: '#fff', background: '#000' }}>Exit ✕ (Esc)</button>
            </div>
            <div className="toolbar" role="toolbar" aria-label="Briefing controls">
              <button className="btn" onClick={() => go2(idx - 1)} disabled={sites.length < 2} aria-label="Previous site">◀</button>
              <button className={`btn ${playing ? 'on' : ''}`} onClick={() => setPlaying((p) => !p)} disabled={sites.length < 2} aria-pressed={playing}>{playing ? '❚❚ Pause tour' : '▶ Play tour'}</button>
              <button className="btn" onClick={() => go2(idx + 1)} disabled={sites.length < 2} aria-label="Next site">▶</button>
              <span style={{ width: 10 }} />
              {(['orbit', 'pen', 'arrow', 'rect'] as Tool[]).map((t) => (
                <button key={t} className={`btn ${tool === t ? 'on' : ''}`} onClick={() => setTool(t)} aria-pressed={tool === t}>{{ orbit: '✥ Orbit', pen: '✎ Pen', arrow: '➚ Arrow', rect: '▭ Box' }[t]}</button>
              ))}
              {COLORS.map((c) => <button key={c} className="btn" onClick={() => setColor(c)} aria-label={`Colour ${c}`} aria-pressed={color === c} style={{ width: 32, padding: 0, height: 32, background: c, outline: color === c ? '3px solid #fff' : 'none', outlineOffset: 2 }} />)}
              <button className="btn" onClick={() => setStrokes((s) => ({ ...s, [key]: (s[key] ?? []).slice(0, -1) }))} disabled={!(strokes[key]?.length)}>↶ Undo</button>
              <button className="btn" onClick={() => setStrokes((s) => ({ ...s, [key]: [] }))} disabled={!(strokes[key]?.length)}>Clear</button>
              <span style={{ width: 10 }} />
              <button className={`btn ${spot ? 'on' : ''}`} onClick={() => setSpot((v) => !v)} aria-pressed={spot}>◎ Spotlight</button>
              <button className={`btn ${hc ? 'on' : ''}`} onClick={() => setHc((v) => !v)} aria-pressed={hc}>◐ Contrast (H)</button>
              <button className={`btn ${fs ? 'on' : ''}`} onClick={toggleFs} aria-pressed={fs}>⛶ Fullscreen</button>
              <button className="btn" onClick={() => setChrome(false)}>Hide panels (T)</button>
            </div>
          </>
        )}
        {!chrome && <button className="btn" style={{ position: 'absolute', right: 12, bottom: 12, zIndex: 8, opacity: 0.35, background: '#000', border: '1px solid #fff' }} onClick={() => setChrome(true)} aria-label="Show panels">⋯</button>}
        <span className="keys">← → sites · Space play/pause · P pen · O orbit · H contrast · T panels · Esc exit</span>
      </div>
    </div>
  );
}
