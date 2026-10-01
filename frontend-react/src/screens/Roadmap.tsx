/* eslint-disable @typescript-eslint/no-explicit-any */
import { useEffect, useMemo, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from '@vendor/OrbitControls.js';
import { api } from '@/api/client';
import { GeoMap, type MapCircle } from '@/components/GeoMap';
import { Panel } from '@/components/Panel';
import { DASH, regionLabel } from '@/fmt';
import { useApi } from '@/hooks/useApi';
import { go } from '@/router';

// ---- everything on this screen is a NON-FUNCTIONAL MOCKUP. Placeholder text is deliberately obvious. ----

const PANELS = [
  { id: 'dossier', name: 'Tactical dossier export', sub: 'PDF preview' },
  { id: 'ingest', name: 'Ad-hoc raster ingestion', sub: 'drag-and-drop' },
  { id: 'vector3d', name: '3D vector-space visualizer', sub: 'embedding scatter' },
  { id: 'threat', name: 'Threat buffer rings', sub: 'radius overlay' },
  { id: 'xai', name: 'Attention / XAI overlay', sub: 'heatmap' },
] as const;
type PanelId = (typeof PANELS)[number]['id'];

// ViT-B/32 on a 224 px input is a 7 x 7 patch grid (224 / 32). Architectural constant, not a measurement.
const VIT_B32_GRID = 224 / 32;

/** Real ground size of one ViT-B/32 patch, derived live from a catalogued tile's footprint (no hardcoded metres). */
function usePatchMetres(): number | null {
  const hit = useApi((s) => api.searchText({ q: 'open water', k: 1 }, s), []);
  const tileId = hit.data?.results[0]?.tile_id ?? null;
  const fp = useApi(tileId ? (s) => api.tiles([tileId], s) : null, [tileId]);
  const t = fp.data?.tiles[0];
  return t ? Math.round((t.width_km * 1000) / VIT_B32_GRID) : null;
}

function Dossier() {
  const [open, setOpen] = useState(false);
  return (
    <Panel title="Tactical dossier export" tier="roadmap" note="requires MGRS conversion + STAC sun-elevation/off-nadir capture at ingest">
      <div className="row" style={{ gap: 8, marginBottom: 12 }}>
        <button className="btn" onClick={() => setOpen((o) => !o)}>{open ? 'Close preview' : 'Open dossier preview'}</button>
        <button className="btn" disabled title="Not implemented">Export PDF (not implemented)</button>
      </div>
      {open ? (
        <div className="paper" aria-label="Dossier preview mockup">
          <h4>TACTICAL CHANGE DOSSIER <span style={{ float: 'right', fontSize: 10, color: '#8a5a00' }}>MOCKUP · PLACEHOLDER DATA</span></h4>
          <table style={{ width: '100%', borderCollapse: 'collapse', marginBottom: 12 }}>
            <tbody>
              <tr><td><b>MGRS</b></td><td className="mono">XXX XX 00000 00000 (placeholder)</td><td><b>UTM</b></td><td className="mono">zone XX · E 000000 · N 0000000 (placeholder)</td></tr>
              <tr><td><b>Site</b></td><td>Placeholder site</td><td><b>Classification</b></td><td>— (placeholder)</td></tr>
            </tbody>
          </table>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8, marginBottom: 12 }}>
            {['BEFORE', 'AFTER', 'DIFFERENCE'].map((c) => <div key={c}><div className="ph" style={{ height: 96 }}>{c} · placeholder</div></div>)}
          </div>
          <div style={{ border: '1px solid #9aa3c4', padding: 8, marginBottom: 12 }}>
            <b>SENSOR PROVENANCE</b>
            <div className="mono" style={{ fontSize: 10.5, marginTop: 4, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 2 }}>
              <span>Platform: — (placeholder)</span><span>Acquired: — (placeholder)</span>
              <span>Sun elevation: — °</span><span>Off-nadir angle: — °</span>
              <span>Processing baseline: —</span><span>Scene ID: —</span>
            </div>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 18 }}>
            <div><div style={{ borderBottom: '1px solid #1a2140', height: 28 }} /><small>Analyst signature (not enabled)</small></div>
            <div><div style={{ borderBottom: '1px solid #1a2140', height: 28 }} /><small>Date / authorising officer (not enabled)</small></div>
          </div>
        </div>
      ) : <div className="empty">Press “Open dossier preview” to see the layout mockup. Nothing is generated or exported.</div>}
    </Panel>
  );
}

function Ingest() {
  const [note, setNote] = useState<string | null>(null);
  return (
    <Panel title="Drag-and-drop raster ingestion" tier="roadmap" note="requires runtime single-scene ingest path">
      <div className="dropzone" onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => { e.preventDefault(); setNote(`${e.dataTransfer.files.length} file(s) dropped — NOT read, NOT uploaded. Ingestion is not implemented.`); }}>
        <div style={{ fontSize: 30 }}>⇪</div>
        <div style={{ fontWeight: 600, fontSize: 14 }}>Drop a GeoTIFF / COG here</div>
        <div style={{ fontSize: 11.5, marginTop: 4 }}>placeholder dropzone — files are ignored</div>
      </div>
      {note && <div className="warnbox" style={{ marginTop: 10 }} role="status">{note}</div>}
      <div style={{ marginTop: 14 }}>
        <div className="dim" style={{ fontSize: 10, letterSpacing: '.1em', textTransform: 'uppercase', marginBottom: 6 }}>Header parse · validation (mockup)</div>
        <div className="row" style={{ flexWrap: 'wrap', gap: 6 }}>
          {['CRS', 'Bands', 'GSD', 'Footprint', 'Nodata', 'Bit depth'].map((k) => <span key={k} className="chip" style={{ borderStyle: 'dashed', color: '#c9a769' }}>{k}: — not run</span>)}
          <span className="chip amber">validation: NOT IMPLEMENTED</span>
        </div>
      </div>
    </Panel>
  );
}

function mulberry32(a: number) {
  return () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

function Scatter3D() {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = ref.current; if (!el) return;
    let renderer: any;
    try { renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true }); } catch { el.textContent = '3D unavailable (no WebGL)'; return; }
    renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    el.appendChild(renderer.domElement);
    const scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera(50, 1, 0.1, 100);
    camera.position.set(5.6, 3.6, 6.2);
    // SYNTHETIC illustrative points: five seeded gaussian blobs. They are NOT derived from the archive.
    const rnd = mulberry32(20260101), gauss = () => Math.sqrt(-2 * Math.log(rnd() + 1e-9)) * Math.cos(2 * Math.PI * rnd());
    const centres = [[-2, 0.6, 0.5], [1.6, 1.2, -1.2], [0.4, -1.4, 1.8], [-0.6, 1.8, -1.6], [2.1, -0.8, 1.0]];
    const cols = ['#4cc9f0', '#f5a524', '#34d399', '#a78bfa', '#ff6b7a'];
    const N = 220, pos = new Float32Array(centres.length * N * 3), col = new Float32Array(centres.length * N * 3);
    centres.forEach((c, k) => { const cc = new THREE.Color(cols[k]); for (let i = 0; i < N; i++) { const o = (k * N + i) * 3; pos[o] = c[0] + gauss() * 0.42; pos[o + 1] = c[1] + gauss() * 0.42; pos[o + 2] = c[2] + gauss() * 0.42; col.set([cc.r, cc.g, cc.b], o); } });
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3)); g.setAttribute('color', new THREE.BufferAttribute(col, 3));
    const pts = new THREE.Points(g, new THREE.PointsMaterial({ size: 0.07, vertexColors: true, transparent: true, opacity: 0.85 }));
    scene.add(pts, new THREE.AxesHelper(2.6), new THREE.GridHelper(8, 16, 0x2a3a7a, 0x16275a));
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true; controls.autoRotate = true; controls.autoRotateSpeed = 0.8;
    const resize = () => { const w = el.clientWidth, h = el.clientHeight; if (w && h) { renderer.setSize(w, h, false); renderer.domElement.style.cssText = 'width:100%;height:100%;display:block'; camera.aspect = w / h; camera.updateProjectionMatrix(); } };
    const ro = new ResizeObserver(resize); ro.observe(el); resize();
    let raf = 0; const loop = () => { raf = requestAnimationFrame(loop); controls.update(); renderer.render(scene, camera); }; loop();
    return () => { cancelAnimationFrame(raf); ro.disconnect(); controls.dispose(); g.dispose(); pts.material.dispose(); renderer.dispose(); renderer.forceContextLoss?.(); renderer.domElement.remove(); };
  }, []);
  return <div ref={ref} style={{ width: '100%', height: 400, borderRadius: 8, background: '#050c20', position: 'relative' }} />;
}

function Vector3D() {
  return (
    <Panel title="3D vector-space visualizer" tier="roadmap" note="requires offline-precomputed UMAP projection">
      <div className="warnbox" style={{ marginBottom: 10 }}>ILLUSTRATIVE SYNTHETIC POINTS — five random blobs generated in the browser. They are not your archive's embeddings; no projection has been computed.</div>
      <Scatter3D />
      <div className="faint" style={{ marginTop: 6, fontSize: 11 }}>Drag to orbit · scroll to zoom. Axes and grid are decorative.</div>
    </Panel>
  );
}

function Threat() {
  const regions = useApi((s) => api.regions(s), []);
  const r0 = regions.data?.regions[0];
  const centre = useMemo(() => (r0 ? { lon: (r0.bbox[0] + r0.bbox[2]) / 2, lat: (r0.bbox[1] + r0.bbox[3]) / 2 } : null), [r0]);
  const circles: MapCircle[] = useMemo(() => (centre ? [500, 1000, 2500, 5000].map((m, i) => ({ ...centre, radiusM: m, color: ['#ff6b7a', '#ff9f6b', '#f5a524', '#e8c24a'][i], dashed: true, label: `Ring ${i + 1} · ${m >= 1000 ? m / 1000 + ' km' : m + ' m'} (illustrative)` })) : []), [centre]);
  const fit = useMemo<[number, number, number, number] | null>(() => (centre ? [centre.lon - 0.07, centre.lat - 0.06, centre.lon + 0.07, centre.lat + 0.06] : null), [centre]);
  return (
    <Panel title="Threat buffer rings" tier="roadmap" note="requires an infrastructure feature layer to intersect against">
      <div className="road-grid" style={{ gridTemplateColumns: 'minmax(0,1fr) 260px' }}>
        <GeoMap ariaLabel="Threat buffer rings mockup" circles={circles} points={centre ? [{ id: 'c', lon: centre.lon, lat: centre.lat, color: '#ff6b7a', radius: 6, label: 'Placeholder site' }] : []} fit={fit} height={380} />
        <div className="col" style={{ gap: 10 }}>
          <div className="mock-box" style={{ padding: 10, textAlign: 'left', placeItems: 'start' }}>
            <div>CENTRE: placeholder site{r0 ? ` (map centre of ${regionLabel(r0.name)})` : ''}</div>
          </div>
          <dl className="kv"><dt>Infrastructure layer</dt><dd>none loaded</dd><dt>Features in ring 1</dt><dd>{DASH}</dd><dt>Features in ring 2</dt><dd>{DASH}</dd><dt>Features in ring 3</dt><dd>{DASH}</dd><dt>Features in ring 4</dt><dd>{DASH}</dd></dl>
          <div className="faint" style={{ fontSize: 11 }}>Rings are drawn at fixed illustrative radii. Nothing is intersected or counted.</div>
        </div>
      </div>
    </Panel>
  );
}

function Xai() {
  const m = usePatchMetres();
  const note = `requires ViT-L/14 migration — current ViT-B/32 resolves to ~${m ?? DASH} m patches`;
  const G = 7;
  const blobs = [[30, 38, 70, 0.9], [66, 62, 55, 0.7], [20, 76, 40, 0.5]];
  return (
    <Panel title="Attention / XAI overlay" tier="roadmap" note={note}>
      <div className="warnbox" style={{ marginBottom: 10 }}>CONCEPT ONLY — the heat blobs below are painted by hand. No attention map is computed from any model or image.</div>
      <div className="row" style={{ gap: 16, flexWrap: 'wrap' }}>
        <div style={{ position: 'relative', width: 360, height: 360, borderRadius: 8, overflow: 'hidden', background: 'repeating-linear-gradient(135deg, #1b2548 0 14px, #16204a 14px 28px)', border: '1px dashed rgba(245,165,36,.6)' }}>
          {blobs.map((b, i) => <div key={i} style={{ position: 'absolute', left: `${b[0]}%`, top: `${b[1]}%`, width: b[2] * 3, height: b[2] * 3, margin: `${-b[2] * 1.5}px 0 0 ${-b[2] * 1.5}px`, borderRadius: '50%', background: `radial-gradient(circle, rgba(255,80,60,${b[3]}) 0%, rgba(255,170,0,${b[3] * 0.5}) 45%, transparent 70%)` }} />)}
          <svg viewBox="0 0 100 100" style={{ position: 'absolute', inset: 0, width: '100%', height: '100%' }} aria-hidden="true">
            {Array.from({ length: G - 1 }, (_, i) => <g key={i}><line x1={((i + 1) * 100) / G} y1="0" x2={((i + 1) * 100) / G} y2="100" stroke="rgba(255,255,255,.28)" strokeWidth=".3" /><line y1={((i + 1) * 100) / G} x1="0" y2={((i + 1) * 100) / G} x2="100" stroke="rgba(255,255,255,.28)" strokeWidth=".3" /></g>)}
          </svg>
          <span className="tier-badge roadmap" style={{ position: 'absolute', left: 8, bottom: 8 }}><i />ILLUSTRATIVE</span>
        </div>
        <div className="col" style={{ flex: 1, minWidth: 240, gap: 8 }}>
          <div className="dim" style={{ fontSize: 10, letterSpacing: '.1em', textTransform: 'uppercase' }}>Why this is on the roadmap</div>
          <div>The current encoder splits each tile into a {G}×{G} grid, so one cell is about <b className="mono">{m ?? DASH} m</b> across — too coarse to point at an individual structure. A ViT-L/14 encoder gives a 16×16 grid; the overlay would be meaningful only after that migration.</div>
          <dl className="kv"><dt>Attention source</dt><dd>none</dd><dt>Overlay opacity</dt><dd>— (disabled)</dd></dl>
        </div>
      </div>
    </Panel>
  );
}

export function Roadmap({ panel }: { panel: string | null }) {
  const cur: PanelId = (PANELS.find((p) => p.id === panel)?.id ?? 'dossier');
  return (
    <div className="col">
      <div className="warnbox" role="note"><b>ROADMAP — NOT IMPLEMENTED.</b> Everything on these five screens is a static mockup with placeholder content. None of it computes, exports, ingests or measures anything. Live features are on the other rail entries.</div>
      <div className="road-grid">
        <div className="road-list" role="tablist" aria-label="Roadmap mockups">
          {PANELS.map((p) => (
            <button key={p.id} role="tab" aria-selected={cur === p.id} className={cur === p.id ? 'on' : ''} onClick={() => go('roadmap', p.id)}>
              <small>ROADMAP — NOT IMPLEMENTED</small><b>{p.name}</b><span className="dim" style={{ fontSize: 11 }}>{p.sub}</span>
            </button>
          ))}
        </div>
        <div>
          {cur === 'dossier' && <Dossier />}
          {cur === 'ingest' && <Ingest />}
          {cur === 'vector3d' && <Vector3D />}
          {cur === 'threat' && <Threat />}
          {cur === 'xai' && <Xai />}
        </div>
      </div>
    </div>
  );
}
