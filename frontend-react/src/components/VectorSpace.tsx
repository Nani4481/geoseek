/* eslint-disable @typescript-eslint/no-explicit-any */
import { useEffect, useMemo, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from '@vendor/OrbitControls.js';
import { api, tileThumb } from '@/api/client';
import type { ProjectionSample } from '@/api/types';
import { useApi } from '@/hooks/useApi';
import { fmtInt, fmtLonLat, regionLabel } from '@/fmt';
import { ErrorNote, Loading, TileImg } from './Widgets';
import { Panel } from './Panel';

export interface PickedTile { tile_id: string; lon: number; lat: number }
type ColorBy = 'region' | 'cluster';
const SAMPLE_POINTS = 20000;

/** Categorical colour i of n: golden-angle hues so neighbours differ, fixed lightness for legibility on the dark canvas. */
const cat = (i: number) => new THREE.Color().setHSL(((i * 137.508) % 360) / 360, 0.68, 0.6);
const GREY = new THREE.Color('#6d7aa3');

function circleTexture() {
  const c = document.createElement('canvas');
  c.width = c.height = 64;
  const g = c.getContext('2d')!;
  g.beginPath(); g.arc(32, 32, 30, 0, Math.PI * 2); g.fillStyle = '#fff'; g.fill();
  const t = new THREE.CanvasTexture(c);
  return t;
}

interface Rig {
  render: () => void;
  setColors: (mode: ColorBy) => void;
  setHits: (xyz: number[]) => void;
  setSelected: (xyz: number[] | null) => void;
  setRotate: (on: boolean) => void;
  dispose: () => void;
}

/**
 * 3-D scatter of the archive's tile embeddings (UMAP, precomputed offline by scripts/compute_projection.py), drawn with the
 * vendored three.js. Points are coloured by region or cluster. Search hits are always drawn (looked up exactly, so they show
 * even when outside the display sample). Clicking a point reports that tile so the map can pan to it.
 *
 * The caption says how many points exist and how many are drawn - sampling is never hidden.
 */
export function VectorSpace({ hits, selected, onPick }: { hits: string[]; selected: string | null; onPick: (t: PickedTile) => void }) {
  const sample = useApi((s) => api.projection(SAMPLE_POINTS, s), []);
  const [colorBy, setColorBy] = useState<ColorBy>('region');
  const [rotate, setRotate] = useState(false);
  const [picked, setPicked] = useState<PickedTile | null>(null);
  const host = useRef<HTMLDivElement>(null);
  const rig = useRef<Rig | null>(null);
  const cb = useRef(onPick);
  cb.current = onPick;
  const d = sample.data?.available ? sample.data : null;

  const hitKey = hits.slice(0, 200).join(',');
  const lookup = useApi(hitKey && d ? (s) => api.projectionLookup(hits.slice(0, 200), s) : null, [hitKey, !!d]);
  const selLookup = useApi(selected && d ? (s) => api.projectionLookup([selected], s) : null, [selected, !!d]);

  const legend = useMemo(() => {
    if (!d) return [];
    const counts = new Map<number, number>();
    const arr = colorBy === 'region' ? d.region : d.cluster;
    for (const v of arr) counts.set(v, (counts.get(v) ?? 0) + 1);
    return [...counts.entries()].sort((a, b) => b[1] - a[1]).map(([k, n]) => ({
      key: k, n, color: k < 0 ? GREY : cat(colorBy === 'region' ? k : k + 3),
      label: colorBy === 'region' ? regionLabel(d.regions[k] ?? String(k)) : k < 0 ? 'no cluster' : `cluster ${k}`,
    }));
  }, [d, colorBy]);

  // build the scene once per data set
  useEffect(() => {
    const el = host.current;
    if (!el || !d) return;
    let renderer: any;
    try { renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true }); } catch { el.textContent = '3D unavailable (WebGL is not available in this browser)'; return; }
    renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    el.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 50);
    camera.position.set(1.25, 0.9, 1.65);
    const n = d.n_shown;
    const pos = new Float32Array(d.xyz);
    const col = new Float32Array(n * 3);
    const tex = circleTexture();
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
    const pts = new THREE.Points(geo, new THREE.PointsMaterial({ size: 3.2, sizeAttenuation: false, vertexColors: true, map: tex, alphaTest: 0.5, transparent: true, opacity: 0.9 }));
    const mkHl = (size: number, color: string) => {
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(0), 3));
      const p = new THREE.Points(g, new THREE.PointsMaterial({ size, sizeAttenuation: false, color, map: tex, alphaTest: 0.5, transparent: true, depthTest: false }));
      p.renderOrder = 2; p.frustumCulled = false;
      return p;
    };
    const hl = mkHl(10, '#ffffff'), sel = mkHl(18, '#f5a524');
    scene.add(pts, hl, sel, new THREE.AxesHelper(1.15));
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = false; controls.autoRotateSpeed = 1.2;

    let raf = 0;
    const render = () => { renderer.render(scene, camera); };
    const resize = () => {
      const w = el.clientWidth, h = el.clientHeight;
      if (!w || !h) return;
      renderer.setSize(w, h, false);
      renderer.domElement.style.cssText = 'width:100%;height:100%;display:block;touch-action:none';
      camera.aspect = w / h; camera.updateProjectionMatrix(); render();
    };
    const ro = new ResizeObserver(resize); ro.observe(el);
    controls.addEventListener('change', render);

    const apply = (g: any, xyz: number[]) => {
      g.geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(xyz), 3));
      g.geometry.computeBoundingSphere(); render();
    };
    rig.current = {
      render,
      setColors: (mode) => {
        const arr = mode === 'region' ? d.region : d.cluster;
        for (let i = 0; i < n; i++) { const k = arr[i]; const c = k < 0 ? GREY : cat(mode === 'region' ? k : k + 3); col[i * 3] = c.r; col[i * 3 + 1] = c.g; col[i * 3 + 2] = c.b; }
        (geo.getAttribute("color") as any).needsUpdate = true; render();
      },
      setHits: (xyz) => apply(hl, xyz),
      setSelected: (xyz) => apply(sel, xyz ?? []),
      setRotate: (on) => {
        controls.autoRotate = on; cancelAnimationFrame(raf);
        if (on) { const loop = () => { raf = requestAnimationFrame(loop); controls.update(); }; loop(); }
      },
      dispose: () => { cancelAnimationFrame(raf); ro.disconnect(); controls.dispose(); geo.dispose(); tex.dispose(); hl.geometry.dispose(); sel.geometry.dispose(); renderer.dispose(); renderer.forceContextLoss?.(); renderer.domElement.remove(); },
    };

    // click (not drag) -> nearest drawn point in screen space
    let down: { x: number; y: number } | null = null;
    const dom = renderer.domElement;
    const project = (i: number) => { const v = new THREE.Vector3(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]).project(camera); const r = dom.getBoundingClientRect(); return { x: r.left + ((v.x + 1) / 2) * r.width, y: r.top + ((1 - v.y) / 2) * r.height, z: v.z }; };
    const onDown = (e: PointerEvent) => { down = { x: e.clientX, y: e.clientY }; };
    const onUp = (e: PointerEvent) => {
      if (!down || Math.hypot(e.clientX - down.x, e.clientY - down.y) > 4) { down = null; return; }
      down = null;
      let best = -1, bd = 14 * 14, bz = 2;
      for (let i = 0; i < n; i++) {
        const p = project(i);
        if (p.z > 1) continue;
        const dd = (p.x - e.clientX) ** 2 + (p.y - e.clientY) ** 2;
        if (dd < bd - 1e-9 || (Math.abs(dd - bd) <= 1e-9 && p.z < bz)) { bd = dd; best = i; bz = p.z; }   // exact nearest; front-most on a tie
      }
      if (best >= 0) {
        const t = { tile_id: d.tile_ids[best], lon: d.lon[best], lat: d.lat[best] };
        setPicked(t); cb.current(t);
      }
    };
    dom.addEventListener('pointerdown', onDown); dom.addEventListener('pointerup', onUp);
    // read-only handle for tools/e2e-console.mjs (picks a real point by its screen position)
    (el as any).__vec = { count: n, project, indexOf: (id: string) => d.tile_ids.indexOf(id), tileId: (i: number) => d.tile_ids[i], lonlat: (i: number) => [d.lon[i], d.lat[i]] };

    rig.current.setColors('region');
    resize();
    return () => { dom.removeEventListener('pointerdown', onDown); dom.removeEventListener('pointerup', onUp); rig.current?.dispose(); rig.current = null; delete (el as any).__vec; };
  }, [d]);

  useEffect(() => { rig.current?.setColors(colorBy); }, [colorBy, d]);
  useEffect(() => { rig.current?.setRotate(rotate); }, [rotate, d]);
  useEffect(() => { rig.current?.setHits((lookup.data?.points ?? []).flatMap((p) => p.xyz)); }, [lookup.data, d]);
  useEffect(() => { const p = selLookup.data?.points[0]; rig.current?.setSelected(p ? p.xyz : null); }, [selLookup.data, d]);

  const m = d?.meta;
  const inSample = useMemo(() => (d && lookup.data ? lookup.data.points.filter((p) => d.tile_ids.includes(p.tile_id)).length : 0), [d, lookup.data]);

  return (
    <Panel title="Embedding space · 3-D"
      actions={d && (
        <div className="row" style={{ gap: 6 }}>
          <div className="tabs" role="group" aria-label="Colour points by">
            {(['region', 'cluster'] as ColorBy[]).map((c) => <button key={c} className={colorBy === c ? 'on' : ''} aria-pressed={colorBy === c} onClick={() => setColorBy(c)}>{c}</button>)}
          </div>
          <button className={`btn sm ${rotate ? 'on' : ''}`} aria-pressed={rotate} onClick={() => setRotate((r) => !r)}>⟳ Rotate</button>
        </div>
      )}>
      {sample.error ? <ErrorNote error={sample.error} onRetry={sample.reload} /> : sample.loading ? <Loading rows={5} /> : !d ? (
        <div className="empty" role="status">{sample.data?.reason ?? 'No projection is available.'}</div>
      ) : (
        <div className="col" style={{ gap: 8 }}>
          <div ref={host} className="vec-stage" role="img" aria-label="3-D scatter of tile embeddings" data-points={d.n_shown} data-total={d.n_total} data-hits={lookup.data?.points.length ?? 0} />
          <div className="vec-caption" data-testid="vec-caption">
            <b>{d.sampled ? `${fmtInt(d.n_shown)}-point sample of ${fmtInt(d.n_total)} tiles` : `all ${fmtInt(d.n_total)} tiles`}</b>
            {d.sampled ? ' — drawn uniformly at random so the picture stays smooth' : ''}. UMAP of the RemoteCLIP embeddings, computed offline
            ({m?.wall_seconds?.total ?? '—'} s on {m?.power_source ?? 'unknown power'}). Near = similar appearance; distances between far-apart groups mean nothing.
            {lookup.data && lookup.data.points.length > 0 && <> <b style={{ color: '#fff' }}>{lookup.data.points.length} search results highlighted</b>{inSample < lookup.data.points.length ? ` (${lookup.data.points.length - inSample} of them outside the sample, drawn from the full set)` : ''}.</>}
            {d.stale && <span className="chip amber" style={{ marginLeft: 6 }}>stale: computed from {fmtInt(m?.n_points)} vectors, index now has {fmtInt(d.current_vectors)}</span>}
          </div>
          <div className="legend-row vec-legend" aria-label="Legend">
            {legend.slice(0, 24).map((l) => <span key={l.key}><i style={{ background: `#${l.color.getHexString()}` }} />{l.label} <span className="mono faint">{fmtInt(l.n)}</span></span>)}
          </div>
          {picked ? (
            <div className="vec-picked row" style={{ gap: 10, alignItems: 'center' }} aria-live="polite">
              <div style={{ width: 64, flex: 'none' }}><TileImg src={tileThumb(picked.tile_id)} alt={`Tile ${picked.tile_id}`} /></div>
              <div className="mono" style={{ fontSize: 11, minWidth: 0 }}><b>{picked.tile_id}</b><br />{fmtLonLat([picked.lon, picked.lat])}<br /><span className="dim">map panned to this tile</span></div>
            </div>
          ) : <div className="faint" style={{ fontSize: 11 }}>Drag to orbit · scroll to zoom · click a point to pan the map to that tile.</div>}
        </div>
      )}
    </Panel>
  );
}

export type { ProjectionSample };
