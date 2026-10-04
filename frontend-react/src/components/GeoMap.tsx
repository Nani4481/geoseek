import { useEffect, useMemo, useRef, useState } from 'react';
import L from 'leaflet';
import '@vendor/leaflet/leaflet.css';
import { api } from '@/api/client';
import type { BBox, BasemapCoverage, Polygon } from '@/api/types';

/** A dot (default) or, with `pin`, a numbered / labelled marker that the hover link and tests can address by id. */
export interface MapPoint {
  id: string; lon: number; lat: number; color?: string; label?: string; radius?: number; opacity?: number;
  pin?: { text: string; kind?: 'hit' | 'seed' | 'sel' };
}
/** `tag` is a short label drawn permanently on the map for a selected footprint; `label` is the hover tooltip. */
export interface MapPolygon { id: string; geometry: Polygon; color: string; label?: string; fill?: number; selected?: boolean; tag?: string }
export interface MapRegion { name: string; bbox: BBox; color?: string; label?: string }
/** `tag` is drawn permanently at the top of the circle (a ring radius). */
export interface MapCircle { lon: number; lat: number; radiusM: number; color: string; label?: string; dashed?: boolean; tag?: string }
/** Tiles of one cluster as [lon, lat, tileCount] cells, drawn on a canvas layer (tens of thousands of cells are cheap there). */
export interface MapCells { id: string; color: string; cells: number[][] }
export interface BasemapSpec { year?: string; scene?: string }

interface Props {
  regions?: MapRegion[];
  points?: MapPoint[];
  polygons?: MapPolygon[];
  circles?: MapCircle[];
  cells?: MapCells[];
  cellDeg?: number;
  activeCell?: string | null;
  /** pointer over / away from a cluster's cells, and a click on them (cluster id) */
  onCellHover?: (id: string | null) => void;
  onCellClick?: (id: string) => void;
  bbox?: BBox | null;
  drawMode?: boolean;
  fit?: BBox | null;
  fitMaxZoom?: number;
  height?: number | string;
  fill?: boolean;
  /** local satellite basemap beneath the vectors; omit for the plain dark canvas with a graticule */
  basemap?: BasemapSpec | false;
  /** group overlapping `hit` pins into one count badge (computed in screen space, so it re-groups on every zoom) */
  clusterPins?: boolean;
  /** draw `regions` as the archive's coverage: a clear outline, a permanent name, a marker for footprints too small to see */
  regionLabels?: boolean;
  /** number of catalogued regions, for the caption (the dark is outside the archive, not a failed load) */
  catalogRegions?: number;
  /** map furniture, all computed from the view: a coordinate graticule with degree labels over the imagery, a scale bar, a north indicator */
  cartography?: boolean;
  /** id of the point whose pin is highlighted (the card / row hovered elsewhere); `onHover` reports hovers on the map */
  hoverId?: string | null;
  onHover?: (id: string | null) => void;
  onBBox?: (b: BBox) => void;
  onCancelDraw?: () => void;
  onMapClick?: (lon: number, lat: number) => void;
  onPointClick?: (id: string) => void;
  onPolygonClick?: (id: string) => void;
  /** right-click on a point / polygon (threat rings are drawn from here) */
  onPointContext?: (id: string) => void;
  onPolygonContext?: (id: string) => void;
  ariaLabel: string;
}

// Leaflet 1.9's canvas renderer can run a queued redraw after its map has been removed and then throw "clearRect of undefined".
// Make the two entry points no-ops once the renderer's context is gone (a removed map has nothing left to draw).
const canvasProto = (L.Canvas as unknown as { prototype: Record<string, (...a: unknown[]) => unknown> & { __guarded?: boolean } }).prototype;
if (!canvasProto.__guarded) {
  for (const name of ['_redraw', '_clear', '_update'] as const) {
    const orig = canvasProto[name];
    canvasProto[name] = function (this: { _ctx?: unknown }, ...a: unknown[]) { return this._ctx ? orig.apply(this, a) : undefined; };
  }
  canvasProto.__guarded = true;
}

const CLUSTER_PX = 26;   // a pin is 26 px wide: closer than that and two pins overlap

/** Highlight the marker that shows `hoverId` (a pin, or the cluster holding it) and lift it above its neighbours. */
function applyPinHover(index: Map<string, L.Marker>, hoverId: string | null) {
  const hot = hoverId ? index.get(hoverId) : undefined;
  for (const mk of new Set(index.values())) {
    const on = mk === hot;
    mk.getElement()?.querySelector('.gm-pin')?.classList.toggle('hl', on);
    mk.setZIndexOffset(on ? 30000 : (mk as L.Marker & { __z?: number }).__z ?? 0);
  }
}

const toBounds = (b: BBox) => L.latLngBounds([b[1], b[0]], [b[3], b[2]]);
const esc = (s: string) => s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c] ?? c);
const EMPTY_PNG = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==';

function graticuleStep(zoom: number) {
  if (zoom >= 11) return 0.02;
  if (zoom >= 9) return 0.1;
  if (zoom >= 7) return 0.5;
  if (zoom >= 5) return 1;
  if (zoom >= 4) return 2;
  if (zoom >= 3) return 5;
  if (zoom >= 2) return 10;
  return 20;
}

/** "27.5°N" - decimals follow the grid step so neighbouring labels never read the same */
const fmtDeg = (v: number, pos: string, neg: string, step: number) => `${Math.abs(v).toFixed(step < 0.1 ? 2 : step < 1 ? 1 : 0)}°${v < 0 ? neg : pos}`;

type CellLayerT = L.Layer & { setData: (g: MapCells[], d: number, a: string | null) => void; pick: (lon: number, lat: number, tolDeg: number) => string | null };

/** A canvas layer for very many small squares (one colour per group). Sits under the vector overlay pane. */
const CellLayer = (L.Layer as unknown as { extend: (o: object) => new (o: object) => CellLayerT }).extend({
  initialize(this: Record<string, unknown>, o: object) { this._o = o; },
  setData(this: Record<string, any>, groups: MapCells[], cellDeg: number, active: string | null) { // eslint-disable-line @typescript-eslint/no-explicit-any
    this._groups = groups; this._cellDeg = cellDeg; this._active = active; this._draw?.();
  },
  onAdd(this: Record<string, any>, map: L.Map) { // eslint-disable-line @typescript-eslint/no-explicit-any
    this._map = map;
    const c = (this._canvas = L.DomUtil.create('canvas', 'gm-cells') as HTMLCanvasElement);
    c.style.position = 'absolute'; c.style.pointerEvents = 'none';
    map.getPane('cells')!.appendChild(c);
    this._draw = () => {
      const size = map.getSize(), dpr = window.devicePixelRatio || 1;
      if (c.width !== Math.round(size.x * dpr) || c.height !== Math.round(size.y * dpr)) { c.width = Math.round(size.x * dpr); c.height = Math.round(size.y * dpr); c.style.width = size.x + 'px'; c.style.height = size.y + 'px'; }
      L.DomUtil.setPosition(c, map.containerPointToLayerPoint([0, 0]));
      const g = c.getContext('2d')!;
      g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, size.x, size.y);
      const groups: MapCells[] = this._groups ?? [], d: number = this._cellDeg ?? 0.03, act: string | null = this._active ?? null;
      const a = map.latLngToContainerPoint([0, 0]);
      const px = Math.max(2.4, Math.abs(map.latLngToContainerPoint([d, d]).x - a.x));
      c.dataset.active = act ?? ''; c.dataset.drawn = '0';
      let drawn = 0;
      for (const grp of groups) {
        g.globalAlpha = act === null ? 0.78 : grp.id === act ? 0.97 : 0.16;   // a selected cluster is lit, the rest dimmed (not hidden)
        g.fillStyle = grp.color;
        for (const [lon, lat] of grp.cells) {
          const p = map.latLngToContainerPoint([lat, lon]);
          if (p.x < -px || p.y < -px || p.x > size.x + px || p.y > size.y + px) continue;
          g.fillRect(p.x - px / 2, p.y - px / 2, px, px); drawn++;
        }
      }
      g.globalAlpha = 1; c.dataset.drawn = String(drawn);
    };
    map.on('move zoom moveend zoomend resize viewreset', this._draw, this);
    this._draw();
  },
  /** the cluster whose cell lies under a lon/lat (within `tolDeg`); later-drawn groups are on top, so they win */
  pick(this: Record<string, any>, lon: number, lat: number, tolDeg: number): string | null { // eslint-disable-line @typescript-eslint/no-explicit-any
    const groups: MapCells[] = this._groups ?? [], tol = Math.max(tolDeg, (this._cellDeg ?? 0.03) / 2);
    for (let i = groups.length - 1; i >= 0; i--) {
      for (const [x, y] of groups[i].cells) if (Math.abs(x - lon) <= tol && Math.abs(y - lat) <= tol) return groups[i].id;
    }
    return null;
  },
  onRemove(this: Record<string, any>, map: L.Map) { // eslint-disable-line @typescript-eslint/no-explicit-any
    map.off('move zoom moveend zoomend resize viewreset', this._draw, this);
    this._canvas?.remove(); this._draw = undefined;
  },
});

const fmtRange = (dates: string[]) => (dates.length === 0 ? '' : dates.length === 1 ? dates[0] : `${dates[0]} → ${dates[dates.length - 1]}`);

function captionFor(cov: BasemapCoverage | null, zoom: number, spec: BasemapSpec, regions?: number): { text: string; tone: 'ok' | 'dark' | 'wait' } {
  if (!cov) return { text: 'Local archive imagery · checking what is staged for this view…', tone: 'wait' };
  // with a known region count the dark is stated as information: imagery exists only inside the catalogued regions
  const scope = regions ? `Archive coverage: ${regions} catalogued region${regions === 1 ? '' : 's'}, outlined and named · imagery exists only inside them, so dark means outside the archive, not a failed load · ` : '';
  const up = zoom > cov.native_max_zoom ? ` · zoomed past the ${cov.gsd_m < 1 ? `${cov.gsd_m.toFixed(1)} m` : `${cov.gsd_m} m`} pixels` : '';
  if (spec.scene) {
    const s = cov.scenes[0];
    if (!cov.available) return { text: `Local archive imagery · the ${s?.platform ?? 'Maxar'} scene of ${s?.date ?? ''} lies outside this view · shown dark`, tone: 'dark' };
    return { text: `Local archive imagery · ${s?.platform ?? ''} ${s?.sensor ?? ''} · acquired ${s?.date ?? ''} · the scene these detections were found on · outside the scene shown dark${up}`, tone: 'ok' };
  }
  if (!cov.available) return { text: `${scope}No staged archive imagery in this view · shown dark (nothing is filled in)`, tone: 'dark' };
  return {
    text: `${scope}Local archive imagery · Sentinel-2 · one representative acquisition per granule (the lowest-cloud one), not the date of the overlaid features · ${cov.scenes.length} granule${cov.scenes.length === 1 ? '' : 's'} in view, ${fmtRange(cov.dates)} · unstaged areas shown dark${up}`,
    tone: 'ok',
  };
}

/** Offline Leaflet map. The basemap is the archive's own imagery served by /ui/basemap (no external tiles); where nothing is
 *  staged the dark canvas shows through. */
export function GeoMap({
  regions, points, polygons, circles, cells, cellDeg = 0.03, activeCell = null, onCellHover, onCellClick, bbox, drawMode, fit, fitMaxZoom = 16, height = 360, fill, basemap, clusterPins, regionLabels, catalogRegions, cartography,
  hoverId = null, onHover, onBBox, onCancelDraw, onMapClick, onPointClick, onPolygonClick, onPointContext, onPolygonContext, ariaLabel,
}: Props) {
  const el = useRef<HTMLDivElement>(null);
  const wrap = useRef<HTMLDivElement>(null);
  const map = useRef<L.Map | null>(null);
  const layers = useRef<{ grat: L.LayerGroup; regions: L.LayerGroup; shapes: L.LayerGroup; pins: L.LayerGroup; box: L.LayerGroup; base: L.TileLayer | null; cells: CellLayerT } | null>(null);
  const pins = useRef(new Map<string, { layer: L.Layer; base: { radius: number; opacity: number; weight: number } }>());   // dots
  const pinIndex = useRef(new Map<string, L.Marker>());          // pin id -> the marker showing it (its own, or the cluster that holds it)
  const pinData = useRef<MapPoint[]>([]);
  const pinOpts = useRef({ cluster: false, hover: null as string | null });
  pinOpts.current = { cluster: !!clusterPins, hover: hoverId };
  const cartoRef = useRef(false);
  cartoRef.current = !!cartography;
  const renderPins = useRef<() => void>(() => undefined);
  const cb = useRef({ onBBox, onMapClick, onPointClick, onPolygonClick, onPointContext, onPolygonContext, onHover, onCancelDraw, drawMode, onCellHover, onCellClick });
  cb.current = { onBBox, onMapClick, onPointClick, onPolygonClick, onPointContext, onPolygonContext, onHover, onCancelDraw, drawMode, onCellHover, onCellClick };
  const [cov, setCov] = useState<BasemapCoverage | null>(null);
  const [zoom, setZoom] = useState(4);
  const [view, setView] = useState(0);                          // ticks on every pan / zoom: region names are laid out against the visible frame
  const spec = useMemo<BasemapSpec | null>(() => (basemap === undefined || basemap === false ? null : basemap), [basemap]);
  const specKey = spec ? `${spec.scene ?? ''}|${spec.year ?? ''}` : 'off';

  // create once
  useEffect(() => {
    if (!el.current) return;
    const m = L.map(el.current, { attributionControl: false, zoomControl: true, worldCopyJump: false, minZoom: 1, maxZoom: 21, zoomSnap: 0.25, preferCanvas: true, zoomAnimation: false });
    m.setView([22, 79], 4);
    m.createPane('cells').style.zIndex = '350';                  // above the basemap tiles (200), below the vector overlays (400)
    const grat = L.layerGroup().addTo(m), reg = L.layerGroup().addTo(m), shapes = L.layerGroup().addTo(m), pinsG = L.layerGroup().addTo(m), box = L.layerGroup().addTo(m);
    const cellLayer = new CellLayer({});
    cellLayer.addTo(m);
    layers.current = { grat, regions: reg, shapes, pins: pinsG, box, base: null, cells: cellLayer };
    map.current = m;
    (el.current as HTMLDivElement & { __leaflet?: L.Map }).__leaflet = m; // read-only handle for tools/e2e-console.mjs

    const drawGrat = () => {
      grat.clearLayers();
      const carto = cartoRef.current;
      if (!carto && wrap.current?.dataset.basemap !== 'off') return;   // without cartography the graticule is the basemap of last resort
      const b = m.getBounds(), step = graticuleStep(m.getZoom()), size = m.getSize();
      const style = (carto ? { color: '#d6e4ff', weight: 1, opacity: 0.22, interactive: false } : { color: '#5b8cff', weight: 1, opacity: 0.13, interactive: false }) as L.PolylineOptions;
      const k0 = (v: number) => Math.floor(v / step), k1 = (v: number) => Math.ceil(v / step);
      const south = Math.max(-85, b.getSouth()), north = Math.min(85, b.getNorth());
      const lats: number[] = [], lons: number[] = [];
      for (let k = k0(south); k <= k1(north); k++) lats.push(+(k * step).toFixed(6));
      for (let k = k0(b.getWest()); k <= k1(b.getEast()); k++) lons.push(+(k * step).toFixed(6));
      for (const la of lats) L.polyline([[la, b.getWest() - 1], [la, b.getEast() + 1]], style).addTo(grat);
      for (const lo of lons) L.polyline([[b.getSouth() - 1, lo], [b.getNorth() + 1, lo]], style).addTo(grat);
      if (!carto) return;
      // degree labels on the frame: latitudes down the right edge, longitudes along the top; clear of the zoom buttons and the north indicator
      let n = 0;
      const tag = (x: number, y: number, text: string, edge: 'lat' | 'lon') => {
        n++;
        L.marker(m.containerPointToLatLng(L.point(x, y)), { interactive: false, keyboard: false, zIndexOffset: -1000,
          icon: L.divIcon({ className: 'gm-icon', html: `<span class="gm-deg ${edge}">${esc(text)}</span>`, iconSize: [0, 0], iconAnchor: edge === 'lat' ? [0, 7] : [-3, -2] }) }).addTo(grat);
      };
      for (const la of lats) {
        if (la <= south || la >= north) continue;
        const y = m.latLngToContainerPoint([la, b.getWest()]).y;
        if (y > 60 && y < size.y - 14) tag(size.x - 5, y, fmtDeg(la, 'N', 'S', step), 'lat');
      }
      for (const lo of lons) {
        if (lo <= b.getWest() || lo >= b.getEast()) continue;
        const x = m.latLngToContainerPoint([b.getNorth(), lo]).x;
        if (x > 56 && x < size.x - 64) tag(x, 3, fmtDeg(lo, 'E', 'W', step), 'lon');
      }
      if (wrap.current) { wrap.current.dataset.graticuleStep = String(step); wrap.current.dataset.graticuleLabels = String(n); }
    };
    if (cartoRef.current) {                                     // scale bar (metric, drawn by Leaflet from the view) and a fixed north indicator
      L.control.scale({ position: 'bottomleft', imperial: false, maxWidth: 110 }).addTo(m);
      const North = (L.Control as unknown as { extend: (o: object) => new (o?: object) => L.Control }).extend({
        onAdd() {
          const d = L.DomUtil.create('div', 'gm-north');
          d.setAttribute('role', 'img'); d.setAttribute('aria-label', 'North is up'); d.title = 'North is up';
          d.innerHTML = '<i class="gm-north-arrow"></i><b>N</b>';
          return d;
        },
      });
      new North({ position: 'topright' }).addTo(m);
    }
    m.on('moveend zoomend', drawGrat); drawGrat();
    (m as L.Map & { __grat?: () => void }).__grat = drawGrat;
    m.on('moveend', () => setView((v) => v + 1));
    m.on('zoomend', () => { setZoom(m.getZoom()); if (wrap.current) wrap.current.dataset.zoom = String(m.getZoom()); });

    // numbered pins. With clustering on, `hit` pins closer than CLUSTER_PX on screen share one count badge; the grouping is redone
    // at every zoom, so zooming in always separates them. Without it every pin is drawn on its own (the other screens).
    renderPins.current = () => {
      pinsG.clearLayers(); pinIndex.current.clear();
      const z = m.getZoom();
      const groups: { members: MapPoint[]; x: number; y: number }[] = [];
      const single = (p: MapPoint) => {
        const kind = p.pin!.kind ?? 'hit';
        const w = kind === 'seed' ? 52 : 26, h = kind === 'seed' ? 24 : 26, zi = kind === 'seed' ? 20000 : kind === 'sel' ? 10000 : 0; // above any neighbour: Leaflet's own z-order is the pixel row
        const mk = L.marker([p.lat, p.lon], {
          keyboard: false, zIndexOffset: zi,
          icon: L.divIcon({ className: 'gm-icon', html: `<span class="gm-pin ${kind}" data-pin="${esc(p.id)}">${esc(p.pin!.text)}</span>`, iconSize: [w, h], iconAnchor: [w / 2, h / 2] }),
        });
        (mk as L.Marker & { __z?: number }).__z = zi;
        mk.on('click', (e) => { L.DomEvent.stopPropagation(e); cb.current.onPointClick?.(p.id); });
        if (p.label) mk.bindTooltip(p.label, { direction: 'top', offset: [0, -10] });
        mk.on('mouseover', () => cb.current.onHover?.(p.id));
        mk.on('mouseout', () => cb.current.onHover?.(null));
        mk.on('contextmenu', (e) => { L.DomEvent.preventDefault((e as L.LeafletMouseEvent).originalEvent); L.DomEvent.stopPropagation(e as L.LeafletMouseEvent); cb.current.onPointContext?.(p.id); });
        mk.addTo(pinsG); pinIndex.current.set(p.id, mk);
      };
      for (const p of pinData.current) {
        if (!pinOpts.current.cluster || (p.pin!.kind ?? 'hit') !== 'hit') { single(p); continue; }
        const pt = m.project([p.lat, p.lon], z);
        const g = groups.find((q) => Math.hypot(q.x - pt.x, q.y - pt.y) < CLUSTER_PX);
        if (g) { const n = g.members.length; g.x = (g.x * n + pt.x) / (n + 1); g.y = (g.y * n + pt.y) / (n + 1); g.members.push(p); } else groups.push({ members: [p], x: pt.x, y: pt.y });
      }
      for (const g of groups) {
        if (g.members.length === 1) { single(g.members[0]); continue; }
        const ids = g.members.map((q) => q.id), nums = g.members.map((q) => q.pin!.text);
        const at = m.unproject(L.point(g.x, g.y), z);
        const mk = L.marker(at, {
          keyboard: false, zIndexOffset: 5000,
          icon: L.divIcon({ className: 'gm-icon', html: `<span class="gm-pin hit cluster" data-cluster="${esc(ids.join(' '))}" data-count="${ids.length}" data-numbers="${esc(nums.join(' '))}">${ids.length}</span>`, iconSize: [34, 34], iconAnchor: [17, 17] }),
        });
        (mk as L.Marker & { __z?: number }).__z = 5000;
        const shown = nums.slice(0, 10).map((n) => '#' + n).join(', ') + (nums.length > 10 ? ` +${nums.length - 10} more` : '');
        mk.bindTooltip(`${ids.length} results overlap here: ${shown} · click to zoom in`, { direction: 'top', offset: [0, -12] });
        mk.on('click', (e) => {
          L.DomEvent.stopPropagation(e);
          const b = L.latLngBounds(g.members.map((q) => L.latLng(q.lat, q.lon)));
          if (b.getNorthEast().equals(b.getSouthWest())) m.setView(b.getCenter(), Math.min(m.getZoom() + 3, 18), { animate: false });
          else m.fitBounds(b, { padding: [70, 70], maxZoom: 17, animate: false });
        });
        mk.addTo(pinsG);
        for (const id of ids) pinIndex.current.set(id, mk);
      }
      applyPinHover(pinIndex.current, pinOpts.current.hover);
    };
    m.on('zoomend', () => { if (pinOpts.current.cluster) renderPins.current(); });

    // drag-to-draw bounding box
    let start: L.LatLng | null = null, rect: L.Rectangle | null = null, suppressClickUntil = 0;
    m.on('mousedown', (e: L.LeafletMouseEvent) => {
      if (!cb.current.drawMode) return;
      start = e.latlng; m.dragging.disable();
      rect = L.rectangle(L.latLngBounds(start, start), { color: '#ffd166', weight: 2, dashArray: '6 4', fillColor: '#ffd166', fillOpacity: 0.22 }).addTo(box);
    });
    m.on('mousemove', (e: L.LeafletMouseEvent) => { if (start && rect) rect.setBounds(L.latLngBounds(start, e.latlng)); });
    const finish = (e: L.LeafletMouseEvent) => {
      if (!start) return;
      const b = L.latLngBounds(start, e.latlng); start = null; rect = null; m.dragging.enable();
      suppressClickUntil = Date.now() + 400; // the browser fires a click after this mouseup; it must not count as a map click
      if (Math.abs(b.getEast() - b.getWest()) > 1e-4 && Math.abs(b.getNorth() - b.getSouth()) > 1e-4)
        cb.current.onBBox?.([b.getWest(), b.getSouth(), b.getEast(), b.getNorth()]);
    };
    m.on('mouseup', finish);
    m.on('click', (e: L.LeafletMouseEvent) => { if (!cb.current.drawMode && Date.now() > suppressClickUntil) cb.current.onMapClick?.(e.latlng.lng, e.latlng.lat); });

    // cluster cells: report the cluster under the pointer (one lookup per animation frame) and a click on it
    const cellAt = (ll: L.LatLng) => {
      if (!cb.current.onCellHover && !cb.current.onCellClick) return null;
      const b = m.getBounds(), perPx = (b.getEast() - b.getWest()) / Math.max(1, m.getSize().x);
      return cellLayer.pick(ll.lng, ll.lat, perPx * 1.5);
    };
    let lastCell: string | null = null, raf = 0, pending: L.LatLng | null = null;
    m.on('mousemove', (e: L.LeafletMouseEvent) => {
      if (!cb.current.onCellHover || cb.current.drawMode) return;
      pending = e.latlng;
      if (raf) return;
      raf = requestAnimationFrame(() => {
        raf = 0;
        const id = pending ? cellAt(pending) : null;
        m.getContainer().style.cursor = id ? 'pointer' : '';
        if (id !== lastCell) { lastCell = id; cb.current.onCellHover?.(id); }
      });
    });
    m.on('mouseout', () => { if (lastCell !== null) { lastCell = null; m.getContainer().style.cursor = ''; cb.current.onCellHover?.(null); } });
    m.on('click', (e: L.LeafletMouseEvent) => { if (cb.current.drawMode || !cb.current.onCellClick) return; const id = cellAt(e.latlng); if (id) cb.current.onCellClick(id); });

    const ro = new ResizeObserver(() => m.invalidateSize());
    ro.observe(el.current);
    return () => {
      ro.disconnect();
      m.remove(); map.current = null; layers.current = null;
    };
  }, []);

  useEffect(() => { const m = map.current; if (m) { m.getContainer().style.cursor = drawMode ? 'crosshair' : ''; } }, [drawMode]);
  useEffect(() => {
    if (!drawMode) return;
    const k = (e: KeyboardEvent) => { if (e.key === 'Escape') cb.current.onCancelDraw?.(); };
    window.addEventListener('keydown', k);
    return () => window.removeEventListener('keydown', k);
  }, [drawMode]);

  // the local basemap layer (beneath every vector overlay)
  useEffect(() => {
    const L_ = layers.current, m = map.current; if (!L_ || !m) return;
    if (L_.base) { m.removeLayer(L_.base); L_.base = null; }
    if (wrap.current) wrap.current.dataset.basemap = spec ? (spec.scene ? 'scene' : 'sentinel-2') : 'off';
    (m as L.Map & { __grat?: () => void }).__grat?.();
    if (!spec) return;
    const q = new URLSearchParams(); if (spec.year) q.set('year', spec.year); if (spec.scene) q.set('scene', spec.scene);
    const qs = q.toString();
    L_.base = L.tileLayer(`/ui/basemap/{z}/{x}/{y}${qs ? '?' + qs : ''}`, {
      tileSize: 256, minZoom: 1, maxZoom: 21, maxNativeZoom: spec.scene ? 19 : 14, keepBuffer: 2, updateWhenIdle: false, className: 'gm-basemap', errorTileUrl: EMPTY_PNG, attribution: '',
    }).addTo(m);
  }, [specKey]); // eslint-disable-line react-hooks/exhaustive-deps

  // honest caption: what the basemap holds for the current view (from /ui/basemap/coverage)
  useEffect(() => {
    const m = map.current; if (!m || !spec) { setCov(null); return; }
    let timer: number | undefined, ctl: AbortController | undefined;
    const ask = () => {
      ctl?.abort(); ctl = new AbortController();
      const b = m.getBounds(), w = Math.max(-180, b.getWest()), e = Math.min(180, b.getEast()), s = Math.max(-85, b.getSouth()), n = Math.min(85, b.getNorth());
      if (!(w < e && s < n)) return;
      api.basemapCoverage({ bbox: [w, s, e, n], year: spec.year, scene: spec.scene }, ctl.signal).then(setCov).catch(() => { /* aborted or unavailable: keep the last caption */ });
    };
    const later = () => { window.clearTimeout(timer); timer = window.setTimeout(ask, 350); };
    m.on('moveend', later); ask();
    return () => { window.clearTimeout(timer); ctl?.abort(); m.off('moveend', later); };
  }, [specKey]); // eslint-disable-line react-hooks/exhaustive-deps

  // region boxes (plain). Built once per change of `regions`, never on pan / zoom: Leaflet's canvas renderer draws and hit-tests the
  // most recently added layer on top, so re-adding these would put a restricted-zone box over the candidate footprints it contains.
  useEffect(() => {
    const g = layers.current?.regions; if (!g || regionLabels) return;
    g.clearLayers();
    for (const r of regions ?? []) {
      const c = r.color ?? '#5b8cff';
      L.rectangle(toBounds(r.bbox), { color: c, weight: 1.2, fillColor: c, fillOpacity: spec ? 0.04 : 0.07, interactive: true }).bindTooltip(r.label ?? r.name, { sticky: true, direction: 'top' }).addTo(g);
    }
  }, [regions, specKey, regionLabels]); // eslint-disable-line react-hooks/exhaustive-deps

  // region boxes as the archive's coverage (`regionLabels`): a clear outline per catalogued region, its name drawn permanently where it
  // fits (greedy, largest footprint first; the rest keep a hover name), and a fixed-size marker for footprints too small to see.
  useEffect(() => {
    const g = layers.current?.regions, m = map.current; if (!g || !m || !regionLabels) return;
    g.clearLayers();
    const list = regions ?? [];
    const MIN = 14, size = m.getSize();
    const items = list.map((r) => {
      const nw = m.latLngToContainerPoint([r.bbox[3], r.bbox[0]]), se = m.latLngToContainerPoint([r.bbox[1], r.bbox[2]]);
      const w = se.x - nw.x, h = se.y - nw.y;
      return { r, w, h, x0: Math.min(nw.x, se.x - MIN), x1: Math.max(se.x, nw.x + MIN), y0: Math.min(nw.y, se.y - MIN), y1: Math.max(se.y, nw.y + MIN), c: r.color ?? '#9db8ff', text: r.label ?? r.name };
    });
    for (const it of items) {
      L.rectangle(toBounds(it.r.bbox), { color: it.c, weight: 1.8, fillColor: it.c, fillOpacity: 0.07, interactive: true }).bindTooltip(it.text, { sticky: true, direction: 'top' }).addTo(g);
      if (it.w < MIN || it.h < MIN) {                              // sub-pixel footprint: a fixed-size square so it can still be seen
        const mk = L.marker(toBounds(it.r.bbox).getCenter(), { keyboard: false, icon: L.divIcon({ className: 'gm-icon', html: `<span class="gm-regdot" style="border-color:${esc(it.c)}"></span>`, iconSize: [MIN, MIN], iconAnchor: [MIN / 2, MIN / 2] }) });
        mk.bindTooltip(it.text, { direction: 'top' }); mk.addTo(g);
      }
    }
    // names go where they are readable: not over another name, not over a result pin, not clipped by the frame
    const placed: [number, number, number, number][] = [];
    for (const p of points ?? []) if (p.pin) { const c = m.latLngToContainerPoint([p.lat, p.lon]); placed.push([c.x - 17, c.y - 17, c.x + 17, c.y + 17]); }
    const clash = (r: readonly number[]) => placed.some(([a0, b0, a1, b1]) => r[0] < a1 && r[2] > a0 && r[1] < b1 && r[3] > b0);
    const area = (it: { x0: number; x1: number; y0: number; y1: number }) => (it.x1 - it.x0) * (it.y1 - it.y0);
    let named = 0;
    for (const it of [...items].sort((p, q) => area(q) - area(p))) {
      const lw = Math.round(it.text.length * 6.1 + 12), lh = 16, cx = (it.x0 + it.x1) / 2, cy = (it.y0 + it.y1) / 2;
      // first free spot wins: centred above / below / beside the footprint, then the corners, then the same ring one step further out
      const cands: { r: [number, number, number, number]; gap: number }[] = [];
      for (const gap of [4, 22, 40]) {
        for (const [x, y] of [[cx - lw / 2, it.y0 - lh - gap], [cx - lw / 2, it.y1 + gap], [it.x1 + gap + 1, cy - lh / 2], [it.x0 - gap - 1 - lw, cy - lh / 2],
          [it.x1 + gap - 6, it.y0 - lh - gap], [it.x0 - gap + 6 - lw, it.y0 - lh - gap], [it.x1 + gap - 6, it.y1 + gap], [it.x0 - gap + 6 - lw, it.y1 + gap]])
          cands.push({ r: [x, y, x + lw, y + lh], gap });
      }
      const pick = cands.find((k) => k.r[0] >= 4 && k.r[2] <= size.x - 4 && k.r[1] >= 4 && k.r[3] <= size.y - 4 && !clash(k.r));
      if (!pick) continue;
      const at = pick.r;
      placed.push(at); named++;
      if (pick.gap > 4) {                                        // displaced: a thin leader keeps the name tied to its own footprint
        const lx = Math.min(Math.max(cx, at[0]), at[2]), ly = Math.min(Math.max(cy, at[1]), at[3]);
        L.polyline([m.containerPointToLatLng(L.point(cx, cy)), m.containerPointToLatLng(L.point(lx, ly))], { color: it.c, weight: 1, opacity: 0.75, interactive: false }).addTo(g);
      }
      L.marker(m.containerPointToLatLng(L.point(at[0], at[1])), { interactive: false, keyboard: false, icon: L.divIcon({ className: 'gm-icon', html: `<span class="gm-reglbl" data-region="${esc(it.r.name)}" style="color:${esc(it.c)}">${esc(it.text)}</span>`, iconSize: [lw, lh], iconAnchor: [0, 0] }) }).addTo(g);
    }
    if (wrap.current) { wrap.current.dataset.regionCount = String(items.length); wrap.current.dataset.regionLabels = String(named); }
  }, [regions, specKey, regionLabels, view, points]); // eslint-disable-line react-hooks/exhaustive-deps

  // points / polygons / circles
  useEffect(() => {
    const g = layers.current?.shapes; if (!g) return;
    g.clearLayers(); pins.current.clear();
    pinData.current = (points ?? []).filter((p) => p.pin);
    renderPins.current();
    for (const c of circles ?? []) {
      const ci = L.circle([c.lat, c.lon], { radius: c.radiusM, color: c.color, weight: 2, dashArray: c.dashed ? '6 5' : undefined, fillOpacity: 0.05, interactive: false }).addTo(g);
      if (c.label) ci.bindTooltip(c.label, { direction: 'top' });
      if (c.tag) {                                          // the radius, printed on the circle itself (not hidden in a tooltip)
        const lat = c.lat + c.radiusM / 110_574;
        L.marker([lat, c.lon], { interactive: false, keyboard: false, icon: L.divIcon({ className: 'gm-icon', html: `<span class="gm-ringlbl" style="color:${esc(c.color)};border-color:${esc(c.color)}">${esc(c.tag)}</span>`, iconSize: [64, 18], iconAnchor: [32, 22] }) }).addTo(g);
      }
    }
    for (const p of polygons ?? []) {
      const sel = !!p.selected;
      const gj = L.geoJSON(p.geometry as unknown as GeoJSON.Polygon, { style: { color: sel ? '#ffffff' : p.color, weight: sel ? 3 : 1.5, fillColor: p.color, fillOpacity: sel ? Math.max(p.fill ?? 0.3, 0.38) : Math.min(p.fill ?? 0.12, 0.14), dashArray: sel ? undefined : '3 3' } }).addTo(g);
      if (p.label) gj.bindTooltip(p.label, { sticky: true });
      gj.on('click', () => cb.current.onPolygonClick?.(p.id));
      gj.on('contextmenu', (e: L.LeafletMouseEvent) => { L.DomEvent.preventDefault(e.originalEvent); L.DomEvent.stopPropagation(e); cb.current.onPolygonContext?.(p.id); });
      if (sel) {
        const ctr = gj.getBounds().getCenter();
        L.marker(ctr, { interactive: false, keyboard: false, zIndexOffset: 900, icon: L.divIcon({ className: 'gm-icon', html: '<span class="gm-centre" data-selected="1"></span>', iconSize: [22, 22], iconAnchor: [11, 11] }) }).addTo(g);
        if (p.tag) L.marker([gj.getBounds().getNorth(), ctr.lng], { interactive: false, keyboard: false, zIndexOffset: 900, icon: L.divIcon({ className: 'gm-icon', html: `<span class="gm-sellbl">${esc(p.tag)}</span>`, iconSize: [150, 20], iconAnchor: [75, 26] }) }).addTo(g);
      }
    }
    for (const p of points ?? []) {
      if (p.pin) continue;                                         // numbered pins are drawn by renderPins
      const r = p.radius ?? 5, op = p.opacity ?? 0.55;
      const wgt = r < 4 ? 0.8 : 1.5;                              // a hairline edge on small dots, so they do not swell past their true size
      const cm = L.circleMarker([p.lat, p.lon], { radius: r, color: p.color ?? '#4cc9f0', weight: wgt, fillColor: p.color ?? '#4cc9f0', fillOpacity: op, opacity: Math.min(1, op + 0.25) });
      cm.on('click', (e) => { L.DomEvent.stopPropagation(e); cb.current.onPointClick?.(p.id); });
      pins.current.set(p.id, { layer: cm, base: { radius: r, opacity: op, weight: wgt } });
      cm.addTo(g);
      if (p.label) cm.bindTooltip(p.label, { direction: 'top' });
      cm.on('mouseover', () => cb.current.onHover?.(p.id));
      cm.on('mouseout', () => cb.current.onHover?.(null));
      cm.on('contextmenu', (e) => { L.DomEvent.preventDefault((e as L.LeafletMouseEvent).originalEvent); L.DomEvent.stopPropagation(e as L.LeafletMouseEvent); cb.current.onPointContext?.(p.id); });
    }
    if (wrap.current) wrap.current.dataset.pointCount = String(points?.length ?? 0);
  }, [points, polygons, circles]);

  // hover link: highlight the addressed pin / dot without rebuilding any layer
  useEffect(() => {
    applyPinHover(pinIndex.current, hoverId);
    for (const [id, r] of pins.current) {
      const on = id === hoverId;
      const cm = r.layer as L.CircleMarker;
      cm.setStyle({ radius: on ? r.base.radius + 4 : r.base.radius, weight: on ? 3 : r.base.weight, color: on ? '#ffffff' : (cm.options.fillColor as string), fillOpacity: on ? 1 : r.base.opacity });
      if (on) cm.bringToFront();
    }
    if (wrap.current) wrap.current.dataset.hover = hoverId ?? '';
  }, [hoverId, points]);

  // cluster cells
  useEffect(() => {
    layers.current?.cells.setData(cells ?? [], cellDeg, activeCell);
    if (wrap.current) wrap.current.dataset.activeCluster = activeCell ?? '';
  }, [cells, cellDeg, activeCell]);

  // the active bbox filter
  useEffect(() => {
    const g = layers.current?.box; if (!g) return;
    g.clearLayers();
    if (bbox) {
      const r = L.rectangle(toBounds(bbox), { color: '#ffd166', weight: 2.5, fillColor: '#ffd166', fillOpacity: 0.16, interactive: false }).addTo(g);
      r.bindTooltip('search box', { permanent: true, direction: 'top', className: 'gm-label' });
    }
  }, [bbox]);

  useEffect(() => {
    if (fit && map.current) map.current.fitBounds(toBounds(fit), { padding: [40, 40], animate: false, maxZoom: fitMaxZoom });
  }, [fit]); // eslint-disable-line react-hooks/exhaustive-deps

  const cap = spec ? captionFor(cov, zoom, spec, catalogRegions) : null;
  return (
    <div className="geomap" ref={wrap} data-basemap={spec ? (spec.scene ? 'scene' : 'sentinel-2') : 'off'} data-point-count={points?.length ?? 0}>
      <div className="geomap-view" style={fill ? { minHeight: height } : { height }}>
        <div ref={el} role="application" aria-label={ariaLabel} />
        {drawMode && <div className="gm-hint" role="status">Drag on the map to draw the search box · Esc cancels</div>}
      </div>
      {cap && (
        <div className={`gm-caption ${cap.tone}`} data-testid="map-caption" data-coverage={cov ? cov.fraction : ''} title={cov?.source}>
          <i />{cap.text}
        </div>
      )}
    </div>
  );
}
