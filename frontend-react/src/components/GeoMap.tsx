import { useEffect, useRef } from 'react';
import L from 'leaflet';
import '@vendor/leaflet/leaflet.css';
import type { BBox, Polygon } from '@/api/types';

export interface MapPoint { id: string; lon: number; lat: number; color?: string; label?: string; radius?: number }
export interface MapPolygon { id: string; geometry: Polygon; color: string; label?: string; fill?: number }
export interface MapRegion { name: string; bbox: BBox; color?: string; label?: string }
export interface MapCircle { lon: number; lat: number; radiusM: number; color: string; label?: string; dashed?: boolean }

interface Props {
  regions?: MapRegion[];
  points?: MapPoint[];
  polygons?: MapPolygon[];
  circles?: MapCircle[];
  bbox?: BBox | null;
  drawMode?: boolean;
  fit?: BBox | null;
  height?: number | string;
  onBBox?: (b: BBox) => void;
  onMapClick?: (lon: number, lat: number) => void;
  onPointClick?: (id: string) => void;
  onPolygonClick?: (id: string) => void;
  ariaLabel: string;
}

const toBounds = (b: BBox) => L.latLngBounds([b[1], b[0]], [b[3], b[2]]);

function graticuleStep(zoom: number) {
  if (zoom >= 11) return 0.02;
  if (zoom >= 9) return 0.1;
  if (zoom >= 7) return 0.5;
  if (zoom >= 5) return 1;
  if (zoom >= 4) return 2;
  return 5;
}

/** Offline Leaflet map: there are no tiles to fetch, so the "basemap" is a graticule on the dark canvas. */
export function GeoMap({ regions, points, polygons, circles, bbox, drawMode, fit, height = 360, onBBox, onMapClick, onPointClick, onPolygonClick, ariaLabel }: Props) {
  const el = useRef<HTMLDivElement>(null);
  const map = useRef<L.Map | null>(null);
  const layers = useRef<{ grat: L.LayerGroup; regions: L.LayerGroup; shapes: L.LayerGroup; box: L.LayerGroup } | null>(null);
  const cb = useRef({ onBBox, onMapClick, onPointClick, onPolygonClick, drawMode });
  cb.current = { onBBox, onMapClick, onPointClick, onPolygonClick, drawMode };

  // create once
  useEffect(() => {
    if (!el.current) return;
    const m = L.map(el.current, { attributionControl: false, zoomControl: true, worldCopyJump: false, minZoom: 2, zoomSnap: 0.25 });
    m.setView([22, 79], 4);
    const grat = L.layerGroup().addTo(m), reg = L.layerGroup().addTo(m), shapes = L.layerGroup().addTo(m), box = L.layerGroup().addTo(m);
    layers.current = { grat, regions: reg, shapes, box };
    map.current = m;
    (el.current as HTMLDivElement & { __leaflet?: L.Map }).__leaflet = m; // read-only handle for tools/e2e-tier1.mjs

    const drawGrat = () => {
      grat.clearLayers();
      const b = m.getBounds(), step = graticuleStep(m.getZoom());
      const style = { color: '#5b8cff', weight: 1, opacity: 0.13, interactive: false } as L.PolylineOptions;
      const lat0 = Math.floor(b.getSouth() / step) * step, lon0 = Math.floor(b.getWest() / step) * step;
      for (let la = lat0; la <= b.getNorth() + step; la += step) L.polyline([[la, b.getWest() - 1], [la, b.getEast() + 1]], style).addTo(grat);
      for (let lo = lon0; lo <= b.getEast() + step; lo += step) L.polyline([[b.getSouth() - 1, lo], [b.getNorth() + 1, lo]], style).addTo(grat);
    };
    m.on('moveend zoomend', drawGrat); drawGrat();

    // drag-to-draw bounding box
    let start: L.LatLng | null = null, rect: L.Rectangle | null = null, suppressClickUntil = 0;
    m.on('mousedown', (e: L.LeafletMouseEvent) => {
      if (!cb.current.drawMode) return;
      start = e.latlng; m.dragging.disable();
      rect = L.rectangle(L.latLngBounds(start, start), { color: '#4cc9f0', weight: 2, dashArray: '5 4', fillOpacity: 0.1 }).addTo(box);
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

    const ro = new ResizeObserver(() => m.invalidateSize());
    ro.observe(el.current);
    return () => { ro.disconnect(); m.remove(); map.current = null; layers.current = null; };
  }, []);

  useEffect(() => { const m = map.current; if (m) { m.getContainer().style.cursor = drawMode ? 'crosshair' : ''; } }, [drawMode]);

  // region boxes
  useEffect(() => {
    const g = layers.current?.regions; if (!g) return;
    g.clearLayers();
    for (const r of regions ?? []) {
      const c = r.color ?? '#5b8cff';
      const rect = L.rectangle(toBounds(r.bbox), { color: c, weight: 1.2, fillColor: c, fillOpacity: 0.07, interactive: !!r.label || true });
      rect.bindTooltip(r.label ?? r.name, { sticky: true, direction: 'top' });
      rect.addTo(g);
    }
  }, [regions]);

  // points / polygons / circles
  useEffect(() => {
    const g = layers.current?.shapes; if (!g) return;
    g.clearLayers();
    for (const c of circles ?? []) {
      const ci = L.circle([c.lat, c.lon], { radius: c.radiusM, color: c.color, weight: 1.5, dashArray: c.dashed ? '6 5' : undefined, fillOpacity: 0.04 }).addTo(g);
      if (c.label) ci.bindTooltip(c.label, { direction: 'top' });
    }
    for (const p of polygons ?? []) {
      const gj = L.geoJSON(p.geometry as unknown as GeoJSON.Polygon, { style: { color: p.color, weight: 2, fillColor: p.color, fillOpacity: p.fill ?? 0.25 } }).addTo(g);
      if (p.label) gj.bindTooltip(p.label, { sticky: true });
      gj.on('click', () => cb.current.onPolygonClick?.(p.id));
    }
    for (const p of points ?? []) {
      const m = L.circleMarker([p.lat, p.lon], { radius: p.radius ?? 5, color: p.color ?? '#4cc9f0', weight: 1.5, fillColor: p.color ?? '#4cc9f0', fillOpacity: 0.55 }).addTo(g);
      if (p.label) m.bindTooltip(p.label, { direction: 'top' });
      m.on('click', (e) => { L.DomEvent.stopPropagation(e); cb.current.onPointClick?.(p.id); });
    }
  }, [points, polygons, circles]);

  // the active bbox filter
  useEffect(() => {
    const g = layers.current?.box; if (!g) return;
    g.clearLayers();
    if (bbox) L.rectangle(toBounds(bbox), { color: '#4cc9f0', weight: 2, dashArray: '5 4', fillOpacity: 0.08, interactive: false }).addTo(g);
  }, [bbox]);

  useEffect(() => {
    if (fit && map.current) map.current.fitBounds(toBounds(fit), { padding: [24, 24], animate: false });
  }, [fit]);

  return <div ref={el} role="application" aria-label={ariaLabel} style={{ height, width: '100%', borderRadius: 8 }} />;
}
