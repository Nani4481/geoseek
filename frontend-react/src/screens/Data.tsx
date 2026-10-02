import { useCallback, useMemo, useRef, useState } from 'react';
import { api } from '@/api/client';
import type { BBox, ConsoleMetrics, RegionListItem } from '@/api/types';
import { GeoMap } from '@/components/GeoMap';
import { Panel } from '@/components/Panel';
import { ErrorNote, Loading } from '@/components/Widgets';
import { DASH, fmtInt, regionLabel } from '@/fmt';
import { useApi } from '@/hooks/useApi';
import { parseRasterHeader, type RasterHeader } from '@/lib/rasterHeader';

const MAX_FILES = 6;

const human = (b: number | null) => (b === null ? DASH : b < 1024 ** 2 ? `${(b / 1024).toFixed(1)} kB` : b < 1024 ** 3 ? `${(b / 1024 ** 2).toFixed(1)} MB` : `${(b / 1024 ** 3).toFixed(2)} GB`);
const bboxArea = (b: BBox) => Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
function overlap(a: BBox, b: BBox): number {
  const w = Math.min(a[2], b[2]) - Math.max(a[0], b[0]), h = Math.min(a[3], b[3]) - Math.max(a[1], b[1]);
  return w > 0 && h > 0 ? w * h : 0;
}

interface Parsed { name: string; header: RasterHeader | null; error: string | null }

function verdict(p: Parsed): { cls: 'green' | 'amber' | 'red'; text: string } {
  if (!p.header) return { cls: 'red', text: 'NOT A READABLE GEOTIFF' };
  const bad = p.header.checks.filter((c) => !c.ok);
  return bad.length === 0 ? { cls: 'green', text: 'HEADER VALID · GEOREFERENCED' } : { cls: 'amber', text: `READABLE · ${bad.length} CHECK${bad.length === 1 ? '' : 'S'} FAILED` };
}

function RasterCard({ p, regions, sensors }: { p: Parsed; regions: RegionListItem[]; sensors: ConsoleMetrics['sensors'] }) {
  const h = p.header;
  const v = verdict(p);
  const ll = h?.lonlatBounds ?? null;
  const overlapping = useMemo(() => (ll ? regions.map((r) => ({ r, frac: overlap(ll, r.bbox) / Math.max(bboxArea(ll), 1e-12), cover: overlap(ll, r.bbox) / Math.max(bboxArea(r.bbox), 1e-12) }))
    .filter((x) => x.frac > 0).sort((a, b) => b.frac - a.frac) : []), [ll, regions]);
  const gsdMatch = h?.resolution ? sensors.filter((s) => Math.abs(s.native_gsd_m - h.resolution![0]) / s.native_gsd_m < 0.01) : [];
  const polys = useMemo(() => (ll ? [{ id: 'file', color: '#f5a524', fill: 0.25, label: p.name, geometry: { type: 'Polygon' as const, coordinates: [[[ll[0], ll[1]], [ll[2], ll[1]], [ll[2], ll[3]], [ll[0], ll[3]], [ll[0], ll[1]]]] } }] : []), [ll, p.name]);
  const boxes = useMemo(() => regions.map((r) => ({ name: r.name, bbox: r.bbox, label: regionLabel(r.name), color: '#4cc9f0' })), [regions]);
  const fit = useMemo<BBox | null>(() => (ll ? [ll[0] - (ll[2] - ll[0]) * 0.6, ll[1] - (ll[3] - ll[1]) * 0.6, ll[2] + (ll[2] - ll[0]) * 0.6, ll[3] + (ll[3] - ll[1]) * 0.6] : null), [ll]);

  return (
    <article className="raster-card" data-verdict={v.cls} aria-label={`Header check for ${p.name}`}>
      <header className="row" style={{ alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
        <b className="mono">{p.name}</b>
        <span className={`chip ${v.cls}`} data-testid="verdict">{v.text}</span>
        {h && <span className="dim mono" style={{ fontSize: 11 }}>{human(h.sizeBytes)}</span>}
      </header>
      {!h ? <div className="err" style={{ padding: '6px 0' }}>The file header could not be parsed: {p.error}. Nothing else can be said about it.</div> : (
        <div className="raster-body">
          <div className="col" style={{ gap: 10, minWidth: 0 }}>
            <div className="row" style={{ flexWrap: 'wrap', gap: 6 }} role="list" aria-label="Header checks">
              {h.checks.map((c) => <span key={c.id} role="listitem" data-check={c.id} data-ok={c.ok ? '1' : '0'} className={`chip ${c.ok ? 'green' : 'red'}`} title={c.detail}>{c.ok ? '✓' : '✕'} {c.label}</span>)}
            </div>
            <dl className="kv raster-kv">
              <dt>Size</dt><dd data-field="size">{fmtInt(h.width)} × {fmtInt(h.height)} px</dd>
              <dt>Bands</dt><dd data-field="bands">{h.bands} × {h.dtype}</dd>
              <dt>CRS</dt><dd data-field="crs">{h.crs.epsg !== null ? `EPSG:${h.crs.epsg}${h.crs.name ? ' · ' + h.crs.name : ''}` : h.crs.kind === 'unknown' ? 'none' : 'user-defined (no EPSG code)'}</dd>
              <dt>Pixel size</dt><dd data-field="resolution">{h.resolution ? `${+h.resolution[0].toPrecision(8)} × ${+h.resolution[1].toPrecision(8)}${h.crs.kind === 'geographic' ? ' °' : ' m'}` : DASH}</dd>
              <dt>Affine transform</dt><dd className="mono" data-field="affine" style={{ fontSize: 10.5 }}>{h.affine ? `[${h.affine.map((x) => +x.toPrecision(10)).join(', ')}]` : DASH}</dd>
              <dt>Bounds (file CRS)</dt><dd className="mono" data-field="bounds" style={{ fontSize: 10.5 }}>{h.bounds ? h.bounds.map((x) => +x.toFixed(3)).join(', ') : DASH}</dd>
              <dt>Bounds (lon/lat)</dt><dd className="mono" data-field="lonlat" style={{ fontSize: 10.5 }}>{ll ? ll.map((x) => x.toFixed(5)).join(', ') : DASH}</dd>
              <dt>Acquisition</dt><dd data-field="acquisition">{h.acquisition ? `${h.acquisition.iso ?? h.acquisition.raw} (${h.acquisition.source})` : 'none in the header'}</dd>
              {h.fileTimestamp && <><dt>File written</dt><dd data-field="filetime">{h.fileTimestamp.iso ?? h.fileTimestamp.raw} (TIFF DateTime — when the file was written)</dd></>}
              <dt>Layout</dt><dd>{h.tiled ? 'internally tiled' : 'striped'}{h.overviews ? ` · ${h.overviews} overview level${h.overviews === 1 ? '' : 's'}` : ' · no overviews'}{h.compression ? ` · ${h.compression}` : ''}</dd>
              <dt>NoData</dt><dd>{h.nodata ?? 'not set'}</dd>
            </dl>
            <div className="raster-note" data-testid="what-it-contains">
              <b>What this file contains.</b> A {fmtInt(h.width)} × {fmtInt(h.height)} px, {h.bands}-band {h.dtype} raster
              {h.crs.epsg !== null ? <> in EPSG:{h.crs.epsg}</> : <> with no usable coordinate reference system</>}
              {h.resolution ? <>, {+h.resolution[0].toPrecision(6)} {h.crs.kind === 'geographic' ? '°' : 'm'} pixels</> : null}
              {ll ? <>, covering lon {ll[0].toFixed(4)}…{ll[2].toFixed(4)}, lat {ll[1].toFixed(4)}…{ll[3].toFixed(4)}</> : null}
              {h.acquisition?.iso ? <>, acquired {h.acquisition.iso}</> : <>; its header carries no acquisition timestamp</>}.
              {' '}{overlapping.length ? <>It overlaps the archive region{overlapping.length > 1 ? 's' : ''} {overlapping.slice(0, 3).map((o) => `${regionLabel(o.r.name)} (${Math.round(o.frac * 100)}% of the file)`).join(', ')}.</> : ll ? <>It does not overlap any archive region.</> : null}
              {gsdMatch.length ? <> Its pixel size matches the {gsdMatch.map((s) => `${s.sensor} ${s.native_gsd_m} m`).join(' / ')} collection{gsdMatch.length > 1 ? 's' : ''} already in the archive.</> : null}
            </div>
          </div>
          <div className="raster-map" aria-label="Footprint on the archive map">
            {ll && fit ? <GeoMap ariaLabel="File footprint and archive regions" regions={boxes} polygons={polys} fit={fit} height={250} />
              : <div className="empty">{h.lonlatNote ?? 'No footprint to draw.'}</div>}
          </div>
        </div>
      )}
    </article>
  );
}

function RasterCheck() {
  const regions = useApi((s) => api.regions(s), []);
  const metrics = useApi((s) => api.metrics(s), []);
  const [items, setItems] = useState<Parsed[]>([]);
  const [busy, setBusy] = useState(false);
  const [over, setOver] = useState(false);
  const input = useRef<HTMLInputElement>(null);

  const handle = useCallback(async (files: FileList | File[]) => {
    const list = [...files].slice(0, MAX_FILES);
    if (!list.length) return;
    setBusy(true);
    const out: Parsed[] = [];
    for (const f of list) {
      try { out.push({ name: f.name, header: await parseRasterHeader(f, f.name), error: null }); }
      catch (e) { out.push({ name: f.name, header: null, error: e instanceof Error ? e.message : String(e) }); }
    }
    setItems(out); setBusy(false);
  }, []);

  return (
    <Panel title="Ad-hoc raster check · GeoTIFF header">
      <div className="col" style={{ gap: 12 }}>
        <div className={`dropzone ${over ? 'over' : ''}`} role="button" tabIndex={0} aria-label="Drop GeoTIFF files here or press Enter to choose"
          onDragOver={(e) => { e.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)}
          onDrop={(e) => { e.preventDefault(); setOver(false); void handle(e.dataTransfer.files); }}
          onClick={() => input.current?.click()} onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); input.current?.click(); } }}>
          <div style={{ fontSize: 28 }}>⇪</div>
          <div style={{ fontWeight: 600, fontSize: 14 }}>Drop a GeoTIFF / COG here, or click to choose</div>
          <div className="dim" style={{ fontSize: 11.5, marginTop: 4 }}>The header is parsed in this browser (up to {MAX_FILES} files). Pixel data is not read and nothing leaves this machine.</div>
          <input ref={input} type="file" multiple accept=".tif,.tiff,.gtiff,image/tiff" style={{ display: 'none' }} aria-label="Choose GeoTIFF files" data-testid="raster-input"
            onChange={(e) => { if (e.target.files) void handle(e.target.files); e.target.value = ''; }} />
        </div>
        {busy && <Loading rows={2} />}
        {metrics.error && <ErrorNote error={metrics.error} />}
        {items.map((p, i) => <RasterCard key={`${p.name}-${i}`} p={p} regions={regions.data?.regions ?? []} sensors={metrics.data?.sensors ?? []} />)}
        {items.length > 0 && (
          <div className="warnbox" role="note" data-testid="ingest-note">
            <b>Not ingested.</b> Checking a header does not add the file to the archive, and no ingest is running. Staging new imagery is done through the
            ingest pipeline: place the per-band GeoTIFFs (B02, B03, B04, B08, B11, SCL) in a scene directory named like <span className="mono">S2B_44RPQ_20190330_1_L2A</span> under
            <span className="mono"> data/datasets/</span>, then run <span className="mono">python -m geoseek.ingest.pipeline ingest &lt;scene_dir&gt;</span>. A single multi-band file must be split into per-band files first.
          </div>
        )}
      </div>
    </Panel>
  );
}

function ArchiveInventory() {
  const m = useApi((s) => api.metrics(s), []);
  const regions = useApi((s) => api.regions(s), []);
  const d = m.data;
  return (
    <Panel title="Archive inventory">
      {m.error ? <ErrorNote error={m.error} onRetry={m.reload} /> : !d ? <Loading rows={6} /> : (
        <div className="col" style={{ gap: 14 }}>
          <dl className="kv">
            <dt>Tiles indexed</dt><dd>{fmtInt(d.counters.tiles_indexed)}</dd>
            <dt>Searchable vectors</dt><dd>{fmtInt(d.counters.vectors_searchable)}</dd>
            <dt>Scenes</dt><dd>{fmtInt(d.counters.scenes)}</dd>
            <dt>Collections</dt><dd>{fmtInt(d.counters.collections)}</dd>
            <dt>Change-pipeline acquisition dates</dt><dd className="mono" style={{ fontSize: 11 }}>{d.observation_dates.join(' · ')}</dd>
          </dl>
          <table className="tbl" aria-label="Collections">
            <thead><tr><th>Collection</th><th>Sensor</th><th className="num">GSD</th><th className="num">Scenes</th></tr></thead>
            <tbody>{d.sensors.map((s) => <tr key={s.collection_id}><td className="mono">{s.collection_id}</td><td>{s.platform} · {s.sensor}</td><td className="num">{s.native_gsd_m} m</td><td className="num">{fmtInt(s.n_scenes)}</td></tr>)}</tbody>
          </table>
          <table className="tbl" aria-label="Regions">
            <thead><tr><th>Region</th><th className="num">Observations</th><th>Bounding box</th></tr></thead>
            <tbody>{(regions.data?.regions ?? []).map((r) => <tr key={r.name}><td>{regionLabel(r.name)}</td><td className="num">{fmtInt(r.n_observations)}</td><td className="mono dim" style={{ fontSize: 10.5 }}>{r.bbox.map((v) => v.toFixed(3)).join(', ')}</td></tr>)}</tbody>
          </table>
        </div>
      )}
    </Panel>
  );
}

export function Data() {
  return (
    <div className="data-grid">
      <RasterCheck />
      <ArchiveInventory />
    </div>
  );
}
