import { useEffect, useMemo, useRef, useState } from 'react';
import { api } from '@/api/client';
import type { BasemapCoverage } from '@/api/types';
import { renderPreview, type PreviewState } from '@/components/RasterPreview';
import { DASH, fmtPct } from '@/fmt';
import { chooseZoom, compositeRGBA, gridForBox, lumaOf, makeProjector, visualDiff, warpToGrid, type Band, type DiffResult, type MercGrid } from '@/lib/rasterMath';
import type { Upload } from '@/lib/uploads';

const MAX_TILES = 16;
const NATIVE_MAX_ZOOM = 14;
const dayDiff = (a: string, b: string) => Math.abs(Date.parse(a + 'T00:00:00Z') - Date.parse(b + 'T00:00:00Z')) / 86_400_000;

/** The archive acquisition nearest in time to the file's date; with no date, the lowest-cloud one (what the basemap shows by default). */
export function nearestAcquisition(acq: NonNullable<BasemapCoverage['acquisitions']>, date: string | null): { pick: NonNullable<BasemapCoverage['acquisitions']>[number]; basis: 'nearest' | 'default' } | null {
  if (!acq.length) return null;
  const byCloud = (a: typeof acq[number], b: typeof acq[number]) => a.mean_cloud - b.mean_cloud || (a.date < b.date ? 1 : -1);
  if (!date) return { pick: [...acq].sort(byCloud)[0], basis: 'default' };
  return { pick: [...acq].sort((a, b) => dayDiff(a.date, date) - dayDiff(b.date, date) || byCloud(a, b))[0], basis: 'nearest' };
}

async function loadTile(url: string): Promise<HTMLImageElement | null> {
  return new Promise((res) => { const im = new Image(); im.onload = () => res(im); im.onerror = () => res(null); im.src = url; });
}

function paint(c: HTMLCanvasElement | null, rgba: Uint8ClampedArray, w: number, h: number) {
  if (!c) return;
  c.width = w; c.height = h;
  c.getContext('2d')?.putImageData(new ImageData(new Uint8ClampedArray(rgba), w, h), 0, 0);
}

interface Result { images: { file: Uint8ClampedArray; archive: Uint8ClampedArray }; year: string; used: BasemapCoverage; z: number; grid: MercGrid; diff: DiffResult; res_m: number; archiveShare: number; fileShare: number }

/** Uploaded raster beside the nearest archive acquisition, with an indicative difference view. Never presents itself as pipeline output. */
export function RasterCompare({ up, preview }: { up: Upload; preview: PreviewState }) {
  const h = up.header, px = up.pixels, ll = h?.lonlatBounds ?? null;
  const [cov, setCov] = useState<BasemapCoverage | null>(null);
  const [covErr, setCovErr] = useState<string | null>(null);
  const [pick, setPick] = useState<string>('');
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [res, setRes] = useState<Result | null>(null);
  const cUp = useRef<HTMLCanvasElement>(null), cAr = useRef<HTMLCanvasElement>(null), cDf = useRef<HTMLCanvasElement>(null);
  const stamp = useRef(0);

  useEffect(() => {
    if (!ll) return;
    const ac = new AbortController();
    setCov(null); setCovErr(null); setRes(null);
    api.basemapCoverage({ bbox: ll }, ac.signal).then(setCov).catch((e: unknown) => { if (!ac.signal.aborted) setCovErr(e instanceof Error ? e.message : String(e)); });
    return () => ac.abort();
  }, [ll?.join(',')]); // eslint-disable-line react-hooks/exhaustive-deps

  const acq = cov?.acquisitions ?? [];
  const nearest = useMemo(() => nearestAcquisition(acq, up.acquired.date), [acq, up.acquired.date]);
  useEffect(() => { if (nearest) setPick(nearest.pick.date); }, [nearest?.pick.date]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { setRes(null); }, [pick, preview, up.id]);

  // the canvases exist only once the result has rendered, so they are painted from an effect
  useEffect(() => {
    if (!res) return;
    paint(cUp.current, res.images.file, res.grid.w, res.grid.h); paint(cAr.current, res.images.archive, res.grid.w, res.grid.h); paint(cDf.current, res.diff.rgba, res.grid.w, res.grid.h);
  }, [res]);

  if (!h || !ll) return null;

  const run = async () => {
    if (!px || !h.affine || !h.crs.epsg || !cov) return;
    const my = ++stamp.current;
    setBusy(true); setErr(null); setRes(null);
    try {
      const proj = makeProjector({ affine: h.affine, epsg: h.crs.epsg, width: h.width, height: h.height });
      if (!proj) throw new Error(`EPSG:${h.crs.epsg} cannot be inverted offline, so this file cannot be placed on the archive grid`);
      const year = pick.slice(0, 4);
      const used = await api.basemapCoverage({ bbox: ll, year });
      const z = chooseZoom(ll, NATIVE_MAX_ZOOM, MAX_TILES);
      const { grid, tiles } = gridForBox(ll, z);
      // archive side: the basemap tiles of that year over the footprint, drawn onto one canvas on the common grid
      const cv = document.createElement('canvas'); cv.width = grid.w; cv.height = grid.h;
      const g = cv.getContext('2d', { willReadFrequently: true })!;
      const jobs: Promise<void>[] = [];
      for (let ty = tiles.y0; ty <= tiles.y1; ty++) for (let tx = tiles.x0; tx <= tiles.x1; tx++) {
        jobs.push(loadTile(`/ui/basemap/${z}/${tx}/${ty}?year=${year}`).then((im) => { if (im) g.drawImage(im, (tx - tiles.x0) * 256, (ty - tiles.y0) * 256); }));
      }
      await Promise.all(jobs);
      const arc = g.getImageData(0, 0, grid.w, grid.h).data;
      const n = grid.w * grid.h;
      const ar = new Float32Array(n), ag = new Float32Array(n), ab = new Float32Array(n);
      for (let i = 0; i < n; i++) { const ok = arc[4 * i + 3] > 127; ar[i] = ok ? arc[4 * i] : NaN; ag[i] = ok ? arc[4 * i + 1] : NaN; ab[i] = ok ? arc[4 * i + 2] : NaN; }
      // file side: nearest-neighbour onto the same grid, shown with the same stretch as the preview above
      const shown = preview.mode === 'rgb' ? preview.rgb : [preview.band, preview.band, preview.band];
      const wb = warpToGrid(shown.map((i) => px.bands[i]) as Band[], px.width, px.height, px.scale, proj, grid, h.nodata);
      const pr = renderPreview(up, preview);
      const ranges = (preview.mode === 'rgb' ? pr!.ranges : [pr!.ranges[0], pr!.ranges[0], pr!.ranges[0]]) as [[number, number], [number, number], [number, number]];
      const fileRGBA = compositeRGBA(wb as unknown as [Band, Band, Band], ranges, null);
      const diff = visualDiff(lumaOf(ar, ag, ab), lumaOf(wb[0], wb[1], wb[2]));    // archive = A, file = B  ->  file minus archive
      if (my !== stamp.current) return;
      const arcRGBA = new Uint8ClampedArray(n * 4);
      for (let i = 0; i < n; i++) { arcRGBA[4 * i] = ar[i]; arcRGBA[4 * i + 1] = ag[i]; arcRGBA[4 * i + 2] = ab[i]; arcRGBA[4 * i + 3] = Number.isFinite(ar[i]) ? 255 : 0; }
      let fileValid = 0, archValid = 0;
      for (let i = 0; i < n; i++) { if (Number.isFinite(wb[0][i])) fileValid++; if (Number.isFinite(ar[i])) archValid++; }
      setRes({ images: { file: fileRGBA, archive: arcRGBA }, year, used, z, grid, diff, res_m: grid.res * Math.cos((((ll[1] + ll[3]) / 2) * Math.PI) / 180), archiveShare: archValid / n, fileShare: fileValid / n });
    } catch (e) { if (my === stamp.current) setErr(e instanceof Error ? e.message : String(e)); }
    finally { if (my === stamp.current) setBusy(false); }
  };

  const none = cov && (!cov.available || acq.length === 0);
  const sel = acq.find((a) => a.date === pick);
  const usedDates = res?.used.dates ?? [];
  return (
    <section className="rcompare" data-testid="raster-compare" aria-label="Indicative comparison with the archive">
      <div className="rc-banner" data-testid="compare-banner" role="note">
        <b>Indicative visual difference — not pipeline output.</b> An ad-hoc upload and the archive differ in coordinate system, pixel size, radiometric scaling and
        registration, and the upload has not been aligned, quality-checked or scored in any way. What follows is a visual aid: it carries no score and is not a finding.
      </div>
      {covErr ? <div className="err">Archive coverage could not be read: {covErr}</div>
        : !cov ? <div className="dim" style={{ fontSize: 12 }} role="status">Checking which archive imagery covers this footprint…</div>
          : none ? (
            <div className="empty" data-testid="compare-none" style={{ textAlign: 'left', padding: '8px 2px' }}>
              <b>The archive has no imagery over this footprint,</b> so there is nothing to compare this file with. (Searched the footprint
              lon {ll[0].toFixed(4)}…{ll[2].toFixed(4)}, lat {ll[1].toFixed(4)}…{ll[3].toFixed(4)}.)
            </div>
          ) : (
            <div className="col" style={{ gap: 10 }}>
              <div className="row rc-pick" style={{ alignItems: 'flex-end', flexWrap: 'wrap', gap: 10 }}>
                <label className="field">Archive acquisition
                  <select className="input" value={pick} data-testid="compare-acq-select" onChange={(e) => setPick(e.target.value)} aria-label="Archive acquisition to compare with">
                    {acq.map((a) => <option key={a.observation_id} value={a.date}>{a.date} · {a.platform} · {fmtPct(a.mean_cloud, 0)} cloud</option>)}
                  </select>
                </label>
                <button className="btn primary" onClick={() => void run()} disabled={busy || !px || !pick} data-testid="compare-run">{busy ? 'Comparing…' : 'Show comparison'}</button>
                <span className="dim" style={{ fontSize: 11.5, maxWidth: 560 }}>
                  {nearest?.basis === 'nearest'
                    ? <>Nearest in time to this file’s date ({up.acquired.date}, {up.acquired.source}) is the archive’s {nearest.pick.date}; the archive covers {fmtPct(cov.fraction, 0)} of the footprint.</>
                    : <>This file carries no acquisition date, so the nearest acquisition cannot be determined; the archive’s default (lowest-cloud) one is pre-selected. Enter a date on the card above, or choose another here. The archive covers {fmtPct(cov.fraction, 0)} of the footprint.</>}
                </span>
              </div>
              {err && <div className="err" role="alert" data-testid="compare-error">{err}</div>}
              {res && (
                <div className="col" style={{ gap: 10 }}>
                  <div className="rc-grid" data-testid="compare-grid" data-z={res.z} data-n={res.diff.n}>
                    <figure><canvas ref={cUp} data-testid="compare-canvas-file" aria-label="Your file on the common grid" /><figcaption>Your file{up.acquired.date ? ` · ${up.acquired.date}` : ''}</figcaption></figure>
                    <figure><canvas ref={cAr} data-testid="compare-canvas-archive" aria-label="Archive imagery on the common grid" /><figcaption data-testid="compare-acq">Archive · acquisition used: {usedDates.length ? usedDates.join(', ') : DASH}</figcaption></figure>
                    <figure><canvas ref={cDf} data-testid="compare-canvas-diff" aria-label="Indicative difference" /><figcaption>Indicative difference (file − archive)</figcaption></figure>
                  </div>
                  <div className="rc-legend" aria-hidden="true"><span>file darker</span><i /><span>about equal</span><i className="r" /><span>file brighter</span></div>
                  <p className="rc-text" data-testid="compare-numbers">
                    Over the {res.diff.n.toLocaleString('en-US')} pixels both images cover ({fmtPct(res.fileShare, 0)} of the grid has file data, {fmtPct(res.archiveShare, 0)} has archive imagery):
                    mean brightness difference <b className="mono">{res.diff.meanAbs.toFixed(3)}</b> on a 0–1 scale, <b className="mono">{fmtPct(res.diff.shareAbove, 1)}</b> of pixels differ
                    by more than {res.diff.threshold}, brightness correlation <b className="mono">{Number.isFinite(res.diff.correlation) ? res.diff.correlation.toFixed(2) : DASH}</b>.
                  </p>
                  <p className="rc-text" data-testid="compare-method">
                    <b>Method.</b> Both images were placed on one Web-Mercator grid (zoom {res.z}, about {res.res_m.toFixed(1)} m per pixel). Your file was resampled nearest-neighbour from its own
                    grid (EPSG:{h.crs.epsg}, {h.resolution ? `${+h.resolution[0].toPrecision(4)} ${h.crs.kind === 'geographic' ? '°' : 'm'} pixels` : 'pixel size unknown'}) — values are never blended.
                    The archive side is its 8-bit true-colour rendering for {res.year}, already display-stretched, served by this console from imagery in the archive.
                    Each side is converted to brightness (Rec. 601 weights) and scaled to its own 2nd–98th percentile over the pixels both cover before subtracting, so a difference in overall
                    brightness or contrast between the two products does not count; the result is file − archive, red where the file is brighter and blue where it is darker.
                  </p>
                  {sel && !usedDates.includes(sel.date) && (
                    <div className="warnbox" role="note">You selected the {sel.date} acquisition, but the archive serves one acquisition per granule per year (its lowest-cloud one), which for {res.year} is {usedDates.join(', ') || 'not available'}. That is the date used here.</div>
                  )}
                </div>
              )}
              <ul className="rc-caveats" data-testid="compare-caveats">
                <li><b>CRS and pixel size differ:</b> the upload’s grid and the archive’s are resampled to a third grid; fine detail is not comparable.</li>
                <li><b>Radiometric scaling differs:</b> the archive image is an 8-bit display rendering; your values are the file’s own. Only relative brightness is compared.</li>
                <li><b>Not co-registered:</b> a shift of a pixel or two reads as a difference at every edge.</li>
                <li><b>Date, season, clouds and sun angle</b> are not matched; differences may be nothing more than that.</li>
              </ul>
            </div>
          )}
    </section>
  );
}
