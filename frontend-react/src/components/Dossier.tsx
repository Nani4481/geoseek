import { useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { api, candidateImage } from '@/api/client';
import type { CandidateDetail, DossierProvenance, Timeline as TL } from '@/api/types';
import { DASH, GATE_LABEL, bandOf, fmtHa, fmtInt, fmtLonLat, fmtPct, isNum, regionLabel } from '@/fmt';
import { useApi } from '@/hooks/useApi';
import { toMGRS, toUTM } from '@/lib/geo';
import { useStore } from '@/state/store';

const when = (iso: string) => { try { return new Date(iso).toLocaleString('en-GB', { timeZone: 'Asia/Kolkata', hour12: false }) + ' IST'; } catch { return iso; } };

function Chip({ src, caption, sub, onLoaded }: { src: string; caption: string; sub: string; onLoaded: () => void }) {
  const [bad, setBad] = useState(false);
  return (
    <figure className="dz-chip">
      {bad ? <div className="dz-noimg">imagery unavailable</div>
        : <img src={src} alt={caption} onLoad={onLoaded} onError={() => { setBad(true); onLoaded(); }} />}
      <figcaption><b>{caption}</b><span>{sub}</span></figcaption>
    </figure>
  );
}

/**
 * Print-ready tactical dossier for one candidate, assembled only from the candidate's own API record, the timeline, the
 * catalog's sensor provenance and the offline coordinate conversion. Export = the browser's print-to-PDF (the print
 * stylesheet shows just this sheet); no PDF service is involved and nothing is uploaded.
 *
 * Fields the catalog does not hold (sun elevation, off-nadir angle) are not rendered at all.
 */
export function Dossier({ d, tl, before, after, onClose }: { d: CandidateDetail; tl: TL | null; before: string | null; after: string | null; onClose: () => void }) {
  const { label } = useStore();
  const regions = useApi((s) => api.regions(s), []);
  const prov = useApi(before && after ? (s) => api.dossier(d.candidate_id, before, after, s) : null, [d.candidate_id, before, after]);
  const [loaded, setLoaded] = useState(0);
  const generated = useRef(new Date());
  const [lon, lat] = d.centroid_lonlat;
  const utm = useMemo(() => toUTM(lon, lat), [lon, lat]);
  const mgrs = useMemo(() => toMGRS(lon, lat, 5), [lon, lat]);
  const region = regions.data?.regions.find((r) => lon >= r.bbox[0] && lon <= r.bbox[2] && lat >= r.bbox[1] && lat <= r.bbox[3]);
  const band = bandOf(d.confidence);
  const trail = [...d.decisions].reverse();
  const nImg = before && after ? 3 : 0;
  const ready = loaded >= nImg && !prov.loading;

  useEffect(() => {
    const prev = document.title;
    document.title = `GeoSeek dossier ${d.candidate_id}`;       // becomes the suggested PDF file name
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', onKey);
    return () => { document.title = prev; window.removeEventListener('keydown', onKey); };
  }, [d.candidate_id, onClose]);

  const sensor = (o: DossierProvenance['observations'][number]) => (
    <div className="dz-sensor" key={o.role} data-role={o.role}>
      <h5>{o.role === 'before' ? 'Before' : 'After'} acquisition</h5>
      <dl>
        <dt>Platform</dt><dd>{o.platform ?? DASH}{o.sensor ? ` · ${o.sensor}` : ''}</dd>
        <dt>Acquired</dt><dd>{o.acquired_at ?? DASH}{o.acquired_at_precision ? ' (date only)' : ''}</dd>
        <dt>Cloud cover</dt><dd>{isNum(o.cloud_fraction) ? fmtPct(o.cloud_fraction, 1) + ' (nearest catalogued tile)' : DASH}</dd>
        <dt>Ground resolution</dt><dd>{isNum(o.native_gsd_m) ? `${o.native_gsd_m} m` : DASH}</dd>
        {o.processing_baseline && <><dt>Processing baseline</dt><dd>{o.processing_baseline}</dd></>}
        <dt>Scene</dt><dd>{o.scene_id ?? DASH}</dd>
        <dt>CRS</dt><dd>{o.crs ?? DASH}</dd>
        {isNum(o.sun_elevation_deg) && <><dt>Sun elevation</dt><dd>{o.sun_elevation_deg.toFixed(1)}°</dd></>}
        {isNum(o.sun_azimuth_deg) && <><dt>Sun azimuth</dt><dd>{o.sun_azimuth_deg.toFixed(1)}°</dd></>}
        {isNum(o.off_nadir_deg) && <><dt>Off-nadir angle</dt><dd>{o.off_nadir_deg.toFixed(1)}°</dd></>}
      </dl>
    </div>
  );

  return createPortal(
    <div className="dossier-overlay" role="dialog" aria-modal="true" aria-label={`Tactical dossier for ${d.candidate_id}`}>
      <div className="dz-toolbar">
        <b>Tactical dossier</b>
        <span className="dim mono">{d.candidate_id}</span>
        <span style={{ flex: 1 }} />
        {!ready && <span className="dim" style={{ fontSize: 11.5 }}>loading imagery and provenance…</span>}
        <button className="btn primary" onClick={() => window.print()} disabled={!ready} title="Opens the browser print dialog — choose “Save as PDF”">⎙ Print / save as PDF</button>
        <button className="btn" onClick={onClose}>Close ✕</button>
      </div>
      <article className="dossier-sheet" data-ready={ready ? '1' : '0'}>
        <header className="dz-head">
          <div><h1>TACTICAL CHANGE DOSSIER</h1><div className="dz-sub">GeoSeek · change candidate <span className="mono">{d.candidate_id}</span></div></div>
          <div className="dz-gen mono">Generated {when(generated.current.toISOString())}<br />from the local catalog (offline)</div>
        </header>

        <section className="dz-sec" data-sec="location">
          <h2>1 · Location</h2>
          <dl className="dz-kv">
            <dt>Latitude / longitude</dt><dd className="mono">{fmtLonLat(d.centroid_lonlat)} <span className="dim">({lat.toFixed(6)}, {lon.toFixed(6)})</span></dd>
            <dt>MGRS (1 m)</dt><dd className="mono" data-field="mgrs">{mgrs?.text ?? DASH}</dd>
            <dt>UTM</dt><dd className="mono" data-field="utm">{utm ? `zone ${utm.zone}${utm.band} (${utm.north ? 'N' : 'S'}) · E ${utm.easting.toFixed(0)} m · N ${utm.northing.toFixed(0)} m` : DASH}</dd>
            <dt>Region</dt><dd>{region ? regionLabel(region.name) : DASH}</dd>
            {d.restricted_zone && <><dt>Restricted zone</dt><dd className="dz-alert">{d.restricted_zone.name} · {d.restricted_zone.alert_level}</dd></>}
            <dt>Area</dt><dd className="mono">{fmtHa(d.area_m2)} · {fmtInt(d.area_m2)} m² · {fmtInt(d.area_px)} px</dd>
            {d.terrain && <><dt>Terrain</dt><dd>{d.terrain.plain_language}</dd></>}
          </dl>
        </section>

        <section className="dz-sec" data-sec="imagery">
          <h2>2 · Imagery</h2>
          {before && after ? (
            <div className="dz-chips">
              <Chip src={candidateImage(d.candidate_id, before.slice(0, 4), 'rgb', 2)} caption="BEFORE" sub={before} onLoaded={() => setLoaded((n) => n + 1)} />
              <Chip src={candidateImage(d.candidate_id, after.slice(0, 4), 'rgb', 2)} caption="AFTER" sub={after} onLoaded={() => setLoaded((n) => n + 1)} />
              <Chip src={candidateImage(d.candidate_id, after.slice(0, 4), 'overlay', 2)} caption="DIFFERENCE" sub={`detected change mask on ${after}`} onLoaded={() => setLoaded((n) => n + 1)} />
            </div>
          ) : <div className="dim">No before / after pair is selected (the chosen date is the baseline acquisition).</div>}
        </section>

        <section className="dz-sec" data-sec="assessment">
          <h2>3 · Assessment</h2>
          <dl className="dz-kv">
            <dt>Change type</dt><dd><b>{label(d.change_type)}</b> <span className="mono dim">({d.change_type})</span></dd>
            <dt>Classification rule</dt><dd>{d.classification?.rule ?? DASH}{d.classification?.detail ? ` — ${d.classification.detail}` : ''}</dd>
            <dt>Confidence</dt><dd className="mono"><b data-field="confidence">{fmtPct(d.confidence, 0)}</b> ({band.label}) · significance {isNum(d.significance) ? d.significance.toFixed(2) : DASH} · model probability {fmtPct(d.mean_model_prob)}</dd>
            <dt>Persistence</dt><dd>{tl?.persistence ?? d.persistence}{tl?.first_detected ? ` — ${tl.n_supporting} of ${tl.n_total} observations support the change` : ''}</dd>
            <dt>Acquisition dates</dt><dd className="mono">{tl ? tl.dates.join(' · ') : DASH}</dd>
            {tl?.first_detected && <><dt>First detected</dt><dd className="mono">{tl.first_detected.date} (change occurred between {tl.first_detected.bracket[0]} and {tl.first_detected.bracket[1]})</dd></>}
            <dt>SAR corroboration</dt><dd>{(d.sar as { available?: boolean }).available ? `Sentinel-1 · ${String((d.sar as { verdict?: string }).verdict ?? DASH)}` : 'not corroborated — optical evidence only'}</dd>
          </dl>
        </section>

        <section className="dz-sec" data-sec="gates">
          <h2>4 · Evidence gate trace</h2>
          <table className="dz-tbl">
            <thead><tr><th>Gate</th><th>Verdict</th><th>Weight</th><th>Detail</th></tr></thead>
            <tbody>
              {d.suppression?.trace?.map((t) => (
                <tr key={t.rule}><td>{GATE_LABEL[t.rule] ?? t.rule}</td><td className={t.verdict === 'pass' ? 'ok' : 'bad'}>{t.verdict}</td><td className="mono">{t.weight.toFixed(2)}</td><td>{t.detail}</td></tr>
              ))}
            </tbody>
          </table>
          <pre className="dz-pre mono">{d.confidence_breakdown?.join('\n')}</pre>
        </section>

        <section className="dz-sec" data-sec="sensor">
          <h2>5 · Sensor provenance</h2>
          {prov.error ? <div className="dz-alert">Provenance unavailable: {prov.error}</div> : !prov.data ? <div className="dim">loading…</div>
            : <div className="dz-sensors">{prov.data.observations.map(sensor)}</div>}
        </section>

        <section className="dz-sec" data-sec="decision">
          <h2>6 · Analyst decision & audit trail</h2>
          <div className="dz-verdict">Current verdict: <b data-field="verdict">{d.effective_decision === 'confirm' ? 'CONFIRMED' : d.effective_decision === 'reject' ? 'REJECTED' : 'PENDING REVIEW'}</b>
            <span className="dim"> · append-only log, {trail.length} row{trail.length === 1 ? '' : 's'}</span></div>
          {trail.length > 0 && (
            <table className="dz-tbl">
              <thead><tr><th>Time (IST)</th><th>Decision</th><th>Analyst</th><th>Note</th><th>Record</th></tr></thead>
              <tbody>
                {trail.map((t) => (
                  <tr key={t.decision_id}><td className="mono">{when(t.created_at)}</td><td><b>{t.decision}</b></td><td>{t.analyst || 'unknown'}</td><td>{t.analyst_note || DASH}</td>
                    <td className="mono">{t.decision_id} · conf {fmtPct(t.confidence_at_decision, 0)} · {t.model_version}</td></tr>
                ))}
              </tbody>
            </table>
          )}
        </section>

        <footer className="dz-foot">
          <div className="dz-sign"><div className="line" /><small>Analyst signature</small></div>
          <div className="dz-sign"><div className="line" /><small>Date / reviewing officer</small></div>
          <div className="dz-trace mono">
            model {d.provenance.model.name} @ {d.provenance.model.threshold} · weights {d.provenance.model.weights_sha256?.slice(0, 16)}… · code {d.provenance.code.git_commit || DASH} · pipeline {d.provenance.code.pipeline_version} · report {when(d.provenance.report_generated_at)}
          </div>
        </footer>
      </article>
    </div>,
    document.body,
  );
}
