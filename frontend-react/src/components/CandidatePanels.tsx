import { useEffect, useMemo, useState } from 'react';
import { api, candidateImage } from '@/api/client';
import type { CandidateDetail, Decision, Timeline as TL } from '@/api/types';
import { useApi } from '@/hooks/useApi';
import { DASH, GATE_LABEL, bandOf, downloadJSON, fmtHa, fmtInt, fmtLonLat, fmtPct, isNum, typeColor } from '@/fmt';
import { href } from '@/router';
import { useStore } from '@/state/store';
import { CompareSlider } from './CompareSlider';
import { GeoMap } from './GeoMap';
import { Panel } from './Panel';
import { Timeline } from './Timeline';
import { ConfidenceRing, DeltaBar, ErrorNote, Loading } from './Widgets';

export interface CandidateBundle {
  detail: CandidateDetail | null;
  timeline: TL | null;
  error: string | null;
  loading: boolean;
  reload: () => void;
}

/** Detail + timeline for one candidate, both from the live API. */
export function useCandidate(id: string | null): CandidateBundle {
  const d = useApi(id ? (s) => api.candidate(id, s) : null, [id]);
  const t = useApi(id ? (s) => api.timeline(id, s) : null, [id]);
  return { detail: d.data, timeline: t.data, error: d.error || t.error, loading: d.loading || t.loading, reload: d.reload };
}

/** Active (after) date + before date, constrained so before < after, always one of the catalog's real dates. */
export function useDatePair(tl: TL | null) {
  const [active, setActive] = useState<string | null>(null);
  const [before, setBefore] = useState<string | null>(null);
  useEffect(() => {
    if (!tl) return;
    setActive(tl.first_detected?.date ?? tl.dates[tl.dates.length - 1] ?? null);
    setBefore(null);
  }, [tl]);
  const dates = tl?.dates ?? [];
  const idx = active ? dates.indexOf(active) : -1;
  const earlier = idx > 0 ? dates.slice(0, idx) : [];
  const effBefore = before && earlier.includes(before) ? before : earlier[0] ?? null;
  return { active, setActive, before: effBefore, setBefore, earlier };
}

const PERSIST_TEXT: Record<string, string> = {
  persistent: 'Appeared early and is still present at the last acquisition.',
  progressive: 'Changing in every interval — consistent with ongoing works.',
  recent: 'Only visible in the most recent interval.',
  transient: 'Appeared, then reverted — treat with caution.',
  inconsistent: 'Intervals contradict each other — low trust.',
  single_pair: 'Only two observations — persistence cannot be assessed.',
  none: 'No change flagged across the acquisition dates.',
};

export function TemporalPanel({ tl, active, onPick, error }: { tl: TL | null; active: string | null; onPick: (d: string) => void; error?: string | null }) {
  return (
    <Panel title="Temporal evidence" tier="surfaced">
      {error ? <ErrorNote error={error} /> : !tl || !active ? <Loading rows={3} /> : (
        <div className="col" style={{ gap: 6 }}>
          <div className="row" style={{ alignItems: 'center', flexWrap: 'wrap', gap: 22 }}>
            <div title={tl.definition}>
              <div className="dim" style={{ fontSize: 10, letterSpacing: '.1em', textTransform: 'uppercase' }}>Persistence</div>
              <div className="mono" style={{ fontSize: 26, fontWeight: 600 }}>
                {tl.first_detected ? <>{tl.n_supporting} <span className="dim" style={{ fontSize: 15 }}>of {tl.n_total}</span></> : DASH}
              </div>
              <div className="dim" style={{ fontSize: 10.5 }}>observations support the change</div>
            </div>
            <div style={{ minWidth: 220, flex: 1 }}>
              <div className="row" style={{ gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
                <span className={`chip ${['persistent', 'progressive'].includes(tl.persistence) ? 'green' : ['transient', 'inconsistent'].includes(tl.persistence) ? 'red' : 'amber'}`}>{tl.persistence}</span>
                {isNum(tl.persistence_confidence) && <span className="chip">persistence confidence {tl.persistence_confidence.toFixed(2)}</span>}
              </div>
              <div style={{ marginTop: 5, color: 'var(--ink-2)' }}>{PERSIST_TEXT[tl.persistence] ?? ''}</div>
              {tl.first_detected && (
                <div className="dim" style={{ fontSize: 11.5, marginTop: 3 }}>
                  First detected <b className="mono" style={{ color: 'var(--amber)' }}>{tl.first_detected.date}</b> — change occurred between {tl.first_detected.bracket[0]} and {tl.first_detected.bracket[1]}.
                </div>
              )}
            </div>
          </div>
          <Timeline tl={tl} active={active} onPick={onPick} />
          {tl.first_detected?.caveat && <div className="faint" style={{ fontSize: 10.5 }}>{tl.first_detected.caveat}</div>}
        </div>
      )}
    </Panel>
  );
}

export function ComparePanel({ id, pair, tl }: { id: string; pair: ReturnType<typeof useDatePair>; tl: TL | null }) {
  const [mask, setMask] = useState(false);
  const { active, before, earlier, setBefore } = pair;
  const ready = !!(tl && active && before);
  return (
    <Panel title="Before / after" tier="live"
      actions={<button className={`btn sm ${mask ? 'on' : ''}`} onClick={() => setMask((m) => !m)} disabled={!ready} aria-pressed={mask} title="Overlay the detected change mask on the after image">Change mask</button>}>
      {!tl || !active ? <Loading rows={4} /> : !before ? (
        <div className="empty">{active} is the baseline acquisition — pick a later date on the timeline to compare against it.</div>
      ) : (
        <div className="col" style={{ gap: 8 }}>
          <CompareSlider
            before={candidateImage(id, before.slice(0, 4), 'rgb')}
            after={candidateImage(id, active.slice(0, 4), mask ? 'overlay' : 'rgb')}
            beforeLabel={before} afterLabel={active + (mask ? ' · mask' : '')} />
          <div className="row" style={{ alignItems: 'center', gap: 10 }}>
            <label className="field" style={{ flex: 1 }}>Before
              <select className="input" value={before} onChange={(e) => setBefore(e.target.value)}>
                {earlier.map((d) => <option key={d} value={d}>{d}</option>)}
              </select>
            </label>
            <span className="dim">→</span>
            <div className="field" style={{ flex: 1 }}>After (timeline)<div className="input mono">{active}</div></div>
          </div>
        </div>
      )}
    </Panel>
  );
}

export function DetailsPanel({ d, error }: { d: CandidateDetail | null; error?: string | null }) {
  const { label } = useStore();
  return (
    <Panel title="Change details" tier="live">
      {error ? <ErrorNote error={error} /> : !d ? <Loading rows={6} /> : (
        <div className="col" style={{ gap: 10 }}>
          <div className="row" style={{ flexWrap: 'wrap', gap: 6 }}>
            <span className="chip" style={{ color: typeColor(d.change_type), borderColor: typeColor(d.change_type) }}>{label(d.change_type)}</span>
            <span className="chip">{d.change_type}</span>
            {d.restricted_zone && <span className={`chip ${d.restricted_zone.alert_level === 'critical' ? 'red' : 'amber'}`} title="Cross-referenced against the demo restricted-zone boxes">⚠ {d.restricted_zone.name}</span>}
          </div>
          <dl className="kv">
            <dt>Candidate</dt><dd>{d.candidate_id}</dd>
            <dt>Area</dt><dd>{fmtHa(d.area_m2)} · {fmtInt(d.area_m2)} m²</dd>
            <dt>Centroid</dt><dd>{fmtLonLat(d.centroid_lonlat)}</dd>
            <dt>Earliest bracket</dt><dd>{d.earliest_supported?.join(' → ') ?? DASH}</dd>
            <dt>Rule</dt><dd>{d.classification?.rule ?? DASH}</dd>
            <dt>Model probability</dt><dd>{fmtPct(d.mean_model_prob)}</dd>
            <dt>Queue score</dt><dd>{isNum(d.queue_score) ? d.queue_score.toFixed(3) : DASH}</dd>
          </dl>
          <div>
            <div className="dim" style={{ fontSize: 10, letterSpacing: '.1em', textTransform: 'uppercase', marginBottom: 5 }} title="Index change minus the scene-wide seasonal change, so a drought-to-green season does not read as change">Spectral type · anomaly vs. season</div>
            <div className="col" style={{ gap: 6 }}>
              <DeltaBar label="NDVI" value={d.classification?.evidence?.ndvi_anomaly} hint="vegetation index anomaly" />
              <DeltaBar label="NDBI" value={d.classification?.evidence?.ndbi_anomaly} hint="built-up index anomaly" />
              <DeltaBar label="NDWI" value={d.classification?.evidence?.ndwi_anomaly} hint="water index anomaly" />
            </div>
            <div className="dim" style={{ fontSize: 11, marginTop: 6 }}>{d.classification?.detail}</div>
          </div>
          {d.terrain?.plain_language && <div className="dim" style={{ fontSize: 11.5 }}><b style={{ color: 'var(--ink-2)' }}>Terrain: </b>{d.terrain.plain_language}</div>}
        </div>
      )}
    </Panel>
  );
}

export function ConfidencePanel({ d, error }: { d: CandidateDetail | null; error?: string | null }) {
  const band = bandOf(d?.confidence);
  const sarOk = !!(d && (d.sar as { available?: boolean }).available);
  return (
    <Panel title="Confidence" tier="live">
      {error ? <ErrorNote error={error} /> : !d ? <Loading rows={6} /> : (
        <div className="col" style={{ gap: 10 }}>
          <div className="row" style={{ alignItems: 'center', gap: 16 }}>
            <ConfidenceRing value={d.confidence} color={band.color} />
            <div style={{ minWidth: 0 }}>
              <div style={{ color: band.color, fontWeight: 700, fontSize: 15 }}>{band.label}</div>
              <div className="dim" style={{ fontSize: 11 }}>significance {isNum(d.significance) ? d.significance.toFixed(2) : DASH}</div>
              <div className="dim" style={{ fontSize: 11 }}>combined down-weight {isNum(d.suppression?.combined_downweight) ? d.suppression.combined_downweight.toFixed(2) : DASH}</div>
            </div>
          </div>
          <ul className="mono" style={{ margin: 0, padding: 0, listStyle: 'none', fontSize: 10.5, color: 'var(--ink-3)', display: 'grid', gap: 2 }}>
            {d.confidence_breakdown?.map((l, i) => <li key={i} style={{ color: l.startsWith('=>') ? 'var(--ink)' : undefined }}>{l}</li>)}
          </ul>
          <div>
            <div className="dim" style={{ fontSize: 10, letterSpacing: '.1em', textTransform: 'uppercase', marginBottom: 3 }}>Evidence gates</div>
            {d.suppression?.trace?.map((t) => (
              <div key={t.rule} className={`gate ${t.verdict === 'pass' ? 'pass' : 'fail'}`} title={t.detail}>
                <i>{t.verdict === 'pass' ? '✓' : '✕'}</i>{GATE_LABEL[t.rule] ?? t.rule}
                <span className="faint mono" style={{ marginLeft: 'auto', fontSize: 10 }}>w {t.weight.toFixed(2)}</span>
              </div>
            ))}
          </div>
          <div>
            <div className="dim" style={{ fontSize: 10, letterSpacing: '.1em', textTransform: 'uppercase', marginBottom: 4 }}>SAR corroboration</div>
            {sarOk ? (
              <div className="row" style={{ gap: 6, flexWrap: 'wrap' }}>
                <span className="chip cyan">Sentinel-1 · {String((d.sar as { verdict?: string }).verdict ?? DASH)}</span>
                <span className="chip">VV median {isNum((d.sar as { vv_median_db?: number }).vv_median_db) ? `${(d.sar as { vv_median_db: number }).vv_median_db.toFixed(1)} dB` : DASH}</span>
                <span className="chip">factor {isNum((d.sar as { factor?: number }).factor) ? (d.sar as { factor: number }).factor.toFixed(2) : DASH}</span>
              </div>
            ) : (
              <div className="row" style={{ gap: 6, alignItems: 'center' }}>
                <span className="chip amber">Not corroborated</span>
                <span className="faint" style={{ fontSize: 11 }}>no Sentinel-1 evidence for this candidate — optical only</span>
              </div>
            )}
          </div>
        </div>
      )}
    </Panel>
  );
}

function when(iso: string) { try { return new Date(iso).toLocaleString('en-GB', { timeZone: 'Asia/Kolkata', hour12: false }) + ' IST'; } catch { return iso; } }

export function DecisionPanel({ d, onChanged }: { d: CandidateDetail | null; onChanged: () => void }) {
  const { analyst, setAnalyst, isBookmarked, toggleBookmark } = useStore();
  const [note, setNote] = useState('');
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);
  useEffect(() => { setNote(''); setMsg(null); }, [d?.candidate_id]);

  const decide = async (decision: 'confirm' | 'reject' | 'reopen') => {
    if (!d) return;
    setBusy(true); setMsg(null);
    try {
      const row = await api.decide(d.candidate_id, decision, note, analyst);
      setMsg({ ok: true, text: `Recorded ${row.decision} as ${row.decision_id} (append-only audit row)` });
      setNote(''); onChanged();
    } catch (e) { setMsg({ ok: false, text: e instanceof Error ? e.message : String(e) }); }
    finally { setBusy(false); }
  };
  const exportOne = async () => {
    if (!d) return;
    try {
      const r = await api.exportGeoJSON([d.candidate_id]);
      downloadJSON(`geoseek_${d.candidate_id}.geojson`, r.geojson);
      setMsg({ ok: true, text: `Exported ${r.count} feature with provenance` });
    } catch (e) { setMsg({ ok: false, text: e instanceof Error ? e.message : String(e) }); }
  };
  const trail: Decision[] = d ? [...d.decisions].reverse() : [];
  const eff = d?.effective_decision;

  return (
    <Panel title="Review & audit trail" tier="live">
      {!d ? <Loading rows={5} /> : (
        <div className="col" style={{ gap: 10 }}>
          <div className="row" style={{ alignItems: 'center', gap: 8 }}>
            <span className="dim">Current verdict</span>
            <span className={`chip ${eff === 'confirm' ? 'green' : eff === 'reject' ? 'red' : 'amber'}`}>{eff === 'confirm' ? 'confirmed' : eff === 'reject' ? 'rejected' : 'pending'}</span>
            <span style={{ flex: 1 }} />
            <button className={`btn sm ${isBookmarked(d.candidate_id) ? 'on' : ''}`} aria-pressed={isBookmarked(d.candidate_id)}
              onClick={() => toggleBookmark({ id: d.candidate_id, label: `${d.change_type} · ${d.candidate_id}`, lonlat: d.centroid_lonlat })}
              title="Bookmark this site for the briefing tour">{isBookmarked(d.candidate_id) ? '★ Bookmarked' : '☆ Bookmark'}</button>
          </div>
          <div className="row" style={{ gap: 8 }}>
            <label className="field" style={{ width: 130 }}>Analyst
              <input className="input" value={analyst} onChange={(e) => setAnalyst(e.target.value)} maxLength={40} />
            </label>
            <label className="field" style={{ flex: 1 }}>Note (optional)
              <input className="input" value={note} onChange={(e) => setNote(e.target.value)} placeholder="Why?" maxLength={300} />
            </label>
          </div>
          <div className="row" style={{ gap: 8, flexWrap: 'wrap' }}>
            <button className="btn ok" disabled={busy} onClick={() => decide('confirm')}>✓ Confirm</button>
            <button className="btn no" disabled={busy} onClick={() => decide('reject')}>✕ Reject</button>
            <button className="btn" disabled={busy || eff === 'undecided'} onClick={() => decide('reopen')} title="Appends a reversal row; the earlier rows are never edited or deleted">↺ Reopen</button>
            <span style={{ flex: 1 }} />
            <button className="btn" onClick={exportOne} title="Download this candidate as GeoJSON with full provenance">⇩ GeoJSON</button>
            <a className="btn" href={href('fingerprints', d.candidate_id)}>Find similar →</a>
          </div>
          {msg && <div style={{ fontSize: 11.5, color: msg.ok ? 'var(--green)' : 'var(--red)' }} role="status">{msg.text}</div>}
          <div>
            <div className="dim" style={{ fontSize: 10, letterSpacing: '.1em', textTransform: 'uppercase', marginBottom: 4 }}>Audit trail · append-only · {trail.length} row{trail.length === 1 ? '' : 's'}</div>
            {trail.length === 0 ? <div className="faint" style={{ fontSize: 11.5 }}>No decisions recorded for this candidate.</div> : (
              <ul style={{ margin: 0, padding: 0, listStyle: 'none', display: 'grid', gap: 5, maxHeight: 150, overflow: 'auto' }}>
                {trail.map((t) => (
                  <li key={t.decision_id} style={{ display: 'grid', gridTemplateColumns: 'auto 1fr', gap: '2px 8px', fontSize: 11, borderLeft: `2px solid ${t.decision === 'confirm' ? 'var(--green)' : t.decision === 'reject' ? 'var(--red)' : 'var(--amber)'}`, paddingLeft: 8 }}>
                    <b>{t.decision}</b><span className="dim mono">{t.analyst || 'unknown'} · {when(t.created_at)}</span>
                    {t.analyst_note && <span style={{ gridColumn: '1 / 3', color: 'var(--ink-2)' }}>{t.analyst_note}</span>}
                    <span className="faint mono" style={{ gridColumn: '1 / 3', fontSize: 9.5 }}>{t.decision_id} · conf {fmtPct(t.confidence_at_decision, 0)} · {t.model_version}</span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
      )}
    </Panel>
  );
}

export function LocationPanel({ d }: { d: CandidateDetail | null }) {
  const polys = useMemo(() => (d ? [{ id: d.candidate_id, geometry: d.geometry, color: typeColor(d.change_type) }] : []), [d]);
  const fit = useMemo(() => {
    if (!d) return null;
    const ring = d.geometry.coordinates[0];
    const xs = ring.map((p) => p[0]), ys = ring.map((p) => p[1]);
    const padX = Math.max(0.01, (Math.max(...xs) - Math.min(...xs)) * 2), padY = Math.max(0.01, (Math.max(...ys) - Math.min(...ys)) * 2);
    return [Math.min(...xs) - padX, Math.min(...ys) - padY, Math.max(...xs) + padX, Math.max(...ys) + padY] as [number, number, number, number];
  }, [d]);
  return (
    <Panel title="Location" tier="live" flush>
      {!d ? <Loading rows={3} /> : <GeoMap ariaLabel="Candidate footprint" polygons={polys} fit={fit} height={230} />}
    </Panel>
  );
}
