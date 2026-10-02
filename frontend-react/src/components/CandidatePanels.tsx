import { useEffect, useMemo, useState } from 'react';
import { api, candidateImage } from '@/api/client';
import type { BBox, CandidateDetail, CandidateExplain, Decision, Timeline as TL } from '@/api/types';
import { useApi } from '@/hooks/useApi';
import { usePlaybackController, usePlaybackSelect, usePlaybackState, type PlaybackController } from '@/hooks/usePlayback';
import { axisPositions, indexAt, layerOpacities } from '@/lib/timelapse';
import { DASH, bandOf, downloadJSON, fmtHa, fmtInt, fmtLonLat, fmtPct, isNum, typeColor } from '@/fmt';
import { go, href } from '@/router';
import { useStore } from '@/state/store';
import { CompareSlider } from './CompareSlider';
import { Dossier } from './Dossier';
import { GeoMap } from './GeoMap';
import { Panel } from './Panel';
import { ThreatRingsResults, useThreatRings } from './ThreatRings';
import { Timeline } from './Timeline';
import { ConfidenceRing, ErrorNote, Loading } from './Widgets';

export interface CandidateBundle {
  detail: CandidateDetail | null;
  timeline: TL | null;
  explain: CandidateExplain | null;
  explainError: string | null;
  error: string | null;
  loading: boolean;
  reload: () => void;
}

/** Detail + timeline for one candidate, both from the live API. */
export function useCandidate(id: string | null): CandidateBundle {
  const d = useApi(id ? (s) => api.candidate(id, s) : null, [id]);
  const t = useApi(id ? (s) => api.timeline(id, s) : null, [id]);
  const x = useApi(id ? (s) => api.explain(id, s) : null, [id]);
  return { detail: d.data, timeline: t.data, explain: x.data, explainError: x.error, error: d.error || t.error, loading: d.loading || t.loading, reload: d.reload };
}

/**
 * Active (after) date + before date, constrained so before < after, always one of the catalog's real dates. Also owns the
 * timeline playback controller for the candidate, which preloads every acquisition's imagery and then sweeps once.
 */
export function useDatePair(tl: TL | null) {
  const [active, setActive] = useState<string | null>(null);
  const [before, setBefore] = useState<string | null>(null);
  const pb = usePlaybackController();
  useEffect(() => {
    if (!tl) return;
    setActive(tl.first_detected?.date ?? tl.dates[tl.dates.length - 1] ?? null);
    setBefore(null);
    pb.load(tl.dates, tl.dates.map((d) => candidateImage(tl.candidate_id, d.slice(0, 4), 'rgb')));
  }, [tl, pb]);
  const dates = tl?.dates ?? [];
  const idx = active ? dates.indexOf(active) : -1;
  const earlier = idx > 0 ? dates.slice(0, idx) : [];
  const effBefore = before && earlier.includes(before) ? before : earlier[0] ?? null;
  return { active, setActive, before: effBefore, setBefore, earlier, pb };
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

function PlaybackControls({ tl, active, pb }: { tl: TL; active: string; pb: PlaybackController }) {
  const st = usePlaybackState(pb);
  const xs = useMemo(() => axisPositions(tl.dates), [tl.dates]);
  const restX = xs[Math.max(0, tl.dates.indexOf(active))] ?? 0;
  const p = st.p ?? restX;
  const i = Math.max(0, indexAt(xs, p));
  const between = st.p !== null && i < tl.dates.length - 1 && xs[i] < p - 1e-9;
  return (
    <div className="lapse-controls" role="group" aria-label="Timeline playback">
      <button className="btn sm" onClick={() => pb.toggle()} disabled={!st.ready} aria-label={st.playing ? 'Pause timeline' : 'Play timeline'}
        title={st.reduced ? 'Reduced motion is on: Play steps through the acquisitions one at a time, without a continuous sweep' : 'Sweep the playhead across the acquisition dates'}>
        {st.playing ? '❚❚ Pause' : '▶ Play'}
      </button>
      <button className="btn sm" onClick={() => pb.replay()} disabled={!st.ready} aria-label="Replay timeline">⟲ Replay</button>
      <input type="range" min={0} max={1000} value={Math.round(p * 1000)} disabled={!st.ready} aria-label="Scrub timeline" aria-valuetext={tl.dates[i]}
        onChange={(e) => pb.scrub(Number(e.target.value) / 1000)} />
      <span className="mono readout" aria-live="off">{tl.dates[i]}{between ? ` → ${tl.dates[i + 1]}` : ''}</span>
    </div>
  );
}

export function TemporalPanel({ tl, pair, error }: { tl: TL | null; pair: ReturnType<typeof useDatePair>; error?: string | null }) {
  const active = pair.active, onPick = pair.setActive;
  return (
    <Panel title="Temporal evidence" actions={tl && active ? <PlaybackControls tl={tl} active={active} pb={pair.pb} /> : undefined}>
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
          <Timeline tl={tl} active={active} onPick={onPick} pb={pair.pb} />
          {tl.first_detected?.caveat && <div className="faint" style={{ fontSize: 10.5 }}>{tl.first_detected.caveat}</div>}
        </div>
      )}
    </Panel>
  );
}

/** The image stack behind the timeline playhead: every acquisition's image, cross-faded by the playhead position. */
function TimelapseStage({ id, tl, pb }: { id: string; tl: TL; pb: PlaybackController }) {
  const st = usePlaybackState(pb);
  const xs = useMemo(() => axisPositions(tl.dates), [tl.dates]);
  const [failed, setFailed] = useState<Set<string>>(new Set());
  const p = st.p ?? 0;
  const ops = layerOpacities(xs, p, st.reduced);
  const i = Math.max(0, indexAt(xs, p));
  const between = i < tl.dates.length - 1 && xs[i] < p - 1e-9;
  return (
    <div className="lapse-stage" role="img" aria-label={`Timelapse of acquisitions, showing ${tl.dates[i]}`} data-p={p.toFixed(4)} data-date={tl.dates[i]}>
      {tl.dates.map((d, k) => (
        <img key={d} src={candidateImage(id, d.slice(0, 4), 'rgb')} alt={`Acquisition ${d}`} draggable={false} data-op={ops[k].toFixed(3)}
          style={{ opacity: failed.has(d) ? 0 : ops[k] }} onError={() => setFailed((f) => new Set(f).add(d))} />
      ))}
      <span className="tag l mono">{tl.dates[i]}{between ? ` → ${tl.dates[i + 1]}` : ''}</span>
      <span className="tag r mono">timelapse · real acquisition dates</span>
    </div>
  );
}

export function ComparePanel({ id, pair, tl }: { id: string; pair: ReturnType<typeof useDatePair>; tl: TL | null }) {
  const [mask, setMask] = useState(false);
  const { active, before, earlier, setBefore, pb } = pair;
  const ready = !!(tl && active && before);
  const lapse = usePlaybackSelect(pb, (s) => s.p !== null);
  return (
    <Panel title="Before / after"
      actions={lapse
        ? <button className="btn sm on" onClick={() => pb.rest()} title="Leave the timelapse and return to the before / after slider">⇆ Back to before / after</button>
        : <button className={`btn sm ${mask ? 'on' : ''}`} onClick={() => setMask((m) => !m)} disabled={!ready} aria-pressed={mask} title="Overlay the detected change mask on the after image">Change mask</button>}>
      {!tl || !active ? <Loading rows={4} /> : !before ? (
        <div className="empty">{active} is the baseline acquisition — pick a later date on the timeline to compare against it.</div>
      ) : (
        <div className="col" style={{ gap: 8 }}>
          {lapse && <TimelapseStage id={id} tl={tl} pb={pb} />}
          <div style={lapse ? { display: 'none' } : undefined} onPointerDownCapture={() => pb.rest()}>
            <CompareSlider
              before={candidateImage(id, before.slice(0, 4), 'rgb')}
              after={candidateImage(id, active.slice(0, 4), mask ? 'overlay' : 'rgb')}
              beforeLabel={before} afterLabel={active + (mask ? ' · mask' : '')} />
          </div>
          <div className="row" style={{ alignItems: 'center', gap: 10 }}>
            <label className="field" style={{ flex: 1 }}>Before
              <select className="input" value={before} onChange={(e) => { pb.rest(); setBefore(e.target.value); }}>
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

const jumpToWhy = () => document.getElementById('why-section')?.scrollIntoView({ behavior: 'smooth', block: 'start' });

export function DetailsPanel({ d, error }: { d: CandidateDetail | null; error?: string | null }) {
  const { label } = useStore();
  return (
    <Panel title="Change details">
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
          <div className="faint" style={{ fontSize: 11 }}>Spectral evidence, terrain and the typing thresholds: <button className="btn sm" onClick={jumpToWhy}>Why this was flagged ↓</button></div>
        </div>
      )}
    </Panel>
  );
}

export function ConfidencePanel({ d, error }: { d: CandidateDetail | null; error?: string | null }) {
  const band = bandOf(d?.confidence);
  const sarOk = !!(d && (d.sar as { available?: boolean }).available);
  return (
    <Panel title="Confidence">
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
          <div className="faint" style={{ fontSize: 11 }}>What each piece of evidence contributed, the gate trace and the raw numbers: <button className="btn sm" onClick={jumpToWhy}>Why this was flagged ↓</button></div>
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

export function DecisionPanel({ d, onChanged, tl, before, after }: { d: CandidateDetail | null; onChanged: () => void; tl: TL | null; before: string | null; after: string | null }) {
  const [dossier, setDossier] = useState(false);
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
    <Panel title="Review & audit trail">
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
            <button className="btn" onClick={() => setDossier(true)} title="Open a print-ready dossier for this candidate (save it as PDF from the print dialog)">⎙ Dossier</button>
            <a className="btn" href={href('fingerprints', d.candidate_id)}>Find similar →</a>
          </div>
          {msg && <div style={{ fontSize: 11.5, color: msg.ok ? 'var(--green)' : 'var(--red)' }} role="status">{msg.text}</div>}
          {dossier && <Dossier d={d} tl={tl} before={before} after={after} onClose={() => setDossier(false)} />}
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

const ZONE_COLOR: Record<string, string> = { critical: '#ff6b7a', warning: '#ff9f6b' };

/** Footprint of the selected candidate with its neighbours and the restricted zones. Right-click any candidate to draw
 *  concentric threat rings around it and list what each ring contains. */
export function LocationPanel({ d }: { d: CandidateDetail | null }) {
  const tr = useThreatRings();
  const zones = useApi((s) => api.restrictedZones(s), []);
  const id = d?.candidate_id ?? null;
  const around = useMemo<BBox | null>(() => (d ? [d.centroid_lonlat[0] - 0.09, d.centroid_lonlat[1] - 0.08, d.centroid_lonlat[0] + 0.09, d.centroid_lonlat[1] + 0.08] : null), [d]);
  const near = useApi(around ? (s) => api.candidates({ bbox: around, sort: 'queue_score', limit: 300 }, s) : null, [around?.join(',')]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { tr.setCenter(null); }, [id]); // eslint-disable-line react-hooks/exhaustive-deps

  const byId = useMemo(() => new Map((near.data?.candidates ?? []).map((c) => [c.candidate_id, c])), [near.data]);
  const polys = useMemo(() => {
    if (!d) return [];
    const others = (near.data?.candidates ?? []).filter((c) => c.candidate_id !== d.candidate_id)
      .map((c) => ({ id: c.candidate_id, geometry: c.geometry, color: typeColor(c.change_type), fill: 0.12, label: `${c.candidate_id} · click to open, right-click for threat rings` }));
    return [...others, { id: d.candidate_id, geometry: d.geometry, color: typeColor(d.change_type), fill: 0.38, selected: true, tag: `Selected · ${fmtHa(d.area_m2)}`,
      label: `${d.candidate_id} (selected) · right-click for threat rings` }];
  }, [d, near.data]);
  const regions = useMemo(() => (zones.data?.zones ?? []).map((z) => ({
    name: z.name, bbox: [z.min_lon, z.min_lat, z.max_lon, z.max_lat] as BBox, color: ZONE_COLOR[z.level] ?? '#ff9f6b', label: `${z.name} (${z.level} restricted zone)`,
  })), [zones.data]);
  const fit = useMemo(() => {
    if (!d) return null;
    const ring = d.geometry.coordinates[0];
    const xs = ring.map((p) => p[0]), ys = ring.map((p) => p[1]);
    const padX = Math.max(0.01, (Math.max(...xs) - Math.min(...xs)) * 2), padY = Math.max(0.01, (Math.max(...ys) - Math.min(...ys)) * 2);
    return [Math.min(...xs) - padX, Math.min(...ys) - padY, Math.max(...xs) + padX, Math.max(...ys) + padY] as BBox;
  }, [d]);

  const centreOn = (cid: string) => {
    const real = cid.startsWith('ring:') ? cid.slice(5) : cid;
    if (!real.startsWith('candidate:') && !byId.has(real) && real !== id) return;
    const key = real.replace(/^candidate:/, '');
    const c = key === d?.candidate_id ? d : byId.get(key);
    if (c) tr.setCenter({ id: `candidate:${key}`, lon: c.centroid_lonlat[0], lat: c.centroid_lonlat[1], label: key });
  };

  return (
    <Panel title="Location & threat rings">
      {!d ? <Loading rows={3} /> : (
        <div className="col" style={{ gap: 10 }}>
          <div style={{ borderRadius: 8, overflow: 'hidden' }}>
            <GeoMap ariaLabel="Candidate footprint and threat rings" polygons={polys} regions={regions} points={tr.overlay} circles={tr.circles}
              fit={tr.fit ?? fit} height={360} basemap={{}} onPolygonClick={(pid) => go('changes', pid)} onPolygonContext={centreOn} onPointContext={centreOn} />
          </div>
          <ThreatRingsResults tr={tr}
            action={(
              <button className="btn primary" onClick={() => centreOn(`candidate:${d.candidate_id}`)} data-testid="draw-rings"
                title="Concentric rings around the selected candidate, with everything inside each ring listed below">◎ Draw threat rings around this candidate</button>
            )}
            hint="You can also right-click any candidate footprint on the map to centre the rings on it; left-click opens that candidate." />
        </div>
      )}
    </Panel>
  );
}
