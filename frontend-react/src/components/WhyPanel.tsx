import { useEffect, useState } from 'react';
import type { CandidateExplain, ExplainTerm, TraceStage } from '@/api/types';
import { GATE_LABEL, bandOf, fmtPct, isNum } from '@/fmt';
import { href } from '@/router';
import { ErrorNote, Loading } from './Widgets';
import { Panel } from './Panel';
import { SpectralEvidence } from './SpectralEvidence';

const sgn = (x: number | null | undefined, d = 2) => (isNum(x) ? `${x >= 0 ? '+' : '−'}${Math.abs(x).toFixed(d)}` : '—');
const pts = (x: number | null | undefined) => (!isNum(x) || Math.abs(x) < 0.05 ? 'no reduction' : `${sgn(x, 1)} pts`);
const INDEX_NAME: Record<string, string> = { ndvi: 'NDVI', ndbi: 'NDBI', ndwi: 'NDWI' };

function EvidenceRow({ t, maxShare }: { t: ExplainTerm; maxShare: number }) {
  const weak = t.strength === 'weak';
  return (
    <li className={`ev-row ${weak ? 'weak' : ''}`} data-term={t.name} title={`${t.raw}\nweight ${t.weight} of ${(t.weight / t.weight_share).toFixed(1)}`}>
      <div className="nm"><b>{t.label}</b><span className="dim">{t.plain}</span></div>
      <div className="meter" aria-label={`weight share ${fmtPct(t.weight_share, 0)}`}>
        <div className="bar"><i style={{ width: `${(t.weight_share / maxShare) * 100}%`, background: 'var(--blue)' }} /></div>
        <span className="mono">{fmtPct(t.weight_share, 0)}</span>
      </div>
      <div className="meter" aria-label={`strength ${t.value.toFixed(2)}`}>
        <div className="bar"><i style={{ width: `${t.value * 100}%`, background: weak ? 'var(--amber)' : 'var(--green)' }} /></div>
        <span className="mono">{t.value.toFixed(2)}</span>
      </div>
      <div className="eff mono" data-effect>{pts(t.effect_points)}</div>
      <span className={`chip ${weak ? 'amber' : 'green'}`}>{weak ? 'weak' : 'supports'}</span>
    </li>
  );
}

/**
 * "Why this was flagged": the one place that explains a candidate. Everything is read from /ui/candidates/{id}/explain
 * (the pipeline's own stored values); the lead sentence is generated server-side from the winning rule. There is no
 * attention map: the embedding's patches are hundreds of metres wide and could not localise a structure.
 */
export function WhyPanel({ ex, error, onShowPixels, pixelsOpen }: { ex: CandidateExplain | null; error?: string | null; onShowPixels: () => void; pixelsOpen: boolean }) {
  const [raw, setRaw] = useState(false);
  const [allTests, setAllTests] = useState(false);
  return (
    <Panel title="Why this was flagged" className="why">
      {error ? <ErrorNote error={error} /> : !ex ? <Loading rows={7} /> : <WhyBody ex={ex} raw={raw} setRaw={setRaw} allTests={allTests} setAllTests={setAllTests} onShowPixels={onShowPixels} pixelsOpen={pixelsOpen} />}
    </Panel>
  );
}

function WhyBody({ ex, raw, setRaw, allTests, setAllTests, onShowPixels, pixelsOpen }: {
  ex: CandidateExplain; raw: boolean; setRaw: (b: boolean) => void; allTests: boolean; setAllTests: (b: boolean) => void; onShowPixels: () => void; pixelsOpen: boolean;
}) {
  const e = ex.evidence;
  const band = bandOf(e.stored_confidence);
  const decisive = ex.spectral.tests.filter((t) => t.decisive);
  const unclassified = ex.change_type === 'other';
  const absent = ex.weak_or_absent;
  const focusLabels = (ex.overlay?.focus_indices ?? []).map((i) => INDEX_NAME[i] ?? i.toUpperCase());
  return (
    <div className="col" style={{ gap: 14 }} data-candidate={ex.candidate_id}>
      <div>
        <p className="why-lead" data-testid="why-lead">{ex.lead.headline}</p>
        <p className="why-lead2" data-testid="why-persistence">{ex.lead.persistence}</p>
        <div className="row" style={{ gap: 6, flexWrap: 'wrap', marginTop: 6 }}>
          <span className="chip mono" title="The typing rule that fired">{ex.lead.rule}</span>
          <span className="chip" style={{ color: band.color, borderColor: band.color }}>{band.label} confidence · {fmtPct(e.stored_confidence, 0)}</span>
        </div>
      </div>

      <section aria-label="Evidence in order of influence">
        <h4 className="why-h">Evidence, largest effect on this candidate's confidence first</h4>
        <div className="ev-head" aria-hidden="true"><span>Evidence</span><span title="How much of the score this evidence is allowed to decide">Weight in score</span><span title="0 = absent, 1 = fully supportive">Strength (0–1)</span><span>Cost to confidence</span><span /></div>
        <ul className="ev-list">{e.terms.map((t) => <EvidenceRow key={t.name} t={t} maxShare={Math.max(...e.terms.map((x) => x.weight_share))} />)}</ul>
        <ul className="ev-list mult">
          {e.multipliers.map((m) => (
            <li key={m.name} className={`ev-row ${m.factor < 0.999 ? 'weak' : ''}`} data-mult={m.name}>
              <div className="nm"><b>{m.label}</b><span className="dim">{m.plain}</span></div>
              <div /><div className="mono" style={{ textAlign: 'right' }}>×{m.factor.toFixed(2)}</div>
              <div className="eff mono">{pts(m.effect_points)}</div>
              <span className={`chip ${m.name === 'sar' && m.factor === 1 ? 'amber' : m.factor < 0.999 ? 'amber' : ''}`}>{m.name === 'sar' && m.factor === 1 ? 'neutral' : m.factor < 0.999 ? 'applied' : m.factor > 1.001 ? 'raised' : 'none'}</span>
            </li>
          ))}
        </ul>
        <div className="why-sum mono" data-testid="why-sum">
          = {fmtPct(e.stored_confidence, 1)} confidence
          <span className={e.reproduces ? 'faint' : ''} style={e.reproduces ? undefined : { color: 'var(--red)' }}>
            {' '}· recomputed from the stored inputs: {fmtPct(e.recomputed_confidence, 1)} {e.reproduces ? '✓' : '— DOES NOT MATCH the stored value'}
          </span>
        </div>
        <div className="faint" style={{ fontSize: 10.5, marginTop: 3 }}>{e.method}</div>
      </section>

      <section aria-label="Spectral changes against thresholds">
        <h4 className="why-h">Spectral change against the typing thresholds</h4>
        <table className="tbl why-spec"><thead><tr><th>Index</th><th className="num">Change</th><th className="num">Scene season</th><th className="num">Vs. season</th><th>Test for “{ex.change_type}”</th></tr></thead>
          <tbody>
            {ex.spectral.anomalies.map((a) => {
              const tests = decisive.filter((t) => t.index === a.index);
              return (
                <tr key={a.index} data-index={a.index}>
                  <td>{a.label}</td><td className="num">{sgn(a.delta)}</td><td className="num">{sgn(a.seasonal)}</td><td className="num"><b>{sgn(a.anomaly)}</b></td>
                  <td>{unclassified ? <span className="faint">no rule matched</span> : tests.length ? tests.map((t, i) => (
                    <span key={i} className={`gate ${t.met ? 'pass' : 'fail'}`} style={{ display: 'flex' }} title={`${t.test} must be ${t.op} ${t.threshold}`}><i>{t.met ? '✓' : '✕'}</i>{t.text} <span className="faint mono">(is {t.test.includes('magnitude') ? Math.abs(t.value).toFixed(2) : sgn(t.value)})</span></span>
                  )) : <span className="faint">not used by this rule</span>}</td>
                </tr>
              );
            })}
            {decisive.some((t) => t.index === 'shape') && decisive.filter((t) => t.index === 'shape').map((t, i) => (
              <tr key={`s${i}`}><td>Shape</td><td className="num" colSpan={3}><span className="faint">footprint elongation</span></td>
                <td><span className={`gate ${t.met ? 'pass' : 'fail'}`} style={{ display: 'flex' }}><i>{t.met ? '✓' : '✕'}</i>{t.text} <span className="faint mono">(is {t.value.toFixed(2)})</span></span></td></tr>
            ))}
          </tbody>
        </table>
        <div className="faint" style={{ fontSize: 10.5, marginTop: 4 }}>{ex.spectral.note} {ex.spectral.rule_detail}</div>
        {unclassified && (
          <div style={{ marginTop: 4 }}>
            <button className="btn sm" aria-expanded={allTests} onClick={() => setAllTests(!allTests)}>{allTests ? 'Hide' : 'Show'} every typing test this candidate failed</button>
            {allTests && <ul className="why-tests mono">{ex.spectral.tests.map((t, i) => <li key={i} className={t.met ? 'met' : ''}>{t.met ? '✓' : '✕'} {t.type}: {t.text} (is {t.test.includes('magnitude') ? Math.abs(t.value).toFixed(2) : t.index === 'shape' ? t.value.toFixed(2) : sgn(t.value)})</li>)}</ul>}
          </div>
        )}
      </section>

      <section aria-label="Where evidence is weak or missing" className="why-gaps" data-testid="why-gaps">
        <h4 className="why-h">Where the evidence is thin or missing</h4>
        {absent.length === 0 ? <div className="dim" style={{ fontSize: 12 }}>Nothing weak: every evidence term is at or above 0.7 and Sentinel-1 corroborates.</div> : (
          <ul>{absent.map((w) => (
            <li key={w.key} className={w.key === 'sar' ? 'sar' : ''} data-gap={w.key}>
              <span className={`chip ${w.level === 'absent' ? 'amber' : 'red'}`}>{w.level === 'absent' ? 'missing' : 'weak'}</span>
              <span>{w.text}{w.key === 'sar' && ex.sar.run_note ? <span className="faint"> Run note: {ex.sar.run_note}.</span> : null}</span>
            </li>
          ))}</ul>
        )}
        {ex.terrain && <div className="dim" style={{ fontSize: 11.5, marginTop: 6 }}><b style={{ color: 'var(--ink-2)' }}>Terrain context: </b>{ex.terrain.plain_language} <span className="faint">(context only — not used in the confidence score)</span></div>}
      </section>

      <div className="row" style={{ gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
        <button className={`btn primary ${pixelsOpen ? 'on' : ''}`} onClick={onShowPixels} disabled={!ex.overlay} aria-pressed={pixelsOpen} data-testid="why-pixels"
          title={ex.overlay ? 'Open the NDVI / NDBI / NDWI maps of the after-date tile with this candidate outlined' : 'No staged tile covers this candidate'}>
          {pixelsOpen ? 'Hide the pixels' : `Show ${focusLabels.join(' · ') || 'the index maps'} on the pixels`}
        </button>
        <button className="btn" aria-expanded={raw} onClick={() => setRaw(!raw)} data-testid="why-raw-toggle">{raw ? 'Hide' : 'Show'} raw decomposition</button>
        <a className="btn" href={href('pipeline')}>How the gates work →</a>
      </div>

      {raw && (
        <section aria-label="Raw decomposition" data-testid="why-raw" className="why-raw">
          <table className="tbl"><thead><tr><th>Term</th><th className="num">Value</th><th className="num">Weight</th><th className="num">Share</th><th className="num">Factor</th><th className="num">Effect</th><th>Engine text</th></tr></thead>
            <tbody>{e.terms.map((t) => <tr key={t.name}><td>{t.name}</td><td className="num">{t.value.toFixed(4)}</td><td className="num">{t.weight}</td><td className="num">{t.weight_share.toFixed(3)}</td><td className="num">{t.factor.toFixed(4)}</td><td className="num">{isNum(t.effect_points) ? sgn(t.effect_points, 1) : '—'}</td><td style={{ whiteSpace: 'normal' }}>{t.raw}</td></tr>)}</tbody></table>
          <ul className="mono why-lines">{e.breakdown_lines.map((l, i) => <li key={i} style={{ color: l.startsWith('=>') ? 'var(--ink)' : undefined }}>{l}</li>)}</ul>
          <div className="faint mono" style={{ fontSize: 10.5 }}>significance {isNum(e.significance) ? e.significance.toFixed(4) : '—'} · queue score {isNum(e.queue_score) ? e.queue_score.toFixed(4) : '—'} · reproduction error {e.max_abs_error}</div>
        </section>
      )}
      <div className="faint" style={{ fontSize: 10.5 }}>{ex.scope}</div>
    </div>
  );
}

/** Moves the reader to the explanation and puts focus on it. Reports whether the target existed, so a caller can never fail silently. */
export function scrollToWhy(): boolean {
  const t = document.getElementById('why-section');
  if (!t) return false;
  t.scrollIntoView({ behavior: 'smooth', block: 'start' });
  t.focus({ preventScroll: true });
  return true;
}

/**
 * The full explanation for one candidate: the evidence breakdown, the pipeline trace and (on request) the index maps behind it.
 * One component for every screen that offers "Why this was flagged", so the control and its target cannot drift apart.
 */
export function ExplanationSection({ candidateId, ex, error }: { candidateId: string; ex: CandidateExplain | null; error?: string | null }) {
  const [pixels, setPixels] = useState(false);
  useEffect(() => { setPixels(false); }, [candidateId]);
  return (
    <section className="col" id="why-section" tabIndex={-1} aria-label="Candidate explanation" style={{ marginTop: 14 }}>
      <div className="why-grid">
        <WhyPanel ex={ex} error={error} pixelsOpen={pixels} onShowPixels={() => setPixels((p) => !p)} />
        <PipelineTracePanel ex={ex} error={error} />
      </div>
      {pixels && ex?.overlay && (
        <SpectralEvidence tileId={ex.overlay.tile_id} query={null} onClose={() => setPixels(false)}
          focus={ex.overlay.focus_indices} footprint={ex.overlay.geometry}
          title={`Index maps behind the explanation · ${ex.overlay.acquired_at ?? ex.overlay.tile_id}`} note={ex.overlay.note} />
      )}
    </section>
  );
}

const VERDICT: Record<string, { mark: string; cls: string; text: string }> = {
  passed: { mark: '✓', cls: 'pass', text: 'passed' },
  rejected: { mark: '✕', cls: 'fail', text: 'rejected' },
  downweighted: { mark: '↓', cls: 'warn', text: 'down-weighted' },
  typed: { mark: '●', cls: 'pass', text: 'typed' },
  unclassified: { mark: '○', cls: 'warn', text: 'left unclassified' },
  supported: { mark: '✓', cls: 'pass', text: 'supported over time' },
  demoted: { mark: '↓', cls: 'fail', text: 'demoted' },
  no_support: { mark: '○', cls: 'warn', text: 'no temporal support' },
  not_assessable: { mark: '○', cls: 'warn', text: 'not assessable' },
  unavailable: { mark: '○', cls: 'warn', text: 'no Sentinel-1 evidence' },
};
const STAGE_LABEL: Record<string, string> = { typing: 'Spectral typing', persistence: 'Temporal persistence', sar: 'SAR corroboration' };

function StageRow({ s }: { s: TraceStage }) {
  const v = VERDICT[s.verdict] ?? { mark: '•', cls: 'warn', text: s.verdict };
  const f = s.effect?.factor;
  const label = GATE_LABEL[s.id] ?? STAGE_LABEL[s.id] ?? s.id;
  const detail = s.id === 'typing' ? `${s.change_type}: ${s.detail}` : s.detail;
  return (
    <li className={`tr-row ${v.cls}`} data-stage={s.id} data-verdict={s.verdict}>
      <i aria-hidden="true">{v.mark}</i>
      <div className="tr-main">
        <div className="top">
          <b>{label}</b>
          <span className="vd">{v.text}{s.verdict === 'downweighted' && isNum(s.gate_weight) ? ` ×${s.gate_weight.toFixed(2)}` : ''}</span>
          <span className="grow" />
          <span className="eff mono" title={s.effect?.text}>{!s.effect || f == null ? 'no effect on confidence' : Math.abs(f - 1) < 5e-4 ? '×1.00' : `×${f.toFixed(2)}${isNum(s.effect.points) && Math.abs(s.effect.points) >= 0.05 ? ` · ${sgn(s.effect.points, 1)} pts` : ''}`}</span>
        </div>
        <div className="dim det">{detail}</div>
      </div>
    </li>
  );
}

/** Every stage this candidate went through, in the order the pipeline applied them, with its verdict and effect on confidence. */
export function PipelineTracePanel({ ex, error }: { ex: CandidateExplain | null; error?: string | null }) {
  const demoted = ex?.trace.find((s) => s.id === 'persistence' && (s.verdict === 'demoted' || s.verdict === 'no_support'));
  return (
    <Panel title="Pipeline trace · how it got through" className="trace" actions={<a className="btn sm" href={href('pipeline')}>Funnel →</a>}>
      {error ? <ErrorNote error={error} /> : !ex ? <Loading rows={7} /> : (
        <div className="col" style={{ gap: 10 }}>
          <div className="dim" style={{ fontSize: 11.5 }}>This candidate survived every rejecting gate (an item that fails one is dropped, not shown). Down-weighting and demotion lower its confidence instead.</div>
          <ol className="tr-list" data-testid="trace-list">{ex.trace.map((s) => <StageRow key={s.id} s={s} />)}</ol>
          {demoted && (
            <div className="warnbox" data-testid="trace-demoted" role="note">
              <b>Demoted, not dropped.</b> {demoted.detail}. It stays in the queue with a lower confidence ({fmtPct(ex.evidence.stored_confidence, 0)}); {isNum(demoted.effect?.points) ? `the repeat-observation evidence alone moves it ${sgn(demoted.effect!.points, 1)} pts.` : ''}
            </div>
          )}
          <div className="faint" style={{ fontSize: 10.5 }}>Queue score {isNum(ex.evidence.queue_score) ? ex.evidence.queue_score.toFixed(3) : '—'}{ex.ranking.queue_score ? ` = ${ex.ranking.queue_score}` : ''}. {ex.scope}</div>
        </div>
      )}
    </Panel>
  );
}
