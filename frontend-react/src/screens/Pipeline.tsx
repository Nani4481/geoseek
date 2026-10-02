import { useMemo, useState } from 'react';
import { api } from '@/api/client';
import type { FunnelGate, FunnelPair, SuppressionFunnel } from '@/api/types';
import { Panel } from '@/components/Panel';
import { ErrorNote, Loading, StatCard } from '@/components/Widgets';
import { DASH, GATE_LABEL, bandOf, fmtInt, fmtPct, isNum, typeColor } from '@/fmt';
import { useApi } from '@/hooks/useApi';
import { href } from '@/router';
import { useStore } from '@/state/store';

const sgn = (x: number | null | undefined, d = 3) => (isNum(x) ? `${x >= 0 ? '+' : '−'}${Math.abs(x).toFixed(d)}` : DASH);
const pairLabel = (p: FunnelPair) => `${p.earlier} → ${p.later}`;

/** One funnel row: the kept share as a solid bar, what this stage removed as a hatched block right after it. */
function FunnelRow({ rule, label, sub, base, input, removed, remaining, right, total }: {
  rule: string; label: string; sub?: string; base: number; input: number; removed: number; remaining: number; right: string; total?: boolean;
}) {
  const keep = base ? (remaining / base) * 100 : 0;
  const cut = base ? (removed / base) * 100 : 0;
  return (
    <div className={`fn-row ${total ? 'total' : ''}`} data-stage={rule} data-removed={removed} data-remaining={remaining} data-input={input}>
      <div className="nm"><span>{label}</span>{sub && <span className="dim">{sub}</span>}</div>
      <div className="fn-track" role="img" aria-label={`${label}: ${fmtInt(remaining)} remain of ${fmtInt(input)}`}>
        <div className="keep" style={{ width: `${keep}%` }} />
        {removed > 0 && <div className="cut" style={{ left: `${keep}%`, width: `${Math.max(cut, 0.4)}%` }} />}
      </div>
      <div className="cnt mono">{right}</div>
    </div>
  );
}

function Stack({ parts, label }: { parts: { key: string; n: number; color: string; title: string }[]; label: string }) {
  const tot = parts.reduce((a, p) => a + p.n, 0);
  return (
    <div role="img" aria-label={label} style={{ display: 'flex', height: 16, borderRadius: 4, overflow: 'hidden', background: 'rgba(255,255,255,.05)' }}>
      {parts.filter((p) => p.n > 0).map((p) => <div key={p.key} title={`${p.title}: ${fmtInt(p.n)}`} style={{ width: `${(p.n / Math.max(tot, 1)) * 100}%`, background: p.color }} />)}
    </div>
  );
}

function GateCards({ f, pair }: { f: SuppressionFunnel; pair: FunnelPair }) {
  const lab = f.labelled_benchmark;
  const byStage = new Map((lab.stages ?? []).map((s) => [s.stage, s]));
  const onSpan = pair.name === f.span_pair;
  return (
    <div className="gate-cards">
      {f.gates.map((g: FunnelGate) => {
        const st = pair.stages.find((s) => s.rule === g.rule);
        const ox = byStage.get(g.rule);
        return (
          <div className="gate-card" key={g.rule} data-gate={g.rule}>
            <h4>{GATE_LABEL[g.rule] ?? g.rule}<span className={`chip ${g.kind === 'reject' ? 'red' : 'amber'}`}>{g.kind === 'reject' ? 'rejects' : 'down-weights only'}</span></h4>
            <div className="what">{g.what}</div>
            <div className="mono faint" style={{ fontSize: 10.5 }}>{Object.entries(g.thresholds).map(([k, v]) => `${k} ${v}`).join(' · ')}</div>
            <div className="eff" data-this-run>
              <b style={{ color: 'var(--ink-2)' }}>This run ({pairLabel(pair)}): </b>
              {g.kind === 'reject'
                ? `removed ${fmtInt(st?.removed)} (${fmtPct(st?.share_of_raw, 1)} of raw components)`
                : `removes none${onSpan && isNum(f.post_gate.downweighted_by_gate[g.rule]) ? `; down-weighted ${fmtInt(f.post_gate.downweighted_by_gate[g.rule])} candidates` : ''}`}
            </div>
            <div className="eff" data-labelled>
              <b style={{ color: 'var(--ink-2)' }}>Labelled benchmark (OSCD): </b>
              {!lab.available ? <span>unavailable ({lab.reason})</span> : ox ? (
                <span>F1 {sgn(ox.delta_f1)} · precision {sgn(ox.delta_precision)} · recall {sgn(ox.delta_recall)}{isNum(ox.delta_f1) && ox.delta_f1 <= -0.005 && <span className="chip amber" style={{ marginLeft: 6 }}>costs F1 off-region</span>}</span>
              ) : <span>not ablated</span>}
            </div>
          </div>
        );
      })}
    </div>
  );
}

export function Pipeline() {
  const { label } = useStore();
  const fn = useApi((s) => api.funnel(s), []);
  const f = fn.data;
  const [sel, setSel] = useState<string | null>(null);
  const [zoom, setZoom] = useState(false);
  const pair = useMemo(() => f?.pairs.find((p) => p.name === (sel ?? f.span_pair)) ?? null, [f, sel]);

  if (fn.error) return <ErrorNote error={fn.error} onRetry={fn.reload} />;
  if (!f || !pair) return <Loading rows={8} />;

  const onSpan = pair.name === f.span_pair;
  const pg = f.post_gate;
  const base = zoom ? pair.stages[0].remaining : pair.raw;
  const gates = new Map(f.gates.map((g) => [g.rule, g]));
  const lab = f.labelled_benchmark;
  const types = onSpan ? pg.typing.by_type : (pair.class_distribution ?? {});
  const typeParts = Object.entries(types).map(([k, n]) => ({ key: k, n, color: typeColor(k), title: label(k) }));
  const pe = pg.persistence;
  const persParts = [
    { key: 'supported', n: pe.supported, color: 'var(--green)', title: 'Supported over time (persistent / progressive / recent)' },
    { key: 'contradicted', n: pe.contradicted, color: 'var(--red)', title: 'Contradicted (transient / inconsistent): demoted' },
    { key: 'none', n: pe.no_support, color: 'var(--amber)', title: 'No interval confirms it' },
    { key: 'other', n: pg.survivors - pe.supported - pe.contradicted - pe.no_support, color: 'var(--ink-4)', title: 'Not assessable' },
  ];
  const bandParts = (['High', 'Medium', 'Low'] as const).map((k) => ({ key: k, n: pg.confidence_bands[k] ?? 0, color: bandOf(k === 'High' ? 0.9 : k === 'Medium' ? 0.7 : 0.1).color, title: `${k} confidence` }));

  return (
    <div className="col" style={{ gap: 14 }} data-testid="pipeline-screen">
      <Panel title="False-alarm suppression · what was filtered out before you saw it"
        actions={(
          <div className="row" style={{ gap: 10, alignItems: 'center' }}>
            <label className="field" style={{ flexDirection: 'row', alignItems: 'center', gap: 8 }}>Pair
              <select className="input" value={pair.name} onChange={(e) => setSel(e.target.value)} aria-label="Acquisition pair" data-testid="pair-select">
                {f.pairs.map((p) => <option key={p.name} value={p.name}>{pairLabel(p)}{p.name === f.span_pair ? ' · span, feeds the review queue' : ''}</option>)}
              </select>
            </label>
            <button className={`btn sm ${zoom ? 'on' : ''}`} aria-pressed={zoom} onClick={() => setZoom(!zoom)} data-testid="funnel-zoom"
              title="Rescale to the components that passed the size floor, so the smaller gates become visible">Zoom past the size floor</button>
          </div>
        )}>
        <div className="col" style={{ gap: 14 }}>
          <div className="dim" style={{ fontSize: 12 }}>
            {f.aoi ? <><b style={{ color: 'var(--ink-2)' }}>{f.aoi}</b>. </> : null}
            The change model fires on {fmtInt(pair.raw)} connected regions for {pairLabel(pair)} (probability ≥ {f.model_threshold ?? DASH}). Five gates decide which of them become candidates; this is what each one removed.
          </div>
          <div className="pipe-stats">
            <StatCard label="Raw model components" value={fmtInt(pair.raw)} sub={`probability ≥ ${f.model_threshold ?? DASH}`} />
            <StatCard label="Rejected by the gates" value={fmtInt(pair.suppressed)} sub={`${fmtPct(pair.raw ? pair.suppressed / pair.raw : null, 1)} of raw`} />
            <StatCard label="Candidates after gates" value={fmtInt(pair.survivors)} sub={`${fmtPct(pair.raw ? pair.survivors / pair.raw : null, 1)} of raw${onSpan ? ' · the review queue' : ''}`} />
            <StatCard label="Demoted, not dropped" value={onSpan ? fmtInt(pe.contradicted) : DASH} sub={onSpan ? 'contradicted by later dates; kept with lower confidence' : 'persistence is scored on the span pair only'} />
          </div>

          <div data-testid="funnel">
            <FunnelRow rule="raw" label="Model components" sub="connected regions above threshold" base={base} input={pair.raw} removed={0} remaining={zoom ? pair.stages[0].remaining : pair.raw}
              right={zoom ? `${fmtInt(pair.stages[0].remaining)} reached the checks` : fmtInt(pair.raw)} />
            {pair.stages.map((s, i) => {
              const g = gates.get(s.rule);
              const dwOnly = g?.kind === 'downweight';
              const hide = zoom && i === 0;
              if (hide) return null;
              return (
                <FunnelRow key={s.rule} rule={s.rule} label={GATE_LABEL[s.rule] ?? s.rule} base={base} input={s.input} removed={s.removed} remaining={s.remaining}
                  sub={dwOnly ? 'down-weights confidence; cannot reject' : i === 0 ? 'area floor, applied first' : 'rejects'}
                  right={dwOnly ? 'removes none' : `−${fmtInt(s.removed)} · ${fmtPct(s.share_of_raw, 1)} of raw · ${fmtPct(s.share_of_input, 1)} of input`} />
              );
            })}
            <FunnelRow rule="survivors" label="Candidates after the gates" base={base} input={pair.raw} removed={0} remaining={pair.survivors} total
              right={`${fmtInt(pair.survivors)} of ${fmtInt(pair.raw)}`} />
          </div>
          {!pair.consistent && <div className="warnbox" role="alert">The per-gate counts for this pair do not add up to its survivors — treat the funnel as unreliable.</div>}
          <div className="faint" style={{ fontSize: 10.5, lineHeight: 1.5 }}>
            {f.attribution_note} {f.morphology_note}
            {isNum(pair.downweighted_survivors) && !onSpan ? ` Down-weighted survivors (all gates together): ${fmtInt(pair.downweighted_survivors)}.` : ''}
          </div>
          <div className="warnbox" data-testid="funnel-scope" role="note">{f.scope}</div>
        </div>
      </Panel>

      <div className="pipe-grid">
        <div className="col">
          <Panel title="After the gates · typed, scored, demoted">
            <div className="col" style={{ gap: 12 }} data-testid="post-gate">
              <div>
                <div className="why-h">Spectral typing · {fmtInt(Object.values(types).reduce((a, b) => a + b, 0))} candidates{onSpan ? '' : ` (${pairLabel(pair)})`}</div>
                <Stack parts={typeParts} label="Change types" />
                <div className="row" style={{ flexWrap: 'wrap', gap: 10, marginTop: 5, fontSize: 11.5 }}>
                  {typeParts.map((p) => <span key={p.key}><i style={{ display: 'inline-block', width: 9, height: 9, borderRadius: 2, background: p.color, marginRight: 5 }} />{p.title} <b className="mono">{fmtInt(p.n)}</b></span>)}
                </div>
                {onSpan && <div className="dim" style={{ fontSize: 11, marginTop: 3 }}>{fmtInt(pg.typing.unclassified)} matched no typing rule and stay 'other' — the model flagged them but the spectral evidence does not explain them.</div>}
              </div>
              {onSpan ? (
                <>
                  <div>
                    <div className="why-h">Temporal persistence · demoted, not dropped</div>
                    <Stack parts={persParts} label="Persistence" />
                    <table className="tbl" style={{ marginTop: 6 }}><thead><tr><th>Class</th><th className="num">Candidates</th><th className="num">Confidence ×</th></tr></thead>
                      <tbody>{Object.entries(pe.by_class).sort((a, b) => b[1] - a[1]).map(([k, n]) => (
                        <tr key={k} data-class={k}><td>{k}</td><td className="num">{fmtInt(n)}</td><td className="num">{isNum(pe.penalty[k]) ? `×${pe.penalty[k].toFixed(2)}` : '—'}</td></tr>
                      ))}</tbody></table>
                    <div className="dim" style={{ fontSize: 11, marginTop: 4 }}>
                      {fmtInt(pe.contradicted)} of {fmtInt(pg.survivors)} candidates are contradicted by later acquisitions (transient / inconsistent) and carry a confidence penalty; {fmtInt(pe.no_support)} have no interval that confirms them. All remain in the review queue.
                    </div>
                  </div>
                  <div>
                    <div className="why-h">SAR corroboration</div>
                    {pg.sar.available
                      ? <div style={{ fontSize: 12 }}>Moved up <b className="mono">{fmtInt(pg.sar.moved_up)}</b> · moved down <b className="mono">{fmtInt(pg.sar.moved_down)}</b> · neutral <b className="mono">{fmtInt(pg.sar.neutral)}</b> (weight only; never overrides the optical detection).</div>
                      : <div className="warnbox" data-testid="funnel-sar">No Sentinel-1 corroboration in this run{pg.sar.note ? ` (run note: ${pg.sar.note})` : ''}. Radar neither supported nor contradicted any candidate.</div>}
                  </div>
                  <div>
                    <div className="why-h">Resulting confidence</div>
                    <Stack parts={bandParts} label="Confidence bands" />
                    <div className="row" style={{ gap: 12, marginTop: 5, fontSize: 11.5 }}>{bandParts.map((p) => <span key={p.key}><i style={{ display: 'inline-block', width: 9, height: 9, borderRadius: 2, background: p.color, marginRight: 5 }} />{p.key} <b className="mono">{fmtInt(p.n)}</b></span>)}</div>
                  </div>
                  {!pg.matches_report && <div className="warnbox" role="alert">The ranked candidate list does not match the report's survivor count for this pair.</div>}
                </>
              ) : <div className="dim" style={{ fontSize: 12 }}>Persistence, SAR and the confidence breakdown are computed for the span pair ({f.span_pair}), which feeds the review queue. Switch the pair back to see them.</div>}
            </div>
          </Panel>

          <Panel title="Demoted candidates · a sample, with the reason">
            <div className="col" style={{ gap: 8 }}>
              <div className="dim" style={{ fontSize: 11.5 }}>{f.rejected.demoted_instead} Strongest model evidence first.</div>
              <table className="tbl" data-testid="demoted-sample"><thead><tr><th>Candidate</th><th>Type</th><th>Time</th><th className="num">Model p</th><th className="num">Confidence</th><th>Why</th></tr></thead>
                <tbody>{f.demoted_sample.map((d) => (
                  <tr key={d.candidate_id} onClick={() => { location.hash = href('changes', d.candidate_id); }} tabIndex={0} onKeyDown={(e) => { if (e.key === 'Enter') location.hash = href('changes', d.candidate_id); }}>
                    <td className="mono">{d.candidate_id}</td>
                    <td><span className="chip" style={{ color: typeColor(d.change_type), borderColor: typeColor(d.change_type) }}>{label(d.change_type ?? 'other')}</span></td>
                    <td><span className="chip red">{d.persistence}</span></td>
                    <td className="num">{isNum(d.mean_model_prob) ? d.mean_model_prob.toFixed(2) : DASH}</td>
                    <td className="num">{fmtPct(d.confidence, 0)} <span className="faint">(≈ {fmtPct(d.confidence_without_penalty, 0)} without ×{d.penalty.toFixed(2)})</span></td>
                    <td style={{ whiteSpace: 'normal' }}>{d.reason}</td>
                  </tr>
                ))}</tbody></table>
              {f.demoted_sample.length === 0 && <div className="empty">No candidate is demoted in this run.</div>}
            </div>
          </Panel>
        </div>

        <div className="col">
          <Panel title="Rejected components · can they be inspected?">
            <div className="col" style={{ gap: 8 }} data-testid="rejected">
              <div className="warnbox"><b>{f.rejected.retained ? 'Retained.' : 'Not retained.'}</b> {f.rejected.text}</div>
              <div className="dim" style={{ fontSize: 11.5 }}><b style={{ color: 'var(--ink-2)' }}>What it would take: </b>{f.rejected.to_emit}</div>
            </div>
          </Panel>
          <Panel title="Same gates on labelled data">
            {!lab.available ? <div className="empty">{lab.reason}</div> : (
              <div className="col" style={{ gap: 8 }} data-testid="labelled">
                <div className="dim" style={{ fontSize: 11.5 }}>{lab.dataset}; operating threshold {lab.threshold}. {lab.order_note}</div>
                <table className="tbl"><thead><tr><th>Stage</th><th className="num">Precision</th><th className="num">Recall</th><th className="num">F1</th><th className="num">ΔF1</th></tr></thead>
                  <tbody>{(lab.stages ?? []).map((s) => (
                    <tr key={s.stage}><td>{s.stage}</td><td className="num">{s.precision.toFixed(3)}</td><td className="num">{s.recall.toFixed(3)}</td><td className="num">{s.f1.toFixed(3)}</td>
                      <td className="num" style={{ color: isNum(s.delta_f1) && s.delta_f1 <= -0.005 ? 'var(--amber)' : undefined }}>{sgn(s.delta_f1)}</td></tr>
                  ))}</tbody></table>
                <div className="warnbox" role="note">{lab.caveat}</div>
                {lab.config_matches_current === false && <div className="warnbox" role="alert">This ablation was run with different gate thresholds than the pipeline now uses.</div>}
                <div className="faint" style={{ fontSize: 10.5 }}>{lab.source} · generated {lab.generated_at}</div>
              </div>
            )}
          </Panel>
        </div>
      </div>

      <Panel title="The five gates, and what each did on this pair">
        <div className="col" style={{ gap: 10 }}>
          <GateCards f={f} pair={pair} />
          <div className="faint" style={{ fontSize: 10.5 }}>Source: {f.source.report} (generated {f.source.report_generated_at}); survivors from {f.source.candidates}. Read-only: nothing on this screen is typed in.</div>
        </div>
      </Panel>
    </div>
  );
}
