import { useEffect, useMemo, useState } from 'react';
import { api } from '@/api/client';
import type { BBox } from '@/api/types';
import { ComparePanel, ConfidencePanel, DecisionPanel, DetailsPanel, LocationPanel, TemporalPanel, useCandidate, useDatePair } from '@/components/CandidatePanels';
import { Panel } from '@/components/Panel';
import { SpectralEvidence } from '@/components/SpectralEvidence';
import { PipelineTracePanel, WhyPanel } from '@/components/WhyPanel';
import { ErrorNote, Loading } from '@/components/Widgets';
import { DASH, bandOf, downloadJSON, fmtHa, fmtInt, fmtPct, regionLabel, typeColor } from '@/fmt';
import { useApi } from '@/hooks/useApi';
import { go } from '@/router';
import { useStore } from '@/state/store';

const PERSISTENCE = ['persistent', 'progressive', 'recent', 'transient', 'inconsistent', 'none'];
const PAGE = 40;

export function Changes({ id }: { id: string | null }) {
  const { selectedId, select, summary, label, takeChangesFilter } = useStore();
  // a filter handed over by the Temporal screen (applied once, here, then forgotten)
  const [opened] = useState(() => takeChangesFilter());
  const [firstDetected, setFirstDetected] = useState(opened?.firstDetected ?? '');
  const [sensor, setSensor] = useState(opened?.sensor ?? '');
  const [originLabel, setOriginLabel] = useState(opened?.label ?? '');
  const [changeType, setChangeType] = useState(opened?.changeType ?? '');
  const [minConf, setMinConf] = useState(0);
  const [persistence, setPersistence] = useState(opened?.persistence ?? '');
  const [decision, setDecision] = useState('');
  const [sort, setSort] = useState('queue_score');
  const [region, setRegion] = useState(opened?.region ?? '');
  const [offset, setOffset] = useState(0);
  const [exportMsg, setExportMsg] = useState<string | null>(null);
  const [pixels, setPixels] = useState(false);

  const regions = useApi((s) => api.regions(s), []);
  const bbox: BBox | null = useMemo(() => regions.data?.regions.find((r) => r.name === region)?.bbox ?? null, [regions.data, region]);
  const query = { change_type: changeType || undefined, min_confidence: minConf || undefined, persistence: persistence || undefined, decision: decision || undefined, sort, bbox, limit: PAGE, offset,
    first_detected: firstDetected || undefined, sensor: sensor || undefined };
  const list = useApi((s) => api.candidates(query, s), [changeType, minConf, persistence, decision, sort, region, regions.data, offset, firstDetected, sensor]);
  useEffect(() => { setOffset(0); }, [changeType, minConf, persistence, decision, sort, region, firstDetected, sensor]);

  const active = id ?? selectedId;
  useEffect(() => { if (id) select(id); }, [id, select]);
  useEffect(() => { if (!active && list.data?.candidates[0]) select(list.data.candidates[0].candidate_id); }, [active, list.data, select]);

  const bundle = useCandidate(active);
  const pair = useDatePair(bundle.timeline);
  useEffect(() => { setPixels(false); }, [active]);

  const doExport = async () => {
    setExportMsg(null);
    try {
      const filters: Record<string, unknown> = { sort };
      if (changeType) filters.change_type = changeType;
      if (minConf) filters.min_confidence = minConf;
      if (persistence) filters.persistence = persistence;
      if (decision) filters.decision = decision;
      if (bbox) filters.bbox = bbox.join(',');
      if (sensor) filters.sensor = sensor;
      if (firstDetected) filters.first_detected = firstDetected;
      const r = await api.exportGeoJSON(undefined, filters);
      downloadJSON('geoseek_change_candidates.geojson', r.geojson);
      setExportMsg(`Exported ${fmtInt(r.count)} features with provenance`);
    } catch (e) { setExportMsg(e instanceof Error ? e.message : String(e)); }
  };

  const d = list.data;
  const types = Object.keys(summary?.change_type_labels ?? {});

  return (
    <>
    <div className="changes-grid">
      <div className="col queue-col">
        <Panel title="Review queue" grow stack actions={<button className="btn sm" onClick={doExport} title="GeoJSON for everything matching the filters">⇩ Export filtered</button>}>
          <div className="filters" style={{ marginBottom: 10 }}>
            <label className="field">Change type
              <select className="input" value={changeType} onChange={(e) => setChangeType(e.target.value)}>
                <option value="">All</option>{types.map((t) => <option key={t} value={t}>{label(t)}</option>)}
              </select>
            </label>
            <label className="field">Region
              <select className="input" value={region} onChange={(e) => setRegion(e.target.value)}>
                <option value="">Whole catalog</option>{regions.data?.regions.map((r) => <option key={r.name} value={r.name}>{regionLabel(r.name)}</option>)}
              </select>
            </label>
            <label className="field">Persistence
              <select className="input" value={persistence} onChange={(e) => setPersistence(e.target.value)}>
                <option value="">Any</option>{PERSISTENCE.map((p) => <option key={p} value={p}>{p}</option>)}
              </select>
            </label>
            <label className="field">Verdict
              <select className="input" value={decision} onChange={(e) => setDecision(e.target.value)}>
                <option value="">Any</option><option value="undecided">Pending</option><option value="confirm">Confirmed</option><option value="reject">Rejected</option>
              </select>
            </label>
            <label className="field">Sort
              <select className="input" value={sort} onChange={(e) => setSort(e.target.value)}>
                <option value="queue_score">Priority</option><option value="confidence">Confidence</option><option value="significance">Significance</option><option value="area_m2">Area</option>
              </select>
            </label>
            <label className="field" style={{ width: 150 }}>Min confidence {minConf ? fmtPct(minConf, 0) : 'any'}
              <input type="range" min={0} max={0.95} step={0.05} value={minConf} onChange={(e) => setMinConf(Number(e.target.value))} aria-label="Minimum confidence" />
            </label>
          </div>
          {originLabel && (
            <div className="row" style={{ alignItems: 'center', gap: 8, marginBottom: 8 }} data-testid="changes-origin">
              <span className="chip cyan" title="This queue was opened from the Temporal screen with these candidates selected">Opened from Temporal · {originLabel}</span>
              <button className="btn sm" onClick={() => { setFirstDetected(''); setSensor(''); setOriginLabel(''); }} aria-label="Clear the Temporal filter">Clear</button>
            </div>
          )}
          {exportMsg && <div style={{ fontSize: 11.5, color: 'var(--green)', marginBottom: 6 }} role="status">{exportMsg}</div>}
          {list.error ? <ErrorNote error={list.error} onRetry={list.reload} /> : !d ? <Loading rows={8} /> : (
            <>
              <div className="queue-scroll">
                <table className="tbl">
                  <thead><tr><th className="num">#</th><th>Type</th><th>Confidence</th><th>Persistence</th><th className="num">Area</th><th>Verdict</th></tr></thead>
                  <tbody>
                    {d.candidates.map((c) => {
                      const band = bandOf(c.confidence);
                      return (
                        <tr key={c.candidate_id} className={c.candidate_id === active ? 'sel' : ''} onClick={() => go('changes', c.candidate_id)}
                          tabIndex={0} onKeyDown={(e) => { if (e.key === 'Enter') go('changes', c.candidate_id); }} aria-selected={c.candidate_id === active}>
                          <td className="num">{c.rank}</td>
                          <td><span className="chip" style={{ color: typeColor(c.change_type), borderColor: typeColor(c.change_type) }}>{label(c.change_type)}</span>{c.restricted_zone && <span title={c.restricted_zone.name} style={{ marginLeft: 5, color: 'var(--red)' }}>⚠</span>}</td>
                          <td><div style={{ display: 'flex', alignItems: 'center', gap: 7 }}><div className="bar" style={{ width: 54 }}><i style={{ width: `${c.confidence * 100}%`, background: band.color }} /></div><span className="mono">{fmtPct(c.confidence, 0)}</span></div></td>
                          <td>{c.persistence}</td>
                          <td className="num">{fmtHa(c.area_m2)}</td>
                          <td><span className={`chip ${c.decision === 'confirm' ? 'green' : c.decision === 'reject' ? 'red' : 'amber'}`}>{c.decision === 'undecided' ? 'pending' : c.decision === 'confirm' ? 'confirmed' : 'rejected'}</span></td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
                {d.candidates.length === 0 && <div className="empty">No candidates match these filters.</div>}
              </div>
              <div className="row" style={{ alignItems: 'center', marginTop: 8 }}>
                <span className="dim mono" style={{ fontSize: 11 }}>{d.total ? `${offset + 1}–${offset + d.count} of ${fmtInt(d.total)}` : DASH}</span>
                <span style={{ flex: 1 }} />
                <button className="btn sm" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE))}>← Prev</button>
                <button className="btn sm" disabled={offset + PAGE >= d.total} onClick={() => setOffset(offset + PAGE)}>Next →</button>
              </div>
            </>
          )}
        </Panel>
      </div>

      <div className="col">
        {!active ? <Panel title="Candidate"><div className="empty">Select a candidate from the queue.</div></Panel> : (
          <>
            <TemporalPanel tl={bundle.timeline} pair={pair} error={bundle.error} />
            <div className="wb-grid">
              <ComparePanel id={active} pair={pair} tl={bundle.timeline} />
              <ConfidencePanel d={bundle.detail} error={bundle.error} />
              <DetailsPanel d={bundle.detail} error={bundle.error} />
              <DecisionPanel d={bundle.detail} tl={bundle.timeline} before={pair.before} after={pair.active} onChanged={() => { bundle.reload(); list.reload(); }} />
            </div>
            <LocationPanel d={bundle.detail} />
          </>
        )}
      </div>
    </div>
    {active && (
      <section className="col" id="why-section" aria-label="Candidate explanation" style={{ marginTop: 14 }}>
        <div className="why-grid">
          <WhyPanel ex={bundle.explain} error={bundle.explainError} pixelsOpen={pixels} onShowPixels={() => setPixels((p) => !p)} />
          <PipelineTracePanel ex={bundle.explain} error={bundle.explainError} />
        </div>
        {pixels && bundle.explain?.overlay && (
          <SpectralEvidence tileId={bundle.explain.overlay.tile_id} query={null} onClose={() => setPixels(false)}
            focus={bundle.explain.overlay.focus_indices} footprint={bundle.explain.overlay.geometry}
            title={`Index maps behind the explanation · ${bundle.explain.overlay.acquired_at ?? bundle.explain.overlay.tile_id}`} note={bundle.explain.overlay.note} />
        )}
      </section>
    )}
    </>
  );
}
