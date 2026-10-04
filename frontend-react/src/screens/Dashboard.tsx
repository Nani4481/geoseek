import { useEffect, useMemo, useState } from 'react';
import { api } from '@/api/client';
import type { Candidate, ConsoleMetrics, LatencyProbe, Notification, RegionInfo } from '@/api/types';
import { ComparePanel, ConfidencePanel, DetailsPanel, TemporalPanel, useCandidate, useDatePair } from '@/components/CandidatePanels';
import { ExplanationSection, scrollToWhy } from '@/components/WhyPanel';
import { Globe, type Focus, type Pin } from '@/components/Globe';
import { Panel } from '@/components/Panel';
import { ErrorNote, Loading, StatCard } from '@/components/Widgets';
import { DASH, bandOf, fmtHa, fmtInt, fmtMs, fmtPct, regionLabel, typeColor } from '@/fmt';
import { go, href } from '@/router';
import { useStore } from '@/state/store';
import { useApi, type ApiState } from '@/hooks/useApi';

function StatRow({ m, lat }: { m: ApiState<ConsoleMetrics>; lat: ApiState<LatencyProbe> }) {
  const d = m.data;
  const analysed = d?.findings_by_region.filter(isAnalysed).length;
  return (
    <div className="stats" aria-label="Operational figures">
      <StatCard label="Tiles indexed" loading={m.loading} error={m.error} value={fmtInt(d?.counters.tiles_indexed)}
        sub={<><b>{fmtInt(d?.counters.vectors_searchable)}</b> searchable vectors</>} />
      <StatCard label="Scenes" loading={m.loading} error={m.error} value={fmtInt(d?.counters.scenes)}
        sub={d?.sensors.map((s) => `${s.n_scenes} ${s.sensor}`).join(' · ')} />
      <StatCard label="Regions" loading={m.loading} error={m.error} value={fmtInt(d?.counters.regions)}
        sub={<><b>{analysed ?? DASH}</b> analysed for change</>} />
      <StatCard label="Sensors" loading={m.loading} error={m.error} value={fmtInt(d?.counters.sensors)}
        sub={d?.sensors.map((s) => s.platform.replace(' Open Data Program', '')).join(' · ')} />
      <StatCard label="Change candidates" loading={m.loading} error={m.error} value={fmtInt(d?.counters.change_candidates)}
        sub={<>awaiting analyst review in <b>Changes</b></>} />
      <StatCard label="Analyst decisions" loading={m.loading} error={m.error} value={fmtInt(d?.counters.analyst_decisions)}
        sub="confirm / reject / reopen rows in the audit log" />
      <StatCard label="Search latency" loading={lat.loading} error={lat.error} value={fmtMs(lat.data?.median_ms)} unit="ms"
        title={lat.data?.source}
        sub={lat.data ? <>median of {lat.data.n_queries} probes · p95 <b>{fmtMs(lat.data.p95_ms)}</b> · measured now</> : undefined} />
    </div>
  );
}

/** A region counts as analysed when the change pipeline's own observation list touches it (catalog), not merely when it has findings. */
const isAnalysed = (r: RegionInfo) => (r.catalog ? r.catalog.analysed : r.candidates > 0);
const monthsOf = (a: string, b: string) => (Date.parse(b) - Date.parse(a)) / 86_400_000 / 30.4375;
const spanText = (a: string, b: string) => { const m = monthsOf(a, b); return m >= 18 ? `${(m / 12).toFixed(1)} years` : `${m.toFixed(1)} months`; };
const sensorName = (platform: string | null) => (platform ?? 'unknown').replace(' Open Data Program', '');
const plural = (n: number, w: string) => `${fmtInt(n)} ${w}${n === 1 ? '' : 's'}`;

/** The figures every region has, straight from the catalog: tiles, observations, acquisition dates, sensor. */
function CatalogFacts({ r }: { r: RegionInfo }) {
  const c = r.catalog;
  if (!c) return <span className="faint">catalog figures unavailable</span>;
  const p = c.sensors[0];
  return (
    <>
      <span data-field="tiles" title="tiles in the catalog"><b>{fmtInt(c.n_tiles)}</b> tiles</span>
      <span data-field="observations" title="catalogued observations (one per scene per date)">{plural(r.n_observations, 'observation')}</span>
      <span data-field="acquisitions" title={`distinct acquisition dates of ${sensorName(p.platform)}`}>{plural(p.n_acquisitions, 'acquisition')}</span>
      <span data-field="dates" className="mono" title="first and last acquisition of that sensor"><i data-first>{p.first_date}</i>{p.n_acquisitions > 1 ? <> → <i data-last>{p.last_date}</i></> : null}</span>
      <span data-field="sensor" title={c.sensors.map((x) => `${x.platform} ${x.sensor}: ${x.n_acquisitions} acquisitions`).join(' · ')}>{c.sensors.map((x) => sensorName(x.platform)).join(' + ')}</span>
    </>
  );
}

/** One sentence, computed from the catalog rows above, saying why change analysis has run where it has and not elsewhere. */
function whyLine(analysed: RegionInfo[], rest: RegionInfo[]): string | null {
  const a = analysed.filter((r) => r.catalog), o = rest.filter((r) => r.catalog);
  if (a.length === 0 || o.length === 0) return null;
  const have = a.map((r) => {
    const p = r.catalog!.sensors[0];
    return `${regionLabel(r.name)} has ${plural(p.n_acquisitions, `${sensorName(p.platform)} acquisition`)} (${p.first_date} → ${p.last_date}, ${spanText(p.first_date, p.last_date)}), ${p.n_coregistered} of ${p.n_observations} with a co-registration record`;
  }).join('; ');
  const ps = o.map((r) => r.catalog!.sensors[0]);
  const acq = ps.map((p) => p.n_acquisitions), days = ps.map((p) => Math.round((Date.parse(p.last_date) - Date.parse(p.first_date)) / 86_400_000));
  const rng = (v: number[]) => (Math.min(...v) === Math.max(...v) ? `${Math.min(...v)}` : `${Math.min(...v)}–${Math.max(...v)}`);
  return `The change pipeline compares acquisitions pixel by pixel, so it needs a multi-date stack co-registered onto one grid. ${have}. The other ${o.length} regions hold ${rng(acq)} acquisitions each, spanning ${rng(days)} days, with ${ps.reduce((n, p) => n + p.n_coregistered, 0)} of ${ps.reduce((n, p) => n + p.n_observations, 0)} observations carrying a co-registration record.`;
}

function RegionsPanel({ m }: { m: ApiState<ConsoleMetrics> }) {
  const { label } = useStore();
  const d = m.data;
  const regions = useMemo(() => d?.findings_by_region ?? [], [d]);
  const done = useMemo(() => regions.filter(isAnalysed).sort((x, y) => y.candidates - x.candidates), [regions]);
  const rest = useMemo(() => regions.filter((r) => !isAnalysed(r)).sort((x, y) => (y.catalog?.n_tiles ?? 0) - (x.catalog?.n_tiles ?? 0) || x.name.localeCompare(y.name)), [regions]);
  const maxC = Math.max(1, ...done.map((r) => r.candidates));
  const why = whyLine(done, rest);
  return (
    <Panel title="Findings by region" stack>
      {m.error ? <ErrorNote error={m.error} onRetry={m.reload} /> : !d ? <Loading rows={6} /> : (
        <div className="regpanel" data-testid="regions-panel">
          <section aria-label="Change analysis complete" data-group="analysed">
            <h4 className="reg-h">Change analysis complete <span className="chip green">{done.length}</span></h4>
            {done.map((r) => (
              <div key={r.name} className="reg-done" data-region={r.name}>
                <div className="brow">
                  <span title={r.name}>{regionLabel(r.name)}</span>
                  <div className="bar"><i style={{ width: `${(r.candidates / maxC) * 100}%`, background: 'var(--amber)' }} /></div>
                  <span className="mono" style={{ textAlign: 'right' }}><b data-field="candidates">{fmtInt(r.candidates)}</b> <span className="faint">candidates</span></span>
                </div>
                <div className="reg-types">
                  {Object.entries(r.by_type).sort((x, y) => y[1] - x[1]).map(([t, n]) => (
                    <span key={t} className="chip" data-type={t} style={{ color: typeColor(t), borderColor: typeColor(t) }}>{label(t)} <b data-field="type-count">{fmtInt(n)}</b></span>
                  ))}
                </div>
                <div className="reg-facts"><CatalogFacts r={r} /></div>
              </div>
            ))}
            {done.length === 0 && <div className="faint" style={{ fontSize: 11.5 }}>The change pipeline has not been run on any region.</div>}
          </section>
          <section aria-label="Indexed and searchable, change analysis not yet run" data-group="not-analysed">
            <h4 className="reg-h">Indexed and searchable · change analysis not yet run <span className="chip">{rest.length}</span></h4>
            <div className="reg-grid" role="table">
              <div className="reg-row head" role="row"><span>Region</span><span className="num">Tiles</span><span className="num">Obs.</span><span className="num">Acq.</span><span>Acquired</span><span>Sensor</span></div>
              {rest.map((r) => {
                const c = r.catalog, p = c?.sensors[0];
                return (
                  <div key={r.name} className="reg-row" role="row" data-region={r.name}>
                    <span title={r.name}>{regionLabel(r.name)}</span>
                    {c && p ? (
                      <>
                        <span className="num mono" data-field="tiles">{fmtInt(c.n_tiles)}</span>
                        <span className="num mono" data-field="observations">{fmtInt(r.n_observations)}</span>
                        <span className="num mono" data-field="acquisitions" title={`distinct acquisition dates of ${sensorName(p.platform)}`}>{fmtInt(p.n_acquisitions)}</span>
                        <span className="mono dim" data-field="dates"><i data-first>{p.first_date}</i>{p.n_acquisitions > 1 ? <> → <i data-last>{p.last_date}</i></> : null}</span>
                        <span data-field="sensor" title={c.sensors.map((x) => `${x.platform} ${x.sensor}: ${x.n_acquisitions} acquisitions`).join(' · ')}>{c.sensors.map((x) => sensorName(x.platform)).join(' + ')}</span>
                      </>
                    ) : <span className="faint" style={{ gridColumn: '2 / -1' }}>catalog figures unavailable</span>}
                  </div>
                );
              })}
            </div>
          </section>
          {why && <div className="reg-why" data-testid="regions-why">{why}</div>}
          <div className="faint" style={{ fontSize: 10.5 }} data-testid="regions-caption">
            Change pipeline run on: {d.change_pipeline_aoi ?? DASH}. A region with no finding count has not been analysed: that is “not run”, not “nothing found”.
          </div>
        </div>
      )}
    </Panel>
  );
}

function Breakdown({ m }: { m: ApiState<ConsoleMetrics> }) {
  const { label } = useStore();
  const d = m.data;
  const types = useMemo(() => Object.entries(d?.findings_by_type ?? {}).sort((a, b) => b[1] - a[1]), [d]);
  const total = types.reduce((s, [, n]) => s + n, 0) || 1;
  return (
    <div className="dash-pair">
      <RegionsPanel m={m} />
      <Panel title="Findings by change type" stack>
        {m.error ? <ErrorNote error={m.error} onRetry={m.reload} /> : !d ? <Loading rows={6} /> : (
          <>
            <div className="even-rows">
              {types.map(([t, n]) => (
                <div key={t} className="brow">
                  <span>{label(t)}</span>
                  <div className="bar"><i style={{ width: `${(n / total) * 100}%`, background: typeColor(t) }} /></div>
                  <span className="mono" style={{ textAlign: 'right' }}>{fmtInt(n)} <span className="faint">{fmtPct(n / total, 0)}</span></span>
                </div>
              ))}
            </div>
            <div className="faint" style={{ fontSize: 10.5, marginTop: 8 }}>
              Share of all {fmtInt(types.reduce((a, [, n]) => a + n, 0))} change candidates, by the class the change pipeline assigned.
            </div>
          </>
        )}
      </Panel>
    </div>
  );
}

interface Alert { key: string; candidateId: string; level: 'critical' | 'warning' | 'watch'; title: string; sub: string; meta: string; lonlat: [number, number] }

function buildAlerts(cands: Candidate[], notes: Notification[], label: (t: string | null) => string): Alert[] {
  const out: Alert[] = [];
  for (const n of notes) {
    const first = n.candidates[0];
    const c = first && cands.find((x) => x.candidate_id === first.candidate_id);
    if (!first) continue;
    out.push({ key: 'n' + (n.notification_id ?? n.watch_id + n.observation_date), candidateId: first.candidate_id, level: 'watch',
      title: `Watch area “${n.watch_name ?? n.watch_id}” matched ${n.candidates.length} candidate${n.candidates.length === 1 ? '' : 's'}`,
      sub: `${label(first.change_type)} · severity ${n.severity}`, meta: n.observation_date ?? '', lonlat: c?.centroid_lonlat ?? [0, 0] });
  }
  for (const c of cands) {
    if (!c.restricted_zone) continue;
    out.push({ key: c.candidate_id, candidateId: c.candidate_id, level: c.restricted_zone.alert_level === 'critical' ? 'critical' : 'warning',
      title: `${label(c.change_type)} inside ${c.restricted_zone.name}`,
      sub: `${bandOf(c.confidence).label} confidence ${fmtPct(c.confidence, 0)} · ${fmtHa(c.area_m2)} · ${c.persistence}`,
      meta: c.candidate_id, lonlat: c.centroid_lonlat });
  }
  return out;
}

export function Dashboard() {
  const { selectedId, select, summary, label } = useStore();
  const metrics = useApi((s) => api.metrics(s), []);
  const latency = useApi((s) => api.latency(s), []);
  const zones = useApi((s) => api.restrictedZones(s), []);
  const top = useApi((s) => api.candidates({ sort: 'queue_score', limit: 300 }, s), []);
  const notes = useApi((s) => api.notifications(s), []);
  const bundle = useCandidate(selectedId);
  const pair = useDatePair(bundle.timeline);
  const [focus, setFocus] = useState<Focus | null>(null);
  const [pinInfo, setPinInfo] = useState<string | null>(null);
  const [whyOpen, setWhyOpen] = useState(false);   // the explanation is collapsed here (the Changes screen always shows it)

  useEffect(() => { if (whyOpen) requestAnimationFrame(() => scrollToWhy()); }, [whyOpen]);
  useEffect(() => { if (!selectedId && top.data?.candidates[0]) select(top.data.candidates[0].candidate_id); }, [top.data, selectedId, select]);

  const alerts = useMemo(() => buildAlerts(top.data?.candidates ?? [], notes.data?.notifications ?? [], label), [top.data, notes.data, label]);
  const regions = metrics.data?.findings_by_region ?? [];

  const pins = useMemo<Pin[]>(() => {
    const out: Pin[] = regions.map((r) => ({
      id: 'r:' + r.name, lon: r.center[0], lat: r.center[1], label: `${regionLabel(r.name)}${r.candidates ? ` · ${r.candidates} findings` : ''}`,
      color: r.candidates ? '#f5a524' : '#4cc9f0', emphasis: r.candidates > 0, size: r.candidates ? 1.5 : 1,
    }));
    for (const z of zones.data?.zones ?? []) {
      out.push({ id: 'z:' + z.name, lon: (z.min_lon + z.max_lon) / 2, lat: (z.min_lat + z.max_lat) / 2, label: `${z.name} (${z.level})`, color: z.level === 'critical' ? '#ff6b7a' : '#ff9f6b', size: 0.8 });
    }
    return out;
  }, [regions, zones.data]);

  const sel = top.data?.candidates.find((c) => c.candidate_id === selectedId);
  const info = pinInfo?.startsWith('r:') ? regions.find((r) => 'r:' + r.name === pinInfo) : null;

  const pick = (a: Alert) => { select(a.candidateId); if (a.lonlat[0]) setFocus({ lon: a.lonlat[0], lat: a.lonlat[1], key: a.key, distance: 1.9 }); };

  return (
    <div className="col" style={{ gap: 14 }}>
      <StatRow m={metrics} lat={latency} />
      <div className="dash-hero">
        <Panel title="Area of interest · globe" flush className="hero">
          <div className="hero-body">
            <Globe pins={pins} focus={focus} selectedPin={null} onPinClick={(id) => { setPinInfo(id); }} />
            <div className="hero-legend legend-row">
              <span><i style={{ background: '#f5a524' }} />region with change findings</span>
              <span><i style={{ background: '#4cc9f0' }} />catalogued region</span>
              <span><i style={{ background: '#ff6b7a' }} />restricted zone</span>
              <span className="faint">drag to rotate · scroll to zoom · click a pin</span>
            </div>
            {pinInfo && (
              <div className="pin-card" role="dialog" aria-label="Pin details">
                <button className="btn sm" style={{ float: 'right' }} onClick={() => setPinInfo(null)} aria-label="Close">✕</button>
                {info ? (
                  <>
                    <b>{regionLabel(info.name)}</b>
                    <div className="dim mono" style={{ fontSize: 11 }}>{info.n_observations} observations · {info.candidates} change findings</div>
                    <div className="row" style={{ gap: 6, marginTop: 6 }}>
                      {info.candidates > 0 && <button className="btn sm primary" onClick={() => go('changes')}>Review findings</button>}
                      <button className="btn sm" onClick={() => go('search')}>Search archive</button>
                    </div>
                  </>
                ) : <b>{pinInfo.slice(2)}</b>}
              </div>
            )}
          </div>
        </Panel>
        <Panel title="Alert feed" className="feed"
          actions={<span className="chip red" title="Candidates whose centroid falls inside a restricted-zone box">{fmtInt(summary?.counters.restricted_zone_alerts)} in restricted zones</span>}>
          {top.error ? <ErrorNote error={top.error} onRetry={top.reload} /> : top.loading ? <Loading rows={6} /> : alerts.length === 0 ? <div className="empty">No alerts.</div> : (
            <ul className="feed-list">
              {alerts.slice(0, 40).map((a) => (
                <li key={a.key}>
                  <button className={`alert ${a.level} ${a.candidateId === selectedId ? 'sel' : ''}`} onClick={() => pick(a)}>
                    <i className="lvl" />
                    <span className="t">{a.title}</span>
                    <span className="s">{a.sub}</span>
                    <span className="m mono">{a.level.toUpperCase()} · {a.meta}</span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </Panel>
      </div>

      <Breakdown m={metrics} />

      <div className="sel-head">
        <span className="dim">Selected candidate</span>
        <b className="mono">{selectedId ?? DASH}</b>
        {sel && <span className="chip" style={{ color: typeColor(sel.change_type), borderColor: typeColor(sel.change_type) }}>{label(sel.change_type)}</span>}
        {selectedId && <a className="btn sm" href={href('changes', selectedId)}>Open in Changes →</a>}
      </div>
      <TemporalPanel tl={bundle.timeline} pair={pair} error={bundle.error} />
      <div className="dash-triple">
        {selectedId ? <ComparePanel id={selectedId} pair={pair} tl={bundle.timeline} /> : <Panel title="Before / after"><Loading rows={4} /></Panel>}
        <DetailsPanel d={bundle.detail} error={bundle.error} />
        <ConfidencePanel d={bundle.detail} error={bundle.error} whyOpen={whyOpen} onWhy={() => setWhyOpen((o) => !o)} />
      </div>
      {whyOpen && selectedId && <ExplanationSection candidateId={selectedId} ex={bundle.explain} error={bundle.explainError} />}
    </div>
  );
}
