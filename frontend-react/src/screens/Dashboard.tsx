import { useEffect, useMemo, useState } from 'react';
import { api } from '@/api/client';
import type { Candidate, ConsoleMetrics, LatencyProbe, Notification } from '@/api/types';
import { ComparePanel, ConfidencePanel, DetailsPanel, TemporalPanel, useCandidate, useDatePair } from '@/components/CandidatePanels';
import { Globe, type Focus, type Pin } from '@/components/Globe';
import { Panel } from '@/components/Panel';
import { ErrorNote, Loading, StatCard } from '@/components/Widgets';
import { DASH, bandOf, fmtFixed, fmtHa, fmtInt, fmtMs, fmtPct, regionLabel, typeColor } from '@/fmt';
import { go, href } from '@/router';
import { useStore } from '@/state/store';
import { useApi, type ApiState } from '@/hooks/useApi';

function StatRow({ m, lat }: { m: ApiState<ConsoleMetrics>; lat: ApiState<LatencyProbe> }) {
  const d = m.data;
  const cm = d?.change_model, det = d?.detector;
  const analysed = d?.findings_by_region.filter((r) => r.candidates > 0).length;
  return (
    <div className="stats" aria-label="Headline figures">
      <StatCard label="Tiles indexed" loading={m.loading} error={m.error} value={fmtInt(d?.counters.tiles_indexed)}
        sub={<><b>{fmtInt(d?.counters.vectors_searchable)}</b> searchable vectors</>} />
      <StatCard label="Scenes" loading={m.loading} error={m.error} value={fmtInt(d?.counters.scenes)}
        sub={d?.sensors.map((s) => `${s.n_scenes} ${s.sensor}`).join(' · ')} />
      <StatCard label="Regions" loading={m.loading} error={m.error} value={fmtInt(d?.counters.regions)}
        sub={<><b>{analysed ?? DASH}</b> analysed for change</>} />
      <StatCard label="Sensors" loading={m.loading} error={m.error} value={fmtInt(d?.counters.sensors)}
        sub={d?.sensors.map((s) => s.platform.replace(' Open Data Program', '')).join(' · ')} />
      <StatCard label="Search latency" loading={lat.loading} error={lat.error} value={fmtMs(lat.data?.median_ms)} unit="ms"
        title={lat.data?.source}
        sub={lat.data ? <>median of {lat.data.n_queries} probes · p95 <b>{fmtMs(lat.data.p95_ms)}</b> · measured now</> : undefined} />
      <StatCard label="Change detection F1" loading={m.loading} error={m.error || (cm && !cm.available ? 'model card not found' : null)}
        value={fmtPct(cm?.f1, 1)} title={cm ? `${cm.dataset}. ${cm.caveat}` : undefined}
        sub={cm?.available ? <>P <b>{fmtPct(cm.precision, 1)}</b> · R <b>{fmtPct(cm.recall, 1)}</b> · OSCD held-out</> : undefined} />
      <StatCard label="Object detection AP50" loading={m.loading} error={m.error || (det && !det.dota_val ? 'eval results not found' : null)}
        value={fmtFixed(det?.dota_val?.small_vehicle_ap50, 3)}
        title={det?.dota_val ? `${det.dota_val.dataset}\n${det.caveat}` : undefined}
        sub={det?.dota_val ? (
          <>small-vehicle · DOTA val<br /><b>{fmtFixed(det.dota_val.ground_vehicles_ap50, 3)}</b> ground vehicles<br />
            {det.xview_test && <>xView test: {fmtFixed(det.xview_test.small_vehicle_ap50, 3)} small · {fmtFixed(det.xview_test.large_vehicle_ap50, 3)} large</>}</>
        ) : undefined} />
      <StatCard label="Change candidates" loading={m.loading} error={m.error} value={fmtInt(d?.counters.change_candidates)}
        sub={<><b>{fmtInt(d?.counters.analyst_decisions)}</b> analyst decisions logged</>} />
    </div>
  );
}

function Breakdown({ m }: { m: ApiState<ConsoleMetrics> }) {
  const { label } = useStore();
  const d = m.data;
  const regions = useMemo(() => [...(d?.findings_by_region ?? [])].sort((a, b) => b.candidates - a.candidates || a.name.localeCompare(b.name)), [d]);
  const types = useMemo(() => Object.entries(d?.findings_by_type ?? {}).sort((a, b) => b[1] - a[1]), [d]);
  const maxR = Math.max(1, ...regions.map((r) => r.candidates));
  const total = types.reduce((s, [, n]) => s + n, 0) || 1;
  return (
    <div className="dash-pair">
      <Panel title="Findings by region" tier="live">
        {m.error ? <ErrorNote error={m.error} onRetry={m.reload} /> : !d ? <Loading rows={6} /> : (
          <>
            <div className="col" style={{ gap: 7 }}>
              {regions.map((r) => (
                <div key={r.name} className="brow" style={{ opacity: r.candidates ? 1 : 0.55 }}>
                  <span title={r.name}>{regionLabel(r.name)}</span>
                  <div className="bar"><i style={{ width: `${(r.candidates / maxR) * 100}%`, background: 'var(--amber)' }} /></div>
                  <span className="mono" style={{ textAlign: 'right' }}>{fmtInt(r.candidates)}</span>
                </div>
              ))}
            </div>
            <div className="faint" style={{ fontSize: 10.5, marginTop: 8 }}>
              Change pipeline run on: {d.change_pipeline_aoi ?? DASH}. Regions showing 0 have not been analysed for change.
            </div>
          </>
        )}
      </Panel>
      <Panel title="Findings by change type" tier="live">
        {m.error ? <ErrorNote error={m.error} onRetry={m.reload} /> : !d ? <Loading rows={6} /> : (
          <div className="col" style={{ gap: 7 }}>
            {types.map(([t, n]) => (
              <div key={t} className="brow">
                <span>{label(t)}</span>
                <div className="bar"><i style={{ width: `${(n / total) * 100}%`, background: typeColor(t) }} /></div>
                <span className="mono" style={{ textAlign: 'right' }}>{fmtInt(n)} <span className="faint">{fmtPct(n / total, 0)}</span></span>
              </div>
            ))}
          </div>
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
        <Panel title="Area of interest · globe" tier="live" flush className="hero">
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
        <Panel title="Alert feed" tier="live" className="feed"
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
      <TemporalPanel tl={bundle.timeline} active={pair.active} onPick={pair.setActive} error={bundle.error} />
      <div className="dash-triple">
        {selectedId ? <ComparePanel id={selectedId} pair={pair} tl={bundle.timeline} /> : <Panel title="Before / after" tier="live"><Loading rows={4} /></Panel>}
        <DetailsPanel d={bundle.detail} error={bundle.error} />
        <ConfidencePanel d={bundle.detail} error={bundle.error} />
      </div>
    </div>
  );
}
