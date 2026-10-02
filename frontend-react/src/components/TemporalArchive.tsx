import { useId, useMemo, useState } from 'react';
import { api } from '@/api/client';
import type { BBox, TemporalArchive as TemporalData, TemporalBucket } from '@/api/types';
import { Panel } from '@/components/Panel';
import { ErrorNote, Loading } from '@/components/Widgets';
import { DASH, fmtInt, fmtPct, regionLabel, typeColor } from '@/fmt';
import { useApi } from '@/hooks/useApi';
import { calendarAxis, niceMax, type CalendarAxis } from '@/lib/calendarAxis';
import { useStore } from '@/state/store';

// ---- geometry shared by the four charts: ONE real calendar axis (pixels proportional to days), observations drawn as marks, never as a continuous line ----
const W = 960, L = 58, R = 22, PLOT_H = 230;
const PERS_COLOR: Record<string, string> = { persistent: '#34d399', progressive: '#4cc9f0', recent: '#5b8cff', transient: '#ff6b7a', inconsistent: '#f5a524', none: '#8d9cc6' };
const PERS_ORDER = ['persistent', 'progressive', 'recent', 'transient', 'inconsistent'];
const km2 = (m2: number) => m2 / 1e6;
const fmtKm2 = (m2: number) => `${km2(m2) >= 10 ? km2(m2).toFixed(1) : km2(m2).toFixed(2)} km²`;

type OnOpen = (f: { firstDetected?: string; changeType?: string; persistence?: string; sensor?: string; region?: string; label: string }) => void;

function Rail({ ax, dates, y }: { ax: CalendarAxis; dates: string[]; y: number }) {
  return (
    <g className="tc-rail" aria-label="Acquisition dates">
      {ax.ticks.map((t) => <g key={t.label}><line x1={t.x} x2={t.x} y1={y - PLOT_H - 4} y2={y} className="tc-year" /><text x={t.x + 3} y={y - PLOT_H + 8} className="tc-yeartxt">{t.label}</text></g>)}
      <line x1={L} x2={W - R} y1={y} y2={y} className="tc-base" />
      {dates.map((d, i) => (
        <g key={d} data-testid="acq-point" data-date={d}>
          <circle cx={ax.x(d)} cy={y} r={5} className="tc-acq" />
          <text x={ax.x(d)} y={y + 20} textAnchor={i === 0 ? 'start' : i === dates.length - 1 ? 'end' : 'middle'} className="tc-datetxt">{d}</text>
        </g>
      ))}
      {dates.slice(1).map((d, i) => {
        const a = dates[i], days = Math.round((Date.parse(d) - Date.parse(a)) / 86_400_000);
        const room = ax.x(d) - ax.x(a);                                          // pixels between the two acquisitions
        return <text key={d} x={(ax.x(a) + ax.x(d)) / 2} y={y + 36} textAnchor="middle" className="tc-gaptxt">{room > 190 ? `${days} days with no observation` : `${days} d`}</text>;
      })}
    </g>
  );
}

function YAxis({ max, ticks, y, fmt }: { max: number; ticks: number[]; y: number; fmt: (v: number) => string }) {
  return (
    <g className="tc-yaxis">
      {ticks.map((t) => { const yy = y - (t / max) * PLOT_H; return <g key={t}><line x1={L} x2={W - R} y1={yy} y2={yy} className="tc-grid" /><text x={L - 8} y={yy + 4} textAnchor="end" className="tc-ytxt">{fmt(t)}</text></g>; })}
    </g>
  );
}

function Legend({ types, label }: { types: string[]; label: (t: string) => string }) {
  return <div className="tc-legend" aria-label="Change types">{types.map((t) => <span key={t}><i style={{ background: typeColor(t) }} />{label(t)}</span>)}</div>;
}

// ------------------------------------------------------------------------------------------------ chart 1: candidates per interval, by type
function CountsChart({ d, basis, onOpen, label, region }: { d: TemporalData; basis: 'stored' | 'pair_run'; onOpen: OnOpen; label: (t: string) => string; region: string }) {
  const hatch = useId().replace(/:/g, '');
  const ax = useMemo(() => calendarAxis(d.dates, L, W - R), [d.dates]);
  const base = PLOT_H + 20, H = base + 56;
  const rows = d.intervals.map((iv) => ({ iv, by: basis === 'stored' ? iv.stored.by_type : iv.pair_run?.by_type ?? null }));
  const totals = rows.map((r) => (r.by ? Object.values(r.by).reduce((a, c) => a + c, 0) : 0));
  const { max, ticks } = niceMax(Math.max(1, ...totals));
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="tc-svg" role="group" aria-label={`Candidates per acquisition interval by type (${basis === 'stored' ? 'stored candidates, by first-detected interval' : 'report counts of the per-interval pair runs'})`}
      data-testid="chart-counts" data-basis={basis}>
      <defs><pattern id={hatch} width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="6" className="tc-hatch" /></pattern></defs>
      <YAxis max={max} ticks={ticks} y={base} fmt={(v) => String(v)} />
      <Rail ax={ax} dates={d.dates} y={base} />
      {rows.map(({ iv, by }, k) => {
        const x0 = ax.x(iv.from) + 4, x1 = ax.x(iv.to) - 4;
        let acc = 0;
        return (
          <g key={iv.id} data-testid="interval-bar" data-interval={iv.id} data-total={by ? totals[k] : ''}>
            <rect x={x0} y={base + 6} width={x1 - x0} height={7} fill={`url(#${hatch})`} className="tc-gap"><title>{`${iv.from} to ${iv.to}: not observed in between`}</title></rect>
            {by ? d.types.map((t) => {
              const n = by[t] ?? 0;
              if (!n) return null;
              const hgt = (n / max) * PLOT_H, y = base - ((acc + n) / max) * PLOT_H;
              acc += n;
              const open = basis === 'stored';
              const txt = `${fmtInt(n)} ${label(t)} · ${iv.from} → ${iv.to}${open ? ' · open in Changes' : ''}`;
              return (
                <rect key={t} x={x0} y={y} width={x1 - x0} height={Math.max(1, hgt - 0.6)} fill={typeColor(t)} className={open ? 'tc-seg click' : 'tc-seg'} data-testid="seg"
                  data-interval={iv.id} data-type={t} data-n={n} role={open ? 'button' : undefined} tabIndex={open ? 0 : undefined} aria-label={txt}
                  onClick={open ? () => onOpen({ firstDetected: iv.id, changeType: t, region, label: `first detected ${iv.from} → ${iv.to} · ${label(t)}` }) : undefined}
                  onKeyDown={open ? (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onOpen({ firstDetected: iv.id, changeType: t, region, label: `first detected ${iv.from} → ${iv.to} · ${label(t)}` }); } } : undefined}>
                  <title>{txt}</title>
                </rect>
              );
            }) : <text x={(x0 + x1) / 2} y={base - 8} textAnchor="middle" className="tc-na">{DASH}</text>}
            {by && <text x={(x0 + x1) / 2} y={base - (totals[k] / max) * PLOT_H - 6} textAnchor="middle" className="tc-total" data-testid="interval-total">{fmtInt(totals[k])}</text>}
          </g>
        );
      })}
    </svg>
  );
}

// ------------------------------------------------------------------------------------------------ not placed in one interval
function Unplaced({ d, onOpen, label, region }: { d: TemporalData; onOpen: OnOpen; label: (t: string) => string; region: string }) {
  const rows: { id: string; title: string; sub: string; b: TemporalBucket; fd: string }[] = [
    ...d.multi_interval.map((m) => ({ id: m.id, fd: m.id, title: `Seen only across ${m.from} → ${m.to}`, sub: `not localised to a single interval · ${fmtInt(m.days)} days wide`, b: m })),
    { id: 'none', fd: 'none', title: 'No supported interval', sub: 'the model fired on the whole-span comparison but on no later date', b: d.no_interval },
  ].filter((r) => r.b.n > 0);
  if (!rows.length) return <div className="dim" style={{ fontSize: 12 }}>Every stored candidate in this scope was placed in a single interval.</div>;
  const mx = Math.max(...rows.map((r) => r.b.n));
  return (
    <div className="tc-unplaced" data-testid="unplaced">
      {rows.map((r) => (
        <div className="tc-urow" key={r.id} data-testid="unplaced-row" data-id={r.id} data-n={r.b.n}>
          <div className="tc-utitle"><b>{r.title}</b><span className="dim">{r.sub}</span></div>
          <div className="tc-ubar" style={{ width: `${Math.max(6, (r.b.n / mx) * 100)}%` }}>
            {d.types.map((t) => { const n = r.b.by_type[t] ?? 0; return n ? (
              <button key={t} className="tc-useg" style={{ flexGrow: n, background: typeColor(t) }} data-testid="useg" data-id={r.id} data-type={t} data-n={n}
                aria-label={`${n} ${label(t)} · ${r.title} · open in Changes`} title={`${n} ${label(t)} · open in Changes`}
                onClick={() => onOpen({ firstDetected: r.fd, changeType: t, region, label: `${r.title.toLowerCase()} · ${label(t)}` })}>{n >= 8 ? n : ''}</button>) : null; })}
          </div>
          <span className="mono tc-un">{fmtInt(r.b.n)} · {fmtKm2(r.b.area_m2)}</span>
        </div>
      ))}
    </div>
  );
}

// ------------------------------------------------------------------------------------------------ chart 2: cumulative area per type
function CumulativeChart({ d, onOpen, label, region }: { d: TemporalData; onOpen: OnOpen; label: (t: string) => string; region: string }) {
  const ax = useMemo(() => calendarAxis(d.dates, L, W - R), [d.dates]);
  const base = PLOT_H + 20, H = base + 56;
  const peak = Math.max(...d.cumulative.flatMap((c) => Object.values(c.area_m2_by_type)), 1);
  const { max, ticks } = niceMax(km2(peak));
  const dodge = (i: number) => (i - (d.types.length - 1) / 2) * 7;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="tc-svg" role="group" aria-label="Cumulative area of stored candidates by type, at each acquisition date" data-testid="chart-cumulative">
      <YAxis max={max} ticks={ticks} y={base} fmt={(v) => `${v} km²`} />
      <Rail ax={ax} dates={d.dates} y={base} />
      {d.types.map((t, ti) => d.cumulative.map((c, ci) => {
        const a = c.area_m2_by_type[t] ?? 0, cx = ax.x(c.date ?? d.dates[ci]) + dodge(ti), cy = base - (km2(a) / max) * PLOT_H;
        const ids = d.intervals.filter((iv) => iv.to <= (c.date ?? '')).map((iv) => iv.id).join(',');
        const base0 = ci === 0;                                                       // the baseline acquisition: nothing can have been detected yet
        const txt = base0 ? `${label(t)} · ${c.date} · baseline acquisition: nothing detected by definition` : `${label(t)} · ${fmtKm2(a)} first detected by ${c.date} (${fmtInt(c.n_by_type[t] ?? 0)} candidates)${a ? ' · open in Changes' : ''}`;
        return (
          <g key={`${t}-${ci}`}>
            <line x1={cx} x2={cx} y1={base} y2={cy} stroke={typeColor(t)} className="tc-stem" />
            <circle cx={cx} cy={cy} r={5} data-testid="cum-point" data-type={t} data-date={c.date} data-area-m2={a} data-n={c.n_by_type[t] ?? 0}
              className={base0 ? 'tc-pt open' : 'tc-pt click'} fill={base0 ? 'var(--panel)' : typeColor(t)} stroke={typeColor(t)} role={!base0 && a ? 'button' : undefined} tabIndex={!base0 && a ? 0 : undefined} aria-label={txt}
              onClick={!base0 && a ? () => onOpen({ firstDetected: ids, changeType: t, region, label: `${label(t)} first detected up to ${c.date}` }) : undefined}
              onKeyDown={!base0 && a ? (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onOpen({ firstDetected: ids, changeType: t, region, label: `${label(t)} first detected up to ${c.date}` }); } } : undefined}>
              <title>{txt}</title>
            </circle>
          </g>
        );
      }))}
    </svg>
  );
}

// ------------------------------------------------------------------------------------------------ chart 3: persistence per interval
function PersistenceChart({ d, onOpen, region }: { d: TemporalData; onOpen: OnOpen; region: string }) {
  const hatch = useId().replace(/:/g, '');
  const ax = useMemo(() => calendarAxis(d.dates, L, W - R), [d.dates]);
  const base = PLOT_H + 20, H = base + 56;
  const { max, ticks } = niceMax(Math.max(1, ...d.intervals.map((iv) => iv.stored.n)));
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="tc-svg" role="group" aria-label="Persistence of the stored candidates, by first-detected interval" data-testid="chart-persistence">
      <defs><pattern id={hatch} width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="6" className="tc-hatch" /></pattern></defs>
      <YAxis max={max} ticks={ticks} y={base} fmt={(v) => String(v)} />
      <Rail ax={ax} dates={d.dates} y={base} />
      {d.intervals.map((iv) => {
        const x0 = ax.x(iv.from) + 4, x1 = ax.x(iv.to) - 4;
        let acc = 0;
        return (
          <g key={iv.id} data-testid="pers-bar" data-interval={iv.id} data-held={iv.stored.held} data-demoted={iv.stored.demoted}>
            <rect x={x0} y={base + 6} width={x1 - x0} height={7} fill={`url(#${hatch})`} className="tc-gap" />
            {PERS_ORDER.map((p) => {
              const n = iv.stored.persistence[p] ?? 0;
              if (!n) return null;
              const hgt = (n / max) * PLOT_H, y = base - ((acc + n) / max) * PLOT_H;
              acc += n;
              const txt = `${fmtInt(n)} ${p} · first detected ${iv.from} → ${iv.to} · open in Changes`;
              return (
                <rect key={p} x={x0} y={y} width={x1 - x0} height={Math.max(1, hgt - 0.6)} fill={PERS_COLOR[p]} className="tc-seg click" data-testid="pseg" data-interval={iv.id} data-class={p} data-n={n}
                  role="button" tabIndex={0} aria-label={txt}
                  onClick={() => onOpen({ firstDetected: iv.id, persistence: p, region, label: `first detected ${iv.from} → ${iv.to} · ${p}` })}
                  onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onOpen({ firstDetected: iv.id, persistence: p, region, label: `first detected ${iv.from} → ${iv.to} · ${p}` }); } }}>
                  <title>{txt}</title>
                </rect>
              );
            })}
            {iv.stored.n > 0 && <text x={(x0 + x1) / 2} y={base - (iv.stored.n / max) * PLOT_H - 6} textAnchor="middle" className="tc-total">{iv.stored.held} held · {iv.stored.demoted} demoted</text>}
          </g>
        );
      })}
    </svg>
  );
}

// ------------------------------------------------------------------------------------------------ chart 4: SAR corroboration coverage per interval
function SarChart({ d, onOpen, region }: { d: TemporalData; onOpen: OnOpen; region: string }) {
  const ax = useMemo(() => calendarAxis(d.dates, L, W - R), [d.dates]);
  const base = 130, H = base + 80;
  const ph = 110, rail = base + 28;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="tc-svg" role="group" aria-label="Share of stored candidates with Sentinel-1 corroboration, by first-detected interval" data-testid="chart-sar">
      <g className="tc-yaxis">{[0, 0.5, 1].map((t) => { const yy = base - t * ph; return <g key={t}><line x1={L} x2={W - R} y1={yy} y2={yy} className="tc-grid" /><text x={L - 8} y={yy + 4} textAnchor="end" className="tc-ytxt">{Math.round(t * 100)}%</text></g>; })}</g>
      <g className="tc-rail">
        {ax.ticks.map((t) => <line key={t.label} x1={t.x} x2={t.x} y1={base - ph} y2={base} className="tc-year" />)}
        <line x1={L} x2={W - R} y1={rail} y2={rail} className="tc-base" />
        {d.dates.map((dt, i) => <g key={dt}><circle cx={ax.x(dt)} cy={rail} r={5} className="tc-acq" /><text x={ax.x(dt)} y={rail + 20} textAnchor={i === 0 ? 'start' : i === d.dates.length - 1 ? 'end' : 'middle'} className="tc-datetxt">{dt}</text></g>)}
      </g>
      {d.intervals.map((iv) => {
        const cx = (ax.x(iv.from) + ax.x(iv.to)) / 2, cov = iv.stored.sar_coverage;
        if (cov === null) return <text key={iv.id} x={cx} y={base - 8} textAnchor="middle" className="tc-na" data-testid="sar-point" data-interval={iv.id} data-n="0" data-covered="">{DASH}</text>;
        const txt = `${iv.stored.sar_available} of ${iv.stored.n} candidates first detected ${iv.from} → ${iv.to} have Sentinel-1 corroboration · open in Changes`;
        return (
          <g key={iv.id}>
            <circle cx={cx} cy={base - cov * ph} r={6} className="tc-pt click" fill={iv.stored.sar_available ? '#4cc9f0' : 'var(--panel)'} stroke="#4cc9f0" data-testid="sar-point" data-interval={iv.id}
              data-n={iv.stored.n} data-covered={iv.stored.sar_available} role="button" tabIndex={0} aria-label={txt}
              onClick={() => onOpen({ firstDetected: iv.id, sensor: 'sentinel-1', region, label: `first detected ${iv.from} → ${iv.to} · with Sentinel-1 corroboration` })}
              onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onOpen({ firstDetected: iv.id, sensor: 'sentinel-1', region, label: `first detected ${iv.from} → ${iv.to} · with Sentinel-1 corroboration` }); } }}><title>{txt}</title></circle>
            <text x={cx} y={base - cov * ph - 12} textAnchor="middle" className="tc-total">{iv.stored.sar_available} of {fmtInt(iv.stored.n)}</text>
          </g>
        );
      })}
    </svg>
  );
}

// ------------------------------------------------------------------------------------------------ the mode
export function TemporalArchiveMode() {
  const { label, openInChanges } = useStore();
  const regions = useApi((s) => api.regions(s), []);
  const [region, setRegion] = useState('');
  const [basis, setBasis] = useState<'stored' | 'pair_run'>('stored');
  const bbox: BBox | null = useMemo(() => regions.data?.regions.find((r) => r.name === region)?.bbox ?? null, [regions.data, region]);
  const t = useApi((s) => api.temporalArchive(bbox, s), [region, regions.data]);
  const d = t.data;
  const onOpen: OnOpen = (f) => openInChanges(f);
  const lab = (x: string) => label(x);

  return (
    <div className="col tc-mode" data-testid="temporal-archive" data-mode="archive">
      <div className="tc-modebanner archive" role="note" data-testid="archive-banner">
        <b>ARCHIVE MODE · pipeline output.</b> Every figure below is read from the change pipeline’s own results for the archive’s real acquisition dates. Nothing is interpolated: the dates are marks on a true calendar axis and the space between them is a gap in what was observed.
      </div>
      <div className="row" style={{ alignItems: 'flex-end', flexWrap: 'wrap', gap: 12 }}>
        <label className="field">Region
          <select className="input" value={region} data-testid="temporal-region" onChange={(e) => setRegion(e.target.value)} aria-label="Region">
            <option value="">Whole change-pipeline area</option>{regions.data?.regions.map((r) => <option key={r.name} value={r.name}>{regionLabel(r.name)}</option>)}
          </select>
        </label>
        <label className="field">Counts in chart 1
          <select className="input" value={basis} data-testid="temporal-basis" onChange={(e) => setBasis(e.target.value as 'stored' | 'pair_run')} aria-label="Basis of the counts">
            <option value="stored">Stored candidates, by the interval they were first detected in</option>
            <option value="pair_run" disabled={!!region}>Report counts from each interval’s own pair run{region ? ' (whole area only)' : ''}</option>
          </select>
        </label>
        {d && <span className="dim mono" style={{ fontSize: 11 }} data-testid="temporal-scope">{d.aoi ?? 'area'} · {fmtInt(d.totals.stored_in_scope)} stored candidates in scope</span>}
      </div>
      {t.error ? <ErrorNote error={t.error} onRetry={t.reload} /> : !d ? <Loading rows={8} /> : !d.available ? <Panel title="Archive mode"><div className="empty">The change pipeline has not produced results for this archive.</div></Panel> : (
        <>
          <Panel title={basis === 'stored' ? 'Candidates per interval, by type · first-detected interval' : 'Survivors per interval, by type · each interval’s own pair run'}>
            <Legend types={d.types} label={lab} />
            <CountsChart d={d} basis={basis} onOpen={onOpen} label={lab} region={region} />
            <p className="tc-note" data-testid="basis-note">
              {basis === 'stored'
                ? <>A bar covers the whole interval between two acquisitions: the pipeline first supported the candidate somewhere inside it, and the exact date is unknown. Click a segment to open exactly those candidates in Changes.</>
                : <>{d.pair_run_note} Report counts for the five-date run are survivors of each consecutive pair’s own gates; they overlap one another and are a different population from the stored candidates.</>}
              {d.span_run && <> The whole-span pair ({d.span_run.from} → {d.span_run.to}) produced {fmtInt(d.span_run.raw)} raw components and {fmtInt(d.span_run.survivors)} survivors; those are the stored candidates.</>}
            </p>
            {basis === 'stored' && <>
              <h4 className="tc-sub">Not placed in a single interval</h4>
              <Unplaced d={d} onOpen={onOpen} label={lab} region={region} />
            </>}
          </Panel>

          <Panel title="Cumulative area first detected, by type">
            <Legend types={d.types} label={lab} />
            <CumulativeChart d={d} onOpen={onOpen} label={lab} region={region} />
            <p className="tc-note" data-testid="cum-note">
              One mark per acquisition date: the area (km²) of stored candidates the pipeline had first supported by that date. The marks are not joined: nothing was observed between them. The first date is the baseline, so it is zero by definition.
              Not included: {fmtInt(d.multi_interval.reduce((a, m) => a + m.n, 0))} candidates ({fmtKm2(d.multi_interval.reduce((a, m) => a + m.area_m2, 0))}) seen only in the whole-span comparison and {fmtInt(d.no_interval.n)} ({fmtKm2(d.no_interval.area_m2)}) with no supported interval.
              Candidates are counted once each and their areas are summed; areas are the pipeline’s own pixel counts at 100 m² per 10 m pixel.
            </p>
          </Panel>

          <Panel title="Persistence per interval · held or demoted">
            <div className="tc-legend" aria-label="Persistence classes">{PERS_ORDER.map((p) => <span key={p}><i style={{ background: PERS_COLOR[p] }} />{p}{d.held_classes.includes(p) ? ' (held)' : ' (demoted)'}</span>)}</div>
            <PersistenceChart d={d} onOpen={onOpen} region={region} />
            <p className="tc-note" data-testid="pers-note">
              <b>Held</b> = later acquisitions still show the change ({d.held_classes.join(' / ')}); <b>demoted</b> = later acquisitions contradict it ({d.demoted_classes.join(' / ')}), so its confidence is penalised, not removed.
              Candidates first detected in the last interval have no later acquisition to test them against, so they can only read “recent”; persistence is not comparable across intervals for that reason.
            </p>
          </Panel>

          <Panel title="Sentinel-1 (SAR) corroboration coverage per interval">
            <SarChart d={d} onOpen={onOpen} region={region} />
            <p className="tc-note" data-testid="sar-note">
              Share of each interval’s stored candidates that carry a SAR corroboration. {d.sar.available_candidates === 0
                ? <b>No candidate in this run has SAR corroboration: </b> : null}{d.sar.note ?? ''}
            </p>
          </Panel>
          <div className="dim" style={{ fontSize: 11 }} data-testid="temporal-source">{d.scope} Source: <span className="mono">{d.source.report}</span> and <span className="mono">{d.source.candidates}</span>, report generated {d.source.report_generated_at.slice(0, 19).replace('T', ' ')} UTC. Held share overall: {fmtPct(d.intervals.reduce((a, iv) => a + iv.stored.held, 0) / Math.max(1, d.intervals.reduce((a, iv) => a + iv.stored.held + iv.stored.demoted, 0)), 0)} of the candidates placed in an interval.</div>
        </>
      )}
    </div>
  );
}
