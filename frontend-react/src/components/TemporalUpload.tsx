import { useEffect, useMemo, useState } from 'react';
import { Panel } from '@/components/Panel';
import { DASH, fmtInt } from '@/fmt';
import { href } from '@/router';
import { calendarAxis } from '@/lib/calendarAxis';
import { computeProfile, selectProfileSet, type Profile, type Selection } from '@/lib/rasterProfile';
import { setUploadDate, useUploads, type Upload } from '@/lib/uploads';

const W = 960, L = 64, R = 24, PLOT_H = 240;
const COLORS = ['#4cc9f0', '#f5a524', '#34d399', '#a78bfa', '#ff9f6b', '#ff6b7a'];

const f4 = (v: number) => (Number.isFinite(v) ? (Math.abs(v) >= 100 ? v.toFixed(1) : v.toFixed(4)) : DASH);

function ProfileChart({ p, metric }: { p: Profile; metric: string }) {
  const rows = p.rows.map((r, i) => ({ r, i, s: r.metrics[metric] })).filter((x) => x.s && x.s.n > 0);
  const dates = rows.map((x) => x.r.date);
  const ax = useMemo(() => calendarAxis(dates, L, W - R, 90), [dates.join('|')]); // eslint-disable-line react-hooks/exhaustive-deps
  const lo0 = Math.min(...rows.map((x) => x.s.p10)), hi0 = Math.max(...rows.map((x) => x.s.p90));
  const pad = (hi0 - lo0 || Math.abs(hi0) || 1) * 0.12, lo = lo0 - pad, hi = hi0 + pad;
  const base = PLOT_H + 20, H = base + 64, y = (v: number) => base - ((v - lo) / (hi - lo)) * PLOT_H;
  const tickVals = Array.from({ length: 5 }, (_, k) => lo + ((hi - lo) * k) / 4);
  const dodge = (i: number) => (i - (rows.length - 1) / 2) * 10;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="tc-svg" role="group" aria-label="Spectral profile: per-file statistics of the overlapping region on each acquisition date" data-testid="profile-chart" data-metric={metric}>
      {tickVals.map((t) => <g key={t}><line x1={L} x2={W - R} y1={y(t)} y2={y(t)} className="tc-grid" /><text x={L - 8} y={y(t) + 4} textAnchor="end" className="tc-ytxt">{f4(t)}</text></g>)}
      {ax.ticks.map((t) => <g key={t.label}><line x1={t.x} x2={t.x} y1={base - PLOT_H} y2={base} className="tc-year" /><text x={t.x + 3} y={base - PLOT_H + 10} className="tc-yeartxt">{t.label}</text></g>)}
      <line x1={L} x2={W - R} y1={base} y2={base} className="tc-base" />
      {rows.map(({ r, i, s }, k) => {
        const cx = ax.x(r.date) + dodge(k), c = COLORS[i % COLORS.length];
        return (
          <g key={r.name + r.date} data-testid="profile-point" data-file={r.name} data-date={r.date} data-mean={s.mean} data-p10={s.p10} data-p90={s.p90} data-n={s.n}>
            <line x1={cx} x2={cx} y1={y(s.p10)} y2={y(s.p90)} stroke={c} strokeWidth="2.5" strokeLinecap="round" />
            <line x1={cx - 5} x2={cx + 5} y1={y(s.p10)} y2={y(s.p10)} stroke={c} strokeWidth="2" /><line x1={cx - 5} x2={cx + 5} y1={y(s.p90)} y2={y(s.p90)} stroke={c} strokeWidth="2" />
            <circle cx={cx} cy={y(s.mean)} r={6} fill={c} stroke="var(--panel)" strokeWidth="1.5"><title>{`${r.name} · ${r.date} · mean ${f4(s.mean)} · p10–p90 ${f4(s.p10)}…${f4(s.p90)} · n ${fmtInt(s.n)}`}</title></circle>
            <circle cx={cx} cy={base} r={5} className="tc-acq" /><text x={cx} y={base + 20} textAnchor="middle" className="tc-datetxt">{r.date}</text>
          </g>
        );
      })}
    </svg>
  );
}

export function TemporalUploadMode() {
  const files = useUploads();
  const sel = useMemo(() => selectProfileSet(files), [files]);
  const dates = new Set(sel.members.map((m) => m.date));
  const ready = sel.members.length >= 2 && dates.size >= 2 && sel.box !== null;
  const [metric, setMetric] = useState('');
  const [profile, setProfile] = useState<Profile | null>(null);
  const [busy, setBusy] = useState(false);
  const key = ready ? sel.members.map((m) => `${m.up.id}:${m.date}`).join('|') : '';

  useEffect(() => {
    if (!ready) { setProfile(null); return; }
    setBusy(true);
    const t = window.setTimeout(() => {                                    // let the "computing" state paint before the (synchronous) pixel work
      try { const p = computeProfile(sel); setProfile(p); setMetric((m) => (p.metricKeys.some((k) => k.key === m) ? m : (p.metricKeys.find((k) => k.key === 'ndvi') ?? p.metricKeys[0]).key)); }
      finally { setBusy(false); }
    }, 30);
    return () => window.clearTimeout(t);
  }, [key]); // eslint-disable-line react-hooks/exhaustive-deps

  const n = files.filter((u) => u.header).length;
  const checks: { ok: boolean; text: string }[] = [
    { ok: n >= 2, text: `Two or more GeoTIFFs dropped on the Data screen (${n} loaded)` },
    { ok: sel.placeable >= 2, text: `Pixels decoded and a footprint that can be placed on the ground, for at least two files (${sel.placeable} qualify)` },
    { ok: sel.dated >= 2, text: `An acquisition date for at least two of them, from the file header or entered by you (${sel.dated} have one)` },
    { ok: sel.members.length >= 2, text: `Footprints that overlap one another (${sel.members.length} of the dated files share a common area)` },
    { ok: dates.size >= 2, text: `At least two different acquisition dates (${dates.size} distinct among them)` },
  ];

  return (
    <div className="col tc-mode" data-testid="temporal-upload" data-mode="upload">
      <div className="tc-modebanner upload" role="note" data-testid="upload-banner">
        <b>UPLOAD MODE · SPECTRAL PROFILE — not change detection.</b> This tab summarises the pixels of the files you dropped, per band and per spectral index, on each file’s date. It is computed in this browser; nothing is uploaded.
        <ul>
          <li>The change model, suppression gates, persistence check and confidence engine are <b>not run</b> on ad-hoc uploads. There are no candidates and no confidence scores here, and none should be inferred.</li>
          <li>The files are <b>not co-registered</b>: they are only resampled onto a common grid so they describe the same patch of ground.</li>
          <li>Index values are <b>not comparable across sensors</b> with different band response functions, and no reflectance scaling or atmospheric correction is applied.</li>
        </ul>
      </div>

      {!ready || !sel.box ? (
        <Panel title="Spectral profile · not available yet">
          <div className="col" style={{ gap: 10 }} data-testid="upload-unavailable">
            <p style={{ margin: 0, fontSize: 13 }}><b>One date cannot produce a temporal series.</b> A profile needs two or more files that overlap on the ground and were acquired on different dates. Nothing is drawn until that is true.</p>
            <ul className="tc-checks" aria-label="What is required">
              {checks.map((c) => <li key={c.text} data-ok={c.ok ? '1' : '0'}><span className={`chip ${c.ok ? 'green' : 'amber'}`}>{c.ok ? '✓' : 'missing'}</span> {c.text}</li>)}
            </ul>
            <FileTable files={files} sel={sel} />
            <div className="dim" style={{ fontSize: 12 }}>Drop files on the <a href={href('data')}>Data screen</a>, then return here.</div>
          </div>
        </Panel>
      ) : (
        <>
          <Panel title="Spectral profile of the overlapping region">
            <div className="col" style={{ gap: 10 }}>
              <div className="row" style={{ alignItems: 'flex-end', flexWrap: 'wrap', gap: 12 }}>
                <label className="field">Metric
                  <select className="input" value={metric} data-testid="profile-metric" onChange={(e) => setMetric(e.target.value)} aria-label="Metric to plot">
                    {(profile?.metricKeys ?? []).map((k) => <option key={k.key} value={k.key}>{k.label}</option>)}
                  </select>
                </label>
                <span className="dim" style={{ fontSize: 11.5 }}>{profile?.metricKeys.find((k) => k.key === metric)?.note}</span>
              </div>
              {busy || !profile ? <div className="dim" role="status" style={{ fontSize: 12 }}>Computing statistics from the decoded pixels…</div> : (
                <>
                  <ProfileChart p={profile} metric={metric} />
                  <p className="tc-note" data-testid="profile-note">
                    Each mark is one file on its own acquisition date (a calendar axis, so spacing is real); the dot is the mean over the overlapping region and the bar runs from the 10th to the 90th percentile.
                    Marks are not joined, since nothing was observed between the dates. Region: lon {sel.box[0].toFixed(4)}…{sel.box[2].toFixed(4)}, lat {sel.box[1].toFixed(4)}…{sel.box[3].toFixed(4)}, sampled on a Web-Mercator grid
                    (zoom {profile.z}, about {profile.resM.toFixed(1)} m per cell, {fmtInt(profile.cells)} cells) by nearest-neighbour from each file’s own grid, so every file is summarised over the same ground.
                    Statistics use only valid pixels (the file’s NoData value is excluded); the standard deviation is the population one.
                  </p>
                  <ProfileTable p={profile} />
                </>
              )}
              <FileTable files={files} sel={sel} />
            </div>
          </Panel>
        </>
      )}
    </div>
  );
}

function ProfileTable({ p }: { p: Profile }) {
  return (
    <div className="queue-scroll" style={{ maxHeight: 320 }}>
      <table className="tbl" aria-label="Statistics per file, band and index" data-testid="profile-table">
        <thead><tr><th>File</th><th>Date</th><th>Metric</th><th className="num">Valid px</th><th className="num">Mean</th><th className="num">Std</th><th className="num">P10</th><th className="num">Median</th><th className="num">P90</th></tr></thead>
        <tbody>
          {p.rows.flatMap((r) => p.metricKeys.filter((k) => r.metrics[k.key]).map((k) => {
            const s = r.metrics[k.key];
            return <tr key={r.name + k.key} data-metric={k.key} data-file={r.name}><td className="mono">{r.name}</td><td className="mono">{r.date}</td><td>{k.key in r.bandIndex ? `${k.label} (band ${r.bandIndex[k.key] + 1})` : k.label.split(' · ')[0]}</td><td className="num">{fmtInt(s.n)}</td><td className="num">{f4(s.mean)}</td><td className="num">{f4(s.std)}</td><td className="num">{f4(s.p10)}</td><td className="num">{f4(s.p50)}</td><td className="num">{f4(s.p90)}</td></tr>;
          }))}
        </tbody>
      </table>
    </div>
  );
}

function FileTable({ files, sel }: { files: Upload[]; sel: Selection }) {
  if (!files.length) return null;
  const used = new Set(sel.members.map((m) => m.up.id));
  const why = new Map(sel.excluded.map((e) => [e.up.id, e.why]));
  return (
    <table className="tbl" aria-label="Files and their dates" data-testid="upload-files">
      <thead><tr><th>File</th><th>Acquisition date</th><th>Used in the profile</th></tr></thead>
      <tbody>
        {files.map((u) => (
          <tr key={u.id} data-used={used.has(u.id) ? '1' : '0'}>
            <td className="mono">{u.name}</td>
            <td>{u.header?.acquisition?.iso ? <span className="mono">{u.acquired.date} <span className="dim">(file header)</span></span> : u.header ? (
              <span className="row" style={{ gap: 6, alignItems: 'center' }}>
                <input type="date" className="input" value={u.acquired.date ?? ''} aria-label={`Acquisition date of ${u.name}`} onChange={(e) => setUploadDate(u.id, e.target.value)} style={{ padding: '2px 6px', fontSize: 11.5 }} />
                <span className="dim" style={{ fontSize: 10.5 }}>{u.acquired.source === 'entered by the analyst' ? 'entered by you' : 'not in the file'}</span>
              </span>) : DASH}</td>
            <td>{used.has(u.id) ? <span className="chip green">used</span> : <span className="chip amber" title={why.get(u.id)}>not used</span>} <span className="dim" style={{ fontSize: 11 }}>{used.has(u.id) ? '' : why.get(u.id) ?? ''}</span></td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
