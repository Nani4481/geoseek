import { useMemo, useState, type ReactNode } from 'react';
import { api } from '@/api/client';
import type { BBox, RingFeature, RingKind, ThreatRings } from '@/api/types';
import { useApi, type ApiState } from '@/hooks/useApi';
import { DASH, fmtInt, fmtLonLat, fmtPct } from '@/fmt';
import { useStore } from '@/state/store';
import type { MapCircle, MapPoint } from './GeoMap';
import { ErrorNote, Loading } from './Widgets';

export interface RingCenter { id: string; lon: number; lat: number; label: string }

export const RING_COLORS = ['#ff6b7a', '#ff9f6b', '#f5a524', '#e8c24a', '#9bd36b', '#4cc9f0', '#5b8cff', '#a78bfa'];
export const KIND_COLOR: Record<RingKind, string> = { restricted_zone: '#ff6b7a', change_candidate: '#f5a524', detection: '#4cc9f0', watch_area: '#a78bfa' };
export const KIND_LABEL: Record<RingKind, string> = { restricted_zone: 'Restricted zone', change_candidate: 'Change candidate', detection: 'Detection', watch_area: 'Watch area' };
const DEFAULT_RADII = '500, 1000, 2500, 5000';
const MIN_M = 10, MAX_M = 200_000, MAX_RINGS = 8;

export const fmtDist = (m: number) => (m < 1000 ? `${Math.round(m)} m` : `${(m / 1000).toFixed(m < 10_000 ? 2 : 1)} km`);

/** Same rules as the backend (`threat_rings.parse_radii`), so a bad entry is explained before any request is made. */
export function parseRadii(text: string): { values: number[] | null; error: string | null } {
  const parts = text.split(/[,\s]+/).filter(Boolean);
  const nums = parts.map(Number);
  if (parts.length === 0) return { values: null, error: 'enter at least one radius in metres' };
  if (nums.some((n) => !Number.isFinite(n))) return { values: null, error: 'radii must be numbers (metres)' };
  const vals = [...new Set(nums)].sort((a, b) => a - b);
  if (vals.length > MAX_RINGS) return { values: null, error: `at most ${MAX_RINGS} rings` };
  if (vals[0] < MIN_M || vals[vals.length - 1] > MAX_M) return { values: null, error: `radii must be between ${MIN_M} m and ${MAX_M / 1000} km` };
  return { values: vals, error: null };
}

export interface RingsController {
  center: RingCenter | null;
  setCenter: (c: RingCenter | null) => void;
  draft: string; setDraft: (s: string) => void; apply: () => void; radiiError: string | null;
  radii: number[];
  query: ApiState<ThreatRings>;
  circles: MapCircle[];
  overlay: MapPoint[];
  fit: BBox | null;
}

/** State for one map's threat rings: the centre picked by right-click, the configurable radii, the live API result and the
 *  map overlays derived from it. */
export function useThreatRings(): RingsController {
  const [center, setCenter] = useState<RingCenter | null>(null);
  const [draft, setDraft] = useState(DEFAULT_RADII);
  const [applied, setApplied] = useState(DEFAULT_RADII);
  const parsedDraft = useMemo(() => parseRadii(draft), [draft]);
  const parsed = useMemo(() => parseRadii(applied), [applied]);
  const radii = parsed.values ?? [];
  const key = `${center?.id}|${center?.lon}|${center?.lat}|${radii.join(',')}`;
  const query = useApi(center && parsed.values ? (s) => api.threatRings({ lon: center.lon, lat: center.lat, radii_m: parsed.values!, exclude: center.id }, s) : null, [key]);

  const circles = useMemo<MapCircle[]>(() => (center ? radii.map((r, i) => ({
    lon: center.lon, lat: center.lat, radiusM: r, color: RING_COLORS[i % RING_COLORS.length], dashed: true, label: `Ring ${i + 1} · ${fmtDist(r)}`, tag: fmtDist(r),
  })) : []), [center, radii.join(',')]); // eslint-disable-line react-hooks/exhaustive-deps

  const overlay = useMemo<MapPoint[]>(() => {
    if (!center) return [];
    const pts: MapPoint[] = [{ id: 'ring-centre', lon: center.lon, lat: center.lat, color: '#ffffff', radius: 6, label: `Ring centre · ${center.label}` }];
    for (const f of query.data?.features ?? []) {
      pts.push({ id: 'ring:' + f.id, lon: f.nearest[0], lat: f.nearest[1], color: KIND_COLOR[f.kind], radius: 4, label: `${KIND_LABEL[f.kind]} · ${f.name} · ${fmtDist(f.distance_m)}` });
    }
    return pts;
  }, [center, query.data]);

  const fit = useMemo<BBox | null>(() => {
    if (!center || radii.length === 0) return null;
    const r = radii[radii.length - 1] * 1.15;
    const dlat = r / 110_540, dlon = r / (111_320 * Math.max(Math.cos((center.lat * Math.PI) / 180), 0.01));
    return [center.lon - dlon, center.lat - dlat, center.lon + dlon, center.lat + dlat];
  }, [center?.lon, center?.lat, radii.join(',')]); // eslint-disable-line react-hooks/exhaustive-deps

  return {
    center, setCenter, draft, setDraft, apply: () => { if (parsedDraft.values) setApplied(parsedDraft.values.join(', ')); },
    radiiError: parsedDraft.error, radii, query, circles, overlay, fit,
  };
}

function detailText(f: RingFeature, label: (t: string | null) => string): string {
  const d = f.detail;
  switch (f.kind) {
    case 'restricted_zone': return `${String(d.level)} alert level`;
    case 'change_candidate': return `${label(String(d.change_type))} · confidence ${fmtPct(Number(d.confidence), 0)}`;
    case 'detection': return `${String(d.class)} · score ${Number(d.score).toFixed(2)}`;
    case 'watch_area': return d.active ? 'active watch' : 'inactive watch';
  }
}

/** The configuration and result list that sit under a map with right-click threat rings. */
export function ThreatRingsResults({ tr, hint, action }: { tr: RingsController; hint: string; action?: ReactNode }) {
  const { label } = useStore();
  const q = tr.query;
  const d = q.data;
  return (
    <div className="col rings-panel" style={{ gap: 10 }} aria-label="Threat buffer rings">
      <form className="row" style={{ alignItems: 'flex-end', gap: 8, flexWrap: 'wrap' }} onSubmit={(e) => { e.preventDefault(); tr.apply(); }}>
        <label className="field" style={{ flex: 1, minWidth: 220 }}>Ring radii (metres, up to {MAX_RINGS})
          <input className="input mono" value={tr.draft} onChange={(e) => tr.setDraft(e.target.value)} aria-label="Ring radii in metres" aria-invalid={!!tr.radiiError} />
        </label>
        <button className="btn sm" type="submit" disabled={!!tr.radiiError}>Apply radii</button>
        {tr.center && <button className="btn sm" type="button" onClick={() => tr.setCenter(null)}>Clear rings</button>}
      </form>
      {tr.radiiError && <div className="err" role="alert" style={{ padding: 0 }}>{tr.radiiError}</div>}

      {!tr.center ? (
        action ? (
          <div className="col" style={{ gap: 6, alignItems: 'flex-start', padding: '4px 0' }}>
            {action}
            <span className="faint" style={{ fontSize: 11 }}>{hint}</span>
          </div>
        ) : <div className="empty" style={{ padding: 10 }}>{hint}</div>
      ) : (
        <>
          <div className="row" style={{ alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
            <span className="dim">Centre</span><b className="mono rings-centre">{tr.center.label}</b>
            <span className="dim mono" style={{ fontSize: 11 }}>{fmtLonLat([tr.center.lon, tr.center.lat])}</span>
          </div>
          {q.error ? <ErrorNote error={q.error} onRetry={q.reload} /> : !d ? <Loading rows={3} /> : (
            <>
              <div className="row ring-chips" style={{ gap: 6, flexWrap: 'wrap' }}>
                {d.rings.map((r, i) => (
                  <span key={r.radius_m} className="chip mono" data-ring={i} data-total={r.total}
                    style={{ borderColor: RING_COLORS[i % RING_COLORS.length], color: RING_COLORS[i % RING_COLORS.length] }}
                    title={Object.entries(r.counts).map(([k, n]) => `${KIND_LABEL[k as RingKind]}: ${n}`).join('\n')}>
                    ≤ {fmtDist(r.radius_m)} · {fmtInt(r.total)}
                  </span>
                ))}
              </div>
              {d.features.length === 0 ? (
                <div className="empty" style={{ padding: 10 }}>Nothing from the searched layers lies within {fmtDist(d.radii_m[d.radii_m.length - 1])} of this point.</div>
              ) : (
                <div className="queue-scroll" style={{ maxHeight: 280 }}>
                  <table className="tbl ring-table">
                    <thead><tr><th>Ring</th><th>Layer</th><th>Feature</th><th className="num">Distance</th><th>Detail</th></tr></thead>
                    <tbody>
                      {d.features.map((f) => (
                        <tr key={f.id} data-kind={f.kind} data-distance={f.distance_m}>
                          <td><i className="ring-dot" style={{ background: RING_COLORS[f.ring_index % RING_COLORS.length] }} />≤ {fmtDist(f.ring_m)}</td>
                          <td><span className="chip" style={{ color: KIND_COLOR[f.kind], borderColor: KIND_COLOR[f.kind] }}>{KIND_LABEL[f.kind]}</span></td>
                          <td className="mono" title={f.id}>{f.name}</td>
                          <td className="num">{f.contains_centre ? 'contains centre' : fmtDist(f.distance_m)}</td>
                          <td className="dim">{detailText(f, label)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
              {d.truncated && <div className="faint" style={{ fontSize: 11 }}>Showing the nearest {d.features.length} of {fmtInt(d.n_features)} features; the ring totals above count all of them.</div>}
              <div className="faint" style={{ fontSize: 10.5 }}>
                Searched: {Object.entries(d.sources).map(([k, s]) => `${KIND_LABEL[k as RingKind].toLowerCase()}s ${fmtInt(s?.features_searched)}`).join(' · ') || DASH}.
                {' '}{d.notes.join(' ')}
              </div>
            </>
          )}
        </>
      )}
    </div>
  );
}
