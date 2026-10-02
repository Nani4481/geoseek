import { Fragment, useEffect, useMemo, useState } from 'react';
import { api, tileThumb } from '@/api/client';
import type { SpectralLayer, TileSpectral } from '@/api/types';
import { useApi } from '@/hooks/useApi';
import { fmtPct } from '@/fmt';
import { ErrorNote, Loading } from './Widgets';
import { Panel } from './Panel';

const ORDER = ['ndwi', 'ndbi', 'ndvi'] as const;
type Idx = (typeof ORDER)[number];

const gradient = (l: SpectralLayer) => {
  const [lo, hi] = l.domain;
  return `linear-gradient(90deg, ${l.stops.map(([v, c]) => `${c} ${(((v - lo) / (hi - lo)) * 100).toFixed(1)}%`).join(', ')})`;
};

function Legend({ l }: { l: SpectralLayer }) {
  const [lo, hi] = l.domain;
  return (
    <div className="spec-legend" aria-label={`${l.label} legend`}>
      <div className="bar" style={{ background: gradient(l) }} />
      <div className="ticks mono">
        <span>{lo}</span><span>{l.stops.find((s) => s[0] === 0) ? '0' : ''}</span><span>{hi}</span>
      </div>
      <div className="dim" style={{ fontSize: 10.5 }}>low: {l.low} · high: {l.high}</div>
    </div>
  );
}

function Figure({ idx, l, tileId, opacity, relevant }: { idx: Idx; l: SpectralLayer; tileId: string; opacity: number; relevant: boolean }) {
  const [failed, setFailed] = useState(false);
  const st = l.stats;
  return (
    <figure className={`spec-fig ${relevant ? 'relevant' : ''}`} data-index={idx} data-relevant={relevant ? '1' : '0'}>
      <div className="spec-img">
        <img className="base" src={tileThumb(tileId)} alt={`True-colour tile ${tileId}`} />
        {!failed && <img className="pix" src={l.image_url} alt={`${l.label} overlay, one pixel per 10 m`} style={{ opacity }} onError={() => setFailed(true)} />}
        <span className="tag mono">{l.label}</span>
        {relevant && <span className="tag rel">relevant to your query</span>}
      </div>
      <figcaption>
        <b>{l.label}</b> <span className="dim">· {l.meaning}</span>
        <div className="mono dim" style={{ fontSize: 10.5 }}>{l.formula}</div>
        {st ? <div className="mono" style={{ fontSize: 11 }} data-stats={idx}>mean {st.mean.toFixed(3)} · p10 {st.p10.toFixed(2)} · p50 {st.p50.toFixed(2)} · p90 {st.p90.toFixed(2)}</div> : <div className="dim" style={{ fontSize: 11 }}>no statistics (too few valid pixels)</div>}
        <Legend l={l} />
      </figcaption>
    </figure>
  );
}

/**
 * "Why did this match?" answered with physical measurements: per-pixel NDWI / NDBI / NDVI of the tile at its native 10 m,
 * from the NIR and SWIR bands the retrieval model never sees. It deliberately does NOT draw an attention map: the model's
 * patches are hundreds of metres across and could not localise a structure.
 */
export function SpectralEvidence({ tileId, query, onClose }: { tileId: string; query: string | null; onClose: () => void }) {
  const spec = useApi((s) => api.spectral(tileId, query, s), [tileId, query]);
  const d: TileSpectral | null = spec.data;
  const relevant = useMemo(() => new Set((d?.relevance.matches ?? []).map((m) => m.index)), [d]);
  const [on, setOn] = useState<Set<Idx>>(new Set());
  const [opacity, setOpacity] = useState(0.8);

  // default: show the indices that bear on the query (or all three when none does)
  useEffect(() => {
    if (!d) return;
    const rel = ORDER.filter((i) => relevant.has(i));
    setOn(new Set(rel.length ? rel : ORDER));
  }, [d, relevant]);

  const toggle = (i: Idx) => setOn((s) => { const x = new Set(s); if (x.has(i)) x.delete(i); else x.add(i); return x; });

  return (
    <Panel title={`Spectral evidence · ${tileId}`} actions={<button className="btn sm" onClick={onClose} aria-label="Close spectral evidence">Close ✕</button>}>
      {spec.error ? <ErrorNote error={spec.error} onRetry={spec.reload} /> : !d ? <Loading rows={5} /> : (
        <div className="col spec-panel" style={{ gap: 12 }} data-tile={tileId}>
          <div className="spec-why" data-testid="spec-why">
            {query ? <>Search: <b>“{query}”</b>. </> : null}
            {d.relevance.matches.length ? d.relevance.matches.map((m) => <span key={m.index} className="spec-match"><b>{m.index.toUpperCase()}</b> — {m.why} (matched: {m.terms.join(', ')}). </span>)
              : <span className="dim">{d.relevance.note ?? 'No query: all three indices are shown.'} </span>}
          </div>
          <div className="row" style={{ alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
            <span className="dim" style={{ fontSize: 11 }}>Overlays:</span>
            {ORDER.map((i) => (
              <button key={i} className={`btn sm ${on.has(i) ? 'on' : ''}`} aria-pressed={on.has(i)} data-toggle={i} onClick={() => toggle(i)} disabled={!d.usable}>
                {relevant.has(i) ? '★ ' : ''}{d.layers[i].label}
              </button>
            ))}
            <label className="field" style={{ flexDirection: 'row', alignItems: 'center', gap: 8, marginLeft: 'auto' }}>Opacity {Math.round(opacity * 100)}%
              <input type="range" min={0.1} max={1} step={0.05} value={opacity} onChange={(e) => setOpacity(Number(e.target.value))} aria-label="Overlay opacity" /></label>
          </div>
          {!d.usable && <div className="warnbox" role="status">{d.unusable_reason}</div>}
          <div className="spec-grid">
            {ORDER.filter((i) => on.has(i)).map((i) => <Figure key={i} idx={i} l={d.layers[i]} tileId={tileId} opacity={opacity} relevant={relevant.has(i)} />)}
            {on.size === 0 && <div className="empty">Pick an index above to overlay it.</div>}
          </div>
          {d.classes && (
            <dl className="kv spec-classes" aria-label="Surface classes">
              {(['water_frac', 'veg_frac', 'dense_veg_frac', 'bare_frac', 'built_frac'] as const).map((k) => (
                <Fragment key={k}><dt title={d.class_rules[k]}>{k.replace('_frac', '').replace('_', ' ')} <span className="faint">({d.class_rules[k]})</span></dt><dd data-class={k}>{fmtPct(d.classes![k], 1)}</dd></Fragment>
              ))}
            </dl>
          )}
          <div className="faint" style={{ fontSize: 10.5, lineHeight: 1.5 }}>
            {d.scene_id} · {d.acquired_at} · {d.width_px}×{d.height_px} px at {d.gsd_m} m · {fmtPct(d.valid_fraction, 1)} valid pixels
            <br />{d.source}. {d.caveat}
          </div>
        </div>
      )}
    </Panel>
  );
}
