import { useState, type ReactNode } from 'react';
import { DASH, isNum } from '@/fmt';

export function StatCard({ label, value, unit, sub, error, loading, warn, title }: {
  label: string; value: ReactNode; unit?: string; sub?: ReactNode; error?: string | null; loading?: boolean; warn?: boolean; title?: string;
}) {
  const missing = !!error;
  return (
    <div className={`stat ${warn ? 'warn' : ''}`} title={missing ? `Unavailable: ${error}` : title}>
      <div className="label"><i style={missing ? { background: 'var(--red)' } : undefined} />{label}</div>
      {loading ? (
        <div className="skeleton" style={{ height: 30, marginTop: 7, width: '60%' }} />
      ) : (
        <div className="value">{missing ? DASH : value}{unit && !missing && <small>{unit}</small>}</div>
      )}
      <div className="sub">{missing ? 'endpoint unavailable' : sub}</div>
    </div>
  );
}

export function ConfidenceRing({ value, label = 'confidence', color }: { value: number | null | undefined; label?: string; color: string }) {
  const r = 46, c = 2 * Math.PI * r;
  const v = isNum(value) ? Math.max(0, Math.min(1, value)) : 0;
  return (
    <div className="ring" role="img" aria-label={`${label} ${isNum(value) ? Math.round(v * 100) + ' percent' : 'unavailable'}`}>
      <svg viewBox="0 0 112 112">
        <circle cx="56" cy="56" r={r} fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth="9" />
        <circle cx="56" cy="56" r={r} fill="none" stroke={color} strokeWidth="9" strokeLinecap="round" strokeDasharray={`${c * v} ${c}`} />
      </svg>
      <div className="ctr"><b>{isNum(value) ? Math.round(v * 100) : DASH}</b><small>{label}</small></div>
    </div>
  );
}

/** Diverging bar for a signed index anomaly in about [-1, 1]. */
export function DeltaBar({ label, value, hint }: { label: string; value: number | undefined; hint?: string }) {
  const v = isNum(value) ? Math.max(-1, Math.min(1, value)) : 0;
  const pos = v >= 0;
  return (
    <div title={hint} style={{ display: 'grid', gridTemplateColumns: '54px 1fr 54px', gap: 8, alignItems: 'center', fontSize: 11.5 }}>
      <span className="dim">{label}</span>
      <div className="dbar"><i style={{ left: pos ? '50%' : `${50 + v * 50}%`, width: `${Math.abs(v) * 50}%`, background: pos ? 'var(--cyan)' : 'var(--amber)' }} /></div>
      <span className="mono" style={{ textAlign: 'right' }}>{isNum(value) ? (value >= 0 ? '+' : '') + value.toFixed(2) : DASH}</span>
    </div>
  );
}

export function TileImg({ src, alt, style }: { src: string; alt: string; style?: React.CSSProperties }) {
  const [bad, setBad] = useState(false);
  if (bad) return <div className="thumb-img empty" style={{ display: 'grid', placeItems: 'center', ...style }}>no preview</div>;
  return <img className="thumb-img" src={src} alt={alt} loading="lazy" style={style} onError={() => setBad(true)} />;
}

export function Loading({ rows = 3 }: { rows?: number }) {
  return <div style={{ display: 'grid', gap: 8, padding: 4 }}>{Array.from({ length: rows }, (_, i) => <div key={i} className="skeleton" style={{ height: 16, width: `${90 - i * 12}%` }} />)}</div>;
}

export function ErrorNote({ error, onRetry }: { error: string; onRetry?: () => void }) {
  return <div className="err">Could not load: {error} {onRetry && <button className="btn sm" onClick={onRetry}>Retry</button>}</div>;
}
