import { useMemo, type KeyboardEvent } from 'react';
import type { Timeline as TL } from '@/api/types';
import { dateMs } from '@/fmt';

const ROLE_TEXT: Record<string, string> = {
  baseline: 'baseline', before_detection: 'no change yet', first_detected: 'first detected', supported: 'still present',
  not_supported: 'not supported', no_change_seen: 'no change',
};

/**
 * Acquisition-date scrubber. Dots sit at their REAL calendar position (the dates are uneven), only dates that exist
 * in the catalog are drawn, the dashed bracket is the window in which the change must have occurred, and the
 * filled amber dot is the first acquisition that shows it.
 */
export function Timeline({ tl, active, onPick }: { tl: TL; active: string; onPick: (date: string) => void }) {
  const geo = useMemo(() => {
    const ms = tl.dates.map(dateMs);
    const lo = Math.min(...ms), hi = Math.max(...ms), span = Math.max(1, hi - lo);
    const x = (d: string) => ((dateMs(d) - lo) / span) * 100;
    return { x };
  }, [tl.dates]);

  const idx = tl.dates.indexOf(active);
  const onKey = (e: KeyboardEvent) => {
    if (e.key === 'ArrowLeft' && idx > 0) { onPick(tl.dates[idx - 1]); e.preventDefault(); }
    if (e.key === 'ArrowRight' && idx < tl.dates.length - 1) { onPick(tl.dates[idx + 1]); e.preventDefault(); }
  };
  const br = tl.first_detected?.bracket;

  return (
    <div className="tl" tabIndex={0} role="slider" aria-label="Acquisition date" aria-valuemin={0} aria-valuemax={tl.dates.length - 1}
      aria-valuenow={Math.max(0, idx)} aria-valuetext={active} onKeyDown={onKey}>
      <div className="axis" />
      {br && (
        <div className="bracket" style={{ left: `${geo.x(br[0])}%`, width: `${geo.x(br[1]) - geo.x(br[0])}%` }}>
          <span>change occurred in this window</span>
        </div>
      )}
      {tl.points.map((p) => p.interval && (
        <span key={'s' + p.date} className={`seg ${p.interval.changed ? 'chg' : ''}`}
          style={{ left: `${(geo.x(p.interval.from) + geo.x(p.interval.to)) / 2}%` }}
          title={`${p.interval.from} → ${p.interval.to}: ${p.interval.changed ? 'change flagged' : 'no change flagged'} (probability ${p.interval.probability.toFixed(3)})`}>
          {p.interval.changed ? '▲' : '·'} p{p.interval.probability.toFixed(2)}
        </span>
      ))}
      {tl.points.map((p) => (
        <span key={p.date}>
          <button className={`dot ${p.role} ${p.date === active ? 'active' : ''}`} style={{ left: `${geo.x(p.date)}%` }}
            onClick={() => onPick(p.date)} aria-label={`${p.date} — ${ROLE_TEXT[p.role]}`} title={`${p.date} — ${ROLE_TEXT[p.role]}`} />
          <span className="lbl" style={{ left: `${geo.x(p.date)}%` }}>{p.date}<small>{ROLE_TEXT[p.role]}</small></span>
        </span>
      ))}
      <i className="thumb" style={{ left: `${geo.x(active)}%` }} />
    </div>
  );
}
