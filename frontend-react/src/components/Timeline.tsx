import { useMemo, useRef, type KeyboardEvent, type PointerEvent } from 'react';
import type { Timeline as TL } from '@/api/types';
import { usePlaybackState, type PlaybackController } from '@/hooks/usePlayback';
import { axisPositions, drawnSpan, indexAt, isLit } from '@/lib/timelapse';

const ROLE_TEXT: Record<string, string> = {
  baseline: 'baseline', before_detection: 'no change yet', first_detected: 'first detected', supported: 'still present',
  not_supported: 'not supported', no_change_seen: 'no change',
};

/**
 * Acquisition-date scrubber. Dots sit at their REAL calendar position (the dates are uneven), only dates that exist
 * in the catalog are drawn, the dashed bracket is the window in which the change must have occurred, and the
 * filled amber dot is the first acquisition that shows it.
 *
 * While the playback controller is running (or scrubbed) a playhead sweeps the axis: each observation node lights up as
 * the playhead passes it, and the bracket is drawn in by the playhead rather than appearing at once. At rest (p = null)
 * the static view is exactly what it always was. Playback is presentation only - it moves along real dates, it adds none.
 */
export function Timeline({ tl, active, onPick, pb }: { tl: TL; active: string; onPick: (date: string) => void; pb: PlaybackController }) {
  const st = usePlaybackState(pb);
  const xs = useMemo(() => axisPositions(tl.dates), [tl.dates]);
  const xOf = (d: string) => (xs[tl.dates.indexOf(d)] ?? 0) * 100;
  const track = useRef<HTMLDivElement>(null);

  const idx = tl.dates.indexOf(active);
  const onKey = (e: KeyboardEvent) => {
    if (e.key === 'ArrowLeft' && idx > 0) { onPick(tl.dates[idx - 1]); e.preventDefault(); }
    if (e.key === 'ArrowRight' && idx < tl.dates.length - 1) { onPick(tl.dates[idx + 1]); e.preventDefault(); }
  };
  const br = tl.first_detected?.bracket;
  const p = st.p;
  const playing = p !== null;

  const scrubFromPointer = (e: PointerEvent<HTMLDivElement>) => {
    const r = track.current?.getBoundingClientRect();
    if (r && r.width > 0) pb.scrub((e.clientX - r.left) / r.width);
  };

  // the bracket, drawn in by the playhead while playing; whole at rest
  const bracket = (() => {
    if (!br) return null;
    const x0 = xOf(br[0]), x1 = xOf(br[1]);
    if (!playing) return { left: x0, width: x1 - x0, full: true };
    const s = drawnSpan(x0 / 100, x1 / 100, p);
    return { left: s.left * 100, width: s.width * 100, full: s.width * 100 >= x1 - x0 - 1e-6 };
  })();

  return (
    <div className="tl" tabIndex={0} role="slider" aria-label="Acquisition date" aria-valuemin={0} aria-valuemax={tl.dates.length - 1}
      aria-valuenow={Math.max(0, idx)} aria-valuetext={active} onKeyDown={onKey} data-playing={playing ? '1' : '0'}>
      <div className="axis" />
      <div ref={track} className="tl-scrub" aria-hidden="true" title="Drag to scrub through the acquisitions"
        onPointerDown={(e) => { e.currentTarget.setPointerCapture(e.pointerId); scrubFromPointer(e); }}
        onPointerMove={(e) => { if (e.buttons) scrubFromPointer(e); }} />
      {bracket && (
        <div className={`bracket ${bracket.full ? 'full' : 'drawing'}`} data-drawn={bracket.width.toFixed(2)}
          style={{ left: `${bracket.left}%`, width: `${bracket.width}%` }}>
          <span>change occurred in this window</span>
        </div>
      )}
      {tl.points.map((pt, i) => pt.interval && (
        <span key={'s' + pt.date} className={`seg ${pt.interval.changed ? 'chg' : ''} ${playing && !isLit(xs[i], p) ? 'unlit' : ''}`}
          style={{ left: `${(xOf(pt.interval.from) + xOf(pt.interval.to)) / 2}%` }}
          title={`${pt.interval.from} → ${pt.interval.to}: ${pt.interval.changed ? 'change flagged' : 'no change flagged'} (probability ${pt.interval.probability.toFixed(3)})`}>
          {pt.interval.changed ? '▲' : '·'} p{pt.interval.probability.toFixed(2)}
        </span>
      ))}
      {tl.points.map((pt, i) => {
        const lit = !playing || isLit(xs[i], p);
        return (
          <span key={pt.date}>
            <button className={`dot ${pt.role} ${pt.date === active ? 'active' : ''} ${lit ? 'lit' : 'unlit'}`} style={{ left: `${xOf(pt.date)}%` }}
              onClick={() => { pb.rest(); onPick(pt.date); }} aria-label={`${pt.date} — ${ROLE_TEXT[pt.role]}`} title={`${pt.date} — ${ROLE_TEXT[pt.role]}`} data-lit={lit ? '1' : '0'} />
            <span className={`lbl ${lit ? '' : 'unlit'}`} style={{ left: `${xOf(pt.date)}%` }}>{pt.date}<small>{ROLE_TEXT[pt.role]}</small></span>
          </span>
        );
      })}
      {playing
        ? <i className="playhead" style={{ left: `${p * 100}%` }} data-p={p.toFixed(4)}><b className="mono">{tl.dates[Math.max(0, indexAt(xs, p))]}</b></i>
        : <i className="thumb" style={{ left: `${xOf(active)}%` }} />}
    </div>
  );
}
