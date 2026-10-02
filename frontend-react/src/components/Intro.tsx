import { useCallback, useEffect, useId, useRef, useState } from 'react';
import { MARK } from '@/components/Brand';
import { useReducedMotion } from '@/hooks/useReducedMotion';

const OFF_KEY = 'geoseek.intro.off';          // localStorage: the analyst asked never to see it again
const SEEN_KEY = 'geoseek.intro.seen';        // sessionStorage: shown once per browser session

export const INTRO_MS = 3400;                 // whole sequence including the dissolve (inside the 2-4 s brief)
export const INTRO_REDUCED_MS = 1400;         // prefers-reduced-motion: no drawing or typing, just a held frame
const DISSOLVE_MS = 500;
const TAGLINE = 'Offline archive search and change review for Earth-observation imagery.';

const get = (store: 'local' | 'session', key: string): string | null => { try { return (store === 'local' ? localStorage : sessionStorage).getItem(key); } catch { return null; } };
const set = (store: 'local' | 'session', key: string, v: string) => { try { (store === 'local' ? localStorage : sessionStorage).setItem(key, v); } catch { /* storage unavailable: nothing is remembered */ } };

/** First load of the session, not opted out, and not a deep link (the analyst asked for a place; give them it). */
export function shouldShowIntro(): boolean {
  if (get('local', OFF_KEY) === '1') return false;
  if (get('session', SEEN_KEY) === '1') return false;
  const route = location.hash.replace(/^#\/?/, '');
  return route === '' || route === 'dashboard';
}

/** The mark drawn on, stroke first (each stroke is a path with pathLength 1 that un-hides along its length), then the satellite appears. */
function DrawnMark({ size }: { size: number }) {
  const id = useId().replace(/:/g, '');
  const stroke = { pathLength: 1 } as const;
  return (
    <svg className="intro-mark" viewBox="0 0 32 32" width={size} height={size} aria-hidden="true" fill="none" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round">
      <defs>
        <mask id={`${id}-back`} maskUnits="userSpaceOnUse" x="-20" y="-20" width="40" height="40">
          <rect x="-20" y="-20" width="40" height="40" fill="#fff" stroke="none" /><circle r="11" fill="#000" stroke="none" />
        </mask>
      </defs>
      <circle className="im-draw d1" cx={MARK.globe.cx} cy={MARK.globe.cy} r={MARK.globe.r} strokeWidth="2" {...stroke} />
      <ellipse className="im-draw d2" cx={MARK.meridian.cx} cy={MARK.meridian.cy} rx={MARK.meridian.rx} ry={MARK.meridian.ry} strokeWidth="1.5" {...stroke} />
      <path className="im-draw d2" d={MARK.parallel} strokeWidth="1.5" {...stroke} />
      <g transform={`translate(16 16) rotate(${MARK.tilt})`}>
        <path className="im-draw d3" d={MARK.orbitBack} strokeWidth="1.7" mask={`url(#${id}-back)`} {...stroke} />
        <path className="im-draw d3" d={MARK.orbitFront} strokeWidth="1.7" {...stroke} />
        <g className="im-sat" transform="translate(9.6 3.9)">
          <g transform="rotate(24)"><rect x="-1.7" y="-1.7" width="3.4" height="3.4" rx="0.6" fill="currentColor" stroke="none" /><path d="M-5.6 0h2.6M3 0h2.6" strokeWidth="1.8" /></g>
        </g>
      </g>
    </svg>
  );
}

/**
 * A short opening sequence over the already-mounted app: dark field, the mark drawn stroke-first, the wordmark typed in with a
 * saffron / white / green sweep, one tagline, then a dissolve into the Dashboard underneath. It never waits for anything: the app
 * beneath is live from the first frame, the skip control is on screen from the first frame, and a click or any key ends it.
 */
export function Intro({ onDone }: { onDone: () => void }) {
  const reduced = useReducedMotion();
  const [leaving, setLeaving] = useState(false);
  const [never, setNever] = useState(false);
  const root = useRef<HTMLDivElement>(null);
  const done = useRef(false);
  const total = reduced ? INTRO_REDUCED_MS : INTRO_MS;

  const finish = useCallback((fast: boolean) => {
    if (done.current) return;
    done.current = true;
    setLeaving(true);
    window.setTimeout(onDone, fast ? 220 : DISSOLVE_MS);
  }, [onDone]);

  useEffect(() => {
    set('session', SEEN_KEY, '1');
    const t = window.setTimeout(() => finish(false), total - DISSOLVE_MS);
    return () => window.clearTimeout(t);
  }, [finish, total]);

  useEffect(() => {
    const k = (e: KeyboardEvent) => {
      if (e.key === 'Tab' || e.key === 'Shift' || e.key === 'Control' || e.key === 'Alt' || e.key === 'Meta') return;      // let a keyboard user reach the checkbox
      if ((e.target as HTMLElement | null)?.dataset?.introNever === '1') return;                                         // Space on the checkbox toggles it
      finish(true);
    };
    window.addEventListener('keydown', k);
    return () => window.removeEventListener('keydown', k);
  }, [finish]);

  const letters = 'GEOSEEK'.split('');
  return (
    <div ref={root} className={`intro ${leaving ? 'leaving' : ''} ${reduced ? 'reduced' : ''}`} role="dialog" aria-label="GeoSeek introduction" data-testid="intro" data-state={leaving ? 'leaving' : 'playing'}
      style={{ ['--intro-ms' as string]: `${total}ms` }} onClick={() => finish(true)}>
      <div className="intro-stage" aria-hidden="true">
        <DrawnMark size={150} />
        <div className="intro-word" data-testid="intro-word">{letters.map((c, i) => <span key={i} style={{ ['--i' as string]: i }}>{c}</span>)}</div>
        <div className="intro-tag" data-testid="intro-tagline">{TAGLINE}</div>
      </div>
      <div className="intro-controls" onClick={(e) => e.stopPropagation()}>
        <label className="intro-never">
          <input type="checkbox" checked={never} data-intro-never="1" data-testid="intro-never" onChange={(e) => { setNever(e.target.checked); if (e.target.checked) set('local', OFF_KEY, '1'); else { try { localStorage.removeItem(OFF_KEY); } catch { /* ignore */ } } }} />
          Don’t show this again
        </label>
        <button type="button" className="btn sm" data-testid="intro-skip" onClick={() => finish(true)}>Skip · click or press any key</button>
      </div>
    </div>
  );
}
