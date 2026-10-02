import { useId } from 'react';
import { useReducedMotion } from '@/hooks/useReducedMotion';

/** Geometry of the mark in one place, so the rail logo and the intro draw exactly the same artwork. */
export const MARK = {
  globe: { cx: 16, cy: 16, r: 8.4 },
  meridian: { cx: 16, cy: 16, rx: 3.6, ry: 8.4 },
  parallel: 'M7.6 16h16.8',
  orbitBack: 'M-14.2 0A14.2 5.3 0 0 1 14.2 0',
  orbitFront: 'M-14.2 0A14.2 5.3 0 0 0 14.2 0',
  /** one full lap of the orbit: out along the front (below the globe's centre line), back along the far side */
  lap: 'M-14.2 0A14.2 5.3 0 0 0 14.2 0A14.2 5.3 0 0 0 -14.2 0',
  tilt: -24,
} as const;

/** One full lap of the satellite takes this long: slow enough to read as a held, living mark on a screen that is never closed. */
export const ORBIT_SECONDS = 48;

function Satellite() {
  return (
    <g transform="rotate(24)">
      <rect x="-1.7" y="-1.7" width="3.4" height="3.4" rx="0.6" fill="currentColor" stroke="none" />
      <path d="M-5.6 0h2.6M3 0h2.6" strokeWidth="1.8" />
    </g>
  );
}

/**
 * GeoSeek mark: a globe (one meridian, one parallel - read together they are also a sighting cross) circled by the track of an
 * imaging satellite that passes in front of the globe and behind it. Drawn with `currentColor` only, with no fills that depend on a
 * second colour, so it reads identically in one ink (white on the rail, black on paper, or a single-colour print). Original artwork;
 * it is not derived from any agency, military or government emblem.
 *
 * `animated`: the satellite flies one slow lap of its orbit (SMIL in the inline SVG - no asset, no script), in front of the globe on the
 * near half of the track and hidden behind it on the far half. Under prefers-reduced-motion it is not animated at all.
 */
export function LogoMark({ size = 32, title = 'GeoSeek', animated = false }: { size?: number; title?: string; animated?: boolean }) {
  const id = useId().replace(/:/g, '');
  const still = useReducedMotion();
  const fly = animated && !still;
  const lap = (
    <animateMotion path={MARK.lap} dur={`${ORBIT_SECONDS}s`} begin={`-${(ORBIT_SECONDS * 0.368).toFixed(2)}s`} repeatCount="indefinite" rotate="0" />
  );
  return (
    <svg className={`logo-mark${fly ? ' orbiting' : ''}`} viewBox="0 0 32 32" width={size} height={size} role="img" aria-label={title} fill="none" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" data-orbit={fly ? 'running' : 'still'}>
      <title>{title}</title>
      <defs>
        {/* the half of the orbit that lies behind the globe is cut where it crosses the disc */}
        <mask id={`${id}-back`} maskUnits="userSpaceOnUse" x="-20" y="-20" width="40" height="40">
          <rect x="-20" y="-20" width="40" height="40" fill="#fff" stroke="none" />
          <circle r="11" fill="#000" stroke="none" />
        </mask>
        {/* the satellite on the far half of its lap: only the far side of the track, and never over the globe */}
        <mask id={`${id}-sat-back`} maskUnits="userSpaceOnUse" x="-20" y="-20" width="40" height="40">
          <rect x="-20" y="-20" width="40" height="20" fill="#fff" stroke="none" />
          <circle r="11" fill="#000" stroke="none" />
        </mask>
        <clipPath id={`${id}-sat-front`} clipPathUnits="userSpaceOnUse"><rect x="-20" y="0" width="40" height="20" /></clipPath>
      </defs>
      <circle cx={MARK.globe.cx} cy={MARK.globe.cy} r={MARK.globe.r} strokeWidth="2" />
      <ellipse cx={MARK.meridian.cx} cy={MARK.meridian.cy} rx={MARK.meridian.rx} ry={MARK.meridian.ry} strokeWidth="1.5" />
      <path d={MARK.parallel} strokeWidth="1.5" />
      <g transform={`translate(16 16) rotate(${MARK.tilt})`}>
        <path d={MARK.orbitBack} strokeWidth="1.7" mask={`url(#${id}-back)`} />
        <path d={MARK.orbitFront} strokeWidth="1.7" />
        {fly ? (
          <>
            <g mask={`url(#${id}-sat-back)`}><g data-testid="logo-sat-back">{lap}<Satellite /></g></g>
            <g clipPath={`url(#${id}-sat-front)`}><g data-testid="logo-sat-front">{lap}<Satellite /></g></g>
          </>
        ) : (
          /* the satellite: a body with two panels, on the front half of the track */
          <g transform="translate(9.6 3.9)"><Satellite /></g>
        )}
      </g>
    </svg>
  );
}

/** The national flag of India, 3:2: saffron / white / green bands, and the Ashoka Chakra (24 spokes) in navy, its diameter 3/4 of the white band. */
const SAFFRON = '#FF9933', GREEN = '#138808', NAVY = '#000080';
const SPOKES = Array.from({ length: 24 }, (_, i) => i * 15);
export function IndiaFlag({ height = 20 }: { height?: number }) {
  const R = 3;                                    // chakra radius: 0.75 * 8 / 2 (band height 8 in a 36 x 24 box)
  return (
    <svg className="india-flag" viewBox="0 0 36 24" width={(height * 3) / 2} height={height} role="img" aria-label="Flag of India" data-testid="india-flag">
      <title>India</title>
      <rect width="36" height="8" y="0" fill={SAFFRON} />
      <rect width="36" height="8" y="8" fill="#FFFFFF" />
      <rect width="36" height="8" y="16" fill={GREEN} />
      <g transform="translate(18 12)" stroke={NAVY} fill="none" strokeLinecap="butt">
        <circle r={R - 0.12} strokeWidth="0.3" />
        {SPOKES.map((deg) => <line key={deg} x1="0" y1="0" x2="0" y2={-(R - 0.12)} strokeWidth="0.17" transform={`rotate(${deg})`} />)}
        <circle r="0.55" fill={NAVY} stroke="none" />
      </g>
    </svg>
  );
}
