import { useEffect, useState, type ReactNode } from 'react';
import { istDate, istTime } from '@/fmt';
import { useOfflineStatus } from '@/hooks/useOfflineStatus';
import { href } from '@/router';
import type { Tier } from './Panel';

const I = (d: string) => (
  <svg viewBox="0 0 24 24" aria-hidden="true"><path d={d} /></svg>
);
const ICONS = {
  dashboard: I('M3 3h8v8H3zM13 3h8v5h-8zM13 10h8v11h-8zM3 13h8v8H3z'),
  search: I('M11 4a7 7 0 1 0 0 14 7 7 0 0 0 0-14zM21 21l-5-5'),
  changes: I('M4 7l8-4 8 4-8 4zM4 12l8 4 8-4M4 17l8 4 8-4'),
  detect: I('M4 8V5a1 1 0 0 1 1-1h3M16 4h3a1 1 0 0 1 1 1v3M20 16v3a1 1 0 0 1-1 1h-3M8 20H5a1 1 0 0 1-1-1v-3M9 9h6v6H9z'),
  discovery: I('M12 5a2 2 0 1 0 0 .01M5 17a2 2 0 1 0 0 .01M19 17a2 2 0 1 0 0 .01M12 7v4M10.5 12.5L6 15.5M13.5 12.5L18 15.5'),
  fingerprints: I('M6 18c0-3 2-5 2-8a4 4 0 0 1 8 0c0 3-1 5-1 8M9 20c0-3 1-5 1-8a2 2 0 0 1 4 0c0 4-1 6-1 8M4 13c0-5 3-9 8-9s8 4 8 9'),
  briefing: I('M3 4h18v12H3zM8 20h8M12 16v4M7 12l3-3 3 2 4-4'),
  roadmap: I('M5 21V4M5 4h11l-2 4 2 4H5'),
};

interface NavItem { id: string; label: string; tier: Tier; icon: keyof typeof ICONS }
export const NAV: NavItem[] = [
  { id: 'dashboard', label: 'Dashboard', tier: 'live', icon: 'dashboard' },
  { id: 'search', label: 'Search', tier: 'live', icon: 'search' },
  { id: 'changes', label: 'Changes', tier: 'live', icon: 'changes' },
  { id: 'detect', label: 'Detect', tier: 'live', icon: 'detect' },
  { id: 'discovery', label: 'Discover', tier: 'live', icon: 'discovery' },
  { id: 'fingerprints', label: 'Similar', tier: 'surfaced', icon: 'fingerprints' },
  { id: 'briefing', label: 'Brief', tier: 'surfaced', icon: 'briefing' },
  { id: 'roadmap', label: 'Roadmap', tier: 'roadmap', icon: 'roadmap' },
];

export function Rail({ active }: { active: string }) {
  return (
    <nav className="rail" aria-label="Primary">
      <div className="logo" aria-hidden="true">GS</div>
      {NAV.map((n, i) => (
        <span key={n.id} style={{ display: 'contents' }}>
          {n.id === 'roadmap' && <div className="sep" />}
          <a href={href(n.id)} className={active === n.id ? 'active' : ''} title={`${n.label} — ${n.tier}`} aria-current={active === n.id ? 'page' : undefined}>
            {ICONS[n.icon]}
            <span>{n.label}</span>
            <i className={`tier-dot ${n.tier}`} />
          </a>
          {i === 4 && <div className="sep" />}
        </span>
      ))}
      <div className="legend">
        <div><i style={{ background: 'var(--green)' }} />Live</div>
        <div><i style={{ background: 'var(--cyan)' }} />Existing backend</div>
        <div><i style={{ border: '1.5px dashed var(--amber)' }} />Roadmap</div>
      </div>
    </nav>
  );
}

function Clock() {
  const [now, setNow] = useState(() => new Date());
  useEffect(() => { const t = window.setInterval(() => setNow(new Date()), 1000); return () => clearInterval(t); }, []);
  return <div className="clock" aria-label="Indian Standard Time">{istTime(now)}<small>IST · {istDate(now)}</small></div>;
}

export function OfflinePill() {
  const s = useOfflineStatus();
  const tip = [
    `Resources loaded this session: ${s.checkedResources}`,
    s.externalRequests.length ? `EXTERNAL: ${s.externalRequests.join(', ')}` : 'External requests observed: 0',
    `CSP violations: ${s.cspViolations}`,
    `Backend: ${s.backendOk === null ? 'checking' : s.backendOk ? 'reachable' : 'unreachable'}`,
    `Host network interface: ${s.hostOnline ? 'up' : 'down'} (informational — this app never uses it)`,
  ].join('\n');
  return <span className={`offline-pill ${s.level}`} title={tip} role="status"><i />{s.label}</span>;
}

export function TopBar({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <header className="topbar">
      <h1>GeoSeek</h1>
      <span className="sub">{title}</span>
      <span className="spacer" />
      {children}
      <OfflinePill />
      <Clock />
    </header>
  );
}
