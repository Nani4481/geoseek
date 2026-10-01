import { useEffect, useState } from 'react';
import { api } from '@/api/client';

export type OfflineLevel = 'ok' | 'warn' | 'bad';
export interface OfflineStatus {
  level: OfflineLevel;
  label: string;
  externalRequests: string[];
  cspViolations: number;
  backendOk: boolean | null;
  hostOnline: boolean;
  checkedResources: number;
}

const isExternal = (url: string) => {
  try { const u = new URL(url, location.href); return !['data:', 'blob:', 'about:'].includes(u.protocol) && u.origin !== location.origin; } catch { return false; }
};

/**
 * The offline indicator is derived from what the browser actually did, not from a flag:
 *   - every resource the page loaded (Resource Timing) must be same-origin;
 *   - the page CSP must not have blocked anything (securitypolicyviolation);
 *   - the backend that serves this console must answer /health.
 * The host's own network-interface state is reported separately and never makes the pill green or red.
 */
export function useOfflineStatus(): OfflineStatus {
  const [external, setExternal] = useState<string[]>([]);
  const [violations, setViolations] = useState(0);
  const [backendOk, setBackendOk] = useState<boolean | null>(null);
  const [hostOnline, setHostOnline] = useState(navigator.onLine);
  const [checked, setChecked] = useState(0);

  useEffect(() => {
    const seen = (entries: PerformanceEntry[]) => {
      setChecked((n) => n + entries.length);
      const bad = entries.map((e) => e.name).filter(isExternal);
      if (bad.length) setExternal((cur) => Array.from(new Set([...cur, ...bad])));
    };
    seen(performance.getEntriesByType('resource'));
    let po: PerformanceObserver | null = null;
    try {
      po = new PerformanceObserver((list) => seen(list.getEntries()));
      po.observe({ type: 'resource', buffered: false });
    } catch { /* Resource Timing unsupported - the CSP + health checks still apply */ }

    const onViolation = () => setViolations((n) => n + 1);
    const on = () => setHostOnline(true), off = () => setHostOnline(false);
    document.addEventListener('securitypolicyviolation', onViolation);
    window.addEventListener('online', on);
    window.addEventListener('offline', off);

    let alive = true;
    const ping = () => api.health().then(() => alive && setBackendOk(true)).catch(() => alive && setBackendOk(false));
    ping();
    const t = window.setInterval(ping, 15000);
    return () => {
      alive = false; clearInterval(t); po?.disconnect();
      document.removeEventListener('securitypolicyviolation', onViolation);
      window.removeEventListener('online', on); window.removeEventListener('offline', off);
    };
  }, []);

  let level: OfflineLevel = 'ok';
  let label = 'OFFLINE · 0 EXTERNAL REQUESTS';
  if (external.length || violations) { level = 'bad'; label = `EXTERNAL REQUEST DETECTED · ${external.length + violations}`; }
  else if (backendOk === false) { level = 'warn'; label = 'BACKEND UNREACHABLE'; }
  else if (backendOk === null) { level = 'warn'; label = 'CHECKING…'; }
  return { level, label, externalRequests: external, cspViolations: violations, backendOk, hostOnline, checkedResources: checked };
}
