import { useEffect, useState } from 'react';

/** `raw` is the hash exactly as typed (without the leading `#/`), so a screen that does not exist can say what it was asked for. */
export interface Route { name: string; param: string | null; raw: string }

/** Other spellings of a screen. `#/brief` is what the rail label, the requirements document and the demo script call it. */
export const ALIASES: Record<string, string> = { brief: 'briefing' };

const safeDecode = (s: string) => { try { return decodeURIComponent(s); } catch { return s; } };

/** `#/changes/2019_2026_004510` -> { name: 'changes', param: '2019_2026_004510' } */
export function parseHash(h: string): Route {
  const parts = h.replace(/^#\/?/, '').split('/').filter(Boolean);
  const first = parts[0] || 'dashboard';          // an EMPTY hash is the home screen; a non-empty one is never silently replaced
  return { name: Object.prototype.hasOwnProperty.call(ALIASES, first) ? ALIASES[first] : first, param: parts[1] ? safeDecode(parts.slice(1).join('/')) : null, raw: parts.join('/') };
}

export function useRoute(): Route {
  const [route, setRoute] = useState<Route>(() => parseHash(location.hash));
  useEffect(() => {
    const on = () => setRoute(parseHash(location.hash));
    window.addEventListener('hashchange', on);
    return () => window.removeEventListener('hashchange', on);
  }, []);
  return route;
}

export const go = (name: string, param?: string | null) => {
  location.hash = `#/${name}${param ? '/' + encodeURIComponent(param) : ''}`;
};
export const href = (name: string, param?: string | null) => `#/${name}${param ? '/' + encodeURIComponent(param) : ''}`;
