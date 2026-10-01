import { useEffect, useState } from 'react';

export interface Route { name: string; param: string | null }

/** `#/changes/2019_2026_004510` -> { name: 'changes', param: '2019_2026_004510' } */
export function parseHash(h: string): Route {
  const parts = h.replace(/^#\/?/, '').split('/').filter(Boolean);
  return { name: parts[0] || 'dashboard', param: parts[1] ? decodeURIComponent(parts.slice(1).join('/')) : null };
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
