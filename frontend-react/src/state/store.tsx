import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { api } from '@/api/client';
import type { LonLat, PresentationSummary } from '@/api/types';

export interface Bookmark { id: string; label: string; lonlat: LonLat }
/** A one-shot filter handed to the Changes queue (from the Temporal screen): applied once when Changes opens, then consumed. */
export interface ChangesFilter { firstDetected?: string; changeType?: string; persistence?: string; sensor?: string; region?: string; label: string }

interface Store {
  selectedId: string | null;
  select: (id: string | null) => void;
  bookmarks: Bookmark[];
  toggleBookmark: (b: Bookmark) => void;
  isBookmarked: (id: string) => boolean;
  analyst: string;
  setAnalyst: (s: string) => void;
  /** change-type labels + acquisition dates, fetched once from /presentation/summary */
  summary: PresentationSummary | null;
  /** set before navigating to #/changes; Changes reads it on mount and clears it */
  openInChanges: (f: ChangesFilter) => void;
  takeChangesFilter: () => ChangesFilter | null;
  label: (changeType: string | null | undefined) => string;
}

const Ctx = createContext<Store | null>(null);

// Browser storage is a per-viewer convenience only (bookmarks, analyst name); every read/write is guarded because
// it can be unavailable (private window, blocked site data).
function readLS<T>(key: string, fallback: T): T {
  try { const v = localStorage.getItem(key); return v ? (JSON.parse(v) as T) : fallback; } catch { return fallback; }
}
function writeLS(key: string, value: unknown) {
  try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* ignore */ }
}

export function StoreProvider({ children }: { children: ReactNode }) {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [bookmarks, setBookmarks] = useState<Bookmark[]>(() => readLS<Bookmark[]>('geoseek.bookmarks', []));
  const [analyst, setAnalystState] = useState<string>(() => readLS<string>('geoseek.analyst', 'analyst'));
  const [summary, setSummary] = useState<PresentationSummary | null>(null);
  const pending = useRef<ChangesFilter | null>(null);

  useEffect(() => {
    const ac = new AbortController();
    api.presentation(ac.signal).then(setSummary).catch(() => { /* labels fall back to raw codes */ });
    return () => ac.abort();
  }, []);

  const toggleBookmark = useCallback((b: Bookmark) => {
    setBookmarks((cur) => {
      const next = cur.some((x) => x.id === b.id) ? cur.filter((x) => x.id !== b.id) : [...cur, b];
      writeLS('geoseek.bookmarks', next);
      return next;
    });
  }, []);
  const openInChanges = useCallback((f: ChangesFilter) => { pending.current = f; location.hash = '#/changes'; }, []);
  const takeChangesFilter = useCallback(() => { const f = pending.current; pending.current = null; return f; }, []);
  const setAnalyst = useCallback((s: string) => { setAnalystState(s); writeLS('geoseek.analyst', s); }, []);

  const value = useMemo<Store>(() => ({
    selectedId, select: setSelectedId, bookmarks, toggleBookmark,
    isBookmarked: (id) => bookmarks.some((b) => b.id === id),
    analyst, setAnalyst, summary, openInChanges, takeChangesFilter,
    label: (t) => (t && summary?.change_type_labels[t]) || t || '—',
  }), [selectedId, bookmarks, toggleBookmark, analyst, setAnalyst, summary, openInChanges, takeChangesFilter]);

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useStore(): Store {
  const s = useContext(Ctx);
  if (!s) throw new Error('useStore outside StoreProvider');
  return s;
}
