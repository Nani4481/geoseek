import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react';
import { api } from '@/api/client';
import type { LonLat, PresentationSummary } from '@/api/types';

export interface Bookmark { id: string; label: string; lonlat: LonLat }

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
  const setAnalyst = useCallback((s: string) => { setAnalystState(s); writeLS('geoseek.analyst', s); }, []);

  const value = useMemo<Store>(() => ({
    selectedId, select: setSelectedId, bookmarks, toggleBookmark,
    isBookmarked: (id) => bookmarks.some((b) => b.id === id),
    analyst, setAnalyst, summary,
    label: (t) => (t && summary?.change_type_labels[t]) || t || '—',
  }), [selectedId, bookmarks, toggleBookmark, analyst, setAnalyst, summary]);

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useStore(): Store {
  const s = useContext(Ctx);
  if (!s) throw new Error('useStore outside StoreProvider');
  return s;
}
