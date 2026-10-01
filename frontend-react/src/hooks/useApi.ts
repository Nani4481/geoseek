import { useCallback, useEffect, useRef, useState } from 'react';

export interface ApiState<T> { data: T | null; error: string | null; loading: boolean; reload: () => void }

/**
 * Run an async fetcher whenever `deps` change; aborts the in-flight request on change/unmount. `null` data + a
 * visible error is the honest failure mode - there is deliberately no fallback value.
 */
export function useApi<T>(fetcher: ((signal: AbortSignal) => Promise<T>) | null, deps: unknown[]): ApiState<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(fetcher !== null);
  const [tick, setTick] = useState(0);
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;
  const reload = useCallback(() => setTick((t) => t + 1), []);

  useEffect(() => {
    const f = fetcherRef.current;
    if (!f) { setData(null); setError(null); setLoading(false); return; }
    const ac = new AbortController();
    setLoading(true); setError(null);
    f(ac.signal)
      .then((d) => { if (!ac.signal.aborted) { setData(d); setLoading(false); } })
      .catch((e: unknown) => {
        if (ac.signal.aborted || (e instanceof DOMException && e.name === 'AbortError')) return;
        setError(e instanceof Error ? e.message : String(e)); setData(null); setLoading(false);
      });
    return () => ac.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, tick]);

  return { data, error, loading, reload };
}
