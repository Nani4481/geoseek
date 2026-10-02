import { useSyncExternalStore } from 'react';

const Q = '(prefers-reduced-motion: reduce)';
const mq = () => (typeof window !== 'undefined' && typeof window.matchMedia === 'function' ? window.matchMedia(Q) : null);

/** True while the OS asks for reduced motion (live: follows the setting without a reload). */
export function useReducedMotion(): boolean {
  return useSyncExternalStore(
    (cb) => { const m = mq(); m?.addEventListener('change', cb); return () => m?.removeEventListener('change', cb); },
    () => !!mq()?.matches,
    () => false,
  );
}
