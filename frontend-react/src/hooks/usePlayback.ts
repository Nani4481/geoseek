import { useEffect, useRef, useSyncExternalStore } from 'react';
import { axisPositions, stepTarget } from '@/lib/timelapse';

/** Total time for the playhead to cross the whole calendar axis (the axis is real time, so uneven gaps take longer). */
const SWEEP_MS = 6000;
const SETTLE_MS = 900;      // after the sweep reaches the last acquisition, hold, then return to the static before/after view
const STEP_MS = 1500;       // reduced motion: dwell on each acquisition
const PRELOAD_TIMEOUT_MS = 6000;

export interface PlaybackState {
  /** playhead on the 0..1 calendar axis; null = at rest (the normal before/after view) */
  p: number | null;
  playing: boolean;
  /** imagery for every acquisition has loaded (or failed / timed out), so the first frame is not blank */
  ready: boolean;
  reduced: boolean;
  /** bumps on every (re)start so consumers can restart one-shot visuals */
  run: number;
}

const prefersReduced = () => typeof window !== 'undefined' && !!window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;

/**
 * Drives the timeline playhead outside React's render cycle (a 60 fps playhead must not re-render the whole screen):
 * components subscribe to just the slice they draw via `usePlaybackState`. The controller owns no data - it is given a
 * Timeline's real acquisition dates and the image URLs to preload for them.
 */
export class PlaybackController {
  private state: PlaybackState = { p: null, playing: false, ready: false, reduced: prefersReduced(), run: 0 };
  private listeners = new Set<() => void>();
  private xs: number[] = [];
  private raf = 0;
  private last = 0;
  private settleTimer = 0;
  private stepTimer = 0;
  private token = 0;
  /** the analyst interacted before the on-load sweep began: never start it behind their back */
  private touched = false;
  private mq: MediaQueryList | null = typeof window !== 'undefined' ? window.matchMedia?.('(prefers-reduced-motion: reduce)') ?? null : null;
  private onMq = () => { this.set({ reduced: prefersReduced() }); if (prefersReduced() && this.state.playing) { this.cancelLoops(); this.scheduleStep(); } };

  /** (re)attach the media-query listener; paired with detach() so React StrictMode's mount/unmount/mount is safe */
  attach() { this.mq?.addEventListener?.('change', this.onMq); }
  detach() { this.cancelLoops(); this.token++; this.mq?.removeEventListener?.('change', this.onMq); }

  subscribe = (l: () => void) => { this.listeners.add(l); return () => { this.listeners.delete(l); }; };
  getSnapshot = () => this.state;
  get positions() { return this.xs; }

  private set(patch: Partial<PlaybackState>) {
    this.state = { ...this.state, ...patch };
    for (const l of this.listeners) l();
  }
  private cancelLoops() {
    cancelAnimationFrame(this.raf); window.clearTimeout(this.settleTimer); window.clearTimeout(this.stepTimer);
    this.raf = 0;
  }

  /** New candidate: reset, preload its imagery, then sweep once (or stay at rest under reduced motion). */
  load(dates: string[], urls: string[]) {
    this.cancelLoops();
    const token = ++this.token;
    this.touched = false;
    this.xs = axisPositions(dates);
    this.set({ p: null, playing: false, ready: false, reduced: prefersReduced() });
    if (dates.length < 2) { this.set({ ready: true }); return; }
    const done = () => {
      if (token !== this.token) return;
      if (prefersReduced() || this.touched) this.set({ ready: true });   // reduced-motion users press Play
      else this.start(0, { ready: true });                               // the on-load sweep
    };
    const timer = window.setTimeout(done, PRELOAD_TIMEOUT_MS);
    Promise.allSettled(urls.map((u) => new Promise<void>((res) => {
      const im = new Image(); im.onload = () => res(); im.onerror = () => res(); im.src = u;
    }))).then(() => { window.clearTimeout(timer); done(); });
  }

  private start(from: number, extra: Partial<PlaybackState> = {}) {
    this.cancelLoops();
    this.set({ ...extra, p: from, playing: true, run: this.state.run + 1 });
    if (this.state.reduced) { this.scheduleStep(); return; }
    this.last = performance.now();
    const tick = (now: number) => {
      const p = Math.min(1, (this.state.p ?? 0) + (now - this.last) / SWEEP_MS);
      this.last = now;
      if (p >= 1) { this.set({ p: 1, playing: false }); this.settleTimer = window.setTimeout(() => this.rest(), SETTLE_MS); return; }
      this.set({ p });
      this.raf = requestAnimationFrame(tick);
    };
    this.raf = requestAnimationFrame(tick);
  }

  private scheduleStep() {
    this.stepTimer = window.setTimeout(() => {
      const next = stepTarget(this.xs, this.state.p ?? -1, 1);
      if (next === null) { this.set({ playing: false }); this.settleTimer = window.setTimeout(() => this.rest(), SETTLE_MS); return; }
      this.set({ p: next }); this.scheduleStep();
    }, this.state.p === null || this.state.p < 0 ? 0 : STEP_MS);
  }

  play() {
    if (!this.state.ready || this.xs.length < 2) return;
    const p = this.state.p;
    if (this.state.reduced) { this.start(p === null || p >= 1 ? 0 : p); return; }
    this.start(p === null || p >= 1 ? 0 : p);
  }
  pause() { this.cancelLoops(); this.set({ playing: false }); }
  toggle() { if (this.state.playing) this.pause(); else this.play(); }
  replay() { if (this.xs.length >= 2) this.start(0); }
  /** User drag / range input: pauses and shows the cross-fade at exactly that position. */
  scrub(p: number) { this.cancelLoops(); this.set({ p: Math.max(0, Math.min(1, p)), playing: false }); }
  /** Back to the static before/after view. */
  rest() { this.touched = true; this.cancelLoops(); this.set({ p: null, playing: false }); }

}

/** A stable controller for the lifetime of the screen. */
export function usePlaybackController(): PlaybackController {
  const ref = useRef<PlaybackController | null>(null);
  if (ref.current === null) ref.current = new PlaybackController();
  useEffect(() => {
    const c = ref.current!;
    c.attach();
    return () => c.detach();
  }, []);
  return ref.current;
}

export function usePlaybackState(pb: PlaybackController): PlaybackState {
  return useSyncExternalStore(pb.subscribe, pb.getSnapshot, pb.getSnapshot);
}

/** Subscribe to a primitive derived from the playback state (re-renders only when that primitive changes). */
export function usePlaybackSelect<T extends string | number | boolean | null>(pb: PlaybackController, select: (s: PlaybackState) => T): T {
  return useSyncExternalStore(pb.subscribe, () => select(pb.getSnapshot()), () => select(pb.getSnapshot()));
}
