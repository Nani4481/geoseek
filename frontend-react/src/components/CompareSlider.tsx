import { useRef, useState } from 'react';

interface Props {
  before: string;
  after: string;
  beforeLabel: string;
  afterLabel: string;
}

const clamp = (v: number) => Math.max(0, Math.min(100, v));

/** Before/after reveal. Left of the handle = BEFORE, right = AFTER. Mouse, touch/pen (pointer events) and
 *  ArrowLeft/ArrowRight (Shift = 10 pts) all drive it. */
export function CompareSlider({ before, after, beforeLabel, afterLabel }: Props) {
  const [pos, setPos] = useState(50);
  const [failed, setFailed] = useState<string | null>(null);
  const ref = useRef<HTMLDivElement>(null);
  const move = (clientX: number) => {
    const r = ref.current?.getBoundingClientRect();
    if (r && r.width > 0) setPos(clamp(((clientX - r.left) / r.width) * 100));
  };
  return (
    <div
      ref={ref}
      className="compare"
      role="slider"
      tabIndex={0}
      aria-label="Before and after comparison"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(pos)}
      aria-valuetext={`${Math.round(pos)}% before`}
      onPointerDown={(e) => { e.currentTarget.setPointerCapture(e.pointerId); move(e.clientX); }}
      onPointerMove={(e) => { if (e.buttons || e.pointerType === 'touch') move(e.clientX); }}
      onKeyDown={(e) => {
        const step = e.shiftKey ? 10 : 2;
        if (e.key === 'ArrowLeft') { setPos((p) => clamp(p - step)); e.preventDefault(); }
        if (e.key === 'ArrowRight') { setPos((p) => clamp(p + step)); e.preventDefault(); }
      }}
    >
      <img src={before} alt={`Before, ${beforeLabel}`} draggable={false} onError={() => setFailed(before)} onLoad={() => setFailed(null)} />
      <div className="after-clip" style={{ clipPath: `inset(0 0 0 ${pos}%)` }}>
        <img src={after} alt={`After, ${afterLabel}`} draggable={false} onError={() => setFailed(after)} />
      </div>
      <div className="handle" style={{ left: `${pos}%` }} />
      <span className="tag l">{beforeLabel}</span>
      <span className="tag r">{afterLabel}</span>
      {failed && <div className="empty" style={{ position: 'absolute', inset: 0, display: 'grid', placeItems: 'center', background: '#02050f' }}>Imagery unavailable for this date</div>}
    </div>
  );
}
