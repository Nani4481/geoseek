import { useEffect, useMemo, useRef, useState } from 'react';
import { guessRoles, type RoleGuess } from '@/lib/rasterPixels';
import { compositeRGBA, greyRGBA, stretchRange, type Role } from '@/lib/rasterMath';
import type { Upload } from '@/lib/uploads';

export type Pct = [number, number];
export const STRETCHES: { label: string; pct: Pct }[] = [
  { label: '2nd – 98th percentile (default)', pct: [2, 98] }, { label: '1st – 99th percentile', pct: [1, 99] },
  { label: '5th – 95th percentile', pct: [5, 95] }, { label: 'minimum – maximum', pct: [0, 100] },
];

export interface PreviewState { mode: 'rgb' | 'grey'; rgb: [number, number, number]; band: number; pct: Pct }

export function defaultPreview(up: Upload): PreviewState | null {
  const h = up.header;
  if (!h || !up.pixels) return null;
  const g = guessRoles(h.bands, h.bandNames);
  return g.rgb ? { mode: 'rgb', rgb: g.rgb, band: 0, pct: [2, 98] } : { mode: 'grey', rgb: [0, 0, 0], band: 0, pct: [2, 98] };
}

export const roleGuess = (up: Upload): RoleGuess => guessRoles(up.header?.bands ?? 0, up.header?.bandNames ?? []);

export function bandLabel(up: Upload, i: number): string {
  const nm = up.header?.bandNames[i];
  const role = (Object.entries(roleGuess(up).roles) as [Role, number][]).find(([, k]) => k === i)?.[0];
  return `Band ${i + 1}${nm ? ` · ${nm}` : role ? ` · ${role}` : ''}`;
}

/** The RGBA bytes of the preview in the current state, plus the stretch values used per displayed band. */
export function renderPreview(up: Upload, st: PreviewState): { rgba: Uint8ClampedArray; ranges: [number, number][]; bands: number[] } | null {
  const px = up.pixels, nodata = up.header?.nodata ?? null;
  if (!px) return null;
  if (st.mode === 'rgb') {
    const ranges = st.rgb.map((i) => stretchRange(px.bands[i], st.pct[0], st.pct[1], nodata)) as [[number, number], [number, number], [number, number]];
    return { rgba: compositeRGBA(st.rgb.map((i) => px.bands[i]) as never, ranges, nodata), ranges, bands: [...st.rgb] };
  }
  const r = stretchRange(px.bands[st.band], st.pct[0], st.pct[1], nodata);
  return { rgba: greyRGBA(px.bands[st.band], r, nodata), ranges: [r], bands: [st.band] };
}

const fmtV = (v: number) => (Math.abs(v) >= 100 ? v.toFixed(0) : v.toPrecision(4));

export function RasterPreview({ up, state, onState }: { up: Upload; state: PreviewState; onState: (s: PreviewState) => void }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const px = up.pixels, h = up.header;
  const r = useMemo(() => (px ? renderPreview(up, state) : null), [up, px, state]);
  const [drawn, setDrawn] = useState('');

  useEffect(() => {
    const c = canvas.current;
    if (!c || !px || !r) return;
    c.width = px.width; c.height = px.height;
    const g = c.getContext('2d');
    if (!g) return;
    g.putImageData(new ImageData(new Uint8ClampedArray(r.rgba), px.width, px.height), 0, 0);
    setDrawn(`${px.width}x${px.height}:${state.mode}:${r.bands.join(',')}:${state.pct.join('-')}`);
  }, [px, r, state]);

  if (!h) return null;
  if (up.decoding) return <div className="dim" style={{ fontSize: 12 }} role="status">Decoding pixels in this browser…</div>;
  if (!px) return <div className="err" style={{ padding: '6px 0' }}>The header parsed, but the pixel data could not be decoded here: {up.pixelError ?? 'unknown error'}. No preview is shown.</div>;

  const opts = Array.from({ length: h.bands }, (_, i) => i);
  const sel = (value: number, set: (i: number) => void, aria: string) => (
    <select className="input" value={value} aria-label={aria} onChange={(e) => set(Number(e.target.value))} style={{ padding: '3px 6px' }}>
      {opts.map((i) => <option key={i} value={i}>{bandLabel(up, i)}</option>)}
    </select>
  );
  const rgbOk = h.bands >= 3;
  const g = roleGuess(up);
  const labels = state.mode === 'rgb' ? ['R', 'G', 'B'] : ['grey'];
  return (
    <div className="rpreview" data-testid="raster-preview" aria-label={`Pixel preview of ${up.name}`}>
      <div className="rp-controls" role="group" aria-label="Preview controls">
        <label className="field">View
          <select className="input" value={state.mode} aria-label="Preview mode" data-testid="preview-mode" onChange={(e) => onState({ ...state, mode: e.target.value as 'rgb' | 'grey' })} style={{ padding: '3px 6px' }}>
            {rgbOk && <option value="rgb">Colour composite (3 bands)</option>}<option value="grey">Single band (grey)</option>
          </select>
        </label>
        {state.mode === 'rgb' ? (
          <>
            <label className="field">Red {sel(state.rgb[0], (i) => onState({ ...state, rgb: [i, state.rgb[1], state.rgb[2]] }), 'Band shown as red')}</label>
            <label className="field">Green {sel(state.rgb[1], (i) => onState({ ...state, rgb: [state.rgb[0], i, state.rgb[2]] }), 'Band shown as green')}</label>
            <label className="field">Blue {sel(state.rgb[2], (i) => onState({ ...state, rgb: [state.rgb[0], state.rgb[1], i] }), 'Band shown as blue')}</label>
          </>
        ) : <label className="field">Band {sel(state.band, (i) => onState({ ...state, band: i }), 'Band shown in grey')}</label>}
        <label className="field">Stretch
          <select className="input" value={`${state.pct[0]}-${state.pct[1]}`} aria-label="Stretch percentiles" data-testid="preview-stretch" onChange={(e) => onState({ ...state, pct: STRETCHES.find((s) => s.pct.join('-') === e.target.value)!.pct })} style={{ padding: '3px 6px' }}>
            {STRETCHES.map((s) => <option key={s.label} value={s.pct.join('-')}>{s.label}</option>)}
          </select>
        </label>
      </div>
      <div className="rp-stage">
        <canvas ref={canvas} data-testid="preview-canvas" data-drawn={drawn} data-mode={state.mode} data-w={px.width} data-h={px.height}
          role="img" aria-label={`${state.mode === 'rgb' ? 'Colour composite' : 'Grey band'} preview of ${up.name}`} />
      </div>
      <p className="rp-caption" data-testid="preview-caption">
        <b>Preview decoded in this browser from the file on your disk — nothing is uploaded.</b> Shown at {px.note}.
        {' '}Each displayed band is stretched linearly between its <b>{state.pct[0]}{ordinal(state.pct[0])} and {state.pct[1]}{ordinal(state.pct[1])} percentile</b> of
        its valid pixels{r ? <> (<span className="mono">{labels.map((l, i) => `${l} ${fmtV(r.ranges[i][0])}–${fmtV(r.ranges[i][1])}`).join(' · ')}</span>)</> : null}, values outside are clipped.
        {h.nodata !== null ? <> Pixels equal to the file’s NoData value ({h.nodata}) are left transparent.</> : <> The file declares no NoData value, so every pixel is treated as valid.</>}
        {' '}Band roles: {g.basis}. This is a visual aid only; no correction, reflectance scaling or co-registration is applied.
      </p>
    </div>
  );
}

function ordinal(n: number) { const t = n % 100; return t >= 11 && t <= 13 ? 'th' : ({ 1: 'st', 2: 'nd', 3: 'rd' } as Record<number, string>)[n % 10] ?? 'th'; }
