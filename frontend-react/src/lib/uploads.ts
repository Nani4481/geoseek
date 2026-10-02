// Session-only holder of the GeoTIFFs dropped on the Data screen, shared with the Temporal screen's upload mode. Everything stays in
// this tab's memory: nothing is sent anywhere, nothing is written to storage, and a page reload empties it.
import { useSyncExternalStore } from 'react';
import { parseRasterHeader, type RasterHeader } from './rasterHeader';
import { decodeRaster, type RasterPixels } from './rasterPixels';

export const MAX_UPLOADS = 6;

export interface UploadDate { date: string | null; source: 'file header' | 'entered by the analyst' | null }
export interface Upload {
  id: string; name: string; header: RasterHeader | null; error: string | null;
  pixels: RasterPixels | null; decoding: boolean; pixelError: string | null;
  acquired: UploadDate;
}

let items: Upload[] = [];
let seq = 0;
const subs = new Set<() => void>();
const emit = () => { items = [...items]; subs.forEach((f) => f()); };
const patch = (id: string, p: Partial<Upload>) => { items = items.map((u) => (u.id === id ? { ...u, ...p } : u)); subs.forEach((f) => f()); };

/** The calendar date (YYYY-MM-DD) of a header acquisition hint, or null. */
export const headerDate = (h: RasterHeader | null): string | null => (h?.acquisition?.iso ? h.acquisition.iso.slice(0, 10) : null);

export async function addFiles(files: FileList | File[]): Promise<void> {
  const list = [...files].slice(0, MAX_UPLOADS);
  if (!list.length) return;
  const fresh: Upload[] = list.map((f) => ({ id: `u${++seq}`, name: f.name, header: null, error: null, pixels: null, decoding: true, pixelError: null, acquired: { date: null, source: null } }));
  items = fresh; emit();                                           // a new drop replaces the previous one
  await Promise.all(list.map(async (f, i) => {
    const id = fresh[i].id;
    try {
      const header = await parseRasterHeader(f, f.name);
      const d = headerDate(header);
      patch(id, { header, acquired: d ? { date: d, source: 'file header' } : { date: null, source: null } });
      try { patch(id, { pixels: await decodeRaster(f), decoding: false }); }
      catch (e) {
        const wasm = header.compression === 'ZSTD' || header.compression === 'LERC';     // these two decoders are WebAssembly, which the page's security policy does not allow
        patch(id, { decoding: false, pixelError: wasm
          ? `this file is ${header.compression}-compressed and that decoder needs WebAssembly, which this console's security policy does not allow (the header, footprint and archive coverage above are unaffected). Re-save it with DEFLATE or LZW compression to preview it, e.g. gdal_translate -co COMPRESS=DEFLATE in.tif out.tif`
          : e instanceof Error ? e.message : String(e) });
      }
    } catch (e) { patch(id, { error: e instanceof Error ? e.message : String(e), decoding: false }); }
  }));
}

export function setUploadDate(id: string, date: string): void {
  patch(id, { acquired: /^\d{4}-\d{2}-\d{2}$/.test(date) ? { date, source: 'entered by the analyst' } : { date: null, source: null } });
}
export function clearUploads(): void { items = []; emit(); }

export function useUploads(): Upload[] {
  return useSyncExternalStore((f) => { subs.add(f); return () => { subs.delete(f); }; }, () => items);
}
