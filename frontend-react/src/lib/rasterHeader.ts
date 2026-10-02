// In-browser GeoTIFF *header* parse (geotiff.js, bundled - nothing is fetched). Only the file directory and GeoKeys are read here;
// pixels are decoded separately, on request, by rasterPixels.ts. Nothing is uploaded, nothing is ingested. The result describes the
// file as written; the console never claims more than that. tests/test_react_lib_selftest.py cross-checks every field against rasterio.
import { fromArrayBuffer, fromBlob } from 'geotiff';
import { fromUTM, utmFromEpsg } from './geo.ts';

export type Affine = [number, number, number, number, number, number]; // GDAL order: a b c d e f  (x = a*col + b*row + c; y = d*col + e*row + f)

export interface AcquisitionHint { source: string; raw: string; iso: string | null }
export interface RasterHeader {
  fileName: string;
  sizeBytes: number | null;
  width: number; height: number; bands: number;
  dtype: string; bitsPerSample: number;
  crs: { kind: 'projected' | 'geographic' | 'unknown'; epsg: number | null; name: string | null };
  affine: Affine | null;
  rotated: boolean;
  /** native-CRS bounds [minx, miny, maxx, maxy] from the four corners (so a rotated grid is still bounded) */
  bounds: [number, number, number, number] | null;
  resolution: [number, number] | null;            // |pixel width|, |pixel height| in CRS units
  lonlatBounds: [number, number, number, number] | null;
  /** the four footprint corners as lon/lat, in order NW, NE, SE, SW of the file grid (so a rotated or projected footprint is drawn as it lies) */
  lonlatCorners: [number, number][] | null;
  /** per-band DESCRIPTION from the GDAL metadata (e.g. B04), null where the file has none */
  bandNames: (string | null)[];
  lonlatNote: string | null;                      // why lon/lat bounds are absent, when they are
  nodata: number | null;
  tiled: boolean; overviews: number; compression: string | null;
  /** an acquisition-like item from GDAL metadata (never the TIFF DateTime tag, which is when the file was written) */
  acquisition: AcquisitionHint | null;
  fileTimestamp: AcquisitionHint | null;          // TIFF DateTime / TIFFTAG_DATETIME
  metadata: Record<string, string>;
  checks: { id: string; ok: boolean; label: string; detail: string }[];
}

const COMPRESSION: Record<number, string> = { 1: 'none', 5: 'LZW', 7: 'JPEG', 8: 'Deflate', 32946: 'Deflate', 34925: 'LZMA', 50000: 'ZSTD', 34887: 'LERC', 32773: 'PackBits' };
const ACQ_KEYS = ['ACQUISITION_DATE', 'ACQUISITIONDATE', 'ACQUISITION_TIME', 'SENSING_TIME', 'SENSINGTIME', 'SENSING_START', 'DATATAKE_SENSING_START',
  'PRODUCT_START_TIME', 'FIRST_LINE_DATE_TIME', 'DATE_ACQUIRED', 'ACQUIRED', 'DATETIME_ACQUIRED', 'ACQUISITION'];

function toIso(raw: string): string | null {
  const s = raw.trim().replace(/^(\d{4}):(\d{2}):(\d{2})/, '$1-$2-$3').replace(' ', 'T');
  const t = Date.parse(/(?:Z|[+-]\d{2}:?\d{2})$/.test(s) || !/T/.test(s) ? s : s + 'Z');
  return Number.isNaN(t) ? null : new Date(t).toISOString();
}

function crsName(kind: string, epsg: number | null): string | null {
  if (epsg === null) return null;
  if (epsg === 4326) return 'WGS 84';
  if (epsg === 3857) return 'WGS 84 / Pseudo-Mercator';
  const u = utmFromEpsg(epsg);
  return u ? `WGS 84 / UTM zone ${u.zone}${u.north ? 'N' : 'S'}` : kind === 'geographic' ? 'geographic CRS' : null;
}

/** Corner-by-corner lon/lat for the CRSs this console can invert offline (WGS 84, Web Mercator, WGS 84 UTM). */
function toLonLat(epsg: number | null, pts: [number, number][]): { ll: [number, number][] | null; note: string | null } {
  if (epsg === null) return { ll: null, note: 'no EPSG code in the file, so its coordinates cannot be placed on the globe' };
  if (epsg === 4326) return { ll: pts, note: null };
  if (epsg === 3857) {
    const R = 6378137;
    return { ll: pts.map(([x, y]) => [(x / R) * (180 / Math.PI), (2 * Math.atan(Math.exp(y / R)) - Math.PI / 2) * (180 / Math.PI)] as [number, number]), note: null };
  }
  const u = utmFromEpsg(epsg);
  if (u) return { ll: pts.map(([x, y]) => { const r = fromUTM(u.zone, u.north, x, y); return [r.lon, r.lat] as [number, number]; }), note: null };
  return { ll: null, note: `EPSG:${epsg} reprojection is not bundled; bounds are shown in the file's own CRS only` };
}

export async function parseRasterHeader(src: Blob | ArrayBuffer, fileName = 'file'): Promise<RasterHeader> {
  const tiff = src instanceof ArrayBuffer ? await fromArrayBuffer(src) : await fromBlob(src);
  const img = await tiff.getImage(0);
  const ifd = img.getFileDirectory();
  // geotiff.js 3 loads large tags lazily: pull exactly the ones this parse needs (still header bytes only, never pixel data)
  const tag = async (name: string): Promise<unknown> => { try { return ifd.hasTag(name as never) ? await ifd.loadValue(name as never) : undefined; } catch { return undefined; } };
  const fd: Record<string, unknown> = {};
  for (const t of ['ModelTiepoint', 'ModelPixelScale', 'ModelTransformation', 'GeoKeyDirectory', 'GeoDoubleParams', 'GeoAsciiParams', 'DateTime',
    'BitsPerSample', 'SampleFormat', 'TileWidth', 'Compression']) fd[t] = await tag(t);
  const gk = (img.getGeoKeys() ?? {}) as Record<string, number>;
  const width = img.getWidth(), height = img.getHeight(), bands = img.getSamplesPerPixel();

  // --- affine transform: ModelTransformation, or tiepoint + pixel scale (GeoTIFF spec); PixelIsPoint shifts by half a pixel like GDAL
  let affine: Affine | null = null;
  const mt = fd.ModelTransformation as ArrayLike<number> | undefined, tp = fd.ModelTiepoint as ArrayLike<number> | undefined, sc = fd.ModelPixelScale as ArrayLike<number> | undefined;
  if (mt && mt.length >= 16) affine = [mt[0], mt[1], mt[3], mt[4], mt[5], mt[7]];
  else if (tp && tp.length >= 6 && sc && sc.length >= 2) affine = [sc[0], 0, tp[3] - tp[0] * sc[0], 0, -sc[1], tp[4] + tp[1] * sc[1]];
  // GDAL writes an identity matrix for a file that is not georeferenced; with no CRS that is pixel space, not a place on Earth
  const identity = !!affine && affine.every((v, i) => v === [1, 0, 0, 0, 1, 0][i]);
  if (identity && Object.keys(gk).length === 0) affine = null;
  if (affine && gk.GTRasterTypeGeoKey === 2) { affine[2] -= (affine[0] + affine[1]) / 2; affine[5] -= (affine[3] + affine[4]) / 2; }
  const rotated = !!affine && (Math.abs(affine[1]) > 1e-12 * Math.abs(affine[0]) || Math.abs(affine[3]) > 1e-12 * Math.abs(affine[4]));

  // --- CRS
  const projected = Number.isInteger(gk.ProjectedCSTypeGeoKey) && gk.ProjectedCSTypeGeoKey !== 32767 ? gk.ProjectedCSTypeGeoKey : null;
  const geographic = Number.isInteger(gk.GeographicTypeGeoKey) && gk.GeographicTypeGeoKey !== 32767 ? gk.GeographicTypeGeoKey : null;
  const hasKeys = Object.keys(gk).length > 0;
  const modelType = gk.GTModelTypeGeoKey;                                    // 1 projected, 2 geographic
  const epsg = modelType === 2 ? geographic : projected ?? (modelType === undefined ? geographic : null);
  const kind: RasterHeader['crs']['kind'] = !hasKeys ? 'unknown' : modelType === 2 || (modelType === undefined && geographic !== null) ? 'geographic' : modelType === 1 ? 'projected' : 'unknown';
  const crs = { kind, epsg, name: crsName(kind, epsg) };

  // --- bounds / resolution from the corners
  let bounds: RasterHeader['bounds'] = null, resolution: RasterHeader['resolution'] = null, corners: [number, number][] = [];
  if (affine) {
    const [a, b, c, d, e, f] = affine;
    corners = ([[0, 0], [width, 0], [width, height], [0, height]] as [number, number][]).map(([col, row]) => [a * col + b * row + c, d * col + e * row + f] as [number, number]);
    const xs = corners.map((p) => p[0]), ys = corners.map((p) => p[1]);
    bounds = [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
    resolution = [Math.hypot(a, d), Math.hypot(b, e)];
  }
  const ll = corners.length ? toLonLat(epsg, corners) : { ll: null, note: 'no geotransform in the file' };
  const lonlatCorners = ll.ll ?? null;
  const lonlatBounds: RasterHeader['lonlatBounds'] = ll.ll ? [Math.min(...ll.ll.map((p) => p[0])), Math.min(...ll.ll.map((p) => p[1])), Math.max(...ll.ll.map((p) => p[0])), Math.max(...ll.ll.map((p) => p[1]))] : null;

  // --- sample type
  const bps = (fd.BitsPerSample as ArrayLike<number> | number | undefined);
  const bitsPerSample = Number(typeof bps === 'object' && bps !== null ? bps[0] : bps ?? 0);
  const sf = (fd.SampleFormat as ArrayLike<number> | number | undefined);
  const fmt = Number(typeof sf === 'object' && sf !== null ? sf[0] : sf ?? 1);
  const dtype = `${fmt === 3 ? 'float' : fmt === 2 ? 'int' : 'uint'}${bitsPerSample}`;

  // --- metadata & acquisition hints
  const md: Record<string, string> = {};
  try {
    const g = ((await img.getGDALMetadata()) ?? {}) as Record<string, unknown>;
    for (const [k, v] of Object.entries(g)) if (typeof v === 'string') md[k] = v;
  } catch { /* malformed GDAL metadata is simply not shown */ }
  const key = (k: string) => Object.keys(md).find((m) => m.toUpperCase() === k);
  const acqKey = ACQ_KEYS.map(key).find(Boolean);
  const acquisition: AcquisitionHint | null = acqKey ? { source: `GDAL metadata ${acqKey}`, raw: md[acqKey], iso: toIso(md[acqKey]) } : null;
  const dt = typeof fd.DateTime === 'string' ? fd.DateTime : md[key('TIFFTAG_DATETIME') ?? ''] ?? null;
  const fileTimestamp: AcquisitionHint | null = dt ? { source: 'TIFF DateTime tag', raw: dt, iso: toIso(dt) } : null;

  const bandNames: (string | null)[] = [];
  for (let i = 0; i < bands; i++) {
    try { const g = (await img.getGDALMetadata(i)) as Record<string, unknown> | null; const d = g?.DESCRIPTION; bandNames.push(typeof d === 'string' && d.trim() ? d.trim() : null); } catch { bandNames.push(null); }
  }

  let nodata: number | null = null;
  try { const n = img.getGDALNoData(); nodata = typeof n === 'number' && Number.isFinite(n) ? n : null; } catch { /* absent */ }
  const comp = Number(fd.Compression);
  const sizeBytes = src instanceof ArrayBuffer ? src.byteLength : (src as Blob).size ?? null;

  const squareOk = !!resolution && Math.abs(resolution[0] - resolution[1]) / Math.max(resolution[0], 1e-12) < 1e-3;
  const checks = [
    { id: 'crs', ok: epsg !== null, label: 'Coordinate reference system', detail: epsg !== null ? `EPSG:${epsg}${crs.name ? ' · ' + crs.name : ''}` : hasKeys ? 'GeoKeys present but no EPSG code (user-defined CRS)' : 'no GeoKeys: the file is not georeferenced' },
    { id: 'transform', ok: !!affine, label: 'Affine geotransform', detail: affine ? `origin (${affine[2].toFixed(3)}, ${affine[5].toFixed(3)}), pixel (${affine[0].toPrecision(8)}, ${affine[4].toPrecision(8)})` : 'no ModelTiepoint / ModelPixelScale / ModelTransformation' },
    { id: 'north_up', ok: !!affine && !rotated, label: 'North-up, axis-aligned grid', detail: !affine ? 'unknown' : rotated ? 'the grid is rotated or sheared' : 'no rotation' },
    { id: 'square', ok: squareOk, label: 'Square pixels', detail: resolution ? `${resolution[0].toPrecision(6)} × ${resolution[1].toPrecision(6)}` : 'unknown' },
    { id: 'lonlat', ok: !!lonlatBounds, label: 'Footprint placeable on the globe', detail: lonlatBounds ? lonlatBounds.map((v) => v.toFixed(4)).join(', ') : ll.note ?? 'unknown' },
  ];
  return {
    fileName, sizeBytes, width, height, bands, dtype, bitsPerSample, crs, affine, rotated, bounds, resolution, lonlatBounds, lonlatCorners, bandNames, lonlatNote: ll.note,
    nodata, tiled: !!fd.TileWidth, overviews: Math.max(0, (await tiff.getImageCount()) - 1), compression: COMPRESSION[comp] ?? (Number.isFinite(comp) ? `code ${comp}` : null),
    acquisition, fileTimestamp, metadata: md, checks,
  };
}
