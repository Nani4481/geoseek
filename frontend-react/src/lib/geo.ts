// WGS84 <-> UTM <-> MGRS, implemented here so the console needs no geodesy dependency (nothing to bundle, nothing to fetch).
// Krüger n-series to 6th order (Karney 2011): millimetre-accurate over a whole UTM zone. No imports: tools/selftest-lib.mjs
// runs this file directly and tests/test_react_lib_selftest.py cross-checks it against pyproj and against the catalog's own
// Sentinel-2 tile names (a tile id such as 44RPQ *is* an MGRS zone + 100 km square).
//
// Not handled, on purpose: points poleward of 84°N / 80°S (UTM does not cover them; UPS is not implemented) -> null.

const A = 6378137.0, F = 1 / 298.257223563, K0 = 0.9996, E0 = 500000, N0_SOUTH = 10_000_000;
const N = F / (2 - F);
const N2 = N * N, N3 = N2 * N, N4 = N3 * N, N5 = N4 * N, N6 = N5 * N;
const RECT = (A / (1 + N)) * (1 + N2 / 4 + N4 / 64 + N6 / 256);
const ALPHA = [
  N / 2 - (2 * N2) / 3 + (5 * N3) / 16 + (41 * N4) / 180 - (127 * N5) / 288 + (7891 * N6) / 37800,
  (13 * N2) / 48 - (3 * N3) / 5 + (557 * N4) / 1440 + (281 * N5) / 630 - (1983433 * N6) / 1935360,
  (61 * N3) / 240 - (103 * N4) / 140 + (15061 * N5) / 26880 + (167603 * N6) / 181440,
  (49561 * N4) / 161280 - (179 * N5) / 168 + (6601661 * N6) / 7257600,
  (34729 * N5) / 80640 - (3418889 * N6) / 1995840,
  (212378941 * N6) / 319334400,
];
const BETA = [
  N / 2 - (2 * N2) / 3 + (37 * N3) / 96 - N4 / 360 - (81 * N5) / 512 + (96199 * N6) / 604800,
  N2 / 48 + N3 / 15 - (437 * N4) / 1440 + (46 * N5) / 105 - (1118711 * N6) / 3870720,
  (17 * N3) / 480 - (37 * N4) / 840 - (209 * N5) / 4480 + (5569 * N6) / 90720,
  (4397 * N4) / 161280 - (11 * N5) / 504 - (830251 * N6) / 7257600,
  (4583 * N5) / 161280 - (108847 * N6) / 3991680,
  (20648693 * N6) / 638668800,
];
const DELTA = [
  2 * N - (2 * N2) / 3 - 2 * N3 + (116 * N4) / 45 + (26 * N5) / 45 - (2854 * N6) / 675,
  (7 * N2) / 3 - (8 * N3) / 5 - (227 * N4) / 45 + (2704 * N5) / 315 + (2323 * N6) / 945,
  (56 * N3) / 15 - (136 * N4) / 35 - (1262 * N5) / 105 + (73814 * N6) / 2835,
  (4279 * N4) / 630 - (332 * N5) / 35 - (399572 * N6) / 14175,
  (4174 * N5) / 315 - (144838 * N6) / 6237,
  (601676 * N6) / 22275,
];
const rad = (d: number) => (d * Math.PI) / 180;
const deg = (r: number) => (r * 180) / Math.PI;

export interface UTM { zone: number; north: boolean; easting: number; northing: number; band: string }

const BANDS = 'CDEFGHJKLMNPQRSTUVWX'; // 8° bands from 80°S; X is 12° (72–84°N); I and O are never used

export function latBand(lat: number): string | null {
  if (lat < -80 || lat > 84) return null;
  return BANDS[Math.min(19, Math.floor((lat + 80) / 8))];
}

/** UTM zone for a point, including the Norway (32V) and Svalbard (31X–37X) exceptions. */
export function utmZone(lon: number, lat: number): number {
  let z = Math.floor((lon + 180) / 6) + 1;
  if (z > 60) z = 60;
  if (z < 1) z = 1;
  if (lat >= 56 && lat < 64 && lon >= 3 && lon < 12) z = 32;
  if (lat >= 72 && lat < 84) {
    if (lon >= 0 && lon < 9) z = 31; else if (lon >= 9 && lon < 21) z = 33; else if (lon >= 21 && lon < 33) z = 35; else if (lon >= 33 && lon < 42) z = 37;
  }
  return z;
}

export const centralMeridian = (zone: number) => (zone - 1) * 6 - 180 + 3;

export function toUTM(lon: number, lat: number, forceZone?: number): UTM | null {
  const band = latBand(lat);
  if (band === null || !Number.isFinite(lon) || !Number.isFinite(lat)) return null;
  const zone = forceZone ?? utmZone(lon, lat);
  const phi = rad(lat);
  let lam = rad(lon - centralMeridian(zone));
  lam = Math.atan2(Math.sin(lam), Math.cos(lam)); // wrap into (-180°, 180°]
  const e = (2 * Math.sqrt(N)) / (1 + N);
  const sp = Math.sin(phi);
  const t = Math.sinh(Math.atanh(sp) - e * Math.atanh(e * sp));
  const xi0 = Math.atan2(t, Math.cos(lam));
  const eta0 = Math.atanh(Math.sin(lam) / Math.sqrt(1 + t * t));
  let xi = xi0, eta = eta0;
  for (let j = 1; j <= 6; j++) {
    xi += ALPHA[j - 1] * Math.sin(2 * j * xi0) * Math.cosh(2 * j * eta0);
    eta += ALPHA[j - 1] * Math.cos(2 * j * xi0) * Math.sinh(2 * j * eta0);
  }
  const north = lat >= 0;
  return { zone, north, band, easting: E0 + K0 * RECT * eta, northing: (north ? 0 : N0_SOUTH) + K0 * RECT * xi };
}

export function fromUTM(zone: number, north: boolean, easting: number, northing: number): { lon: number; lat: number } {
  const xi = (northing - (north ? 0 : N0_SOUTH)) / (K0 * RECT);
  const eta = (easting - E0) / (K0 * RECT);
  let xi0 = xi, eta0 = eta;
  for (let j = 1; j <= 6; j++) {
    xi0 -= BETA[j - 1] * Math.sin(2 * j * xi) * Math.cosh(2 * j * eta);
    eta0 -= BETA[j - 1] * Math.cos(2 * j * xi) * Math.sinh(2 * j * eta);
  }
  const chi = Math.asin(Math.sin(xi0) / Math.cosh(eta0));
  let phi = chi;
  for (let j = 1; j <= 6; j++) phi += DELTA[j - 1] * Math.sin(2 * j * chi);
  const lam = Math.atan2(Math.sinh(eta0), Math.cos(xi0));
  return { lon: centralMeridian(zone) + deg(lam), lat: deg(phi) };
}

/** EPSG:326zz (north) / 327zz (south) -> {zone, north}; anything else -> null. */
export function utmFromEpsg(epsg: number): { zone: number; north: boolean } | null {
  if (epsg >= 32601 && epsg <= 32660) return { zone: epsg - 32600, north: true };
  if (epsg >= 32701 && epsg <= 32760) return { zone: epsg - 32700, north: false };
  return null;
}

const COL_SETS = ['ABCDEFGH', 'JKLMNPQR', 'STUVWXYZ'];
const ROW_LETTERS = 'ABCDEFGHJKLMNPQRSTUV';

export interface MGRS { zone: number; band: string; square: string; easting: number; northing: number; text: string }

/** MGRS reference, e.g. `44R PQ 12345 67890`. `digits` = 5 -> 1 m, 4 -> 10 m, ... (truncated, never rounded, per the standard). */
export function toMGRS(lon: number, lat: number, digits = 5): MGRS | null {
  const u = toUTM(lon, lat);
  if (!u) return null;
  const col = COL_SETS[(u.zone - 1) % 3][Math.floor(u.easting / 100000) - 1];
  const rowIdx = (Math.floor(u.northing / 100000) + (u.zone % 2 === 0 ? 5 : 0)) % 20;
  const square = col + ROW_LETTERS[rowIdx];
  const e = Math.floor(u.easting % 100000), n = Math.floor(u.northing % 100000);
  const cut = (v: number) => String(Math.floor(v / 10 ** (5 - digits))).padStart(digits, '0');
  const text = `${String(u.zone).padStart(2, '0')}${u.band} ${square}${digits ? ` ${cut(e)} ${cut(n)}` : ''}`;
  return { zone: u.zone, band: u.band, square, easting: e, northing: n, text };
}

/** The Sentinel-2 style tile token (`44RPQ`) that contains a point: zone + band + 100 km square. */
export function s2TileToken(lon: number, lat: number): string | null {
  const m = toMGRS(lon, lat, 0);
  return m ? `${String(m.zone).padStart(2, '0')}${m.band}${m.square}` : null;
}
