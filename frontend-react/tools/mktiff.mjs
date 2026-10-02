// A tiny classic-TIFF writer for the e2e test: produces a real, valid GeoTIFF whose header fields are KNOWN, so the console's
// parse can be compared with the values the file was built from (ground truth independent of the parser).
//
//   makeGeoTiff({ width, height, epsg, originX, originY, resX, resY, acquisition })  ->  Buffer
//   epsg = null writes a TIFF with no GeoKeys / geotransform at all (an un-georeferenced image).
export function makeGeoTiff({ width = 32, height = 24, epsg = 32644, originX = 600000, originY = 2950000, resX = 10, resY = 10, acquisition = null } = {}) {
  const pixels = Buffer.alloc(width * height);
  for (let i = 0; i < pixels.length; i++) pixels[i] = (i * 7) % 251;
  const entries = []; // [tag, type, count, valueBuffer]
  const SHORT = 3, LONG = 4, DOUBLE = 12, ASCII = 2;
  const u16 = (...v) => { const b = Buffer.alloc(2 * v.length); v.forEach((x, i) => b.writeUInt16LE(x, 2 * i)); return b; };
  const u32 = (...v) => { const b = Buffer.alloc(4 * v.length); v.forEach((x, i) => b.writeUInt32LE(x, 4 * i)); return b; };
  const f64 = (...v) => { const b = Buffer.alloc(8 * v.length); v.forEach((x, i) => b.writeDoubleLE(x, 8 * i)); return b; };
  entries.push([256, LONG, 1, u32(width)], [257, LONG, 1, u32(height)], [258, SHORT, 1, u16(8)], [259, SHORT, 1, u16(1)], [262, SHORT, 1, u16(1)]);
  entries.push([273, LONG, 1, null /* strip offset, patched below */], [277, SHORT, 1, u16(1)], [278, LONG, 1, u32(height)], [279, LONG, 1, u32(pixels.length)]);
  if (epsg !== null) {
    entries.push([33550, DOUBLE, 3, f64(resX, resY, 0)], [33922, DOUBLE, 6, f64(0, 0, 0, originX, originY, 0)]);
    entries.push([34735, SHORT, 16, u16(1, 1, 0, 3, 1024, 0, 1, 1, 1025, 0, 1, 1, 3072, 0, 1, epsg)]);
  }
  if (acquisition) {
    const xml = `<GDALMetadata>\n  <Item name="ACQUISITION_DATE">${acquisition}</Item>\n</GDALMetadata>\0`;
    entries.push([42112, ASCII, Buffer.byteLength(xml), Buffer.from(xml, 'latin1')]);
  }
  entries.sort((a, b) => a[0] - b[0]);
  const ifdSize = 2 + entries.length * 12 + 4;
  let dataOff = 8 + ifdSize;
  const blobs = [];
  const placed = entries.map(([tag, type, count, val]) => {
    if (tag === 273) return { tag, type, count, inline: null, patch: true };
    if (val.length <= 4) return { tag, type, count, inline: Buffer.concat([val, Buffer.alloc(4 - val.length)]) };
    const off = dataOff; dataOff += val.length + (val.length % 2); blobs.push([off, val]);
    return { tag, type, count, inline: u32(off) };
  });
  const pixelOff = dataOff;
  const out = Buffer.alloc(pixelOff + pixels.length);
  out.write('II', 0, 'latin1'); out.writeUInt16LE(42, 2); out.writeUInt32LE(8, 4);
  out.writeUInt16LE(entries.length, 8);
  placed.forEach((e, i) => {
    const o = 10 + i * 12;
    out.writeUInt16LE(e.tag, o); out.writeUInt16LE(e.type, o + 2); out.writeUInt32LE(e.count, o + 4);
    if (e.patch) out.writeUInt32LE(pixelOff, o + 8); else e.inline.copy(out, o + 8);
  });
  out.writeUInt32LE(0, 10 + entries.length * 12);
  for (const [off, val] of blobs) val.copy(out, off);
  pixels.copy(out, pixelOff);
  return out;
}
