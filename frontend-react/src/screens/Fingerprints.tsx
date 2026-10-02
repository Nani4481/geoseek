import { useEffect, useMemo, useState } from 'react';
import { api, tileThumb } from '@/api/client';
import type { LonLat, TileFootprint } from '@/api/types';
import { GeoMap, type MapPoint } from '@/components/GeoMap';
import { Panel } from '@/components/Panel';
import { ErrorNote, Loading, TileImg } from '@/components/Widgets';
import { DASH, fmtLonLat, fmtMs, haversineKm, isNum } from '@/fmt';
import { useApi } from '@/hooks/useApi';
import { go } from '@/router';
import { useStore } from '@/state/store';

interface Row { tile_id: string; score: number; date: string; lonlat: LonLat; cluster: number | null; isSeed: boolean }

// tile ids embed their acquisition date: <scene>_<tile>_<YYYYMMDD>_...  ->  YYYY-MM-DD
const seedDate = (id: string) => { const m = id.match(/_(\d{4})(\d{2})(\d{2})_/); return m ? `${m[1]}-${m[2]}-${m[3]}` : ''; };

function Axis({ name, bar, value, hint }: { name: string; bar: number | null; value: string; hint: string }) {
  return (
    <div className="axis-row" title={hint}>
      <span className="dim">{name}</span>
      <div className="bar" style={{ width: '100%' }}><i style={{ width: `${(isNum(bar) ? Math.max(0, Math.min(1, bar)) : 0) * 100}%`, background: 'var(--cyan)' }} /></div>
      <span className="mono" style={{ textAlign: 'right' }}>{value}</span>
    </div>
  );
}

export function Fingerprints({ seed }: { seed: string | null }) {
  const { selectedId, label } = useStore();
  const top = useApi((s) => api.candidates({ sort: 'queue_score', limit: 30 }, s), []);
  const seedKey = seed ?? selectedId ?? top.data?.candidates[0]?.candidate_id ?? null;
  const isTile = !!seedKey?.startsWith('tile:');
  const sim = useApi(seedKey ? (s) => (isTile ? api.similarTile(seedKey!.slice(5), 16, s) : api.similarToCandidate(seedKey!, 16, s)) : null, [seedKey]);

  const rows: Row[] = useMemo(() => {
    if (!sim.data) return [];
    const seg = sim.data;
    return seg.results.filter((r) => r.tile_id !== seg.seed_tile_id).map((r) => ({ tile_id: r.tile_id, score: r.score, date: r.acq_date, lonlat: r.centroid_lonlat, cluster: r.cluster, isSeed: false }));
  }, [sim.data]);
  const ids = useMemo(() => (sim.data ? [sim.data.seed_tile_id, ...rows.map((r) => r.tile_id)] : []), [sim.data, rows]);
  const fps = useApi(ids.length ? (s) => api.tiles(ids, s) : null, [ids.join(',')]);
  const fpBy = useMemo(() => new Map<string, TileFootprint>((fps.data?.tiles ?? []).map((t) => [t.tile_id, t])), [fps.data]);

  const [cmp, setCmp] = useState<string | null>(null);
  const [hover, setHover] = useState<string | null>(null);   // 'seed' or a tile id: the card or map pin under the pointer
  useEffect(() => { setCmp(rows[0]?.tile_id ?? null); }, [rows]);
  const seedFp = sim.data ? fpBy.get(sim.data.seed_tile_id) : undefined;
  const seedCenter: LonLat | null = seedFp ? [(seedFp.bbox[0] + seedFp.bbox[2]) / 2, (seedFp.bbox[1] + seedFp.bbox[3]) / 2] : null;
  const cur = rows.find((r) => r.tile_id === cmp);
  const curFp = cmp ? fpBy.get(cmp) : undefined;
  const maxDist = useMemo(() => (seedCenter ? Math.max(1, ...rows.map((r) => haversineKm(seedCenter, r.lonlat))) : 1), [rows, seedCenter]);
  const dist = cur && seedCenter ? haversineKm(seedCenter, cur.lonlat) : null;
  const footRatio = seedFp && curFp ? Math.min(seedFp.area_km2, curFp.area_km2) / Math.max(seedFp.area_km2, curFp.area_km2) : null;
  const same = cur && sim.data ? cur.cluster !== null && cur.cluster === sim.data.seed_cluster : null;

  const points: MapPoint[] = useMemo(() => {
    const p: MapPoint[] = rows.map((r, i) => ({
      id: r.tile_id, lon: r.lonlat[0], lat: r.lonlat[1], pin: { text: String(i + 1), kind: r.tile_id === cmp ? 'sel' : 'hit' },
      label: `#${i + 1} · similarity ${r.score.toFixed(3)} · ${r.date} · click to compare with the seed`,
    }));
    if (seedCenter) p.push({ id: 'seed', lon: seedCenter[0], lat: seedCenter[1], pin: { text: 'Seed', kind: 'seed' }, label: `Seed tile ${sim.data?.seed_tile_id ?? ''} · ${fmtLonLat(seedCenter)}` });
    return p;
  }, [rows, cmp, seedCenter, sim.data?.seed_tile_id]);

  // fit only when the SET of tiles changes - not when the user merely selects a different tile
  const setKey = ids.join(',');
  const fit = useMemo<[number, number, number, number] | null>(() => {
    if (!seedCenter && rows.length === 0) return null;
    const xs = [...rows.map((r) => r.lonlat[0]), ...(seedCenter ? [seedCenter[0]] : [])];
    const ys = [...rows.map((r) => r.lonlat[1]), ...(seedCenter ? [seedCenter[1]] : [])];
    const px = Math.max(0.05, (Math.max(...xs) - Math.min(...xs)) * 0.1), py = Math.max(0.05, (Math.max(...ys) - Math.min(...ys)) * 0.1);
    return [Math.min(...xs) - px, Math.min(...ys) - py, Math.max(...xs) + px, Math.max(...ys) + py];
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [setKey, !!seedCenter]);

  return (
    <div className="col">
      <div className="row" style={{ alignItems: 'flex-end', gap: 14, flexWrap: 'wrap' }}>
        <label className="field" style={{ minWidth: 320 }}>Seed
          <select className="input" value={seedKey ?? ''} onChange={(e) => go('fingerprints', e.target.value)}>
            {isTile && seedKey && <option value={seedKey}>tile · {seedKey.slice(5)}</option>}
            {top.data?.candidates.map((c) => <option key={c.candidate_id} value={c.candidate_id}>#{c.rank} · {label(c.change_type)} · {c.candidate_id}</option>)}
          </select>
        </label>
        {sim.data && <span className="chip green mono">{fmtMs(sim.data.latency_ms)} ms · {rows.length} similar tiles</span>}
        <span className="dim" style={{ fontSize: 11.5, maxWidth: 620 }}>Tiles are ranked by structural similarity of their appearance; select one to compare it with the seed along each axis the catalog can actually measure.</span>
      </div>
      {sim.error ? <ErrorNote error={sim.error} /> : (
        <div className="fp-grid">
          <Panel title="Similar tiles" grow>
            {sim.loading || !sim.data ? <Loading rows={6} /> : (
              <div className="fp-tiles">
                <button className={`fp-tile seed ${hover === 'seed' ? 'hl' : ''}`.trim()} style={{ cursor: 'default' }} aria-label="Seed tile" onMouseEnter={() => setHover('seed')} onMouseLeave={() => setHover(null)}>
                  <TileImg src={tileThumb(sim.data.seed_tile_id)} alt="Seed tile" />
                  <div className="cap"><b style={{ color: 'var(--amber)' }}>SEED</b><span>{seedDate(sim.data.seed_tile_id) || DASH}</span></div>
                </button>
                {rows.map((r, i) => (
                  <button key={r.tile_id} className={`fp-tile ${r.tile_id === cmp ? 'sel' : ''} ${hover === r.tile_id ? 'hl' : ''}`.replace(/\s+/g, ' ').trim()} data-tile={r.tile_id} data-n={i + 1}
                    onClick={() => setCmp(r.tile_id)} onMouseEnter={() => setHover(r.tile_id)} onMouseLeave={() => setHover(null)} aria-pressed={r.tile_id === cmp} aria-label={`Similar tile ${i + 1}`}>
                    <TileImg src={tileThumb(r.tile_id)} alt={`Similar tile ${i + 1}`} />
                    <div className="cap"><span>#{i + 1} · {r.score.toFixed(3)}</span><span>{r.date}</span></div>
                  </button>
                ))}
              </div>
            )}
          </Panel>
          <div className="col">
            <Panel title="Similarity comparison">
              {!sim.data || !cur ? <Loading rows={5} /> : (
                <div className="col" style={{ gap: 12 }}>
                  <div className="cmp-pair">
                    <figure style={{ margin: 0 }}><TileImg src={tileThumb(sim.data.seed_tile_id)} alt="Seed" /><figcaption className="dim mono" style={{ fontSize: 10.5, marginTop: 3 }}>seed · {fmtLonLat(seedCenter)}</figcaption></figure>
                    <figure style={{ margin: 0 }}><TileImg src={tileThumb(cur.tile_id)} alt="Selected" /><figcaption className="dim mono" style={{ fontSize: 10.5, marginTop: 3 }}>{cur.date} · {fmtLonLat(cur.lonlat)}</figcaption></figure>
                  </div>
                  <Axis name="Appearance" bar={cur.score} value={cur.score.toFixed(3)} hint="Exact similarity between the two tiles' vectors (1 = identical appearance)" />
                  <Axis name="Proximity" bar={dist === null ? null : 1 - dist / maxDist} value={dist === null ? DASH : `${dist.toFixed(1)} km`} hint="Great-circle distance between the two tile centres; the bar is relative to the farthest tile in this set" />
                  <Axis name="Footprint" bar={footRatio} value={seedFp && curFp ? `${seedFp.area_km2.toFixed(2)} · ${curFp.area_km2.toFixed(2)} km²` : DASH} hint="Ground area covered by each tile (catalog geometry); the bar is the smaller area over the larger" />
                  <Axis name="Cluster" bar={same === null ? null : same ? 1 : 0} value={same === null ? DASH : same ? 'same' : 'different'} hint="Whether both tiles belong to the same archive cluster" />
                  <dl className="kv">
                    <dt>Seed cluster</dt><dd>{sim.data.seed_cluster ?? DASH}</dd>
                    <dt>Selected cluster</dt><dd>{cur.cluster ?? DASH}</dd>
                    <dt>Cloud cover</dt><dd>{seedFp ? Math.round(seedFp.cloud_fraction * 100) + '%' : DASH} · {curFp ? Math.round(curFp.cloud_fraction * 100) + '%' : DASH}</dd>
                  </dl>
                </div>
              )}
            </Panel>
            <Panel title="Spatial spread" flush fill grow>
              <GeoMap ariaLabel="Spatial spread of similar tiles" points={points} height={340} fill fit={fit} fitMaxZoom={14} basemap={{}} hoverId={hover} onHover={setHover}
                onPointClick={(id) => { if (id !== 'seed') setCmp(id); }} />
            </Panel>
          </div>
        </div>
      )}
    </div>
  );
}
