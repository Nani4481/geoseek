# React console (`/react/`)

A **parallel** analyst console built with Vite + React + TypeScript, styled as a dark-navy dashboard (icon rail, 3D globe
hero with region pins, stat cards, alert feed, animated temporal evidence, before/after slider, change-details and confidence
panels, threat rings, tactical dossier, embedding-space view, spectral evidence, raster header check). It is served by the
same FastAPI process as the existing frontend and makes **zero external requests**. Every screen is a working feature: there
is no mock-up, roadmap or "not implemented" chrome anywhere.

* Existing frontend: `/app/` — **unchanged** (nothing under `src/geoseek/analyst/web/` is modified; it is only *read*,
  see §6).
* React console: `/react/` — source in `frontend-react/`, build output in `src/geoseek/analyst/web_react/`.

## 1. Build and serve

| | |
|---|---|
| Toolchain used | Node **v24.14.0**, npm **11.9.0**, Python 3.11.16 (the `geoseek` conda env) |
| Runtime dependencies | `react` 19.3.0, `react-dom` 19.3.0 and `geotiff` 3.0.5 (MIT; used only to parse GeoTIFF *headers* for the Data screen) with its own eight small runtime dependencies, all bundled into the build — no router, state, chart, CSS or map library |
| Build dependencies | `vite` 8.3.2, `@vitejs/plugin-react` 6.1.1, `typescript` 5.9.3, `@types/react` / `@types/react-dom` 19.3.0, `@types/leaflet` 1.9.22 |
| Pinning | exact versions in `package.json` + `package-lock.json` (a test asserts exact pins, the two-package runtime set, and that the lockfile matches the pinned build) |
| Reused, already-vendored libraries | three.js, OrbitControls, Leaflet 1.9.4, the Blue Marble textures and Inter / JetBrains Mono fonts — aliased straight from `analyst/web/vendor` and `analyst/web/fonts`, so the shipped bytes are the already-hash-pinned ones |
| Charts | hand-rolled SVG/CSS bars (no chart library, so nothing extra to audit) |

```bash
cd frontend-react
npm ci                     # exact versions from package-lock.json
npm run build              # tsc --noEmit && vite build  ->  ../src/geoseek/analyst/web_react/
npm run build:pinned       # build, then re-pin:  python -m geoseek.staging.react_build_pins
node tools/selftest-lib.mjs  # unit checks of src/lib (also run from pytest, see §5)
npm run dev                # Vite dev server on :5173, proxying the API to 127.0.0.1:8000
npm run verify:offline     # runtime network audit against a running backend (see §4)
```

Serve: `uvicorn geoseek.search.api:app --host 127.0.0.1 --port 8000`, then open **`http://127.0.0.1:8000/react/`**.
`GET /` also lists `"react_ui": "/react/"`.

**The build output is committed**, not built at deploy time. An air-gapped machine therefore needs Python only — no Node,
no npm, no network — and the committed bytes are exactly what the tests pin. Rebuild only when `frontend-react/` changes,
then re-pin (`npm run build:pinned`). `.gitattributes` marks `web_react/**` as `-text` so no checkout converts line endings
and breaks the hashes; `node_modules/` is git- and docker-ignored.

## 2. What the console does

Rail: **Dashboard · Search · Changes · Pipeline · Detect · Discover · Similar · Brief · Data · Settings**. Every panel has the same
treatment (title + optional actions); a feature is either wired to the backend or it is not in the product.

| Feature | Screen | Backend (all read-only unless noted) |
|---|---|---|
| Situation dashboard: tiles / scenes / regions / sensors / change candidates / analyst decisions / search latency, findings by region and by change type, globe, alert feed | Dashboard | `GET /ui/metrics`, `/ui/latency`, `/notifications`, `/restricted-zones`, `/candidates` |
| **Animated temporal evidence** (§2.1) | Changes, Dashboard | `GET /ui/candidates/{id}/timeline`, `/candidates/{id}/imagery` |
| Candidate workbench: filterable queue, before/after slider, change mask, spectral type, confidence ring and five evidence gates, SAR status, bookmark, review (confirm / reject / reopen, append-only audit trail), GeoJSON export | Changes | `GET /candidates`, `/candidates/{id}`; `POST /candidates/{id}/decision`, `POST /export` (the existing write paths) |
| **Why this was flagged**, per-candidate **pipeline trace**, and the **suppression funnel** (see the last section) | Changes (full-width row below the workbench), Pipeline | `GET /ui/candidates/{id}/explain`, `/ui/pipeline/funnel` |
| **Tactical dossier export** (§2.2) | Changes → *Dossier* | `GET /ui/candidates/{id}/dossier` + the candidate record |
| **Threat buffer rings** (§2.3) | Changes (candidate map), Detect (detection map) | `GET /ui/threat-rings`, `/ui/detections/{obs}/points` |
| Semantic search with date / cloud / box filters, more-like-this, click-the-map search | Search | `GET /search/text`, `POST /search/image`, `GET /discovery/similar` |
| **3-D vector-space visualiser** (§2.4) | Search (beside the map) | `GET /ui/projection`, `/ui/projection/lookup` |
| **Spectral evidence** (§2.5) | Search (*Evidence* on a result) | `GET /ui/tiles/{id}/spectral`, `/ui/tiles/{id}/spectral/{index}.png` |
| Object detection: stored oriented boxes over the scene, class toggles, measured accuracy and the backend's caveats | Detect | `GET /detect/*` |
| Discovery / clustering, structural-similarity gallery, briefing mode (keyframe tour, annotation, spotlight) | Discover, Similar, Brief | `GET /discovery/*`, `/candidates/{id}/similar`, `/ui/tiles` |
| **Ad-hoc raster header check** (§2.6) and archive inventory | Data | none for the check (it runs in the browser); `GET /ui/metrics`, `/regions` for the inventory |
| **System / model performance**: change-model F1 / precision / recall / IoU and detector AP50 (DOTA val + xView transfer) with their caveats; system status; analyst profile | Settings | `GET /ui/metrics`, `/health` |

The model-evaluation figures live under **Settings → System / model performance**, not on the analyst's home screen: they are a
real strength of the product but an operator does not need a benchmark on their situation screen.

### 2.1 Animated temporal evidence

The playhead sweeps the **real** acquisition dates on load and on *Play*; nodes light as it passes; the "change occurred in this
window" bracket is drawn in by the playhead; the before/after imagery cross-fades by playhead position. There are *Play / Pause /
Replay*, a scrub range input and mouse-drag scrubbing on the timeline. Under `prefers-reduced-motion` there is no automatic
sweep and no blending: *Play* steps date by date. The animation is presentation only — the time axis is the real, uneven
calendar (`src/lib/timelapse.ts`: positions proportional to elapsed days), only catalogued dates are drawn, and any
interaction with the slider or the date pickers cancels a sweep that has not started yet. Playback state lives outside React
(`hooks/usePlayback.ts`) so a 60 fps playhead does not re-render whole screens.

### 2.2 Tactical dossier

*Dossier* on the workbench opens a paper-style sheet built only from the candidate's own record: coordinates with **MGRS and UTM
computed offline** (`src/lib/geo.ts`, Krüger series; no dependency), region, before / after / difference chips for the dates
selected in the workbench, change type, confidence, persistence ("N of M"), acquisition dates, the gate trace, **sensor provenance**
(platform, acquisition date, cloud cover of the nearest catalogued tile, ground resolution, processing baseline, scene id, CRS),
the analyst decision and its append-only audit trail. Export is the browser's own print-to-PDF: a print stylesheet shows just the
sheet (`@page` A4) and the button calls `window.print()`. No PDF service, nothing uploaded.

**Sun elevation and off-nadir angle are not shown, because the catalog does not hold them.** The endpoint scans every catalogued
metadata blob (scene, observation, collection, tile flags) for `sun_elevation` / `sun_azimuth` / `off_nadir` and reports a field only
if it finds a number; for the current archive none exists (checked across all JSON columns of `tiles.sqlite`: 0 matches), so the
dossier omits them entirely rather than printing placeholders. To capture them, the ingest step would need to copy the standard
STAC `view:` properties from the source item into the scene's `metadata` (`view:sun_elevation`, `view:sun_azimuth`,
`view:off_nadir`, where the provider's item carries them) - the catalogued scene metadata currently holds only the
footprint source, acquisition-time precision and BOA-offset notes (Sentinel-2) and `stac_item` / orbit fields (Sentinel-1). The dossier will pick them up unchanged once they are in the catalog.

### 2.3 Threat buffer rings

Right-click a candidate (Changes) or a detection (Detect): concentric rings at configurable radii (default 500 m / 1 / 2.5 / 5 km, up
to eight, 10 m – 200 km, validated in the UI and on the server) are drawn on the vendored Leaflet, and everything inside each ring is
listed with its **exact distance**. The layers intersected are the ones the system actually has: the restricted zones that produce the
restricted-zone alerts (and their names), every change-candidate footprint, every stored object detection and the watch areas. There
is **no road / building / infrastructure layer in the catalog**, so none is searched or implied; the response lists each layer and
how many features it holds. Distances are point-to-footprint geodesic distances in an azimuthal-equidistant projection centred on
the point (0 m = the footprint contains it), checked in `tests/test_threat_rings.py` against an independent `pyproj.Geod`.

### 2.4 3-D vector-space visualiser

`scripts/compute_projection.py` reduces the tile embeddings with PCA-50 and UMAP (3-D, cosine, seed 42) **offline** and writes
`data/discovery/projection_3d.npz` + `.meta.json` (and a provenance record). The console draws it with the vendored three.js,
coloured by region or cluster; **text-search results are highlighted** in the scatter (looked up exactly, so they show even if they
fall outside the drawn sample); **clicking a point pans the Search map to that tile**. 105,245 points would render, but the console
draws a deterministic **20,000-point uniform sample** and says so in the caption (`20,000-point sample of 105,245 tiles`); a stale
projection (index has grown since) is flagged. UMAP preserves neighbourhoods, not distances - the caption says that too.
UMAP is an optional extra (`pip install umap-learn`); the console only reads the finished artifact.

### 2.5 Spectral evidence (replaces the "attention overlay" idea)

There is deliberately **no attention heat-map**: ViT-B/32 sees a 224 px input as a 7 × 7 grid, so over a 2.56 km tile one patch is
~366 m — far too coarse to localise a structure or a riverbank, and an overlay implying otherwise would be false. Instead *Evidence*
on a search result shows **per-pixel NDWI, NDBI and NDVI at the native 10 m**, computed from the B03/B04/B08/B11 bands (NIR and SWIR
the retrieval model never sees) with the **same functions and valid-pixel mask as the catalog's per-tile spectral descriptor** (the
statistics shown equal the stored `tile_spectral` row to 1e-6, asserted on real tiles). Each index is toggleable with a legend drawn from
the same colour stops as the PNG, mean / p10 / p50 / p90, and the descriptor's surface-class fractions with their rules. For a text
query the indices that bear on its words are pre-selected and flagged (water → NDWI; building / urban / road → NDBI; vegetation → NDVI;
bare / dry / sand → NDVI + NDBI), with the matched terms and reason shown. A tile without NIR/SWIR on disk (Maxar, Sentinel-1) answers
404 "not staged" - no overlay is fabricated. The patch size quoted in the caveat is computed from the tile's own footprint.

### 2.6 Ad-hoc raster header check

On *Data*, drop (or choose) up to six GeoTIFFs. `geotiff.js` (bundled) first reads **the header** in the browser (for a 160 MB staged
band: under a second; the pixel preview described at the end of this file follows separately): size, bands, sample type, CRS/EPSG, the affine transform, native bounds, lon/lat
bounds (WGS 84, Web Mercator and WGS 84 UTM invert offline; other CRSs are reported as "not bundled", never guessed), pixel size,
nodata, layout, and an acquisition timestamp **only if the header carries one** (a GDAL metadata item such as `ACQUISITION_DATE`; the
TIFF `DateTime` tag is shown separately as "when the file was written"). The validation badge is the real result: CRS present,
geotransform present, north-up, square pixels, footprint placeable. After validation the panel says plainly what the file contains,
which archive regions it overlaps (from `/regions`) and that **nothing was ingested** - staging is done through the ingest pipeline
(`python -m geoseek.ingest.pipeline ingest <scene_dir>` with per-band GeoTIFFs). There is no upload and no progress bar. Every field
is checked against rasterio in `tests/test_react_lib_selftest.py`.

### Deliberately not built

No UI exists for **InSAR / interferometric fringes** (the system ingests Sentinel-1 **GRD**; interferometric phase is not
recoverable from GRD) or for **change-rate / velocity / completion-horizon estimation** (the catalog holds five acquisition dates,
over an uneven 7-year span, which cannot support a defensible rate estimate). An end-to-end test asserts none of those terms appears
on any content route.

## 3. Where every displayed number comes from

There are no hardcoded statistics: each card fetches at runtime, shows `—` plus a visible "unavailable" state if its
endpoint fails, and (for the figures with a measurement behind them) carries its source in the tooltip.

| Card | Value shown | Endpoint → source |
|---|---|---|
| Tiles indexed | tile count (+ searchable vector count) | `/ui/metrics` → `repo.count_tiles()`, engine vector count |
| Scenes / Regions / Sensors | scene count; region count; distinct sensors | `/ui/metrics` → `repo.list_scenes()`, `list_regions()`, distinct `Collection.sensor` |
| Search latency | median of 5 fixed probe queries, p95 beside it, "measured now" | `/ui/latency` → engine-reported `latency_ms` after a discarded warm-up; cached 30 s |
| Change candidates / Analyst decisions | candidate count; decision-row count | `/ui/metrics` |
| *(Settings)* Change detection F1, P, R, IoU | held-out OSCD test figures at the deployed 0.80 operating point | `/ui/metrics` → `model_card.eval.test_at_precision_favouring` embedded in the change-model checkpoint |
| *(Settings)* Object detection AP50 | **0.845** small-vehicle, **0.854** ground vehicles (DOTA official val, 458 images, with bootstrap CI), xView transfer 0.531 / 0.315 | `/ui/metrics` → `data/detect_eval/eval_results.json#val_full_image_v15` |
| Threat-ring distances and counts | per request | `/ui/threat-rings` (geometry computed per request) |
| Dossier coordinates | MGRS / UTM of the candidate centroid | computed in the browser; cross-checked against pyproj and the catalog's own Sentinel-2 tile names |
| Dossier sensor block | platform, acquisition date, cloud cover, resolution, baseline, scene | `/ui/candidates/{id}/dossier` (catalog) |
| Embedding-space counts | points drawn / points in the projection / run time / power source | `/ui/projection` (the artifact's own metadata) |
| Spectral statistics and legends | index statistics, class fractions, colour stops | `/ui/tiles/{id}/spectral` (computed from the staged bands) |
| Raster header fields | everything on the Data card | parsed from the dropped file in the browser |
| Findings by region / by change type | candidates per region (point-in-box) and per change type, as bar lists on the Dashboard | `/ui/metrics` → `findings_by_region`, `findings_by_type` |
| Every confidence / persistence / area figure | per candidate | `/candidates`, `/candidates/{id}`, `/ui/candidates/{id}/timeline` |

The AP50 figures were checked against `docs/EVALUATION_REPORT.md` §15 (0.8454 and 0.8536) and against the stored eval JSON.

### Things the numbers say

* **"Object Detection 53.1%" in the reference mockup is not an object-detection number.** 53.1 % is the change-detection
  *validation precision at the 0.80 threshold* (report §8.1). The card is wired to the measured AP50 instead.
* **The DOTA-val AP50 does not transfer everywhere.** On the independent xView test half the same detector scores
  **0.531** (small-vehicle) and **0.315** (large-vehicle); on the staged Maxar tiles it finds aircraft and large objects but
  only a few percent of visible cars. Settings prints the xView figures beside the DOTA one, and the Detect screen
  shows the backend's own caveat text.
* **All 841 change candidates are in one region** (the Ayodhya AOI): the change pipeline has only been run there, so the
  other 11 regions show `0` findings. The numbers are real; the console does not dress them up.
* **SAR corroboration reads "Not corroborated"** for every candidate that has no Sentinel-1 evidence (currently all of
  them), rather than a neutral-looking blank.
* The **alert feed** is built from real watch-area notifications plus candidates inside the backend's restricted-zone
  boxes. Those boxes are the existing **demo** zones hard-coded in `analyst/service.py` (`RESTRICTED_ZONES`); the console
  displays them as the backend serves them.

## 4. Offline verification

Four independent layers, strongest first.

1. **The browser refuses.** `index.html` carries `Content-Security-Policy: default-src 'self'; script-src 'self';
   connect-src 'self'; img-src 'self' data: blob:; font-src 'self'; …` with no `unsafe-eval` and no inline scripts, so a
   cross-origin request is blocked by the browser itself. A test pins this (every fetch-type directive limited to
   `'self'`/`data:`/`blob:`).
2. **Static scan + hash pin** (`tests/test_frontend_offline.py`, extended; same rules as the vendor directory):
   * every file in `web_react/` is SHA-256-pinned in `frontend-react/build-pins.json`; any edit, added file or missing file
     fails (verified by mutation: an injected `fetch("https://…")`, an edited chunk and a stray file were each caught);
   * every external-looking URL string in a shipped text file must be classified in
     `geoseek.staging.react_build_pins.ALLOWED_URLS`; protocol-relative strings, remote `@import` and remote
     `fetch`/XHR/`Image` targets are never allowed; binary assets may contain no `http(s)://`;
   * the bundled three.js / OrbitControls / Leaflet are proven to be the **same bytes as the provenance manifest's pins**
     for the existing frontend, and every copied texture/font is byte-identical to its `analyst/web` source;
   * the app **source** (`frontend-react/src`) contains no external URL at all and never creates a Leaflet tile layer.

   **Mutation tests** (a guard that has never failed proves little). Each mutation was applied by hand, the suite run, and the
   tree restored; every one was caught:

   | Mutation | Caught by |
   |---|---|
   | `https://…` string, a remote `fetch("https://…")` and `L.tileLayer("https://…")` added to a file under `frontend-react/src` | `test_react_app_source_contains_no_external_url_and_no_tile_layer` (reported all three, by file) |
   | `fetch("https://evil…")` appended to a built chunk | `test_react_file_is_hash_pinned`, `test_react_text_asset_has_no_unclassified_external_url` |
   | a stray extra file left in the build directory | `test_react_pinned_file_set_equals_build_output` (and its own hash-pin test) |
   | thumbnail 404 fix reverted | 3 of the 6 tests in `test_tile_thumbnail_missing.py` |

   The only URL strings in the shipped bundle are nine distinct inert ones (ten file-level occurrences), each classified:
   five W3C XML-namespace identifiers (`createElementNS`), a React error-doc string, a three.js console-warning string, a
   geotiff.js error-message string (the 64-bit-offset error text), and Leaflet's attribution `href` (never rendered: every map
   is created with `attributionControl: false`). None is ever dereferenced.

   **geotiff.js was vetted before it was added:** it is bundled as-is (no CDN, no runtime download). Since the pixel preview shipped
   (see the last section) the pure-JS decoders (raw, DEFLATE, LZW, PackBits, JPEG) do run and are served as local chunks; the LERC /
   ZSTD chunks contain WebAssembly, which the page CSP (no `wasm-unsafe-eval`, `connect-src 'self'`) refuses to instantiate, so a
   ZSTD / LERC file gets its header and footprint but no preview, and says why and how to fix it (re-save as DEFLATE / LZW). The CSP was
   deliberately not loosened for this. (The earlier "no decoder is ever requested" e2e assertion no longer holds and was replaced.) The lockfile hash and the versions of `geotiff` and its eight dependencies are pinned in
   `build-pins.json`.
3. **Runtime request log** — `frontend-react/tools/verify-offline.mjs` drives headless Chrome over the DevTools protocol (no
   npm dependency), loads the console from the running backend, visits every route, and records every request. Result on
   the final build: **0 external requests** (about 300 requests across a full interaction run, 80–95 for a plain visit of every
   route), 0 CSP blocks.
4. **The indicator is derived, not decorative.** The top-bar pill is computed from Resource Timing (every loaded resource
   must be same-origin), `securitypolicyviolation` events, and `GET /health`. Green = `OFFLINE · 0 EXTERNAL REQUESTS`;
   an external request or CSP block turns it red with a count; an unreachable backend turns it amber. The host's own
   network-interface state is shown in the tooltip only and never changes the colour. The IST clock uses the browser's
   built-in `Intl` time-zone data — no network.

Offline map: there are no tiles, so Leaflet draws a dark canvas with an adaptive lat/lon graticule, region boxes, points and
polygons; candidate imagery is served by the backend. The Earth globe uses the vendored textures.

## 5. Testing

| Suite | What it covers |
|---|---|
| `tests/test_frontend_offline.py` (extended) | the React rules in §4 layer 2, alongside the existing vendor rules |
| `tests/test_react_console.py` | `/react/` is served and `/app/` is unchanged; timeline roles and "N of M" logic (persistent / transient / recent / none); footprints from catalog geometry; latency probe uses engine-measured times and the cache; metric values derive from their sources, and a missing source yields "unavailable", never a number |
| `tests/test_tile_thumbnail_missing.py` | a catalogued tile whose band rasters are not staged is a **404 with a reason**, an unknown tile stays 404, a staged tile still renders a JPEG, and a present-but-corrupt raster is *not* disguised as 404 (it stays a 500). Mutation-checked: 3 of its 6 tests fail with the fix reverted |
| `tests/test_threat_rings.py` | ring geometry against an independent `pyproj.Geod` (distances, ring membership, zone containment, exclusion, truncation, cumulative totals, detections, watch areas, HTTP validation) |
| `tests/test_dossier.py` | sensor provenance comes from the catalog; sun elevation / off-nadir appear **only** when catalogued (booleans and text are never mistaken for angles) |
| `tests/test_projection.py` | the projection artifact: honest sampling (deterministic, both counts reported, rows stay aligned), exact lookups, staleness, unavailable-state |
| `tests/test_spectral_evidence.py` | query → index relevance; colours land exactly on the legend stops; PNG pixels equal the index values at 10 m with clouds transparent; statistics equal the descriptor; unusable / unstaged tiles; **statistics equal the stored `tile_spectral` rows of real catalogued tiles** |
| `tests/test_react_lib_selftest.py` | runs `src/lib` under Node: timelapse geometry; **UTM vs pyproj to < 1 mm** (600 random points + landmarks) and round-trip; MGRS squares vs the catalog's own Sentinel-2 tile names; **GeoTIFF header parse vs rasterio** over UTM N/S, WGS 84, Web Mercator, rotated, non-square, un-georeferenced, un-invertible CRS, tagged and garbage files, plus a real 160 MB staged band |
| `frontend-react/tools/e2e-console.mjs` | 42 steps in headless Chrome with **real mouse and keyboard input** (assertions poll for the expected UI state rather than sleeping): operational stat cards equal the API and no model metric is on the dashboard; **no tier / roadmap chrome on any route**; Settings metrics equal the API; **timeline animation** (every sampled frame: playhead monotonic, nodes lit iff passed, bracket drawn progressively, layer opacities equal the cross-fade function, only catalog dates, pause freezes, scrub lands on the right date, reduced-motion steps without blending); **dossier** (MGRS equals the catalog's tile, UTM, chips and dates, gate / audit row counts, sensor block equals the API, no placeholder wording, print stylesheet, a real PDF from `Page.printToPDF`); **threat rings** (right-click via real events; chips, rows and distances equal `/ui/threat-rings`; centre not self-listed; radii validated / applied / cleared; detection map); **vector space** (20,000 of N points with caption, colour by cluster, search hits highlighted, click pans the map to that tile's footprint); **spectral evidence** (relevant index per query, stats equal the API, overlay pixels all on the legend ramp, toggles, opacity, 404 for unstaged); **Data** (known-ground-truth GeoTIFFs: drop and file-chooser, valid / no-CRS / junk, nothing-ingested note, no progress element, a real 160 MB band, no decoder fetched); text search, more-like-this, region box, map clicks; queue filter; slider; confirm → audit row → reopen → reject; exports; detection boxes; discovery; fingerprints; briefing; scope check; offline pill |

Confirm / reject / reopen append **permanent** audit rows, so the e2e script refuses to write unless `--allow-writes` is
given and must be pointed at a **scratch** backend:

```bash
cp data/index/tiles.sqlite /tmp/scratch.sqlite
DATABASE_URL=sqlite:////tmp/scratch.sqlite uvicorn geoseek.search.api:app --port 8001
node frontend-react/tools/e2e-console.mjs --base http://127.0.0.1:8001/react/ --allow-writes
```

(`POST /export` also writes a file under `data/change_model/exports/` — that is the existing endpoint's behaviour.)

## 6. Existing frontend untouched

`git diff --stat -- src/geoseek/analyst/web/` and `git status --short src/geoseek/analyst/web/` are empty. The only edits to
existing files are: `src/geoseek/search/api.py` (a `/react` static mount, the `/ui/*` read-only routes, and a
`react_ui` key on `GET /`), `src/geoseek/search/engine.py` (the thumbnail 404 fix, §7), the extended
`tests/test_frontend_offline.py`, and `.gitignore` / `.gitattributes` / `.dockerignore`.

## 7. Findings worth knowing about

* **A figure in the brief did not match the repo.** The XAI badge was specified as "ViT-B/32 resolves to ~800 m patches".
  The catalog's tile footprint is 2.58 km and ViT-B/32 on a 224 px input is a 7×7 patch grid, so one patch is about
  **366 m**. The panel computes that live from a catalogued tile's footprint (`/ui/tiles`) instead of hardcoding either number.
* **Pre-existing backend defect, now fixed:** `GET /tile/{id}/thumbnail` returned **500** for the staged Maxar Van Nuys tiles
  because their band rasters (`data/datasets/maxar/wildfires-losangeles-jan-2025/.../B04.tif`) are not on this machine.
  `SearchEngine.get_tile_thumbnail_png` now checks that the RGB band files exist and raises the `KeyError` the route already
  maps to **404** (with a reason naming the missing bands); only *absence* is translated, so a corrupt file is still a 500.
  The console shows "no preview" for such a tile. (The browser console still logs the 404 — browsers log every 4xx.)
* **Briefing-mode Undo: a canvas defect, not an app logic bug.** Under the e2e's scripted input (Undo clicked ~4 ms after a
  stroke redraw) the annotation canvas sometimes kept a *partial stroke after a full `clearRect`* — about half the runs. The
  instrumented call log showed the app's last operation was a correct full-canvas clear, and the same residue (same pixel
  signature) reproduces in a **bare page with no app code** under a software-GL load (1–2 % of trials), where a screenshot
  confirms the stale stroke is genuinely displayed, not just returned by `getImageData`. So it is a Chrome software-canvas
  behaviour that occasionally drops a clear which follows draw ops in the same frame; a later clear removes it. Mitigation in
  `Briefing.tsx`: when nothing remains to draw, reset the bitmap and do so once more a frame later (guarded by a generation
  token so it can never erase a stroke drawn in between; the e2e asserts that too). Measured: in-app repro on fresh pages
  went from ~50 % residue to **20/20 clean**; the e2e is stable over 5 consecutive runs. Not claimed: that the browser
  defect is eliminated everywhere — only that the app now recovers from it within a frame, and the bare-page rate shows it
  is rare outside scripted input.
* Leaflet's stylesheet references `images/layers.png` (the layers control). The console never creates that control, so the
  reference is unused; Vite reports it as unresolved at build time.
* The reference dashboard image was not received with the request; the layout follows its written description. The only
  design material in the repository (`design_handoff_geoseek/`) is a different, near-black design and was not used.

* **No sun elevation / off-nadir in the catalog** (see §2.2). The brief allowed them only if catalogued; they are not, so the dossier
  omits them. Capturing them at ingest is a small change (copy the STAC `view:` properties into the scene metadata).
* **The projection is a picture of local neighbourhoods.** 1,156 of the 105,245 tiles have no cluster label (the stored
  `tile_clusters.json` was computed for 104,089 tiles, before the index grew); they are drawn grey as "no cluster" rather than
  guessed. Re-running `scripts/cluster_at_scale.py` refreshes the labels; re-running `scripts/compute_projection.py` refreshes the
  coordinates.
* **UMAP run (this machine):** 105,245 × 512 → PCA-50 → UMAP-3D, `n_neighbors=15`, `min_dist=0.1`, cosine, seed 42: **222.4 s total**
  (load 2.9 s, PCA 1.2 s, UMAP 210.7 s), measured **on battery power (34 %)**, so it is not an AC benchmark (the project's standing AC-power benchmark rule) —
  the script records the power source in the artifact's metadata and warns when it is not AC. UMAP 0.5.12 / numba 0.68 installed
  cleanly into the existing environment (no existing package changed).
* **A canvas-renderer race in Leaflet 1.9** (a queued redraw firing after the map was removed → "clearRect of undefined") surfaced
  once the detection map used the canvas renderer; `GeoMap` makes the renderer's redraw entry points no-ops once its context is gone.
* **Auto-sweep vs interaction.** The first e2e run caught a real UX bug: if imagery preloading was slow, the on-load sweep could begin
  *after* the analyst had started using the slider and yank the view away. Any interaction now cancels a pending auto-sweep.


## Map pass: layout primitives and the local basemap

**Layout primitive.** Every multi-panel grid (`.dash-pair`, `.dash-triple`, `.wb-grid`, `.changes-grid`, `.search-grid`, `.detect-grid`,
`.disc-grid`, `.fp-grid`, `.settings-grid`, `.data-grid`) uses `align-items: stretch`, so panels in one row share a height. A column
that stacks panels gives its leftover height to the panel marked `grow`; `stack` makes a body a flex column; `fill` + `<GeoMap fill>` makes
a map take the rest of its panel. Lists keep their own max-height and scroll. `tools/e2e-console.mjs` asserts equal row heights, three
equal unclipped result-card buttons (down to the grid's 196 px minimum card) and uniform fingerprint cards.

**Basemap backend** (`geoseek.analyst.basemap`; routes `/ui/basemap/{z}/{x}/{y}`, `/ui/basemap/coverage`, `/ui/basemap/stats`,
`/ui/clusters/geo`). Web-Mercator tiles assembled from imagery already in the archive; fully transparent where nothing is staged (the dark
canvas shows through, nothing is filled in). Nothing is written to `data/` and no new dataset is introduced, so there is no provenance entry.

* Sentinel-2: one acquisition per MGRS granule - the lowest mean tile cloud fraction, newest on a tie. The map caption states this rule and
  that it is *not* the date of the overlaid features. z >= 12 is composited from the catalog's tile thumbnails; below that, from per-granule
  overviews decoded lazily from the same band rasters and held in byte-capped LRUs (overviews 160 MiB, rendered tiles 96 MiB). Three levels,
  each about as coarse as the map pixel it serves: **mid** 1/8 for z 11 (4 x 4 real pixels averaged per 8 x 8 block), **fine** 1/16 for z 8-10
  (4 x 4 per 16 x 16), **coarse** 1/64 for z <= 7 (8 x 8 per 64 x 64). Only the sampled rows are inflated - area-averaging forces a full
  inflate of every row of a ~170 MB strip-deflate file and took 116 s on a cold whole-archive view; the three bands are read in parallel.
  Sampling error against the full-area average of the same band, mean absolute over 6 scenes: **mid 1.7 %, fine 2.1 %, coarse 1.7 %**
  (cheaper samplings measured and rejected: coarse 4 x 4 = 4.0 %, 1 x 1 = 19.7 %; mid 2 x 2 = 4.7 %, 1 x 1 = 13.6 %).
* **Nothing is warmed at boot.** The catalog summary behind the basemap (per-granule date, cloud, footprint hull; ~0.9-1.6 s) is built by the
  first map request, once, under a lock; the cluster geography (`/ui/clusters/geo`, ~2.7 s) by the first Discovery visit. Boot is unchanged.
  Measured: a background-thread warm-up made the first search 2-3x slower (median 138 ms vs 59 ms: GIL contention with the catalog read) but
  never near the 1 s that `test_search` allows, even with 14 busy processes on 16 cores (max 249 ms); the lazy and boot-time variants first
  searched in 59-73 ms. Boot-time warming cost +3.8 s of start-up for nothing a search needs.
* `?scene=<observation_id>` serves a staged Maxar scene (R/G/B bands), looked up in the catalog - never used as a path. The Detect map uses
  it: the Sentinel-2 archive has no coverage where those scenes lie, and the scene is the imagery the detections were found on.
* Measured (this machine; AC power verified at the end of the sweep - charging, battery 30 %, not re-checked per run; the OS file cache could
  not be flushed, so "cold" means cold process caches, with files read in earlier runs possibly still in the OS cache). Fresh server, lazy
  summary included: whole-archive first view 3.9 s (4.6 s on another run); z6-7 0.04-0.4 s; first z8 view of one region 0.6-1.7 s; z9-10
  0.1-0.6 s; **z11 first view (6 tiles) 1.7 s** (3.8 s when it still used thumbnails); first z12 view (12 tiles, thumbnails) 3.0 s, z13 0.8 s;
  warm anywhere 20-30 ms. The z11 gain moved the cold thumbnail cost one zoom level deeper: z11 no longer warms the thumbnail cache for z12.
  Resident for all 15 granules at all three levels: coarse 1.5 MB + fine 24.0 MB + mid 96.1 MB = 121.6 MB of the 167.8 MB cap, no evictions;
  server RSS 2.46 GB at that point.

**Maps** (all through `components/GeoMap.tsx`): `basemap={{}}` adds the local layer and an honest caption (granules in view and their date range
come from `/ui/basemap/coverage`); `pin` points are numbered / labelled markers addressed by id, with `hoverId` / `onHover` for card <-> pin
linking; polygons can be `selected` (stronger stroke, centre mark, permanent tag); circles carry a permanent radius tag; `cells` draws tens of
thousands of cluster cells on a canvas layer. The only tile layer in the source is the same-origin `/ui/basemap` one
(`tests/test_frontend_offline.py` enforces exactly that).

## Explainability and false-alarm suppression

Two features on top of the existing change pipeline. **No model changes and no new claims**: both are read-only views of what the
pipeline already stored, in `src/geoseek/analyst/explain.py`, behind `GET /ui/candidates/{id}/explain` and `GET /ui/pipeline/funnel`.

### Why this was flagged (Changes → the full-width row under the workbench)

The pieces already existed but were scattered (rule, spectral deltas, gates, confidence lines, SAR, terrain, the spectral overlay) and
read as debug output. They are now one panel, with the older panels reduced to what they own (the Confidence panel keeps the ring, band
and SAR status; Change details keeps the facts and the rule; both link down to the explanation).

- **Lead sentence**, generated server-side from the rule that fired and the stored anomalies, e.g. *"Built-up index rose sharply (+0.38 vs.
  season) while vegetation fell (−0.09 vs. season) — consistent with new construction. Present on 4 of 5 acquisition dates since
  2021-03-04."* The words ("sharply" / "clearly" / "only just past the 0.05 threshold") come from where the value sits relative to the
  classifier's own thresholds and the confidence engine's own ramp widths; every number is a stored value. An unclassified candidate
  says that no rule matched. A demoted one quotes the pipeline's own trajectory note and the penalty factor.
- **Evidence in order of influence.** The six confidence terms, largest realised effect on *this* candidate first, each with its weight
  share, its 0–1 strength, its cost in confidence points and a supports / weak chip; then the three multipliers (gate down-weights,
  temporal-support penalty, SAR). Confidence is a weighted geometric mean, so the decomposition is exact: `confidence = Π term^(w/Σw) ×
  down-weights × penalty × SAR`, and a term's cost is the change in the final number if that term alone were neutral.
- **Not stored, so recomputed and checked.** The pipeline stores only text lines for the confidence terms. The terms are re-derived by calling
  `compute_confidence` on the inputs the sidecar did keep, and the panel shows *"recomputed from the stored inputs: 91.4% ✓"*. If that ever
  stopped matching the stored value it would say so in red. A test asserts it for **all 841 stored candidates (max error 0.0004)**.
- **Spectral change against thresholds**: per index the change, the scene's seasonal change and the anomaly, and every test of the
  assigned type with its threshold, the candidate's value and met / not met.
- **Where the evidence is thin or missing** gets its own amber block, as plain as the strong evidence: *"No Sentinel-1 coverage for this
  candidate"* first, then weak terms, then "no rule matched" / "no terrain sampled" where they apply. Terrain is labelled as context, not
  evidence (it does not enter the confidence).
- **On the pixels**: *Show NDBI · NDVI on the pixels* opens the existing NDVI / NDBI / NDWI maps for the after-date tile covering the
  candidate with its footprint outlined and the indices that decided the type starred. They show the end state; the change is the
  difference between dates, and the panel says so.
- **Raw decomposition** (the engine's own text lines, factors to four places, reproduction error) is behind a toggle.
- There is **no attention map or saliency** over the embedding: its patches are ~366 m across (§2.5).

### Pipeline trace (beside it) and the suppression funnel (rail: **Pipeline**)

*Per candidate*: eight stages in the order the pipeline applied them (shape & size, image quality, alignment, brightness match,
seasonal check, spectral typing, temporal persistence, SAR), each with verdict and effect on confidence. A demoted candidate gets a note
saying why and that it was demoted, not dropped.

*Pipeline level*: raw model components → after each gate → candidates, per acquisition pair, read from
`ayodhya_change_report.json#pairs[*].suppression`; below it what happens to the survivors (typing, persistence classes with their
penalties, SAR, confidence bands, counted from the retained candidates), the per-gate cards (what it checks, thresholds, what it removed
on this pair, and its measured effect on the labelled OSCD benchmark from `ablation_study.json`), and a sample of demoted candidates
with their reasons.

Things the panel is explicit about, because they are true of the artifacts:

- **Counts are for the current 5-date run**, not the 3-date run described in `EVALUATION_REPORT.md` §9.3 (13,563 → 1,104). On the span
  pair 2019-03-30 → 2026-03-08 the area floor removes 87.9% of raw components (the 91% in §9.3 is the older pair); the other five pairs
  range 85.6% – 94.9%. Do not compare the two sets of figures.
- **First-rejecting-gate attribution.** The report counts each rejected component against the first gate that rejected it. The area floor
  is applied during component extraction (before the other four checks, which sub-floor components never receive), and the report stores it
  merged with any later morphology rejection as one number. They cannot be separated without the pipeline emitting them separately.
- **Registration and radiometric never reject** (they down-weight); the funnel shows them as "removes none", with the down-weighted
  count taken from the retained candidates' traces.
- **Gates are Ayodhya-calibrated.** A scope note says so wherever an effect is stated. The labelled benchmark table shows why: on OSCD the
  phenology gate costs ≈ −0.19 F1 (it is tuned to Ayodhya's seasonal signal) and the card flags it. The panel also warns if the ablation
  was run with different thresholds from the pipeline's current ones.
- **Rejected components are not retained.** The pipeline writes only counts per gate; sub-floor components never had a trace and the others
  were dropped from memory once counted. The panel says that and names what would have to be emitted (per suppressed component: id,
  bounding box, area, first rejecting gate and its trace values, and the area-floor drops counted separately). No storage was added, and no
  control suggests rejected components can be browsed. What *is* inspectable are the demoted candidates: all of them are in the queue.
- **SAR**: the current run staged no Sentinel-1, so the corroboration term is neutral for every candidate; both screens say so.

Tests: `tests/test_explain.py` (29: sentences follow the stored values, decomposition reproduces the confidence for synthetic cases and
for every stored candidate, ordering, down-weights, SAR present / absent, funnel counts sequential and flagged when they do not add up,
labelled-benchmark deltas and config mismatch, rejected-not-retained) and 12 new e2e steps (every figure in the Why panel, trace and funnel
is compared with the API response; selecting another pair changes every figure, which is the proof they are read rather than typed).

## Discovery explains itself, Data aligns, brand mark (Oct 2026)

No model or data changes. Two read-only additions to the backend, the rest is the React console.

**Discovery.** The screen now answers "why am I looking at this".
- *Interpretation line* (`lib/clusterInsight.ts`, node-tested in `tools/selftest-lib.mjs`): counted from `/discovery/clusters`, never templated.
  Line 1: clusters, tiles, how many are *concentrated* (one region holds ≥ 80 % of the cluster) vs spread. Line 2: the most notable fact - the
  closest-concept label shared by the most clusters that each sit in a *different* region (live run: 7 of 9 "bare dry open ground" clusters, in 5
  regions); if no label does that it falls back to the most widely spread cluster. The two thresholds (80 %, and 5 % for "a region is spanned")
  are printed under the line.
- *Per-cluster summary*: hover = preview, click = selected (zoom + dim the others, Esc or "Show all clusters" clears). Tiles, share, closest concept
  (+ similarity), regions spanned (≥ 5 % each) and touched, extent (overall bbox **and** the middle-95 % core, because a few stray tiles stretch
  a cluster's bbox across the archive; the map zooms to the core), region mix, and 4 example tiles. Hover and select also work from the map
  (cell hit-test in `GeoMap`) and from the matrix. The summary has a fixed height so hovering never resizes the grid row.
- *Region x cluster matrix*: share of each cluster's tiles per region (rows sum to 100 %), rows grouped by closest concept.
- *Examples* are chosen by a fixed rule (`ui_support._pick_examples`): the lowest-cloud tiles, taken in turn from the regions holding ≥ 5 % of the
  cluster, tie-broken by a hash of the tile id. They are examples, not the most typical members, and the panel says so.
- Backend: `/discovery/clusters` now also returns the stored `region_purity` (per-cluster, per-region tile counts already in `tile_clusters.json`);
  `/ui/clusters/geo` gained `examples` per cluster (built in the same single catalog pass as the cells).
- The basemap caption (one representative acquisition per granule, unstaged areas dark) is unchanged.

**Data.** Archive-inventory tables use fixed layouts with `colgroup` widths sized to the longest real value, headers never wrap, GSD is rounded
(`0.31 m`, exact `0.30517578 m` in the tooltip), long sensor / region strings are cut with an ellipsis and a tooltip; the acquisition dates are a
chip row of their own.

**Shell.** Inline-SVG India flag by the IST clock (3:2, #FF9933 / #FFFFFF / #138808, 24-spoke Ashoka Chakra in #000080). New original logo mark
(`components/Brand.tsx`: a globe crossed by a satellite's track) drawn in `currentColor` only, so it reads in one ink; `public/favicon.svg` uses it.
The GEOSEEK wordmark carries a saffron -> white -> green gradient via `background-clip: text`, drifting once per 32 s (~4 px/s, white and green
softened to sit near the page ink); with `prefers-reduced-motion` it is the same gradient, static.

Tests: `tests/test_map_support.py` (+3: examples spread over regions, repeatable, clearest-first; `/discovery/clusters` region counts),
`tools/selftest-lib.mjs` (+cluster insight, core box, GSD), e2e +4 steps (interpretation = API counts, summary fields = API and hover / select /
dim / Esc, matrix cells = API and map-cell click, inventory layout, shell flag/logo/wordmark incl. reduced motion). Screenshots (ignored by git):
`docs/screenshots/ui4/{before,after}`.

## Raster preview, Temporal tab, logo orbit, intro (Oct 2026)

No model, gate or pipeline change; no new claim about the pipeline. New runtime dependency: none (`geotiff` was already bundled; its
pure-JS decoder chunks are now actually fetched, still same-origin). Offline pins regenerated (`build-pins.json`), `test_frontend_offline`
green.

**1. Data: the file is visible.** `lib/rasterPixels.ts` decodes the dropped file in the browser as a *preview*: native resolution when the
longest side is <= 1536 px, else the file's own largest overview that fits, else a nearest-neighbour decimation; the caption states which
(and that nothing is uploaded). `lib/rasterMath.ts` (pure, node-run) does the arithmetic: per-band percentile stretch (default 2nd-98th,
also 1-99, 5-95, min-max; the percentiles and the per-band values are printed), 3-band = RGB, 4-band = B04/B03/B02 true colour (assumed
Sentinel-2 order B02,B03,B04,B08, stated; band DESCRIPTIONs in the file win), any band selectable, single-band grey. The footprint is drawn
(as its four ground corners) on the `/ui/basemap` layer zoomed to the file's bounds. *Comparison* (`components/RasterCompare.tsx`): the
nearest archive acquisition in time (from `coverage.acquisitions`, new) is pre-selected; the file and that year's basemap tiles are put on
one Web-Mercator grid (file resampled nearest-neighbour from its own CRS, never blended); each is converted to brightness and scaled to its
own 2nd-98th percentile over the pixels both cover; the result is file - archive on a blue-white-red scale. It is labelled **indicative
visual difference - not pipeline output**, lists the four ways an ad-hoc upload differs from the archive (CRS / pixel size, radiometric
scaling, registration, date / season / clouds), names the archive date actually served (the archive serves one acquisition per granule per
year), and reuses none of the pipeline's wording, scores or gate chrome. No coverage -> it says so and offers nothing.
Limits: the preview's statistics are computed on the preview, not the full-resolution file; ZSTD / LERC files cannot be previewed (above);
the archive side is an 8-bit display rendering, and the 2024 and 2026 renderings carry strong colour casts (mean RGB 51/52/75 and
137/163/165 vs 77/78/89 for 2019 on the same tile), which a brightness-only comparison does not remove.

**2. Temporal tab** (rail "Temporal", `#/temporal` and `#/temporal/upload`; two modes, never in one chart).
*Archive mode*: new `GET /ui/temporal/archive[?bbox]` (`analyst/temporal.py`) derives everything from the stored ranked candidates and the
report: per acquisition interval the candidate count by type, area by type, persistence class, held vs demoted, SAR corroboration; a
cumulative area per type at each acquisition date; the report's own per-interval pair-run survivor counts as a **separate** basis. All
stored candidates come from the single 2019-2026 span run, so they are placed in the interval in which the pipeline *first supported* them
(`earliest_supported`); 337 are visible only across the whole span and 35 in no interval, and are shown apart, never folded into one.
Live finding: all 337 span-only candidates are transient (demoted). Persistence of last-interval candidates cannot be tested yet (stated).
SAR: no Sentinel-1 is staged in this run, so coverage is 0 of N in every interval and the API's note says so. The x axis is a real
calendar (`lib/calendarAxis.ts`, pixels proportional to days): bars span the interval between two acquisitions, hatching marks the gap, the
cumulative and SAR marks are points that are not joined. Clicking a mark opens Changes with exactly that filter
(`/candidates?first_detected=FROM_TO[,...]|none` is new; the chip says where the queue came from and clears it).
*Upload mode* (`lib/rasterProfile.ts`): needs >= 2 dropped files with a placeable footprint, a date (file header, or entered by the analyst
and labelled so), overlap, and >= 2 distinct dates - otherwise it states what is missing and draws nothing. Per-band (matched across files by
band *role*, not position) and NDVI / NDWI / NDBI (where the bands allow) mean, std, p10 / p50 / p90 over the ground the files share,
sampled on one Web-Mercator grid. Labelled **SPECTRAL PROFILE - not change detection**; it states that the model, gates, persistence and
confidence engine are not run on uploads, that files are not co-registered, and that index values are not comparable across sensors. It
emits no candidates or scores.

**3. Logo.** The rail logo's satellite flies one 48 s lap of its orbit (SMIL `animateMotion` in the inline SVG; in front of the globe on
the near half, hidden behind it on the far half; no asset). Not animated at all under `prefers-reduced-motion`.

**4. Intro.** `components/Intro.tsx`: dark field, the mark drawn stroke-first, GEOSEEK typed in with a saffron / white / green sweep, one
tagline, dissolve into the Dashboard (3.4 s; 1.4 s and still under reduced motion). The app is mounted and live beneath it; the skip control
and "Don't show this again" (localStorage) are on screen from the first frame; click or any key skips (Tab is left alone so the checkbox is
reachable). Shown once per session (sessionStorage) and never on a deep link. The vendored three.js globe was **not** reused: it pulls in
the Earth textures and the three chunk, which is not cheap; the mark already contains a globe.

**Verification against a server-side computation** (`tests/test_react_raster_pixels.py`, fixtures cut from the archive's own staged Ayodhya
Sentinel-2 scenes by `tests/raster_fixtures.py`; the TypeScript runs under Node and every number is compared with numpy / rasterio / pyproj):
band statistics of the whole file equal numpy to 2.4e-13 relative; NDVI / NDWI statistics within 3e-8 relative (browser holds an index as
float32); stretched composite pixels equal exactly; the Mercator resample equals an exact pyproj nearest-pixel reference on > 99.99 % of
pixels (GDAL's warper, which uses a 0.125 px approximate transformer, agrees on ~94.5 % - that is GDAL's tolerance, not an error here); the
spectral profile over the shared ground has the same grid (zoom 14, 511,225 cells), identical valid-pixel counts and identical statistics;
the preview modes (native / overview / decimated) return the file's own pixels. The e2e repeats the composite and profile checks on the DOM.

Tests: `tests/test_temporal_archive.py` (6), `tests/test_react_raster_pixels.py` (9), basemap coverage `acquisitions`, `selftest-lib` calendar
axis, e2e +26 steps (intro x6, logo, dropzone caption, preview, bands, footprint on basemap, comparison labelling + a real difference, no
coverage, ZSTD, archive-mode data = API, click-through, pair-run basis / region, upload mode x4). `tools/shots-temporal.mjs` makes the
screenshots.
