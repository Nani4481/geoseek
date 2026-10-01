# React console (`/react/`)

A **parallel** analyst console built with Vite + React + TypeScript, styled as a dark-navy dashboard (icon rail, 3D globe
hero with region pins, stat cards, alert feed, temporal strip, before/after slider, change-details and confidence
panels). It is served by the same FastAPI process as the existing frontend and makes **zero external requests**.

* Existing frontend: `/app/` — **unchanged** (nothing under `src/geoseek/analyst/web/` is modified; it is only *read*,
  see §6).
* React console: `/react/` — source in `frontend-react/`, build output in `src/geoseek/analyst/web_react/`.

## 1. Build and serve

| | |
|---|---|
| Toolchain used | Node **v24.14.0**, npm **11.9.0**, Python 3.11.16 (the `geoseek` conda env) |
| Runtime dependencies | `react` 19.3.0, `react-dom` 19.3.0 — nothing else (no router, state, chart, CSS or map library) |
| Build dependencies | `vite` 8.3.2, `@vitejs/plugin-react` 6.1.1, `typescript` 5.9.3, `@types/react` / `@types/react-dom` 19.3.0, `@types/leaflet` 1.9.22 |
| Pinning | exact versions in `package.json` + `package-lock.json` (a test asserts exact pins, the two-package runtime set, and that the lockfile matches the pinned build) |
| Reused, already-vendored libraries | three.js, OrbitControls, Leaflet 1.9.4, the Blue Marble textures and Inter / JetBrains Mono fonts — aliased straight from `analyst/web/vendor` and `analyst/web/fonts`, so the shipped bytes are the already-hash-pinned ones |
| Charts | hand-rolled SVG/CSS bars (no chart library, so nothing extra to audit) |

```bash
cd frontend-react
npm ci                     # exact versions from package-lock.json
npm run build              # tsc --noEmit && vite build  ->  ../src/geoseek/analyst/web_react/
npm run build:pinned       # build, then re-pin:  python -m geoseek.staging.react_build_pins
npm run dev                # Vite dev server on :5173, proxying the API to 127.0.0.1:8000
npm run verify:offline     # runtime network audit against a running backend (see §4)
```

Serve: `uvicorn geoseek.search.api:app --host 127.0.0.1 --port 8000`, then open **`http://127.0.0.1:8000/react/`**.
`GET /` also lists `"react_ui": "/react/"`.

**The build output is committed**, not built at deploy time. An air-gapped machine therefore needs Python only — no Node,
no npm, no network — and the committed bytes are exactly what the tests pin. Rebuild only when `frontend-react/` changes,
then re-pin (`npm run build:pinned`). `.gitattributes` marks `web_react/**` as `-text` so no checkout converts line endings
and breaks the hashes; `node_modules/` is git- and docker-ignored.

## 2. The three tiers — what is live, what is surfaced, what is a mockup

Every screen section is a `Panel` that carries its tier **in its own header**, and the rail marks each destination:
**green dot = live**, **cyan dot = live UI over existing backend**, **dashed amber marker = roadmap**.

| Tier | Badge on the panel | Dashed frame? | Features |
|---|---|---|---|
| **1 — LIVE** (wired to the real API) | `● LIVE` | no | see below |
| **2 — LIVE UI · EXISTING BACKEND** | `● LIVE UI · EXISTING BACKEND` | no | see below |
| **3 — ROADMAP — NOT IMPLEMENTED** | amber `ROADMAP — NOT IMPLEMENTED` pill + a "requires …" strip inside the panel chrome + diagonal watermark | **yes**, amber, hatched header | see below |

### Tier 1 — live

| Feature | Screen | Backend |
|---|---|---|
| Semantic text search, ranked, with date / cloud / box filters; measured latency per query | Search | `GET /search/text` |
| Find more like this — from a result tile, or by clicking any map point | Search | `POST /search/image`, `GET /discovery/similar` |
| Bounding-box spatial filter — drag a box on the map, or pick a region | Search, Changes | `bbox` param on the above and on `GET /candidates`; `GET /regions` |
| Change candidates: filterable queue, before/after slider (mouse, touch, arrow keys), spectral type + index anomalies, confidence ring with its breakdown and five evidence gates, SAR-corroboration status | Changes, Dashboard | `GET /candidates`, `GET /candidates/{id}`, `GET /candidates/{id}/imagery` |
| Object detection: stored oriented boxes drawn over the scene, class toggles, hover details, measured accuracy and the backend's own caveats | Detect | `GET /detect/*` |
| Discovery / clustering: 19-cluster table, cluster map, seed explorer | Discover | `GET /discovery/clusters`, `/discovery/cluster-map.png`, `/candidates/{id}/similar` |
| Review queue: confirm / reject / reopen with a note, append-only audit trail, GeoJSON export (one candidate or the whole filtered set) | Changes | `POST /candidates/{id}/decision`, `GET /audit`, `POST /export` |
| Dashboard stat cards, alert feed, globe pins | Dashboard | `GET /ui/metrics`, `/ui/latency`, `/notifications`, `/restricted-zones`, `/candidates` |

### Tier 2 — live UI over backend computation that already existed

| Feature | What is rendered | Where it comes from |
|---|---|---|
| **Temporal timeline + onset slider** | The catalog's real acquisition dates, placed at their true calendar position (they are uneven); a dashed bracket for the window in which the change must have occurred; the first date that shows it; per-interval changed/stable with probability; persistence as **"N of M observations"** (e.g. `4 of 5`). Only dates that exist in the catalog are drawn. | `GET /ui/candidates/{id}/timeline`, a read-only projection of the change pipeline's own trajectory (`geoseek.temporal.persistence`) |
| **Structural fingerprint gallery** | Visually similar tiles, with a similarity comparison panel along only the axes the catalog can measure: appearance (exact vector similarity), **proximity** (great-circle km between tile centres), **footprint** (tile ground area from catalog geometry), cluster membership, cloud cover. **There is no thermal axis** — the system has no thermal band. | `GET /candidates/{id}/similar`, `/discovery/similar`, `GET /ui/tiles` |
| **Briefing mode** | Full-screen; control panels hidden on demand; high-contrast theme; pointer spotlight; a keyframe camera tour that arcs between sites (bookmarked sites, or the featured findings the backend itself highlights); canvas annotation (pen, arrow, box, undo, clear — kept per site). **Pure frontend.** | none (bookmarks live in the viewer's `localStorage`, guarded) |

### Tier 3 — roadmap mockups (non-functional, placeholder content)

| Panel | Badge note (shown in the panel chrome) | What the mockup does |
|---|---|---|
| Tactical dossier export (PDF preview) | requires MGRS conversion + STAC sun-elevation/off-nadir capture at ingest | Static paper layout with placeholder MGRS/UTM, before/after/diff chips, sensor-provenance block, sign-off block. The export button is disabled. |
| Drag-and-drop ad-hoc raster ingestion | requires runtime single-scene ingest path | Dropzone that **reads nothing and uploads nothing** (it says so); header-parse badges all read "— not run". |
| 3D vector-space visualizer | requires offline-precomputed UMAP projection | three.js scatter of **synthetic** random blobs, labelled as such. No projection exists in the archive, so none is shown. |
| Threat buffer rings | requires an infrastructure feature layer to intersect against | Concentric circles at fixed illustrative radii on the offline map; feature counts read "—". |
| Attention / XAI overlay | requires ViT-L/14 migration — current ViT-B/32 resolves to **~366 m** patches (computed live, see §5) | Hand-painted heat blobs on a patch grid, labelled as concept only. |

### Deliberately not built

No UI exists — not as a mockup, a disabled button or a menu entry — for **InSAR / interferometric fringes** (the system
ingests Sentinel-1 **GRD**; interferometric phase is not recoverable from GRD) or for **change-rate / velocity /
completion-horizon estimation** (the catalog holds five acquisition dates, over an uneven 7-year span, which cannot support
a defensible rate estimate). An end-to-end test asserts none of those terms appears on any of the 11 content routes.

## 3. Where every displayed number comes from

There are no hardcoded statistics: each card fetches at runtime, shows `—` plus a visible "unavailable" state if its
endpoint fails, and (for the figures with a measurement behind them) carries its source in the tooltip.

| Card | Value shown | Endpoint → source |
|---|---|---|
| Tiles indexed | tile count (+ searchable vector count) | `/ui/metrics` → `repo.count_tiles()`, engine vector count |
| Scenes / Regions / Sensors | scene count; region count; distinct sensors | `/ui/metrics` → `repo.list_scenes()`, `list_regions()`, distinct `Collection.sensor` |
| Search latency | median of 5 fixed probe queries, p95 beside it, "measured now" | `/ui/latency` → engine-reported `latency_ms` after a discarded warm-up; cached 30 s |
| Change detection F1 | held-out OSCD test F1 at the deployed 0.80 operating point, with P / R | `/ui/metrics` → `model_card.eval.test_at_precision_favouring` embedded in the change-model checkpoint |
| Object detection AP50 | **0.845** small-vehicle, **0.854** ground vehicles (DOTA official val, 458 images, with bootstrap CI) | `/ui/metrics` → `data/detect_eval/eval_results.json#val_full_image_v15` |
| Change candidates | candidate count, analyst-decision count | `/ui/metrics` |
| Findings by region / by change type | candidates per region (point-in-box) and per change type, as bar lists on the Dashboard | `/ui/metrics` → `findings_by_region`, `findings_by_type` |
| Every confidence / persistence / area figure | per candidate | `/candidates`, `/candidates/{id}`, `/ui/candidates/{id}/timeline` |

Both AP50 figures were checked against `docs/EVALUATION_REPORT.md` §15 (0.8454 and 0.8536) and against the stored eval JSON.

### Things the numbers say that a mockup would not

* **"Object Detection 53.1%" in the reference mockup is not an object-detection number.** 53.1 % is the change-detection
  *validation precision at the 0.80 threshold* (report §8.1). The card is wired to the measured AP50 instead.
* **The DOTA-val AP50 does not transfer everywhere.** On the independent xView test half the same detector scores
  **0.531** (small-vehicle) and **0.315** (large-vehicle); on the staged Maxar tiles it finds aircraft and large objects but
  only a few percent of visible cars. The Dashboard card prints the xView figures under the DOTA one, and the Detect screen
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

   The only URL strings in the shipped bundle are eight distinct inert ones (nine file-level occurrences), each classified:
   five W3C XML-namespace identifiers (`createElementNS`), a React error-doc string, a three.js console-warning string, and
   Leaflet's attribution `href` (never rendered: every map is created with `attributionControl: false`). None is ever
   dereferenced.
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
| `frontend-react/tools/e2e-tier1.mjs` | 21 steps in headless Chrome with **real mouse and keyboard input** (assertions poll for the expected UI state rather than sleeping; 5 consecutive full passes on the final build): stat cards and findings-by-region/type equal the API; text search, more-like-this, region box, drag-drawn box, map-point click; queue filter equals `/candidates`; slider drag and arrow keys; timeline pick and change mask; confirm → append-only audit row → reopen → reject; GeoJSON export counts; detection boxes and class toggle; discovery; fingerprint axes; briefing tour / pen / undo / spotlight / contrast / hide / Esc; all five roadmap badges; scope check; offline pill |

Confirm / reject / reopen append **permanent** audit rows, so the e2e script refuses to write unless `--allow-writes` is
given and must be pointed at a **scratch** backend:

```bash
cp data/index/tiles.sqlite /tmp/scratch.sqlite
DATABASE_URL=sqlite:////tmp/scratch.sqlite uvicorn geoseek.search.api:app --port 8001
node frontend-react/tools/e2e-tier1.mjs --base http://127.0.0.1:8001/react/ --allow-writes
```

(`POST /export` also writes a file under `data/change_model/exports/` — that is the existing endpoint's behaviour.)

## 6. Existing frontend untouched

`git diff --stat -- src/geoseek/analyst/web/` and `git status --short src/geoseek/analyst/web/` are empty. The only edits to
existing files are: `src/geoseek/search/api.py` (a `/react` static mount, the four `/ui/*` read-only routes, and a
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
