# GeoSeek Frontend Audit (recon only, no code changed)

Server used for live checks: `uvicorn geoseek.search.api:app` on `127.0.0.1:8000`,
production catalog (105,245 vectors, 841 ranked candidates). Raw JSON samples are
in `docs/api_samples/`. All paths below are relative to
`src/geoseek/analyst/web/` unless stated otherwise.

---

## SECTION 1 — File inventory

| file | lines | purpose |
|---|---|---|
| `app.js` | **1904** ⚑ | The entire SPA: hash router, 7 view controllers, `CoordMap` 2D canvas class, `api()` fetch wrapper, guided-demo script. |
| `style.css` | **711** ⚑ | Whole design system + every view's styling, in one file. |
| `globe.js` | **564** ⚑ | Interactive three.js globe, dynamically imported by `app.js` only when Overview mounts. |
| `index.html` | 443 | DOM shell for all 7 views, header nav, footer, demo overlay. |
| `mosaic_index.json` | 255 | Index (bbox/file/date/observation_id) of 18 real satellite WebP mosaic tiles. |
| `basemap.json` | 1 (minified, 407,650 bytes) | Natural Earth vector layers: coastline (134 features), countries (9), states (36), rivers (135), cities (212). |
| `vendor/OrbitControls.js` | **1417** ⚑ | Vendored three.js camera-orbit controls (third-party, not house code). |
| `vendor/three.module.min.js` | 6 (minified, 670,681 bytes) | Vendored three.js core. |
| `vendor/earth/day.jpg` / `night.jpg` / `specular.jpg` | — | NASA Blue Marble globe textures, 351,650 / 50,257 / 73,885 bytes. |
| `mosaic/*.webp` (18 files) | — | The actual satellite imagery tiles, 2.5 MB total, 6.6 KB–291 KB each. |

⚑ = over 500 lines: `app.js`, `style.css`, `globe.js`, `vendor/OrbitControls.js`.

**Total CSS: 711 lines** (one file). **Total house-authored JS: 2,468 lines** (`app.js` + `globe.js`). Including vendored JS: 3,891 lines.

---

## SECTION 2 — CSS audit

### Custom-property (design-token) layer

Yes, a real one already exists, all defined in `:root` in `style.css:5-83`:

- **Surfaces** (7): `--bg #0a0d13`, `--bg-grad #0d1119`, `--surface-0..4` (`#10141d`,`#141924`,`#1b2230`,`#232b3c`,`#2b3448`)
- **Ink** (4): `--ink #e9edf5`, `--ink-dim #aab3c4`, `--muted #7c879b`, `--muted-2 #5b6579`
- **Borders** (3): `--line #232b3a`, `--line-strong #323d52`, `--line-accent #3a5a99`
- **Accent/alert** (5): `--accent #4c8dff`, `--accent-dim #2a4d8f`, `--accent-ink #06111f`, `--alert #ff5d73`, `--alert-dim #7a2632`
- **Semantic** (4): `--ok #35d68c`, `--warn #ffb454`, `--bad #ff5d73` (== `--alert`), `--chip` (= `var(--surface-2)`)
- **Change-type palette** (6): `--water_gain #4c8dff`, `--water_loss #b892ff`, `--construction #ff9d47`, `--clearance #ffd166`, `--road #93a1b8`, `--other #35d68c`
- **Basemap geo-layers** (6, all rgba): `--geo-land`, `--geo-border`, `--geo-state`, `--geo-coast`, `--geo-river`, `--geo-grid` (`style.css:50-55`)
- **Type scale** (8): `--fs-display 32px` → `--fs-micro 10.5px` (`style.css:58-65`)
- **Spacing rhythm** (7): `--sp-1..7`, 4px→36px (`style.css:68-69`)
- **Radii** (5): `--r-sm 6px`, `--r-md 10px`, `--r-lg 14px`, `--r-xl 20px`, `--r-pill 999px` (`style.css:72`)
- **Elevation** (4): `--shadow-1..3`, `--focus-ring` (`style.css:75-78`)
- **Motion** (1): `--ease: cubic-bezier(.2,.7,.3,1)` (`style.css:80`)

Total: **53 custom properties**, all in one `:root` block, no per-component overrides, no `[data-theme]` or dark/light split (the app is dark-only, `color-scheme: dark` at `style.css:82`).

### Colour literals outside the token layer

Non-var hex in `style.css`: `#fff` (7×: lines 104,147,152,204,229,418,453,560), `#06101f` (1×, badge text colour, line 289), `#241a00` (1×, clearance badge text, line 293), `#040608` (1×, globe canvas bg, line 388). All four are small, deliberate one-offs (readable text-on-colour-chip contrasts + a WebGL clear colour), not systemic leakage.

`rgba()` literals not routed through a token: `rgba(255,255,255,.15)` inset highlight on `button.primary` (line 230), `rgba(6,10,18,.78)` featured-card image-label backdrop (line 559). The rest of the rgba usage is the 6 `--geo-*` tokens plus the 3 `--shadow-*` tokens — already tokenized.

Colour is **duplicated outside CSS entirely** in JS — see Section 3, it's a correctness issue, not a style one.

### Type scale escapees

A `--fs-*` scale exists (8 tokens) but **23 raw `font-size` px/em values bypass it**: `9px`(633), `10px`(152,560), `10.5px`(315,364,368), `11px`(658), `11.5px`(141,500), `12px`(101,603,646), `12.5px`(519), `13px`(630), `13.5px`(329,564), `14px`(312,660), `15px`(132,514), `16px`(348), `17px`(659), `30px`(538) — line numbers in `style.css`. `font-weight` has **no token at all**; 4 raw values are used throughout (`500`,`600`,`700`,`800`), consistently but never through a variable.

### Spacing / radius / shadow / transition

A `--sp-*` rhythm (7 steps) and `--r-*` radii (5 steps) exist and are used for most layout, but raw pixel escapees exist too — e.g. `border-radius: 8px` (line 141), `7px` (199), `6px`×several (164,291,338,560,569), `3px`×several (327,364,370,707) — mostly small marker/dot radii that were never meant to be on the panel-radius scale, plus two raw `999px` (108, 656) that duplicate `--r-pill` instead of referencing it. `box-shadow` is fully tokenized (`--shadow-1/2/3`, only 2 ad-hoc inset shadows: line 230, 521). `transition` timing is well disciplined: essentially every transition in the file uses `var(--ease)`; durations are hand-picked per rule (`.12s`,`.15s`,`.18s`,`.2s`,`.22s`,`.35s`) with no duration token.

### Dead CSS

Cross-checked all 144 distinct class selectors and every real `#id` selector (`#toast`, `#data-credit`, `#demo`, `#ov-alerts`, `#dt_canvas`, `#dt_classes`, `#dt_counts`, `#dt_hover` — `style.css:469,488,642,683,693,707,709`) against `index.html` + `app.js` + `globe.js`. **No dead selectors found.** The only selectors that don't appear as a literal substring in the JS/HTML are the 6 `.b-water_gain` / `.b-water_loss` / `.b-construction` / `.b-clearance` / `.b-road` / `.b-other` badge classes (`style.css:292-294`) — they're built dynamically via template strings (`` `badge b-${ct}` `` at `app.js:1054`, `` `b-${c.change_type}` `` at `app.js:997`, `` `b-${esc(f.change_type)}` `` at `app.js:837`), so a naive text search misses them; they are in fact the only reason those 6 rules exist.

---

## SECTION 3 — Change-type colour consistency (correctness issue)

Three **independent, hand-maintained copies** of the same 6-colour change-type palette exist. Two agree; one has already drifted:

| type | CSS `style.css:39-44` | JS `CTYPE_COLOR` `app.js:7-10` | Globe `CTYPE_RGB` `globe.js:346-349` (as hex) |
|---|---|---|---|
| water_gain | `#4c8dff` | `#4c8dff` | `[0.30,0.55,1.0]` → `#4c8cff` ✅ match |
| **water_loss** | **`#b892ff`** | **`#b892ff`** | **`[0.54,0.57,1.0]` → `#8a91ff`** ❌ **different colour** |
| construction | `#ff9d47` | `#ff9d47` | `[1.0,0.62,0.28]` → `#ff9e47` ✅ (off by 1) |
| clearance | `#ffd166` | `#ffd166` | `[1.0,0.82,0.40]` → `#ffd166` ✅ match |
| road | `#93a1b8` | `#93a1b8` | `[0.58,0.63,0.72]` → `#94a1b8` ✅ (off by 1) |
| other | `#35d68c` | `#35d68c` | `[0.21,0.84,0.55]` → `#36d68c` ✅ (off by 1) |

**`water_loss` on the 3D globe's candidate points is a visibly different, bluer colour** than the same change type's badge, canvas-map marker, and legend swatch everywhere else in the app. This is a real, present-day bug, not a hypothetical redesign risk — quote the actual literals: CSS/JS use `#b892ff` (184,146,255); globe.js's float triple `[0.54,0.57,1.0]` renders as (138,145,255). The R channel is off by 46/255.

A fourth palette, `DET_CLASS_COLOR` (`app.js:1277-1281`, Object Detection tab), is a *different taxonomy* (vehicle/plane/ship/etc., not change type) so it's not a mismatch per se — but it silently reuses the same 8 literals as `REGION_PALETTE` (`app.js:20-21`) in the same order, so it's really one shared categorical palette wearing two names.

None of these three JS colour tables reference the CSS custom properties — they are separate `const` object literals. Changing `--water_loss` in `style.css` will not touch any of them.

---

## SECTION 4 — JS structure

**Routing** (`app.js:653-670`): `go(route)` sets `location.hash = "#/" + route` (line 655). `router()` (lines 656-668) parses the hash, validates against a fixed `views` array (line 654: `["overview","search","queue","detail","discovery","detect","watch"]`), then does two things: (1) toggles every `.view` section's `active` class so exactly one is visible via CSS `display:none`/`block` (`style.css:170-171`) — **views are never removed from the DOM or truly "unmounted,"** only hidden; (2) calls that route's `ensureX()` function. `window.addEventListener("hashchange", router)` (line 669) is the only trigger. Each `ensureX()` (`ensureOverview`, `ensureSearch`, `ensureQueue`, `ensureDiscovery`, `ensureDetect`, `ensureWatch`) is guarded by a module-level boolean (`overviewInit`, `searchInit`, etc.) so DOM construction + `addEventListener` calls run exactly once per page load; subsequent visits to that route just re-run its data loader (`loadQueue()`, `loadOverview()`, …).

**Shared state**: no store object — plain module-level `let`/`const` globals scattered through the file: `PRES` (line 17, cached `/presentation/summary`), `OBS_DATES` (18), `REGIONS` (19, cached `/regions`), `LAST_LATENCY`/`_inflight` (28-29), `detailState` (1035), `queueIdsOverride`/`queueSort` (935-936), `detState` (1283-1286), `watchEditingId`/`REGION_STATS` (1443,1446), `DEMO`/`DEMO_STEPS` (1708-1769), `globeInstance`/`ovViewMode` (1830). Views read/write each other's globals directly (e.g. Watch reads `REGIONS` that Overview/Search/Queue also populate via `loadRegions()`).

**Data fetching**: one wrapper, `api(path, opts)` (`app.js:30-48`) — wraps `fetch`, records latency into `LAST_LATENCY`, drives the header latency chip (`setChipBusy`/`setChip`, 49-58), throws `Error` on non-2xx (parses `detail` from JSON body if present). Used for effectively all analyst/search endpoint calls. **Bypasses it** (raw `fetch`, `app.js`): `basemap.json` (76), `mosaic_index.json` (88), and the fire-and-forget image prewarm (858, 1785) — deliberately, since those don't want the latency chip/error toast. `globe.js:190-195` also bypasses `api()` with 4 raw `fetch()` calls of its own (`basemap.json`, `mosaic_index.json`, `/regions`, `/candidates?limit=900`) — it does **not** reuse `app.js`'s cached `REGIONS`/`loadBasemap()`, so the globe and the 2D map each independently download the 407 KB basemap and the mosaic index, and the globe's candidate/region counts are a separate point-in-time snapshot from whatever the rest of the app is showing.

Every distinct endpoint the frontend calls, with call site:

| endpoint | file:line |
|---|---|
| `GET /health` | `app.js:1883` |
| `GET /presentation/summary` | `app.js:746, 1792, 1890` |
| `GET /regions` | `app.js:631`; `globe.js:193` |
| `GET /candidates` (list, various query strings) | `app.js:756, 988, 1483`; `globe.js:194` |
| `GET /candidates/{id}` | `app.js:1041, 1208` |
| `GET /candidates/{id}/imagery` | via `<img src>` template, `app.js:1165` |
| `POST /candidates/{id}/decision` | `app.js:1202` |
| `GET /candidates/{id}/similar` | `app.js:1219` |
| `GET /search/text` | `app.js:894` |
| `POST /search/image` | `app.js:906` |
| `GET /tile/{id}/thumbnail` | via `<img src>`, `thumbHTML()` `app.js:644` |
| `POST /export` | `app.js:1021` |
| `GET /discovery/clusters` | `app.js:1237` |
| `GET /discovery/cluster-map.png` | via `<img src>`, `app.js:1245` |
| `GET /discovery/similar` | `app.js:1258` (constructed path) |
| `GET /detect/model-info` | `app.js:1302` |
| `GET /detect/observations` | `app.js:1340` |
| `GET /detect/observations/{id}/tiles` | `app.js:1356` |
| `GET /detect/observations/{id}/tiles/{row}/{col}` | `app.js:1371` |
| `GET /detect/observations/{id}/tiles/{row}/{col}/image.png` | via `<img src>`, `app.js:1380` |
| `GET /watch-areas`, `POST /watch-areas`, `PUT /watch-areas/{id}`, `DELETE /watch-areas/{id}` | `app.js:1643, 1611, 1607, 1636` |
| `GET /notifications`, `POST /notifications/{id}/seen` | `app.js:1643/1877, 1686` |
| `basemap.json`, `mosaic_index.json` (static, same origin) | `app.js:76,88`; `globe.js:191-192` |

**Animation/transition code in JS**: yes, three independent implementations. `CoordMap.flyTo` (`app.js:332-346`) — cubic ease-out over 420ms via `requestAnimationFrame`. `animateCount` (793-803) — counter roll-up, same ease shape, duration passed in (900ms on first paint). `Globe._flyTo` (`globe.js:465-491`) — slerp+lerp camera flight, duration 1200ms default or computed in `_diveToRegion` (446-449, 1500-3200ms). None share a tweening helper; each hand-rolls `1 - Math.pow(1-t,3)`.

**Icon approach**: inline `<svg class="icon">` paths, hand-authored, directly in `index.html` (nav buttons, panel headers) and generated as template strings in `app.js` (`emptyState()` line 645-648, the warning triangle in `renderDetCaveats()` line 1312, the home/layers icons built in the `CoordMap` constructor lines 157-162). No icon font, no external icon library, no sprite sheet — every icon is its own inline `<path>`.

---

## SECTION 5 — Canvas map (`CoordMap` class, `app.js:137-624`)

- **Natural Earth data**: `basemap.json`, 407,650 bytes, fetched once via `loadBasemap()` (`app.js:74-84`), cached in a module-level promise, shared by every `CoordMap` instance on the page (but **not** shared with `globe.js`, which re-fetches its own copy — see Section 4).
- **Projection math** (`_proj()`, `app.js:370-385`): a plain equirectangular (plate carrée) transform, no library. Given `bbox=[w,s,e,n]`, scale `k = min(W/(e-w), H/(n-s))` (aspect-preserving, letterboxed), with `x(lon) = ox + (lon-w)*k`, `y(lat) = H - oy - (lat-s)*k`. **Zoom and pan are both represented purely as `this.bbox`** (a 4-tuple in degrees) — there is no separate zoom-level integer or center+scale pair; panning translates the bbox (`_mmove`, 267-285), wheel/`_zoomStep`/dblclick all mutate the bbox around the cursor (`_zoomAt`, 305-317).
- **Markers**: point items are **grid-clustered in screen space every frame** (`_clusterPoints`, `app.js:435-448`, 46px cells, recomputed on each `draw()` — no persistent quadtree/zoom-level cache). A cluster of 1 draws as a circle sized `(it.r || 4.5) + selected-bump` (line 525) — **marker size does vary**, driven by `it.r`, which callers set from `sqrt(area_m2)` (Overview, line 765) or `score` (Search, line 929) or a flat radius (Queue). A cluster of N>1 draws as a bigger circle (`11 + sqrt(n)*2.2`, capped 22px) with a count label.
- **Hit-testing**: yes, `_hitAt(mx,my)` (216-230) — checks cluster circles first (radius+2px tolerance), then individual screen items: polygon items via ray-casting `pointInPoly()` (617-624), point items via `Math.hypot` distance against `max(9, r+3)`. Used for both click (`_click`, 575-583) and hover-tooltip (`_hover`, 231-252).
- **Redraw strategy**: full redraw every time, no dirty-region tracking, no persistent `requestAnimationFrame` loop — `draw()` is called on-demand from event handlers (pan/zoom/data-change) and once per `flyTo` animation frame. A `ResizeObserver` on the canvas also calls `draw()` (line 206).
- **CRITICAL — real imagery on the canvas**: **yes.** `_drawBasemapImages()` (`app.js:402-410`) does `ctx.drawImage(t.img, x0,y0,x1-x0,y1-y0)` for every mosaic tile whose bbox intersects the current view, drawn *underneath* the vector borders/coastline/rivers/city labels (draw order in `draw()`, lines 463-476: land tint → **satellite raster** → borders/coastline/states → rivers → city labels → graticule → data markers). These are real WebP crops of the staged Sentinel-2/Maxar mosaics (`mosaic_index.json`), reprojected to plain EPSG:4326 so pixels line up 1:1 with the map's own lon/lat, toggled by the "Satellite" layer checkbox (`this.layers.sat`, line 466, wired at 177-181).

---

## SECTION 6 — Globe (`globe.js`, `Globe` class)

- **three.js**: vendored at `vendor/three.module.min.js` (670,681 bytes, minified single line) + `vendor/OrbitControls.js` (1417 lines, unminified). Included via an **import map** in `index.html:10-12` (`{"imports":{"three":"./vendor/three.module.min.js"}}`) resolving the bare `"three"` specifier `OrbitControls.js` imports (`globe.js:26-27: import * as THREE from "three"`). No `<script src>` tag for three.js itself — it's an ES module, dynamically `import("./globe.js")`'d from `app.js:1837` only when Overview mounts, so its ~700KB payload never loads for the other 6 views.
- **Earth construction**: one `THREE.SphereGeometry(1, 96, 96)` (`globe.js:248`) with a **custom `ShaderMaterial`** (`EARTH_VERT_SHADER`/`EARTH_FRAG_SHADER`, lines 96-127) that composites day/night/specular textures with a smoothstep terminator (`smoothstep(-0.16,0.16,ndotl)`) and an ocean-only specular highlight, then alpha-blends a separately-rendered borders/India-tint canvas texture on top (`buildBordersTexture()`, 67-94, drawn from the same `basemap.json` vectors used by the 2D map). No procedural/drawn continents — land shapes come from the photographic day texture plus the borders overlay, not from extruded/rendered vector geometry.
- **Textures used**: yes — `vendor/earth/day.jpg` (NASA Blue Marble day), `night.jpg` (night lights), `specular.jpg` (ocean mask), loaded via `THREE.TextureLoader` (237-243), plus one dynamically-generated `CanvasTexture` for the borders overlay and one per staged AOI imagery patch (see below).
- **Camera**: `THREE.PerspectiveCamera(42, 1, 0.05, 100)` (`globe.js:207`) — 42° FOV, near 0.05, far 100. Starting distance `START_DIST = 3.4` world units (line 31); zoom is clamped `MIN_DIST 1.12` → `MAX_DIST 6.2` via `OrbitControls.minDistance/maxDistance` (368). The base sphere has radius **1 world unit = Earth's radius**, so 1 world unit ≈ 6,371 km (never stated explicitly in code, but implied by `latLonToVec3(lat,lon,r=1)` at line 45-48 placing the surface at radius 1).
- **Region pins**: one glowing sprite pair (pin + halo ring) per staged **region** (the coarser grouping, not per imagery tile) at each region bbox's centroid (`globe.js:309-338`), scaled by that region's candidate count (`scale = 0.03 + min(0.05, n/maxCount*0.05)`, line 313) and coloured amber if it has candidates, pale blue otherwise (line 314). A name-badge sprite (`labelTexture()`) fades in only once the camera is close (`_tick`, `labelT` calc at line 534).
- **Fly-to** (`_flyTo`, `globe.js:465-491`): interpolates camera **direction via spherical linear interpolation (`slerpDir`, 55-61) and distance via linear lerp, separately** — chosen specifically so the flight arcs just above the surface instead of cutting a straight chord through the globe (explained in the file's own header comment, lines 49-54). Cubic ease-out (`1-Math.pow(1-p,3)`). Duration: 1200ms default, or for a region dive (`_diveToRegion`, 446-464) computed as `max(1500, min(3200, 1300 + (camDist-finalDist)*480))` — longer for a higher starting altitude. It stops at `finalDist = MIN_DIST + 0.02` (≈1.14 world units, i.e. essentially at the zoom-in limit) pointed at the target region's lat/lon centroid, then (452-462) fades the canvas opacity to 0 over 450ms and calls `opts.onOpenRegion(bbox, name)` — which in `app.js:1840` is wired to `openQueueForBbox`, handing off into the Review Queue filtered to that region's bbox.
- **Logarithmic depth buffer**: **no** — `new THREE.WebGLRenderer({canvas, antialias:true, alpha:false, powerPreference:"high-performance"})` (`globe.js:201`) does not pass `logarithmicDepthBuffer: true`. Given the scene's actual depth range (near 0.05, far 100, everything drawn within ~1-3.4 units of origin, star field at 40-70 units), this is unlikely to cause visible z-fighting in practice, but it is not explicitly guarded against.
- **After fly-to**: yes, hand-off happens — see the fly-to bullet above. On a **pin click** (not a candidate point click), `_diveToRegion` runs, and on completion the canvas fades out and `app.js`'s `openQueueForBbox()` navigates to `#/queue` with that region's bbox filter applied (`app.js:1863-1868`). On a **candidate point click** (`_onClick`, `globe.js:428-441`, only reachable once `POINTS_VISIBLE_DIST` is crossed), `opts.onOpenCandidate(id)` fires immediately (no fly animation) → `app.js:1841: go("detail/"+id)`.

---

## SECTION 7 — The imagery question (answered carefully)

Read in full: `src/geoseek/search/api.py` (504 lines).

**Every endpoint that returns image bytes**, with signature:

```python
@app.get("/tile/{tile_id}/thumbnail")
def tile_thumbnail(tile_id: str):
    # returns a PNG thumbnail for ONE already-indexed tile, by its tile_id

@app.get("/candidates/{candidate_id}/imagery")
def candidate_imagery(
    candidate_id: str,
    date: Optional[str] = None,      # one of the ingested observation dates, default = latest
    view: str = "rgb",               # "rgb" | "overlay"
    scale: int = Query(1, ge=1, le=4),
):
    # returns a PNG crop for ONE already-flagged candidate's own footprint

@app.get("/discovery/cluster-map.png")
def discovery_cluster_map():
    # returns ONE pre-generated static PNG (the HDBSCAN cluster scatter plot)

@app.get("/detect/observations/{observation_id}/tiles/{row}/{col}/image.png")
def detect_tile_image(observation_id: str, row: int, col: int):
    # returns a PNG for ONE stored Maxar detection tile, addressed by (obs, row, col)
```

- **Can you query tiles by bounding box?** No. `/tile/{tile_id}/thumbnail` takes only an opaque `tile_id` string (a pre-computed index key like `S2B_44RPQ_20190330_1_L2A_scaled_r020_c009`), not coordinates.
- **By lon/lat point?** No image endpoint accepts lon/lat directly. (`/discovery/similar` accepts `lon`/`lat` at lines 422-427, but that returns JSON *search results referencing existing tile_ids*, not image bytes for an arbitrary point.)
- **Zoom/scale parameter?** Only `/candidates/{id}/imagery` has one, and it is **not a geographic zoom** — `scale: int, 1-4` (line 242) is a pure integer *upsample* of the already-rendered native-resolution crop (LANCZOS resize, lines 256-265), for crisper on-screen presentation. It does not change what geographic extent is fetched or let you request a different pixel-per-degree density from the source COG.
- **Does `/candidates/{id}/imagery` accept anything else?** Only `date` (must be one of that candidate's own ingested observation dates, else `400`, line 254) and `view` (`rgb`|`overlay`). No bbox override, no arbitrary-extent parameter — the geographic extent rendered is always that specific candidate's own stored footprint (`svc._by_id.get(candidate_id)`, line 245; the actual crop geometry lives inside `render_candidate_imagery()` in `geoseek/analyst/imagery.py`, not exposed to the caller).

**Plain answer**: with the API exactly as it is today, **it is not possible** to build a slippy-map-style satellite basemap that fills the viewport as the user zooms and pans freely. Every image endpoint is keyed to a pre-existing identity (a `tile_id`, a `candidate_id`, or a fixed `(observation_id,row,col)`) — none accept an arbitrary `(bbox, zoom)` pair and crop the underlying imagery on demand. This is exactly why the current offline map (`CoordMap._drawBasemapImages`, Section 5) works the way it does: it pre-stages a small, fixed set of whole-mosaic WebP images (`mosaic_index.json`, 18 tiles) client-side and draws whichever ones intersect the current view — it does **not** ask the server for imagery per-viewport.

**Smallest backend addition** to make a real slippy-basemap possible: a new endpoint like `GET /imagery/tile?observation_id=...&bbox=w,s,e,n&max_px=512` (or a standard XYZ `/imagery/tile/{z}/{x}/{y}.png` keyed to one of the already-ingested observation COGs) that opens the source COG with `rasterio`/GDAL, windows-reads the requested extent, and returns a resized PNG/WebP crop — the ingest pipeline already has COG paths and CRS info in the catalog (see `provenance.observations[*].scene.source_url` in Section 8), so the missing piece is purely "crop-on-request," not new imagery.

---

## SECTION 8 — API response shapes

Raw JSON saved to `docs/api_samples/`: `presentation_summary.json`, `candidates_list.json`, `candidate_detail.json`, `regions.json`, `detect_observations.json`, `notifications.json`, `stats.json`.

### `/candidates/{id}` — complete field tree (candidate `2019_2026_004510`, a 196,000 m² `water_gain`)

Top level: `rank:int, candidate_id:str, pair:str, centroid_lonlat:[float,float], area_px:int, area_m2:float, change_type:str, confidence:float, significance:float, queue_score:float, persistence:str, earliest_supported:[str], bbox_rc:[int×4], label:int, centroid_rc:[float,float], mean_model_prob:float, classification:{…}, confidence_breakdown:[str], suppression:{…}, trajectory:{…} (legacy, mirrors temporal_trajectory), sar:null|{…}, terrain:{…}, geometry:{…}, provenance:{…}, decisions:[{…}], current_decision:{…}|null, temporal_trajectory:{…}, imagery:{…}`.

- **`classification.evidence`**: `d_ndvi/d_ndbi/d_ndwi` (raw index deltas, float, e.g. `-0.4641`), `ndvi_anomaly/ndbi_anomaly/ndwi_anomaly` (delta minus scene-wide seasonal delta, e.g. `-0.5212`), `scene_d_ndvi/scene_d_ndbi/scene_d_ndwi` (the scene-wide seasonal delta itself), `elongation:float` (e.g. `1.35`), `fill_ratio:float` (e.g. `0.67`). Sibling fields `rule:str` (e.g. `"ndwi_anomaly_up"`), `detail:str` (human sentence).
- **`suppression`**: `candidate_id, suppressed:bool, suppressed_by:str|null, combined_downweight:float, trace:[{rule:str, verdict:"pass"|other, weight:float, detail:str, values:{…rule-specific floats/bools…}}]`. The 5 gates always present: `quality` (`bad_scl_fraction_earlier/later:float, valid_fraction:float`), `registration` (`coreg_residual_px:float, coreg_corrected:bool`), `radiometric` (`index_band_min_corr:float, low_confidence:bool`), `phenology` (the same `d_*`/`*_anomaly` sextet as classification.evidence), `morphology` (`area_px:int, area_m2:float`).
- **`provenance`**: `observations:[{role:"before"|"after", observation_id:str, acquired_at:"YYYY-MM-DD", scene:{scene_id, platform, source_url (real `https://sentinel-cogs.s3...` URL), license:str, processing_baseline:null, crs:"EPSG:32644", checksums:{B04,B03,B02,SCL,B08,B11: sha256 hex str}}, collection:{collection_id, sensor:"MSI", platform, bands:[str], native_gsd_m:float}, representative_tile:{tile_id, …scene fields repeated…, radiometry:{fixed_true_color:{method, replaces, reflectance_scale_dn_per_unit, clip_reflectance:[float,float], clip_dn_equivalent:[int,int], gamma, boa_add_offset:…}}}}], model:{name:"FCSiamDiff", checkpoint:str (local path), threshold:float, weights_sha256:str (64-char hex)}, code:{git_commit:"ecf0943", pipeline_version:"0.1.0"}, probability_raster:str (local .tif path), sar_corroboration:null|{vv_median_db, confidence_factor, verdict, speckle_filter}, report_generated_at:ISO8601`.
- **`decisions`** / **`current_decision`**: each decision is `{decision_id:"dec_...", candidate_id, decision:"confirm"|"reject", analyst_note:str, analyst:str, created_at:ISO8601, model_version:str, weights_sha256, git_commit, pipeline_version, confidence_at_decision:float, evidence_snapshot:{…a full COPY of the top-level candidate record at decision time…}}`. **This is the single largest field** — for this candidate (1 recorded decision), `decisions`/`current_decision` alone are ~54 KB each of the 141 KB total payload, because `evidence_snapshot` embeds the entire candidate object (including its own nested `classification`/`suppression`/`confidence_breakdown`) per decision.
- **`terrain`**: `elevation_m:float, slope_deg:float, aspect_deg:float, aspect_compass:"N"|"S"|"E"|"W"|…, distance_to_water_m:float, distance_to_built_up_m:float, provenance:{<same 5 keys>: str (method sentence, "measured"|"derived (...)")}, plain_language:str`.
- **`imagery`**: `before_dates:[str] (e.g. ["2019","2021","2024","2025"]), after_date:str ("2026"), dates:[str] (dup of before_dates), views:["rgb","overlay"], url_template:str`.
- **`temporal_trajectory`**: `persistence:str, persistence_confidence:float, intervals:[{window:[str,str], changed:bool, probability:float, comparable:bool, kind:"consecutive"|"span"}], earliest_supported_change:{window:[str,str], supporting_observations:[str], imagery_quality_note:str, caveat:str}, notes:[str]`.
- **`geometry`**: standard GeoJSON `Polygon`, always a 5-point closed ring (4 corners + repeat).

Other endpoints (top-level keys only — full samples in `docs/api_samples/`): `presentation_summary` → `offline, aoi, span_pair, observation_dates, counters, latest_alerts, regions, change_type_distribution, change_type_labels, featured, demo`. `regions` → `{regions:[{name,bbox,n_observations}]}`. `detect_observations` → `{observations:[{observation_id, aoi_name, n_detections, by_class, n_tiles, n_tiles_with_detections}]}`. `notifications` → `{notifications:[{notification_id, watch_id, watch_name, severity, severity_score, candidates:[…], observation_id, observation_date, created_at, seen}]}`. `stats` → `{index, change_pipeline, model, audit, sar, build}`.

---

## SECTION 9 — Tests

### The no-external-URL test

`tests/test_phase6_presentation.py:173-189`, inside `test_existing_analyst_endpoints_unaffected`. It fetches exactly **3 files** — `/app/app.js`, `/app/style.css`, `/app/index.html` — lower-cases the text, and asserts:

```python
assert "http://" not in low and "https://" not in low
assert "fonts.googleapis" not in low and "cdn." not in low
assert 'src="//' not in low and "url(http" not in low
```

It is a **literal substring scan**, not a URL-extraction/allowlist check.

- A local relative path like `assets/textures/earth_day.jpg` → **passes** (no `http`/`cdn.`/`fonts.googleapis` substring).
- A `data:` URI → **passes** (contains neither `http` nor the other banned substrings, unless the base64 payload happens to spell one of them by chance).
- An inline `<svg>` → **passes trivially** — the app already ships dozens of these.
- **Gap**: the scan **does not cover `globe.js`** (which itself does 4 raw `fetch()` calls, Section 4) **or anything under `vendor/`**. `vendor/three.module.min.js` already contains literal `http://` and `https://` substrings today (an XML namespace URI, two `https://discourse.threejs.org/...` strings in the library's own deprecation-warning text, and a URL-parsing regex) — none of these fire actual network calls, but the fact remains that ~700 KB of what the browser actually loads under `/app/vendor/` is outside what this test inspects. If a redesign adds a CDN font or script to `globe.js` or to a new vendored file instead of to `app.js`/`style.css`/`index.html`, the suite stays green.

### DOM structure / selector dependencies

**None found.** There is no Playwright/Selenium/Puppeteer test anywhere in `tests/`. Every `test_phase6*.py` test drives the backend via FastAPI's `TestClient` (real HTTP against the real analyst service, no browser, no DOM). The only HTML-aware assertions are the substring scan above and `test_ui_bundle_is_served_and_offline` (`test_phase6.py:341-344`), which only checks `"<" in r.text` for `/app/`. **No test asserts on any specific id, class name, or DOM structure** — a redesign is free to rename every id/class in the CSS/HTML without breaking `pytest`. The corresponding risk (nothing will catch it) is in Section 10.

### Latency tests

No frontend-specific latency test. `tests/test_search.py` asserts `latency_ms < 1000.0` on `/search/text`/`/search/image` (backend-only, lines 97,138,209,235) — not scoped to the UI. A separate manual script, `scripts/verify_offline_perf.py`, exists for offline perf verification but is not part of the `pytest` suite (not `test_*.py`, not collected by default).

---

## SECTION 10 — Honest risk list

### Most likely to break when a token system is applied across all 7 screens, in priority order

1. **`hexA()` assumes every colour custom property is a literal `#rrggbb` string — and there are two independent copies of this assumption.** `app.js:599-602` and `globe.js:129-132` both do `hex.replace("#","")` → `parseInt(..., 16)`. Both are fed CSS custom properties at draw time (`css.getPropertyValue("--accent")` etc., e.g. `app.js:535,550`). The moment a redesign moves any colour token to `color-mix()`, `oklch()`, `hsl()`, or any non-hex syntax — which this very file already does for `--shadow-*` and several `rgba()` tokens — `hexA()` silently returns `rgba(NaN,NaN,NaN,a)`: an invisible or garbage-coloured fill, no thrown error, no console warning. This is the single highest-blast-radius landmine in the whole redesign, because every canvas marker, cluster badge, AOI-drag rectangle, and globe pin/label glow goes through one of these two functions.

2. **Change-type colour is duplicated in 3 places and has already drifted once** (Section 3: `water_loss` on the globe is visibly wrong today). A redesign that repaints `--water_gain`..`--other` in `style.css:39-44` will not touch `CTYPE_COLOR` (`app.js:7-10`), `CTYPE_RGB` (`globe.js:346-349`), or `DET_CLASS_COLOR`/`REGION_PALETTE` — all hand-copied `const` literals with zero reference back to the CSS tokens. Any palette change must be applied in (at least) 4 places by hand, and there is no test that would catch a 5th drift like `water_loss`'s.

3. **Canvas/WebGL text and marker sizing is not driven by the CSS type scale at all.** `ctx.font = "11px system-ui..."` (`app.js:416`), `"10px..."`-class strings, `"700 " + n + "px system-ui..."` (539) are separate literal numbers baked into JS. Retuning `--fs-*` in `style.css` will leave city labels, cluster-count text, and the globe's tooltip/legend sizing untouched — they'll visually mismatch the new type scale until someone finds and edits each JS literal by hand.

4. **The offline-URL test only scans `app.js`/`style.css`/`index.html`** (Section 9) — `globe.js` and everything in `vendor/` are unguarded. If the redesign restructures code into new files (a common side-effect of "apply tokens everywhere"), whatever isn't named exactly those 3 filenames silently falls outside the safety net that is the whole reason this product can claim "no external calls."

5. **Zero DOM/selector test coverage** means renaming ids is safe for `pytest` but not for the app itself: `app.js` calls `$("#some_id")` (a raw `querySelector` wrapper, line 23) roughly 150+ times with no null-guards. A single id typo/rename during a markup pass throws inside whatever event handler or `ensureX()`/`boot()` call touches it first — `boot()` (`app.js:1881-1904`) runs several unguarded `$()` calls very early, so a broken id there can silently prevent `router()` from ever being called, i.e. the whole SPA never renders past a blank shell. Nothing in CI catches this class of break; it would only surface by opening the app in a browser.

6. **Views are never unmounted** (`ensureX()`+`Init` boolean guards, Section 4) — event listeners attach exactly once. A redesign pass that re-renders a view's inner HTML (common when swapping in new component markup) without also re-running that view's `ensureX()` listener-attachment code will produce dead buttons/inputs that look right but do nothing.

7. **`CoordMap` assumes `canvas.parentElement` is its intended wrapper** (`app.js:144: const wrap = canvas.parentElement || document.body`) and appends its tooltip/scale-bar/coord-readout/controls/layers-panel as siblings of the `<canvas>` there. A markup restructure that adds an extra wrapping `<div>` around `<canvas class="map">` without updating this assumption will make those overlay elements land in `document.body`, unstyled and mispositioned, without any error.

8. **`globe.js` duplicates data-fetching instead of sharing `app.js`'s caches** (Section 4) — separate `fetch("basemap.json")`, `fetch("mosaic_index.json")`, `fetch("/regions")`, `fetch("/candidates?limit=900")`, none cached across the two. Beyond the wasted bandwidth, this means the globe's pin candidate-counts and the 2D map/queue's counts can visibly disagree if candidates change between the two independent fetches — a "why don't the numbers match" bug a redesign reviewer is likely to trip over and misdiagnose as a backend bug.

9. **23 raw font-size values and several raw radius/spacing values already escape the existing token layer** (Section 2) — a systematic "replace hardcoded values with tokens" pass has to hunt these down one at a time; simply redefining the `--fs-*`/`--r-*`/`--sp-*` variables will not sweep them.

### Genuinely fragile/poorly structured, unprompted

- **No central state store.** ~15 module-level mutable globals (`PRES`, `REGIONS`, `OBS_DATES`, `detailState`, `detState`, `queueIdsOverride`, `DEMO`, `watchEditingId`, `REGION_STATS`, `globeInstance`/`ovViewMode`, …) are read and written across view boundaries with no ownership contract — e.g. `REGIONS` is populated by `loadRegions()` (called from Search/Queue/Watch/Overview) and consumed by all of them plus `globe.js`'s own separate copy.
- **Duplicated algorithms, not just duplicated data.** `pointInPoly()` (`app.js:617-624`) and `detPointInPoly()` (`app.js:1421-1428`) are the identical ray-casting point-in-polygon test, copy-pasted for two different features (map hit-testing vs. detection-box hover) instead of shared.
- **`evidence_snapshot` bloat in the audit trail** (Section 8): every analyst decision embeds a full copy of the entire candidate record, so `/candidates/{id}` payload size grows without bound with decision count (54 KB of a 141 KB payload from a *single* decision on this sample candidate). Not a frontend bug today, but any redesign that fetches this endpoint more eagerly (hover previews, prefetch-on-row-hover in the Queue table) will feel real jank, and it's the kind of payload a naive "just fetch the candidate for a tooltip" feature would silently make much worse.
- **No `ResizeObserver`/listener teardown anywhere** — harmless today only because views are never destroyed (point 6 above); a real hazard the moment someone "fixes" that by making views actually unmount.
- **`api()`'s error path is minimal**: a non-2xx response becomes a generic `Error(status + detail)` with no structured error type, so every call site's `catch` block does its own ad hoc string-interpolation into a toast/inline message — consistent in style, but there's no shared error-rendering component to redesign once and have it apply everywhere.
