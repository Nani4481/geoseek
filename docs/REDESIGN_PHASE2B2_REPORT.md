# Frontend Redesign — Phase 2B-2 Report

Scope: the Globe view only (`globe.js` + the textures it loads +
`scripts/stage_earth_textures.py`). Backend untouched. three.js stays the
already-vendored `r160`-pinned build. This phase does **not** build the
continuous fly-in choreography past the zoom-in limit (an actual "descent"
into imagery/terrain) — see "Handoff point for a future descent" below for
exactly where that attaches.

Most of what a from-scratch "globe visual overhaul" would ask for already
existed going into this phase (real Blue Marble day/night/specular textures,
a day↔night terminator shader, an ocean specular highlight, an atmospheric
rim-glow shell, glowing region pins with halo rings and hover tooltips, a
continuous slerp fly-in to a picked region) — that work predates this report
and isn't re-described here except where it was found broken or genuinely
missing. Two things were: idle auto-rotate didn't exist at all (a prior
session had deliberately built it *out*), and a real rendering bug in the
India highlight. Both are fixed below.

---

## 1. Idle auto-rotate (new)

Requirement: slow, restrained auto-rotate; pauses on interaction; resumes
after idle; no frantic spin. The globe previously had an explicit code
comment arguing *against* auto-rotate ("should stay exactly where it lands").
That decision is superseded here per this phase's brief.

- `AUTO_ROTATE_SPEED = 0.4` against OrbitControls' own documented baseline
  ("30s/orbit at speed 2.0 @ 60fps") — target ≈150s (2.5 min) per revolution.
- Starts once the initial India framing completes; pauses on `OrbitControls`'
  own `start` event (fires for drag, wheel-zoom, and touch alike — no need
  for separate listeners per input type); resumes `AUTO_ROTATE_IDLE_MS =
  3000` ms after the matching `end` event, not the instant the interaction
  ends.
- Explicitly paused for the duration of a region dive (`_diveToRegion`) —
  `OrbitControls.update()` applies `autoRotate` unconditionally, regardless
  of `controls.enabled`, so left alone it would fight the dive's own
  programmatic slerp camera path.
- Respects `prefers-reduced-motion: reduce` (checked once at module load,
  matching the same media query `tokens.css` already uses elsewhere) — the
  rotation never starts at all under that setting, no toggle needed.

**A real, measured bug this surfaced**: `controls.update()` was being called
with no `deltaTime` argument, so `OrbitControls`' fallback auto-rotation path
assumes a fixed 60fps and advances a constant angle *per call*, not per real
elapsed second. Verified empirically in headless Chrome (software-rendered,
~30fps): the globe rotated at roughly **half** the intended rate (measured
≈282s/revolution against a ≈150s target) until fixed. Fixed by tracking real
elapsed time in `_tick()` (clamped to 250ms to avoid one giant jump after a
backgrounded tab) and passing it to `controls.update(dt)`. Re-measured after
the fix: ≈148s/revolution — matches the target regardless of the render
frame rate, so the rotation speed will read the same on a 30Hz, 60Hz or
144Hz display instead of drifting with it.

Verified via a temporary debug hook (`window.__globeDebug`, added and removed
within this session — not present in the shipped file) reading
`controls.autoRotate` and the camera's azimuthal angle directly, across:
idle (rotating, `autoRotate: true`), immediately after a drag (`false`), +1s
and +2s post-drag (still `false`, correctly inside the 3s window), and +3.5s
post-drag (`true` again, angle advancing). All five checkpoints matched the
intended state machine exactly.

## 2. India highlight — real rendering bug, fixed

The India "tint" was a flat `rgba(120,175,255,.28)` canvas fill drawn under
the whole country polygon, then alpha-composited over the *already-lit*
day/night shader output (`color = mix(color, borders.rgb, borders.a)`).
Screenshotted live at 1920×1080 (headless Chrome, real render, not assumed):
this flattens every bit of underlying photographic and night-light detail
into a single flat pale-blue silhouette — it reads as a pasted-on cutout
sticker, not "a real textured Earth," and directly undermines this phase's
own stated goal. It also happened to be close in hue to the `info`-token
colour used for "no candidates" pins, reducing pin/background contrast over
exactly the region where most pins live.

Fixed in `buildBordersTexture()`: India no longer gets a fill at all. It
gets a highlighted **outline** instead — the same border-line pass every
other country gets, redrawn a second time on top at 2.2px / `rgba(150,200,
255,.95)` (vs. the baseline 1px / 50% opacity every other country's border
gets). Zero interior pixels of India's own photography are touched; the
country still reads as "the highlighted one" via a crisp, bright boundary
instead of a wash. Screenshotted before/after — the after state shows full
terrain detail and existing night-lights inside the outline, both cyan
"no-candidate" pins and the amber "has-candidates" pulse pin now clearly
legible against it.

## 3. Textures — brought up to the letter of spec, still ~2.7x under budget

`day.jpg` was already 2048×1024. `night.jpg` was previously halved to
1024×512 "to save bundle weight where it costs nothing visually" — re-
examined against this phase's explicit "2048x1024 day + 2048x1024 night"
ask and re-staged at full resolution from the same already-cached source
(`data/earth_textures_src/earth_lights_2048.png`, no re-fetch needed).
`specular.jpg` stays halved (1024×512) — it's a low-frequency ocean-glint
mask only, never named in the spec, and halving it is invisible in the
render.

| file | before | after |
|---|---|---|
| day.jpg | 2048×1024, 343KB | unchanged |
| night.jpg | 1024×512, 49KB | **2048×1024, 192KB** |
| specular.jpg | 1024×512, 72KB | unchanged |
| **total** | **0.46 MB** | **0.62 MB** (budget: ~1.5MB) |

Manifest (`data/provenance_manifest.json`, `earth_textures_vendor` entry)
rebuilt with fresh SHA256s for the changed files, source URL (three.js
`r160` redistribution of NASA Visible Earth Blue Marble / Black Marble) and
public-domain licence unchanged. `scripts/stage_earth_textures.py`'s own
docstring and `HALVE` set updated to match.

## 4. Handoff point for a future descent

Marked directly in code (`globe.js`, inside `_diveToRegion`, bracketed
`// ---- ZOOM ENDPOINT ----` / `// ---- end ZOOM ENDPOINT ----` comments,
currently around line 487-508): the `_flyTo(...)` completion callback fires
the instant the camera reaches `finalDist` (≈`MIN_DIST`, the zoom-in limit)
pointed at the picked region's lat/lon. At that exact point `this.camera`,
`this.globeMesh`, `this.pinGroup` etc. are all still live and rendered;
`d.bbox`/`d.name` identify the target AOI. Today that callback fades the
canvas to black over 450ms and calls `opts.onOpenRegion(d.bbox, d.name)`,
which `app.js` wires to `openQueueForBbox` (a hard cut into the Review
Queue). A future descent phase replaces exactly that fade-and-cut with a
continuous camera path past this same point — no upstream change needed.
Explicitly **not** built here, per the brief: no logarithmic depth buffer, no
multi-stage camera path.

Investigated whether a "muddy blurry crossfade" / "giant text-slab over
low-res tiles" pattern (named in the brief as the thing to avoid) already
existed anywhere in the zoom-in or region-open path — it doesn't. The
existing dive is a single continuous slerp+lerp camera flight ending in a
plain opacity fade, and the post-dive Queue view has no oversized label or
blurry-tile transition. Confirmed by reading `_diveToRegion`/`openQueueForBbox`
and by screenshotting a mid-zoom state live; no change was needed here.

---

## Verification

- **`pytest`** (full suite): 412 passed, 3 skipped — includes
  `tests/test_frontend_offline.py`, which scans `globe.js` and everything
  under `vendor/` (not just `app.js`/`style.css`/`index.html`) for external
  URLs. The regenerated textures + updated manifest entries were rescanned
  and pass; the vendor-allowlist check that gates every vendored binary
  against a manifest entry passes for the new `night.jpg` hash.
- **`scripts/verify_offline_perf.py`** against the real 841-candidate
  dataset: `functional checks: ALL PASS`, `no process made (or attempted) a
  non-loopback network call`. Two pre-existing, unrelated endpoints
  (`/candidates/{id}/similar`, `/discovery/similar`) sit over the 1s budget —
  both are Discovery-view embedding-similarity lookups, untouched by this
  phase, and were already over budget before it.
- **Offline boot with textures present**: headless Chrome (software-
  rendered/swiftshader), 3 cold-navigation runs, timing the app's own
  `#ov_globe_status` "hidden" signal (i.e. `Globe.ready` resolved: data
  fetched, all 3 textures loaded, shader compiled, initial India framing
  done) from navigation start: **101ms / 124ms / 365ms** — all comfortably
  under the 1s budget, on a software GL fallback slower than a real GPU.
  Zero console errors, zero failed requests, zero non-loopback requests
  observed across all runs.
- **Headless Chrome renders, both required resolutions** (1920×1080 and
  1440×900): textured Earth confirmed live — day-side Blue Marble
  photography, night-side with visible city-light texture, atmospheric rim
  glow, pins legible against both the india outline and open ocean, one
  zoomed-in state captured via scroll-zoom (crisp, no blur/smear, no
  oversized label).
- **Frame timing**: ~30-34 fps sustained in headless software rendering
  (swiftshader has no real GPU behind it, so this is a floor, not a
  representative number for the target RTX 4060 laptop — no such laptop was
  available in this session to benchmark directly). No dropped-frame stalls
  or long-tail spikes observed across the 60-frame sample windows measured
  at both resolutions.

## Renames

None.
