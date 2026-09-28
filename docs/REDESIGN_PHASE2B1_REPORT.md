# Frontend Redesign — Phase 2B-1 Report

Scope: Candidate Detail and Object Detection only. Neither screen's information
architecture changed — every field that was visible before is still visible;
only weight and arrangement changed, imagery now dominant on both.

---

## Carry-ins

### (a) 44px row height — reduced to 34px

Re-examined with the stated framing (mouse-and-keyboard laptop console, not a
touch target). The 44px number came from fitting `.c-conf-ring`'s 40px
diameter without clipping. Rather than accept that as a fixed cost, I shrank
the ring specifically for the Review Queue's table-row context: added a
`.c-conf-ring.xs` variant (26px diameter, `--text-2xs` figure) and switched
the Queue's `Components.confRing()` call to it. **Row height is now 34px**,
matching the original dense-scanning rationale (34px floor, not 44px). The
larger default ring (40px) and `.lg` (56px) remain unchanged for Candidate
Detail's summary panel and map popovers, where they aren't fighting a
dense-table constraint. Re-screenshotted the Queue — confirmed 34px rows with
the smaller ring still reads clearly at both resolutions (see screenshots).

### (b) The darkened `other` colour next to the other five — screenshotted together

Six-badge row captured live (not simulated) at both resolutions; see the
gallery screenshot referenced below. `other`'s darker green reads as
"darker green," not disabled/greyed-out, next to `road`'s light blue-grey,
`water_gain`'s blue, `water_loss`'s violet, `construction`'s orange and
`clearance`'s yellow — the badge's own dot (solid, saturated) and its
label text (full-opacity `--text-primary`) both look identical in treatment
to the other five; only the hue/lightness differs, which is the point. No
opacity, no muting, no disabled-looking affordance anywhere in `.c-badge`.

---

## Layout bugs found and fixed (verified live, not assumed)

Both are worth stating plainly because the CSS *looked* correct and I
initially trusted it — the brief's instruction to "confirm rendering, do not
assert it" caught both.

1. **`align-items: start` on both new grids silently prevented the columns
   from stretching to the specified `height`.** A child's `height: 100%`
   inside such a column had nothing definite to resolve against, so the
   frame/rail grew to fit their own content instead of the container —
   measured overflow up to 1709px tall inside a 948px box before the fix.
2. **An implicit (`auto`) grid row track sizes itself to its content's
   max-content height, not the grid container's specified `height`.** Fixing
   (1) alone did not fix this — `grid-template-rows: minmax(0, 1fr)` was
   also required, the same `minmax(0, ...)` pattern already used for the
   column dimension, just missing on the row axis.
3. **A hardcoded height offset (`calc(100vh - 260px)`) for Object Detection
   was wrong at 1440×900** — its 3 filter rows wrap differently at narrower
   widths, so the fixed constant caused a real, measured ~120px overflow past
   the viewport bottom. Replaced both screens' fixed offsets with
   `fitImageryLayoutHeight()`, a small JS helper that measures the layout
   element's actual `getBoundingClientRect().top` and sets `.style.height`
   from that, re-run on window resize and whenever content above the layout
   could change height (the caveat banner appearing/disappearing, a fresh
   candidate/observation render). This is measured, not assumed, by
   construction — it cannot drift out of sync with whatever is actually above
   it.

---

## Candidate Detail

**Layout**: `.detail-layout`, a 2-column grid (`minmax(0,1fr) 380px`),
escapes `<main>`'s 1520px content cap for this view only
(`main:has(#view-detail.active) { max-width: none; }`) so "two thirds of the
viewport" means the viewport, not a capped column. The imagery column is a
flex layout (toolbar → frame → date strip → caveat line → decision bar); the
rail scrolls independently and collapses to width 0 via a "Panels" button,
state held in a module-level variable (`d_railCollapsed`) that survives
navigating to other candidates within the session, per the brief.

**Imagery region**:
- Segmented Before/After/Overlay control (2A's `.c-segmented`), keyboard
  1/2/3 in addition to native Tab+click, guarded off when a text input has
  focus.
- A unified 5-date strip (2019/2021/2024/2025/2026), each showing its full
  ISO acquisition date (derived client-side from `temporal_trajectory`'s
  interval window boundaries — see the API gap below), sensor + cloud % where
  available, and a flag (⚠ / ☁) when `comparable: false` or cloud fraction
  exceeds 20% (a UI-chosen threshold — the backend doesn't define a
  "heavily clouded" cutoff). Clicking a date switches Before/After
  automatically; the active date/segment stays visually unmistakable
  (accent border + fill).
- Progressive load: `scale=1` fetched and painted first, `scale=2` fetched
  immediately behind it and swapped in on arrival — verified this is the
  right order by reading `geoseek/analyst/imagery.py`: the LRU cache
  (`_render_cached`, keyed on candidate/date/view, NOT scale) is populated by
  whichever scale is requested first, so the `scale=1` request also warms the
  cache for the `scale=2` request right behind it, making the "upgrade" arrive
  faster than a cold `scale=2` would alone. Confirmed via
  `verify_offline_perf.py`: cold single-res render p95 ≈ 68-82ms, warm p95 ≈
  1.2ms. `.c-skel` shimmer shown only while no image has painted yet, removed
  the instant either resolution arrives — no spinner anywhere. Verified live
  (delayed the imagery route in Playwright, read `classList`/computed
  `animation-name`/`animation-play-state` programmatically, not just eyeballed
  a screenshot mid-animation): `c-skel` present, `c-shimmer` `running` during
  load, both gone with the correct `src` set after.
- **Footprint outline, toggleable — uses the server's own pixel-accurate
  overlay render, not a client-drawn shape.** See the named API gap below for
  why. The checkbox swaps the frame's `view` parameter between `rgb` and
  `overlay` for whichever date is currently selected (this works for every
  date, before or after — verified `view=overlay` returns 200 for all 5
  dates, not just the pair's own after-date). Disabled when the Overlay
  segment itself is already selected (redundant in that state). The
  overlay-vs-raw distinction and what red/yellow mean, plus the backend's own
  `earliest_supported_change.caveat`, are always visible under the date strip
  (`#d_imgnote`) — this existed pre-2B-1 and I nearly dropped it converting
  the 3-image grid into one switchable frame; caught it and restored it as a
  permanent, non-collapsible line.

**Decision controls**: Confirm/Reject live in the imagery column's own flex
flow (not the rail), so they never scroll out of view independent of rail
scroll position. A verdict chip in the toolbar (top-right, always visible)
shows undecided/confirm/reject at a glance before acting, colour-coded on the
semantic ramp (confirm = success green, reject = danger red, undecided = a
neutral amber-ish pill, deliberately distinct from both). `C`/`R`/`Esc` are
bound globally while the view is active, guarded against firing while
`INPUT`/`TEXTAREA`/`SELECT` has focus; the buttons show the key hints inline
(`<kbd>`). Post-decision: buttons disable for the duration of the write (no
double-submit), the toast reads "**CONFIRM** written to the permanent audit
record" (not a generic "saved") and the verdict chip updates from the
re-fetched candidate, not optimistically — so what's shown is always what the
server actually holds.

**Secondary rail** (all `.c-panel`, all independently collapsible):
Summary (badge/ring/band from 2A, persistence humanised, earliest-change
window) · Temporal trajectory · Why did the system flag this (evidence, now
tabular-figures-formatted) · Suppression trace (rebuilt, see below) ·
Provenance (rebuilt as a chain, see below) · Decision history · Discovery.

**Suppression trace, rebuilt**: each gate is now its own card with a
pass/fail pill on the semantic ramp (not a plain coloured word) and, wherever
the gate's free-text `detail` string parenthesises a threshold cleanly (4 of
5 gates: quality, registration, radiometric, morphology), the actual value
and the threshold are pulled apart into two separate, side-by-side fields.
The 5th gate (phenology) is a multi-axis rule that doesn't reduce to one
number pair — shown as its full sentence rather than force-split into a
misleading pair. This is a regex-based best-effort split
(`splitTraceDetail()`), same family of approach the file already used for
`pct()`/`regpx()`/`radv()`; it degrades to showing the whole sentence rather
than guessing wrong.

**Provenance, rebuilt as a visible chain**: before → after → model/pipeline,
each a distinct node, connected by a literal ↓ between them rather than a
2-column flat `<dl>` grid. Hashes/ids (scene id, tile id, checksum, weights
sha256) are truncated (`truncateMiddle()`, keeps head+tail) in the mono
token, full value on hover via `title` — "available on demand" via the
browser's native tooltip rather than adding a click-to-copy control this
phase.

---

## Object Detection

**Layout**: `.detect-layout`, same 2-column pattern as Candidate Detail. The
caveat banner (`#dt_caveatpanel`) stays **outside** the grid entirely, full
width, at the very top, unchanged in styling — not touched, not collapsible,
not relocated. Per-class detection counts, detector metrics, and observation
totals moved into the collapsible rail; the confidence-margin slider,
class-filter chips, and observation/tile pickers stayed above the grid
(they're inputs that decide what the tile shows, not secondary readouts).

**Boxes toggleable**: a new "Show boxes" checkbox; `drawDetectCanvas()` skips
the stroke loop entirely when off, and the legend text says "(boxes hidden)"
rather than silently showing a stale count.

**Slider live-filtering**: already wired to the `input` event (not `change`),
already redrawing on every tick. Filtering `detState.tileData.detections` — a
plain-array `.filter()` over at most a few hundred entries — and recomputing
the visible count happen in the same synchronous call as the box redraw.
**No debounce was added, deliberately**: this is well under a frame budget at
the data sizes this product handles, and the brief's own fallback condition
("if too slow to feel instant") didn't hold once actually measured (verified
live: slider drag from 0→90 margin correctly went from 97/98 to 0/98
detections shown, redrawn each tick, no lag observed at 1522×1522 canvas
resolution).

**Per-class colours now resolved through Tokens** — this was a real,
previously-unmigrated gap the Phase 2A report had already flagged
(`DET_CLASS_COLOR`, a hand-copied hex object, was never routed through
Tokens in either Phase 1 or 2A). Added 8 new `--detect-*` tokens to
`tokens.css` (same hues as the old hex values, converted to oklch — no visual
change, just now resolved through the same bridge as everything else), a
`Tokens.detectClass(cls)` accessor mirroring `changeType()`'s graceful
fallback (an unrecognised class from live `/detect/model-info` data warns and
falls back to neutral grey rather than throwing — a data surprise, not a
config bug), and migrated all 5 remaining `DET_CLASS_COLOR[...]` call sites
in `app.js`. `DET_CLASS_COLOR` itself is deleted.

### Language audit — every wording change

| where | before | after |
|---|---|---|
| Tile panel `.mapnote` | "Every box is a **REAL** stored detection from... no live re-run, no illustrative boxes." | "Every box is a stored **detection**... **— not a verified count.**" (also dropped the shouty "REAL", redundant once "not a verified count" states the actual caveat) |
| Rail panel title | "Per-class counts — this tile" | "Per-class **detection** counts — this tile" |
| Rail panel title | "Observation totals" | "**Detection** totals — this observation" |
| Counts table column header | "count" | "detections" |

Not changed, and checked explicitly for the same failure mode: the metrics
panel's "precision"/"recall"/"AP50" language (these are genuine, ground-truth
xView TEST-half evaluation metrics, correctly attributed as such, not a claim
about this tile); the hover readout (`class · conf 0.xxx` — already says
"conf", not "confirmed"); the margin-slider label ("showing everything
stored" / "showing only the top N%..." — already says "stored", not "found"
or "verified").

---

## Screenshots

Captured live via Playwright + headless Chrome at both 1920×1080 and
1440×900 (the browser extension was not connected this session): Candidate Detail
with the rail expanded, with the rail collapsed, and mid-load (skeleton
+ shimmer confirmed both visually and programmatically, per above); Object
Detection with the confidence margin at 0 (97/98 detections shown) and at 90
(0/98 shown), the caveat banner visible unchanged in both. Zero console/page
errors in any of the eight captures beyond one benign `favicon.ico` 404
(pre-existing since Phase 1, the app declares no favicon).

---

## Measured imagery-area percentage (real, not claimed)

Measured via `getBoundingClientRect()` on the actual frame/canvas element
against `window.innerWidth/innerHeight`.

| screen | resolution | width % | height % | **area %** |
|---|---|---|---|---|
| Candidate Detail, rail expanded | 1920×1080 | 77.3% | 68.8% | **53.2%** |
| Candidate Detail, rail collapsed | 1920×1080 | 97.1% | 70.5% | **68.5%** |
| Candidate Detail, rail expanded | 1440×900 | 69.7% | 62.6% | **43.6%** |
| Candidate Detail, rail collapsed | 1440×900 | 96.1% | 62.6% | **60.1%** |
| Object Detection tile | 1920×1080 | 79.3% | 50.2% | **39.8%** |
| Object Detection tile | 1440×900 | 72.4% | 38.4% | **27.8%** |

(Candidate Detail's height % is ~4 points lower than an earlier pass because
restoring the dropped caveat/legend line under the date strip — see "the
red/yellow legend and the backend's own caveat... must not silently
disappear" above — correctly costs the frame a little vertical space. Kept
the honest, current number rather than the higher one from before that fix.)

**Candidate Detail is clearly dominant** by width at both resolutions (comfortably
over the "roughly two thirds" target, and over 96% once the rail is
collapsed) and takes up the clear majority of the frame vertically once the
mandatory chrome (segmented control, date strip, decision bar) is accounted
for.

**Object Detection is width-dominant (72-79%) but not area-dominant** —
saying so plainly rather than claiming otherwise, per the brief. The
shortfall is vertical: Object Detection carries the caveat banner plus three
full filter rows (observation/tile pickers, class chips, confidence slider)
above the tile-dominant grid, none of which I removed or consolidated (out of
scope this phase — the brief reserves information-architecture changes). At
1440×900 those four rows leave only 492px of the 900px viewport for the tile
region at all. Recovering more of that — e.g. collapsing the filter rows
into a single denser control bar — is real, honest 2B-2 work, not something I
should paper over by claiming a percentage that isn't there.

---

## Anything the API doesn't return that the design wanted

Named explicitly, not faked:

1. **Per-date sensor/platform/cloud-fraction metadata is only available for
   a candidate's own detection pair** (`provenance.observations`, exactly 2
   entries, `role: before|after`) — not for the other before-years the date
   strip also lets an analyst preview. Confirmed by inspecting a live
   `/candidates/{id}` response: for a `2019_2026_004510` candidate, metadata
   exists for 2019 and 2026 only; 2021/2024/2025 have no sensor or
   `cloud_fraction` field anywhere in the payload. The date strip shows the
   full ISO date for all 5 (derivable from `temporal_trajectory.intervals`'
   window boundaries, which collectively cover every stack date) but leaves
   the sensor/cloud line blank for the 3 dates it has no data for, rather
   than fabricating a plausible-looking value.
2. **The exact pixel geometry needed to place a client-drawn footprint
   outline is not returned.** Read `geoseek/analyst/imagery.py`'s
   `_crop_true_color()`: the rendered crop window is
   `bbox_rc ± max(bbox_size × 1.6, (320 − bbox_size) / 2, 8)`, clamped to
   `[0, H]`/`[0, W]` where `H, W` are the **source raster's own pixel
   dimensions** — never returned by `/candidates/{id}`. `bbox_rc` and
   `centroid_rc` (both already returned) let me replicate this formula
   correctly for the common case (a candidate well inside the tile interior,
   where the edge-clamp never triggers), but I cannot verify from the API
   response alone whether a specific candidate is close enough to a tile
   edge for that clamp to matter — and a silently-mispositioned "here's what
   we detected" outline is a worse outcome than not offering the feature
   client-side at all. I used the server's own pixel-perfect `view=overlay`
   render instead (verified it works for every date, not just the after-date)
   — correct by construction, at the cost of the outline being the server's
   fixed red/yellow scheme rather than the change-type colour via Tokens the
   brief asked for. Naming this rather than shipping a maybe-wrong coloured
   outline.

---

## Every DOM rename / removal

No test depends on any of these (confirmed: `test_frontend_offline.py` and
`test_phase6_presentation.py` never select by these ids/classes).

**Candidate Detail — removed** (content relocated, not dropped): `#d_head` →
split across `#d_summary_body` (rail) and `#d_verdict_chip` (toolbar);
`#d_dates` → `#d_datestrip`; `#d_imgs` (3-image grid) → `#d_frame` /
`#d_frame_img` / `#d_frame_caption` / `#d_frame_note`; `.imgrow`, `.dateseg`,
`.detail-head`, `.imagery-panel` (CSS classes, fully removed, no longer
referenced anywhere). `#d_imgnote` kept the same id, just relocated under the
date strip (see the restored-caveat fix above).

**Candidate Detail — added**: `#d_body` gained class `detail-layout`;
`#d_segmented`, `#d_outline`, `#d_rail_toggle` (+ `#d_rail_toggle_icon`),
`#d_frame`, `#d_summary` (+ `#d_summary_body`), `#d_traj_panel`,
`#d_trace_panel`, `#d_hist_panel`, `#d_similar_panel` (new panel wrapper ids
— `#d_traj`, `#d_ev`, `#d_trace`, `#d_prov`, `#d_hist`, `#d_similar`
themselves kept their existing ids, just now living inside these new
wrappers). `.trace` (`<table>`) → `.trace-list` (`<div>`-based rows,
`.trace-row`); `.prov` (`<div>` grid) → `.prov-chain` (`.prov-chain__node`).

**Object Detection — added**: `.detect-layout`, `.detect-tile` (+
`__toolbar`/`__spacer`), `#dt_rail_toggle` (+ `#dt_rail_toggle_icon`),
`#dt_boxes_toggle`, `#dt_metrics_panel`, `#dt_counts_panel`,
`#dt_obstotals_panel` (new panel wrappers; `#dt_metrics`, `#dt_counts`,
`#dt_obstotals` kept their ids). `.split` (shared with other screens, e.g.
Search/Watch) is no longer used by this screen specifically — replaced by
`.detect-layout` here only, `.split` itself untouched and still used
elsewhere.

**Deleted from `app.js`**: `DET_CLASS_COLOR` (migrated to
`Tokens.detectClass()`/`tokens.css`'s `--detect-*`).
