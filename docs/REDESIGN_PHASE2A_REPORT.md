# Frontend Redesign — Phase 2A Report

Scope: the shared component layer (`components.css`, `components.js`), a dev
gallery at `#/dev-gallery`, mechanical swaps of existing markup onto the new
components, and two carry-in fixes from Phase 1. No screen's layout or
information architecture was restructured (that is 2B).

---

## Carry-ins

### (a) tokens.js probe safety — real gap found and fixed, not just re-confirmed

The concern was correct, and digging into it turned up a **worse** problem
than the one stated. Verified empirically (a throwaway HTML page + Playwright,
not assumed):

- Canvas `fillStyle` assignment of an unparseable string is **silently
  ignored** — the property keeps its previous value, no exception. The old
  `try/catch` around the assignment could never fire; nothing throws.
- Worse: an **undefined** custom property doesn't make `color: var(--x)`
  produce an invalid/black/transparent result at all — it makes the
  declaration fall back to whatever colour the element would have **inherited
  anyway**. In the empirical test, `color: var(--totally-undefined-xyz)` on an
  element inside a `color: rgb(9,9,9)` body computed to `rgb(9, 9, 9)` — a
  perfectly plausible, non-zero, non-transparent colour. The original
  all-zero-readback check could never catch a typo'd or missing token name
  this way, because the failure doesn't look like a failure.

Fixed in `tokens.js`'s `resolveOne()` with two independent guards:
1. `getComputedStyle(document.documentElement).getPropertyValue('--name')` —
   checked **before** touching canvas at all. This is a direct existence
   check with no inheritance ambiguity: it is the empty string if and only if
   nothing in the cascade defines the property. Throws immediately if empty.
2. A sentinel (`rgba(1,2,3,0.004)`, chosen to never collide with a real
   resolved token) is set on `ctx.fillStyle` immediately before the real
   assignment; if `ctx.fillStyle` still reads back as the sentinel afterwards,
   the real value was rejected as unparseable — throws.

Verified both ends: a deliberately broken copy of `tokens.css` (one token
commented out) now throws `token "--accent-active" is not defined anywhere in
the cascade` and never sets `window.Tokens` at all (confirmed via
`page.on("pageerror")` in headless Chrome — the failure is loud and total, not
partial). The real, unmodified files still boot with zero errors.

### (b) verify_offline_perf.py — real numbers against the real dataset

The two failing checks were both a hardcoded `> 1000` candidate-count
assertion, stale against the current production dataset (841 candidates,
same 1104→841 shift `tests/test_phase6.py` already carries a comment about).
Changed both to `>= 500`, matching the floor the project's own pytest suite
already uses for the identical number. Full current run, real 841-candidate
dataset:

```
app booted offline in 6.6s
COLD start: search view first paint 542.7 ms  OK
OVERVIEW   /presentation/summary                    p95   19.5 ms
SEARCH     /health                                  p95   13.8 ms
           /stats                                   p95   18.6 ms
           /search/text                             p95   33.8 ms
           /tile/{id}/thumbnail (warm)               p95    1.6 ms
           /search/image (POST)                      p95   34.4 ms
QUEUE      /candidates (400)                         p95   45.0 ms
           /candidates (filtered)                    p95   11.9 ms
DETAIL     /candidates/{id}                          p95   34.1 ms
           .../imagery rgb (cold)                    p95   62.1 ms
           .../imagery rgb (warm/LRU)                p95    1.7 ms
           .../imagery overlay (cold)                p95   74.4 ms
           .../imagery rgb x2 (cold)                 p95  152.8 ms
DECISION   decision (POST)                           p95   31.7 ms
           /audit                                    p95   28.0 ms
           /export (filtered ~30)                    p95   41.2 ms
           /export (all 1104)                        p95  238.4 ms
DISCOVERY  /discovery/clusters                       p95   98.3 ms
           /candidates/{id}/similar                  p95  428.9 ms
           /discovery/similar (lon,lat)              p95  447.1 ms
```

All 10 functional checks PASS, every path under the 1s budget, no non-loopback
network call attempted. This run is faster across the board than Phase 1's
captured numbers (less system contention this session, not a code change —
no backend Python was touched besides the two assertions above).

---

## Components built

All in `components.css` (styling, tokens-only) + `components.js`
(`window.Components`, classic script, loaded after `tokens.js` and before
`app.js`). Gallery at `#/dev-gallery` (not in `<nav>`, not in `views[]` —
a separate `HIDDEN_VIEWS` list in the router) renders every state of every
component in one screenshot; see below.

1. **Change-type badge** (`.c-badge[data-type]`) — solid dot + low-alpha fill
   + label, one `[data-type]` attribute selector instead of six hand-written
   classes.
2. **Confidence display** — `.c-conf-ring` (compact/lg) and `.c-conf-bar`,
   sharing `Components.confBand()`/`bandPill()`.
3. **Data table** (`.c-table`) — sticky uppercase tertiary header, tabular
   numeric columns right-aligned, mono for ids, hover/focus/selected states,
   no zebra striping, sort-direction chevron.
4. **Panel/card** (`.c-panel`) — header/body/footer, optional collapse with
   in-memory (session-only) state via event delegation.
5. **Buttons and controls** — primary/secondary/ghost/danger, segmented,
   toggle, select, input, range, checkbox, and `.c-btn--decide` for
   confirm/reject.
6. **Skeleton** (`.c-skel`) — imagery-only shimmer, disabled under
   `prefers-reduced-motion`.
7. **Empty state** (`.c-empty` + `Components.emptyStates.*`) — see copy below.
8. **Stat/counter** (`.c-stat`) — ports the exact existing `animateCount()`
   easing, animates once on first render only.
9. **Latency badge** (`.c-latency`) — 120ms-delayed in-flight reveal,
   semantic-token colour coding against the 1s budget.
10. **Icon set** — see audit below.

---

## Confidence band thresholds — where they came from

Searched first, per the brief. Found in `geoseek/analyst/service.py:63-65`:

```python
def _confidence_band(v: float | None) -> str:
    v = float(v or 0.0)
    return "High" if v >= 0.85 else ("Medium" if v >= 0.60 else "Low")
```

**High ≥ 0.85, Medium ≥ 0.60, else Low.** The backend already ships a
precomputed `confidence_band` string on `/presentation/summary`'s featured
cards; nowhere else does an endpoint expose the band for a raw confidence
float, so the frontend still has to recompute it — but now in exactly one
place (`Components.confBand()`) instead of two: app.js had its own,
independently-matching `confBand()`/`RING_COL`, which is now deleted. Bands
render on the semantic ramp (`--success`/`--warning`/`--danger`), never the
change-type palette, as required.

**A real bug this surfaced, fixed, not just noted**: notification *severity*
(Watch Areas) reuses the identical "High/Medium/Low" vocabulary as confidence,
but with the **opposite polarity** — high severity is bad (red), high
confidence is good (green). My first pass wired both through the same
`.c-band[data-band="High"]` styling, which rendered a "High" severity alert in
success-green — telling an analyst the opposite of what it meant. Fixed with
an explicit `opts.kind: "severity"` flag on `Components.bandPill()` and an
inverted CSS variant (`[data-kind="severity"]`); verified in a live screenshot
that the alert banner is red, not green, after the fix.

---

## Data table row height: 44px

Not the number I designed to at first. The confidence ring (`.c-conf-ring`,
40px diameter) sits in every Review Queue row's confidence column, and a CSS
table-cell `height` is a **floor**, not a cap — a cell cannot render shorter
than its tallest child without clipping it. My first draft set `height: 34px`
reasoning purely from text density; verified against the actual rendered page
and found the ring was already forcing real rows to ~40-44px regardless,
silently overriding the CSS value. **44px** is the honest number: the 40px
ring plus 2px of breathing room top and bottom (`vertical-align: middle`). A
text-only row could sit around 30px, but this table always carries a ring, so
that number was never actually available — 44px is still well under a typical
default table row (~56-64px), which is what matters for scanning hundreds of
rows at once.

---

## Icon audit

17 icons cover everything the seven screens (plus the presentation layer) use.
15 already existed somewhere in `index.html`/`app.js` as hand-written inline
SVG; only **2 are new** (`chevron`, `close`) — both needed by components that
didn't exist before (panel collapse / sort direction; the reject button /
dismiss). One genuine **consolidation**: the product had two different
"layers" glyphs (an exploded-boxes design on Overview's map panel header, a
stacked-diamonds design on every `CoordMap`'s layers-toggle button) — now one
design, used both places (`index.html`'s static header icon was updated to
match).

| icon | used by |
|---|---|
| `overview` | Overview nav |
| `search` | Search nav, Search results panel header, "no search results" empty state |
| `queue` | Review Queue nav, "no candidates match filters" empty state |
| `detail` | Candidate Detail nav, imagery panel header |
| `discovery` | Discovery nav, "no neighbours found" empty states (×2) |
| `detect` | Object Detection nav |
| `watch` | Watch Areas nav, "Define a watch area" panel header, "no watch areas" empty state |
| `play` | Guided Tour button |
| `alerts` | Overview "Latest alerts" panel header, "no alerts"/"no notifications" empty states |
| `layers` | Overview "Where the change candidates are" panel header + every map's layers-toggle control (consolidated) |
| `featured` | Overview "Featured findings" panel header, "no featured findings" empty state |
| `check` | "Analyst decision" panel header, Confirm button |
| `grid` | Object Detection "Tile" panel header |
| `home` | Every map's reset-view control |
| `warning` | Object Detection caveat banner |
| `chevron` | *(new)* panel collapse control, table sort-direction indicator |
| `close` | *(new)* Reject button |

Single sprite as a JS module (`components.js`'s `ICONS` object + `icon()`
helper), one 24×24 viewBox, one stroke width (1.8), `currentColor`, rendered
at 16/20/24px via a size option. No icon library imported. **Not unified**
with the older, still-in-use `svg.icon` CSS rule (`style.css`, 14px, backs
every static nav/panel-header SVG that wasn't touched this phase) — see the
2B list below.

---

## CVD verification — actually simulated, not assumed

Rendered the resolved sRGB values (as `Tokens.js` actually resolves them, not
the authored oklch numbers) through a Machado et al. (2009) deuteranopia
matrix and a protanopia matrix, and measured Euclidean distance between every
pair (flagging anything under 30, a common minimum-separable-difference
heuristic) — script output, not eyeballing:

The two pairs the brief named were both already safely separated (the Phase 1
lightness-delta design worked): `water_gain` vs `water_loss` 124.4,
`construction` vs `clearance` 102.0 under deuteranopia.

**But the sweep found a pair the brief didn't name that failed**: `road` vs
`other`, distance **14.2** under deuteranopia — well under the 30 threshold,
and clearly distinguishable to normal vision (blue-grey vs. green) but not to
deuteranopia, where both collapse toward the same mid-grey. Fixed by lowering
`--change-other`'s lightness from 75% to 60% (same hue, same chroma) —
re-verified all 15 pairs under both deuteranopia and protanopia, all now clear
30 with margin (`road` vs `other` now 73.4 deuteranopia / 59.0 protanopia, the
new worst case). Applied to `tokens.css`, re-verified live in Chrome
(screenshot below), all 412 pytest cases still pass.

---

## Screenshots

All captured live via Playwright + headless Chrome (claude-in-chrome was not
connected this session).

**Dev gallery** (`#/dev-gallery`, full page) — every component, every state,
one screenshot: badges (post-fix, `other` now visibly darker/distinct from
`road`), confidence ring/bar, the data table with hover/focus/selected rows
side by side, static/open/closed panels, all four button kinds ×
default/hover/focus/disabled plus the confirm/reject pair, segmented/toggle/
select/input/range/checkbox, the imagery skeleton placeholder, all four named
empty states with their exact copy, three stat cards (accent, +delta,
-delta), all three latency-badge states, and the full 17-icon set at three
sizes.

**Overview** — counters (now `.c-stat`), the "High" severity alert correctly
red after the polarity fix, globe with region pins, all rendering with zero
console errors beyond a benign `favicon.ico` 404 (the app declares none;
unrelated to this work, present since Phase 1).

**Review Queue** — the actual before/after this phase was built for: the
table now reads as a spreadsheet at a glance (green/amber confidence rings
instead of bare decimals + a warn→ok gradient bar, badges with dots instead
of solid-fill pills, right-aligned tabular numerics, hairline dividers, no
zebra stripes, a live sort-direction chevron on the `queue` column).

**Candidate Detail** — head strip shows the new badge/ring(lg)/band together
(`water_gain` · ring "91" green · band "High"), verdict "reject" in red; the
two former `<details class="disclosure">` panels ("Why did the system flag
this?", "Where did this data come from?") are now real `.c-panel`s and their
collapse click was verified working on the live page, not just the gallery.

(Screenshot files themselves are session-local artifacts, not part of the
repo; described above since they aren't renderable in this report format.)

---

## Every DOM/class rename

Per the rules, renames are listed in full (no test depends on any of these):

| old | new | where |
|---|---|---|
| `.badge` / `.b-water_gain` etc. (6 classes) | `.c-badge[data-type="…"]` | style.css (deleted) → components.css |
| `.confring` / `.confring.sm` | `.c-conf-ring` / `.c-conf-ring.lg` | style.css (deleted) → components.css |
| `.meter` | *(removed — replaced by `.c-conf-ring`)* | style.css (deleted) |
| `.band` / `.band.high/.medium/.low` | `.c-band[data-band="High/Medium/Low"]` (+ `[data-kind="severity"]` variant) | style.css (deleted) → components.css |
| `.emptystate` / `.t` / `.s` | `.c-empty` / `.c-empty__cause` / `.c-empty__action` | style.css (deleted) → components.css |
| `.stat` / `.stat.accent` / `.n` / `.l` (Overview counters) | `.c-stat` / `.c-stat--accent` / `.c-stat__value` / `.c-stat__label` | style.css (rule body deleted) → components.css |
| `button.confirm` / `button.reject` | `.c-btn.c-btn--decide.c-btn--confirm` / `…--reject` | style.css (deleted) → components.css |
| `.skel` | `.c-skel` | style.css (deleted) → components.css; every call site in `app.js` (`IMGLOAD`, `thumbHTML`, detail imagery figures, `.dt-canvaswrap`) and the one `.ff-im` site in `index.html`-rendered markup |
| `<details class="panel disclosure" id="d_why">` / `id="d_provwrap"` | `<div class="c-panel" id="d_why">` (same ids kept) | index.html; `app.js`'s guided-demo step updated from `d.open = false` to `Components.setPanelCollapsed(id, true)` |
| `<button class="confirm" id="d_confirm">` / `class="reject" id="d_reject">` | `class="c-btn c-btn--decide c-btn--confirm/reject"` (same ids kept) | index.html |
| `<span class="chip" id="latencyChip">` | `<span class="c-latency" id="latencyChip">` (same id kept) | index.html |
| `#q_table`'s `class="tablewrap"` wrapper | `class="c-table-wrap"` | index.html |
| `<table id="q_table">` | `<table id="q_table" class="c-table">` (same id kept) | index.html |
| header icon path for "Where the change candidates are" | swapped to the consolidated `layers` glyph | index.html |
| `nav button[data-route]` set | unchanged, but `dev-gallery` deliberately **not** added here | index.html |

No `id` was renamed anywhere — only classes changed, and only on elements no
test selects by class.

---

## Spinners removed

All 9 (2 static in `index.html`, 7 dynamic in `app.js`) — replaced with plain
"Loading…" text, since `.c-skel` is scoped to imagery only per the brief and
these were all text/card loading states, not images. `.spinner`/`@keyframes
sp`/`.chip .spinner` deleted from `style.css`.

---

## Where a component could not absorb an existing usage without a layout
decision — the 2B queue

- **Confidence bar (`.c-conf-bar`) has no real placement yet.** It's built,
  tokenized, and demonstrated in the gallery, but nothing in the current
  markup has a "wide, detail-panel-shaped" slot for it without restructuring
  `#d_head`'s flex row (currently a compact `.kv` chip strip, sized for the
  ring). Candidate Detail's headline confidence stays a ring this phase — a
  same-slot, genuinely mechanical swap. Placing the bar is a 2B layout call.
- **`svg.icon` (14px, style.css) vs. `svg.c-icon` (16/20/24px,
  components.css) are two separate rules.** Every *static* nav/panel-header
  icon across all seven screens still uses the old 14px rule; unifying them
  means touching every header simultaneously — a broad, visually-global
  change I judged too high-blast-radius for a "swap it in place" phase.
- **~15 other `.panel` instances were not converted to `.c-panel`.** Only the
  two that were *already* collapsible (`<details>`) got converted — that's
  the only case where the swap is unambiguous. Whether each remaining panel
  *should* become collapsible (and what that does to a screen's information
  hierarchy) is exactly the kind of judgment call reserved for 2B.
- **Overview's stat-card hover-lift (`translateY(-3px)` + a top accent line
  on hover) was not carried over to `.c-stat`.** It wasn't required by the
  component spec, and I judged it decorative rather than functional; noting
  it here rather than silently dropping it without a record.
- **The two map tooltip/label spots that show a bare "conf 0.94"-style string**
  (`app.js`'s `label:` fields feeding `CoordMap`'s hover tooltip, Overview and
  Queue map markers) still show a raw decimal — a tooltip is plain text, not
  a place a ring/bar component can be mechanically inserted without redesigning
  the tooltip itself.
- **Change-type colour still has two identities for badges specifically vs.
  everything else.** `tokens.css`'s `--change-*` (canvas/WebGL, via
  `Tokens.changeType()`) and `.c-badge`'s use of the same tokens are now
  unified — but this was already true after Phase 1 for canvas; the
  first-order fix this phase was closing `style.css`'s **separate** hex
  copy (`.b-water_gain` etc.), which is done. There is no remaining second
  copy anywhere in the shipped CSS/JS now.
