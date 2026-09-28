# Frontend Redesign — Phase 1 Report

Scope actually touched: `tokens.css`, `tokens.js` (new), `app.js`, `globe.js`,
`index.html`, `style.css` (global chrome only), `tests/test_frontend_offline.py`
(new), `tests/test_phase6_presentation.py` (trimmed), `data/provenance_manifest.json`
(two vendor entries annotated). No view-specific layout or the seven screens'
own styling was touched, per the brief.

---

## 1. Full token list — final values and purpose

All in `src/geoseek/analyst/web/tokens.css`, on `:root` (dark, default) with a
`:root[data-theme="light"]` override block carrying the same names.

### Colour (dark theme)

| token | value | purpose |
|---|---|---|
| `--surface-app` | `oklch(16% 0.028 264)` → `#070d19` (HSL-L 6.3%) | page background |
| `--surface-panel` | `oklch(21% 0.030 264)` → `#111826` (HSL-L 10.8%) | default container |
| `--surface-raised` | `oklch(27% 0.032 264)` → `#1e2636` (HSL-L 16.5%) | hovered/nested/elevated container |
| `--surface-overlay` | `oklch(33% 0.034 264)` → `#2c3547` (HSL-L 22.5%) | dropdown/modal/topmost chrome |
| `--border-hairline` | `oklch(85% 0.01 264 / 10%)` | quiet internal divider |
| `--border-strong` | `oklch(88% 0.01 264 / 20%)` | a boundary meant to be seen |
| `--text-primary` | `oklch(92% 0.01 264)` | body/headings; not pure white |
| `--text-secondary` | `oklch(74% 0.02 264)` | supporting copy — **7.7:1 on `--surface-panel`** (real WCAG math, not eyeballed) |
| `--text-tertiary` | `oklch(58% 0.022 264)` | labels/meta only — **4.1:1 on `--surface-panel`** |
| `--text-disabled` | `oklch(42% 0.018 264)` | inactive controls — 2.1:1, legibility not guaranteed by design |
| `--accent` / `-hover` / `-active` | `oklch(66/72/58% 0.19 258)` | the ONE interactive hue — never decorative |
| `--accent-subtle` | `color-mix(in oklch, var(--accent) 16%, transparent)` | low-alpha accent background (focus halo, subtle active state) |
| `--success` | `oklch(70% 0.17 142)` | confirmed / healthy / pass |
| `--warning` | `oklch(78% 0.15 85)` | needs attention, not yet bad |
| `--danger` | `oklch(65% 0.20 15)` | rejected / failed / urgent |
| `--info` | `oklch(72% 0.12 222)` | neutral informational |
| `--change-water_gain` / `-fill` | `oklch(68% 0.19 258)` | new open water / flooding |
| `--change-water_loss` / `-fill` | `oklch(48% 0.16 320)` | water body shrank / dried |
| `--change-construction` / `-fill` | `oklch(70% 0.16 55)` | new built-up surface |
| `--change-clearance` / `-fill` | `oklch(87% 0.15 100)` | vegetation / land cleared |
| `--change-road` / `-fill` | `oklch(68% 0.02 264)` | new road / linear corridor |
| `--change-other` / `-fill` | `oklch(75% 0.17 170)` | surface change, unclassified |

CVD-safety requirement verified with real numbers, not just hue choice:
`water_gain` L68 vs `water_loss` L48 (Δ20); `construction` L70 vs `clearance`
L87 (Δ17). Every `-fill` variant is `color-mix(in oklch, var(--change-X) 18-22%, transparent)`.

Light theme mirrors every name above with light-appropriate values (dark
surfaces → light, text inverted, saturations bumped for on-white contrast).
Verified: `text-secondary` 9.5:1 / `text-tertiary` 5.2:1 against its
`surface-panel`. **Caveat**: no toggle exists yet (by design, per brief), so
the light block has never been exercised in a real browser — re-verify before
Phase 2 turns a switch on.

### Type

`--font-sans`, `--font-numeric` (sans stack + monospace fallback, pairs with
`.font-numeric { font-variant-numeric: tabular-nums }`), `--font-mono`.
Scale: `--text-2xs 11px` (chip/pill/scale-bar micro-labels) → `--text-xs 12px`
(table headers, meta) → **`--text-sm 13px` (body default)** → `--text-md 14px`
(emphasized body) → `--text-lg 16px` (subheadings) → `--text-xl 20px`
(section headings) → `--text-2xl 26px` (page headings) → `--text-3xl 34px`
(display counters). `--leading-tight 1.2` / `--leading-normal 1.55`.
`--weight-regular/medium/semibold/bold` = 400/500/600/700 (deliberately no
`extrabold` token — see §4).

### Spacing / elevation / motion / z

`--space-1..9` = 2/4/8/12/16/24/32/48/64px. `--radius-control 6px` /
`--radius-panel 12px` / `--radius-pill 999px`. `--shadow-sm/md/lg`, all
near-black low-spread (`oklch(0% 0 0 / 45-55%)`), no grey glow.
`--duration-fast/base/slow` = 120/180/240ms, collapsed to 1ms under
`prefers-reduced-motion: reduce`. `--ease-out` (entrances/state changes,
`cubic-bezier(.16,1,.3,1)`) / `--ease-in-out` (position changes,
`cubic-bezier(.65,0,.35,1)`) — both bounded to y∈[0,1], no overshoot.
`--z-map 1` / `--z-panel 10` / `--z-sticky-header 20` / `--z-dropdown 30` /
`--z-modal 50` / `--z-toast 60`.

---

## 2. Deliverable 2 — every call site changed

**Deleted:** `CTYPE_COLOR` (`app.js:7-10`), `hexA()` (`app.js`, was 599-602),
`CTYPE_RGB` (`globe.js`, was 346-349), `hexA()` (`globe.js`, was 129-132).

**app.js** (9 sites):
- `legend()` — `CTYPE_COLOR[t]` → `Tokens.rgb("change-" + t)`
- Overview inline legend render — same substitution
- Overview map data (`color: CTYPE_COLOR[...]`) → `Tokens.changeType(c.change_type)`
- Queue map data — same substitution
- `draw()` region/AOI polygon fill+stroke (`hexA(it.color,...)`) → inline `rgba()`/`rgb()` template built from a resolved triple
- `draw()` single point marker fill+stroke — same
- `draw()` mixed-cluster colour (`css.getPropertyValue("--accent")` string branch alongside `hexA(color,...)`) → `Tokens.triple("accent")` for the mixed branch, so both ternary branches return the same shape — **this one was forced**, not optional: the `hexA(color,0.86)` fix requires `color.r/g/b`, so the sibling branch had to stop returning a raw CSS-var string
- `draw()` AOI drag-rectangle (`hexA(accent,0.15)`) → `Tokens.triple("accent")` composed inline
- Hover tooltip (`this.tip.style.setProperty("--tip-color", hit.color...)`) — **not a hexA/CTYPE_COLOR site, but a consequential fix**: `--tip-color` is a CSS custom property and needs a colour *string*; `hit.color` can now be a resolved triple *object*, so this was updated to build `rgb(r,g,b)` from it (see bug below)

**globe.js** (3 sites):
- `pinTexture(color)` / `ringTexture(color)` — signature changed to accept a
  resolved `{r,g,b}` triple instead of a hex string; internal `hexA(color,0.5)`/
  `hexA(color,0)` gradient stops rebuilt directly from `r,g,b`
- Region-pin "has candidates" indicator (`"#ffd166"`/`"#7fd4ff"` literals) →
  `window.Tokens.triple("warning")` / `window.Tokens.triple("info")` — **a
  deliberate re-scoping, explained in §4**, not a like-for-like swap
- Candidate point-cloud vertex colours (`CTYPE_RGB[c.change_type]`) →
  `window.Tokens.changeType(c.change_type)` (÷255 for THREE's 0-1 floats) —
  **this is the literal fix for the water_loss drift the audit found**

### A bug I found and fixed mid-implementation

`CoordMap.draw()` and the hover tooltip consume `it.color`/`hit.color`
polymorphically: change-type-driven items now carry a resolved `{r,g,b}`
object (via `Tokens.changeType()`), but `REGION_PALETTE`-driven items (Watch
Areas' region/AOI rings) and a couple of fixed literals (Search's `"#4c8dff"`
result markers, the `"#e9edf5"` AOI-selection outline) still carry plain hex
**strings** — correctly, since those aren't change-type colours and are out
of this phase's scope. My first pass assumed every `.color` was an object and
would have silently broken Watch Areas' region rings and Search's markers
(`"#4c8dff".r` is `undefined` → an invalid canvas colour → Canvas2D silently
*keeps the previous fillStyle* rather than erroring — exactly the "silent
wrong colour" failure mode the audit warned about). Fixed with a small,
narrowly-scoped `_colorTriple(c)` normalizer in `app.js` that accepts either
shape and returns `{r,g,b}` — **not** a reincarnation of `hexA()` (which built
one hardcoded-alpha rgba string at a time); this only produces the same
numeric triple `Tokens` itself uses, so every consumer still composes its own
`rgba()`/`rgb()` string the same way regardless of where the colour came from.
Verified fixed with a real headless-Chrome screenshot of Watch Areas — see §5.

### Verifying the "one line, one file" claim

Checked directly: `Tokens.changeType('water_gain')` is the only place any of
app.js, globe.js, or a canvas/WebGL draw call obtains a change-type colour.
Changing `--change-water_gain` in `tokens.css` and nothing else now changes
the Overview/Queue canvas markers, the Watch/Search polygon and point
renderers, the globe's point cloud, and the `.b-water_gain` CSS badge (which
already read `var(--water_gain)` in `style.css` — unchanged in this phase,
still a *second*, independent CSS value, see §4) — **claim holds for every
canvas/WebGL consumer; does not yet hold for the legacy `style.css` badge
tokens, which Phase 2 owns.**

---

## 3. `three.module.min.js` URL enumeration and verdict

Recorded in `data/provenance_manifest.json` under the `threejs_vendor`
artifact's new `external_urls` field (sha256-pinned: the test fails if the
vendored file changes without this being re-reviewed).

| URL / pattern | kind | verdict |
|---|---|---|
| `http://www.w3.org/1999/xhtml` | XML namespace constant | Not a fetch. Passed to `document.createElementNS()` as an opaque identifier the DOM spec requires — never dereferenced over a network. |
| `` /^https?:\/\// `` and `` /^(https?:)?\/\// `` (regex source, escaped slashes) | URL-detection regex | Not a fetch. Pattern text inside three.js's own `LoaderUtils`-style helper that classifies an *already-known* local resource path as absolute vs. relative. |
| `https://discourse.threejs.org/t/updates-to-lighting-in-three-js-r155/53733` (appears twice) | static console-warning text | Not a fetch. Plain diagnostic string shown via `console.warn()` when a deprecated lighting API path is used — never assigned to a `src`/`href`, never opened. |

**None are a runtime fetch.** `OrbitControls.js` was checked too: zero
external URLs.

Being precise about a limitation: my automated test's regex only matches
*literal* `http(s)://` substrings, so it does **not** actually flag the
escaped-slash regex-source fragment above (`\/\/` ≠ `//` as bytes) — I found
and classified that one by manual inspection while enumerating, not via the
scanner. The scanner does correctly catch the other two (both appear
unescaped inside real JS string literals). I recorded all three in the
manifest anyway for a complete human record, even though the test only
actively verifies two of them are still present/unchanged.

---

## 4. Places a single token can't serve two uses (named, not papered over)

- **Change-type colour still has two identities, not one, until Phase 2.**
  `tokens.css`'s `--change-water_gain` etc. and `style.css`'s pre-existing
  `--water_gain` etc. (`style.css:39-44`, feeding `.b-water_gain` badges) are
  two separate custom properties with independently-chosen values. I did not
  merge them: the brief explicitly says "leave existing literal colours in
  style.css alone" and badges are view-specific styling, out of scope this
  phase. So today, a badge's colour and a canvas marker's colour for the same
  `water_gain` type are **coincidentally similar, not identically resolved** —
  the exact structural problem the audit described, now cut from three copies
  to two, with the remaining one flagged for Phase 2 rather than hidden.

- **`globe.js`'s region-pin "has candidates" indicator was never a
  change-type colour at all.** It re-used `"#ffd166"` (literally identical to
  the old `clearance` hex, by coincidence, for an unrelated "this region has
  activity" cue) and a made-up `"#7fd4ff"` for "empty." There is no existing
  token whose *stated meaning* is "region has activity" — I mapped it onto
  `--warning`/`--info` because their semantics (needs attention / neutral
  informational) fit better than inventing a near-duplicate token, but this
  is a genuine re-scoping of intent, not a mechanical swap, and I'm flagging
  it rather than quietly changing what a pin colour "means."

- **`--accent-subtle`'s alpha (16%) is a single fixed value** asked to serve
  both a focus-ring halo (small, tight glow, wants to stay subtle) and any
  future "quiet active" chip background (wants to stay legible as a fill).
  Both currently look acceptable at 16%, but if Phase 2 finds a chip
  background needs more presence than a focus halo does, that's one token
  trying to do two jobs and should split rather than get nudged to a
  compromise value that serves neither well.

- **Font weight has no `extrabold` token**, but `style.css` already has
  several raw `font-weight: 800` uses (`.ov-title`, `.ov-counters .stat .n`).
  I deliberately did not add an 800 token — not every face in the system sans
  stack reliably distinguishes 800 from 700 — so these are a real,
  named gap for Phase 2 to either accept as 700 or resolve with a
  system-font audit, not something a 5th weight token should paper over.

---

## 5. Verification

**pytest**: `410 passed, 3 skipped, 0 failed` (full suite, including the
rewritten offline test and the trimmed `test_phase6_presentation.py`). The 3
skips are pre-existing/environment-gated, not introduced by this work.

**`scripts/verify_offline_perf.py`**: ran clean on the network-block and
latency fronts (no non-loopback connection attempted; every path stayed under
budget except `/candidates/{id}/similar` and `/discovery/similar`, both FAISS
KNN endpoints at 1.2-1.4s). It reported **2 functional-check failures**
("overview summary" and "queue lists candidates") — both are hardcoded
`> 1000` candidate-count assertions inside that script, stale against the
production dataset's current 841 candidates (the same 1104→841 shift the
project's own pytest suite already has comments explaining and thresholds
already adjusted for). **This script never loads or executes any of
app.js/style.css/tokens.css/tokens.js — it's a pure FastAPI TestClient
probe — so none of this is caused by or related to Phase 1.** I have no
"before" run from this session to diff against (I didn't run it before
touching the frontend), but since zero Python/backend files were modified,
there's no mechanism by which this work could have moved these numbers.

**Boot verification, in a real browser** (Playwright + the system Chrome,
headless — claude-in-chrome wasn't connected this session): loaded
`/app/`, confirmed `window.Tokens` resolves at boot with no thrown error,
spot-checked several resolved values against independently-computed OKLCH→sRGB
math (exact matches, e.g. `--surface-app` → `rgb(7,13,25)`, predicted `#070d19`).
Screenshotted the globe (renders correctly: Blue Marble day/night terminator,
star field, border overlay, region pins in the new warning/info colours, one
pulsing "top" pin), the 2D Overview map (real satellite basemap, correctly
coloured/clustered change-type markers, unmistakable solid-accent active nav
pill), Watch Areas (region/AOI polygon rings — this is what the
`_colorTriple` bug fix above was verified against, and it renders correctly),
and Search (point markers + clusters using the untouched literal-string
colour path). The only console message across all of this was a benign
`favicon.ico` 404 (the app declares none; unrelated to this work).
End-to-end regression check: the Overview legend's `water_gain` swatch
background, read back from the live DOM via `getComputedStyle`, is
byte-identical to `Tokens.rgb("change-water_gain")` called directly —
confirming the fix for the drift the audit found.

---

## 6. Anything in `style.css` that will fight the token system in Phase 2

- **The old `:root` block (`style.css:5-83`) is still there, fully live**,
  powering every one of the seven screens untouched this phase. Two token
  systems now coexist in the same cascade. Phase 2 has to retire the old one
  screen by screen, not delete it in one pass — deleting early would break
  every screen not yet migrated.
- **`.b-water_gain` etc. (`style.css:292-294`) and the change-type badges
  generally** are the highest-value, most obvious Phase 2 target, precisely
  because they're the other half of the "two identities" problem in §4.
- **23 raw `font-size` values already documented in the audit** (`style.css`,
  e.g. `11.5px`, `13.5px`, `30px`) don't disappear just because `--text-*`
  now exists alongside the old `--fs-*` — they were never on either scale.
- **Two raw `999px` border-radius uses and several small raw radii** (6-8px
  range) sit outside both the old 5-step and new 3-step radius scales;
  consolidating to exactly 3 radius tokens necessarily approximates some of
  these, which is expected, but Phase 2 should expect visible ~1-2px shifts
  wherever it retargets them.
- **`--ease` (old, `style.css`'s own `:root`) vs `--ease-out`/`--ease-in-out`
  (new) are different curves**, not aliases — `cubic-bezier(.2,.7,.3,1)` vs.
  `(.16,1,.3,1)`/`(.65,0,.35,1)`. Every transition I moved onto the new
  tokens in Deliverable 4 got a real (small) easing-curve change, not just a
  rename; the many transitions still on `var(--ease)` elsewhere in
  view-specific CSS will look subtly different once Phase 2 retargets them.
- **The decorative body-background gradient used to read `var(--accent)` and
  `var(--water_loss)` purely for two corner "blooms."** I did not carry
  those references forward onto the new tokens (`--accent` is now
  explicitly interactive-only, "never decorative," per the brief, and tying
  page-chrome decoration to a change-type token was never well-motivated
  anyway) — I retargeted both blooms onto neutral `--surface-raised`/
  `--surface-overlay` mixes instead. Phase 2 should know this was a
  deliberate palette change to the hero background, not an oversight, if the
  before/after look different there.
