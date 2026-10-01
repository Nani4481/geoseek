# Frontend Redesign — Plain-Language / Non-Expert Usability Report

Scope: a plain-language + guidance layer on top of the existing seven analyst
screens (Overview, Search, Review Queue, Candidate Detail, Discovery, Object
Detection, Watch Areas). No feature, filter, field or endpoint was removed —
every change either renames/explains something already shown, or adds
guidance on top of it. Hard constraints unchanged: fully offline (no CDN, no
web fonts, no map tiles, no icon library, no build step), vanilla JS/CSS,
every view under 1s, tests green.

Persona: a satellite-imagery analyst who is not an ML engineer. They
understand maps, places, "what changed," "is it real" — not embeddings,
vectors, HDBSCAN, SIG, or queue_score.

---

## 1. Purpose header + first step, every screen

A one-sentence `.view-purpose` block (new shared rule, `components.css`) at
the top of all seven screens (`index.html`), e.g. Review Queue: "Detected
changes, most important first. **Click any row** to see the before/after
evidence and confirm or reject it." Candidate Detail and Object Detection
also got one even though they're not top-level entry points, since a
first-time analyst can land there directly from a Queue row or the Overview
globe.

## 2. Jargon renamed or explained

| jargon | now reads | where |
|---|---|---|
| "vectors indexed" | "searchable image tiles" | sidebar footer, Overview stat context |
| Queue header "sig" | "significance" + hover "How large and how unusual this change is." | Review Queue table |
| Queue header "queue" (queue_score) | "priority" + hover "Our overall ranking — higher means review this first." | Review Queue table + Candidate Detail's "sorted by" line (was leaking the raw field name `queue_score`, found while verifying — see §6) |
| bare "conf" | "confidence" ring **+ High/Medium/Low band pill next to the number**, everywhere a ring appears (was ring-only in the Queue table before this pass) | Review Queue, Candidate Detail |
| persistence codes (`persistent`/`progressive`/…) | plain phrases ("Confirmed over time," "Still growing," "Appeared recently," …), each with a one-line hover — new `persistenceShort()` | Review Queue table (was showing the raw code verbatim), filter dropdown option text |
| raw `change_type` codes shown as the badge label | the existing `CTYPE_HUMAN` phrase is now what's actually rendered, not just computed alongside it | Review Queue table + map tooltip (both showed the raw code), Candidate Detail summary badge, `legend()` (Queue/Search map legends) |
| "HDBSCAN cluster map" + a technical concepts line | "We grouped every indexed location into **19 visual types** — things like trees and dense vegetation, …" with the algorithm/metric/per-cluster breakdown moved behind a **"How this works"** expander | Discovery |
| raw numeric `cluster N` next to a result | "type: `<plain concept label>`" via a new `clusterLabel()` lookup, seeded from the same `/discovery/clusters` response | Discovery results, Discovery meta line, Candidate Detail's "Discovery — more like this" panel |
| bare significance/priority decimals | number **+ a simple neutral bar** (new `Components.miniBar`, distinct colour from the confidence ramp so it never reads as good/bad) | Review Queue table |
| raw area in m² for large changes | hectares once ≥ 1 ha (`humanArea()`), e.g. `196,000 m²` → `19.6 ha` | Review Queue table, Candidate Detail |
| candidate id as the queue's primary "where" | replaced with a resolved **place name** (nearest staged region, via new `regionForPoint()`) or coordinates as a fallback; the id is not lost — it's the row's `title` tooltip and is still the first field shown once you open the candidate | Review Queue table |
| `sorted by queue_score` | `sorted by priority` (new `sortFieldLabel()` map) | Review Queue summary line |
| "operating point" | "Confidence margin ⓘ" with a hover explaining what a per-class operating point means | Object Detection |

Every one-line hover uses either a native `title` attribute directly, or the
new `Components.hint(text)` "?" glyph (small, always-visible, not just an
invisible title on the label) — offline-only, no tooltip library, per the
brief. Added to every Review Queue column header and filter, the Object
Detection confidence-margin control, the Watch Areas table header and min-
confidence field, and every Candidate Detail summary/rail label
(confidence/significance/persistence + the five rail panel titles) and
Overview stat tile.

## 3. Review Queue reads like a task list

Columns, in order: **# · What changed** (badge, human phrase + colour) ·
**Where** (place name / coordinates) · **Confidence** (ring + High/Med/Low) ·
**Significance** (number + bar) · **Priority** (number + bar) · **How big**
(hectares/m²) · **Seen** (persistence, plain phrase) · **First detected**
(earliest-change window) · **Decision**. The "Where" column isn't
sortable server-side (it's a client-side derived field, not one of the
API's own sort keys) — its header carries no sort chevron and the click
handler now explicitly skips headers with no `data-k`, rather than silently
sending a meaningless `sort=` value to the backend (checked: the backend's
`sort` key is a permissive `dict.get`, so this wouldn't have errored — it
would have quietly done nothing, which is worse to leave in).

## 4. Candidate Detail — plain-English verdict line

A new one-sentence line (`#d_verdict_sentence`, above the imagery) built
from data already on the candidate: `"{CTYPE_HUMAN phrase} detected near
{place}, first visible {between the earliest-change window, or a
persistence phrase if no window is available}. Confidence: {band}."` E.g.,
live against the real dataset: *"New open water / flooding detected near
ayodhya, first visible between 2019-03-30 and 2021-03-04. Confidence:
High."* — matches the brief's own example almost verbatim. `placeLabel()`
reuses the same region-bbox lookup as the Queue's "Where" column.

## 5. First-run walkthrough (separate from the existing Guided Tour)

The existing "Guided Tour" button (`DEMO_STEPS`) is a scripted, four-step
**presenter's demo** ("Ask in plain language" → a specific candidate → "find
more like it" → "all of this ran offline") — useful for showing the product
off, but not what the brief asked for, and left untouched. Added a second,
much shorter overlay (`ONBOARD_STEPS`, `#onboard` in `index.html`) aimed
specifically at a first-time analyst:

1. "Your change queue" — spotlights `#q_table`.
2. "Open one to see the evidence" — spotlights the first real row.
3. "Confirm or reject it" — actually opens that real candidate (the same
   `openDetail()` the rest of the app uses — no mock data) and spotlights
   the real confirm/reject controls.
4. "Or just ask" — navigates to Search and spotlights the query box.

Shown automatically once (`localStorage["geoseek_onboarding_seen_v1"]`),
"Skip" or finishing either step marks it seen; a new **"Help"** link next to
"Guided Tour" reopens it any time, ignoring the flag. `prefers-reduced-motion`
disables the spotlight's pulse (`.is-onboard-spot`), matching the rest of the
product's motion policy.

## 6. A real bug this surfaced, fixed in passing

Verifying the Queue's new "priority" header against the summary line above
the table ("sorted by …") turned up a genuine mismatch: the header now reads
"priority" but the summary line was still interpolating the raw backend
field name (`d.sort`, e.g. `queue_score`) directly into the page — exactly
the kind of leftover jargon the brief is about, just not one of the eight
named requirements. Fixed with a small `sortFieldLabel()` map so the two
pieces of UI never disagree about what the current sort is called.

## Verification

- **`pytest`**: 412 passed, 3 skipped (full suite, twice — once before the
  `sortFieldLabel` fix, once after). No Python file changed except a
  pre-existing, unrelated modification already in the working tree
  (`detections.py`); this pass touched `index.html`/`app.js`/`style.css`/
  `components.css`/`components.js` only.
- **`tests/test_frontend_offline.py`**: 35/35 passed — the no-external-URL
  scan (every text/binary file under the web root, not just the three
  original hand-picked ones) is clean after every edit in this pass.
- **`scripts/verify_offline_perf.py`** against the live 841-candidate
  dataset: all 10 functional checks PASS, every path under the 1s budget
  (same two pre-existing FAISS-KNN endpoints over budget as every prior
  phase report — `/candidates/{id}/similar` and `/discovery/similar`,
  untouched by this work), zero non-loopback network calls attempted.
- **Live headless-Chrome verification** (Playwright wasn't installed in any
  Python env on this machine and the browser-automation extension wasn't connected this
  session — drove real Chrome via raw CDP over `websockets` instead, the
  same workaround used in the Phase 8F detect-UI session): booted the real
  FastAPI server against the real dataset, walked all seven views plus the
  dev gallery is untouched, opened a real candidate from the real queue,
  ran a real Discovery lookup, and stepped through the new onboarding
  overlay end-to-end (including its real `openDetail()` call) — **zero
  console errors or exceptions anywhere**, at both 1920×1080 and 1440×900.
  Confirmed via the DOM directly (not just visually): every view's
  `.view-purpose` text, the Queue's badge/where/confidence/verdict cells,
  the Candidate Detail verdict sentence, the Discovery plain-language
  paragraph, and the onboarding overlay's per-step title + spotlighted
  element id all read exactly as designed. Object Detection's tile canvas
  (untouched drawing code) still renders real bounding boxes correctly —
  checked directly since a screenshot taken too early in an earlier pass
  showed a blank canvas and needed to be ruled out as a regression (it
  wasn't; the tile image just hadn't finished loading yet at that
  screenshot's timing).

## Renames

All renames are copy/label changes or new non-conflicting ids/classes
(`.view-purpose`, `.c-hint`, `.c-minibar`, `#onboard*`, `.is-onboard-spot`,
`.help-launch`) — no test selects any of them (confirmed:
`test_frontend_offline.py` and `test_phase6_presentation.py` never select by
id/class). The Review Queue's `<th data-k="candidate_id">` was removed
(replaced by an unsorted "Where" header); the candidate id itself is not
removed from the row — it moved to the row's `title` attribute — and remains
the first field on Candidate Detail's own summary panel.
