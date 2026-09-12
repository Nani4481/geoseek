# Phase 8 polish round 2 — watch-area input UX, cluster map legibility, cluster quality

Three fixes, all against the live 104,089-tile / 841-candidate index. Numbers
below are from real runs on this machine; the sweep JSON is committed under
`data/discovery/cluster_quality_sweep.json` (gitignored like all of `data/`,
but reproducible via `scripts/cluster_quality_sweep.py`).

| # | problem | fix |
|---|---|---|
| 1 | Watch-area AOI required typing a raw bbox string; same on Queue/Search. | New `/regions` endpoint (bbox per catalog region) backs a one-click region dropdown on all three views, plus click-drag-a-rectangle on a new map on the watch-area form. |
| 2 | Discovery's cluster map rendered as an unreadable ~29,000×500px strip. | Replaced the per-observation strip layout with a single true lon/lat scatter (`save_geographic_cluster_map`) - legible at any observation count. |
| 3 | One cluster held 86-99% of all tiles; two clusters shared the identical label "bare dry open ground". | Swept `cluster_selection_method` (eom/leaf), `min_cluster_size`, PCA pre-reduction, and sample size; switched the default to leaf + PCA-50 (19 well-balanced clusters, largest 18.4%) and added region-aware label disambiguation so no two clusters ever display the same string. |

---

## Part 1 — watch-area / Queue / Search AOI input

**Root cause**: the only way to set an AOI anywhere in the app was typing
`west,south,east,north` into a text box, with a placeholder that showed a
real-looking coordinate string (easy to mistake for a pre-filled value).

**Fix**:
- `AnalystService.list_regions()` / `GET /regions`: for every region already
  in the catalog (`_region_of(aoi_name)`, the same collapsing Phase 6's
  presentation layer uses), unions the footprint bounds of every observation
  in that region into one bbox. Pure read over the existing catalog seam, no
  new storage.
- Watch-area form: a region `<select>` (fills the bbox field in one click), a
  new canvas map (`CoordMap` with `onDrawRect`) showing every region as a
  faint coloured rectangle - click-drag draws a custom AOI, clicking a
  region's rectangle picks it, and the bbox text field stays live-editable
  for precision. Placeholder text no longer looks like real data
  ("optional - draw on the map or pick a region above" instead of a raw
  coordinate string). `min confidence` now defaults to 0.5 instead of blank.
  A watch area was already creatable with just a name (bbox/query/types are
  all optional server-side) - this makes that path discoverable rather than
  hidden behind an unlabelled optional field.
- `CoordMap` gained generic drag-to-rectangle support (`_dragStart/_dragMove/
  _dragEnd`, plus `lon()`/`lat()` inverse-projection helpers) that coexists
  with its existing click-to-pick behavour (a plain click/tap never moves
  more than a few px, so `onDrawRect` only fires on an actual drag).
- The same region dropdown was added to the Review Queue's and Search view's
  AOI bbox filters (`#f_region`, `#s_region`) - one `fillRegionSelect()` /
  `loadRegions()` (cached after first fetch) helper shared by all three.

Screenshots: `data/change_model/ui_screens/` (gitignored) - watch form empty,
a region picked (map highlights it, bbox field fills), and a hand-drawn
rectangle (bbox field updates live).

## Part 2 — cluster map legibility

**Root cause**: `save_cluster_map` lays out one grid panel per **observation**
side by side. At 3-5 observations (its original design point) that's fine; at
77 observations (Phase 8's 5-date archive) it produced a 29,388×489px image -
displayed at `width:100%` in the Discovery view, that is a few-pixel-tall
smear.

**Fix**: `save_geographic_cluster_map` (`geoseek/discovery/cluster.py`) plots
every tile at its real lon/lat (matplotlib scatter, coloured by cluster,
legend listing every cluster's id / size / share / label). Since the 9 AOI
regions are geographically far apart across India, clusters naturally appear
as distinct, spatially coherent blobs - directly showing the
region-generalization result rather than requiring 77 side-by-side panels.
Output is a normal ~1300×1150px image, legible at `width:100%` on any screen.
`save_cluster_map`'s per-observation layout is kept (documented as
QA/small-run only) for `scripts/cluster_tiles.py`, which nothing else uses.

`AnalystService.discovery_clusters()`'s stale advice ("run
scripts/cluster_tiles.py" - the old, no-longer-practical script) was also
fixed to point at `cluster_at_scale.py`.

## Part 3 — cluster quality (balance + duplicate labels)

**The problem, quantified**: the previous default (HDBSCAN's library-default
`eom` cluster-selection rule, `min_cluster_size=40`, no dimensionality
reduction, 20k-tile stratified sample) produced 6 clusters, one holding
89,035/104,089 tiles (85.5%), with two different cluster ids independently
landing on the identical top concept "bare dry open ground" (only 15 fixed
RemoteCLIP-text concept phrases exist to label against - see `CONCEPTS`).

**What was tried** (`scripts/cluster_quality_sweep.py`, same corpus/seed,
varying only the knob under test):

| config | sample | clusters | largest cluster | dup top-labels | runtime |
|---|--:|--:|--:|--:|--:|
| eom, mcs=40 (previous default) | 10k | 2 | 98.8% | 0 | 73s |
| eom, mcs=40 (previous default) | 20k | 6 | 85.5%¹ | 1 | 243s |
| eom, mcs=100 | 10k | 2 | 92.4% | 0 | 72s |
| eom, mcs=15 (finer) | 10k | 9 | 54.4% | 1 | 71s |
| **leaf**, mcs=40 | 10k | 7 | 30.5% | 1 | 72s |
| **leaf**, mcs=15 (finer) | 10k | 17 | 19.7% | 2 | 71s |
| **leaf**, mcs=40 | 20k | 11 | 30.2% | 1 | 241s |
| eom, mcs=40 + **PCA-50** | 10k | 2 | 98.9% | 0 | 4.7s |
| leaf, mcs=40 + **PCA-50** | 10k | 9 | 34.5% | 1 | 5.1s |
| leaf, mcs=40 + **PCA-50** | 20k | 17 | 17.3% | 2 | 7.8s |
| eom, mcs=40 + PCA-50 | 40k | 10 | 51.0% | 1 | 16.9s |
| **leaf, mcs=40 + PCA-50 (chosen default)** | **40k** | **19** | **18.4%** | 3→**0** after disambiguation | **10.9s** |

¹ from the actual production run at the time (`cluster_at_scale_100k.json`),
same config as the "eom, mcs=40, 20k" sweep row.

**Findings**:
- `eom` (the HDBSCAN library default) is both consistently imbalanced on this
  corpus (51-99% in one cluster, every config tried) *and* unstable across
  sample size for the identical config (2 clusters at 10k, 6 at 20k) - not a
  reliable choice for a discovery feature meant to expose structure.
- `leaf` selection is dramatically more balanced (18-35% largest cluster
  across every sample size and `min_cluster_size` tried) and gets *more*
  balanced as the sample grows, the opposite of `eom`'s behaviour.
- PCA-50 before HDBSCAN is a strict win: 15-25x faster (the curse of
  dimensionality at raw 512-d is the actual bottleneck, not corpus size -
  confirms Phase 7b's finding that single-pass HDBSCAN at full dimensionality
  doesn't finish in practical time) with equal-or-better balance at every
  sample size tested, since it makes larger samples cheap enough to run.
  Concept labeling and the final nearest-centroid assignment against the full
  104,089-tile corpus both still happen in the native 512-d unit-norm space -
  PCA axes are meaningless to RemoteCLIP's text tower and centroids must stay
  comparable to un-reduced corpus vectors.
- **Kept**: `cluster_selection_method="leaf"`, `pca_dim=50`,
  `min_cluster_size=40` (unchanged), `sample_size=40_000` (up from 20k - PCA
  makes this cheap). Production run: **19 clusters, 0 noise, largest 18.4%**
  (c16, "a small rural village settlement", dehradun-dominant).
- Region purity in the production run is high for most clusters (13 of 19
  clusters are >65% one region, several >95%) - the split is finding real
  region-correlated structure, not noise. It is genuinely NOT possible to get
  a "nice" even split across 15 fixed concept categories over 9 very
  different landscapes (jaisalmer desert vs. kerala backwaters vs. sundarbans
  mangrove) without either an unbalanced mega-cluster (eom) or several
  clusters sharing a top label because they really are visually similar in
  RemoteCLIP's space (several "bare dry open ground" clusters, one per
  region) - stated honestly rather than forced apart with an arbitrary
  vocabulary expansion.
- **Duplicate labels**: `disambiguate_cluster_labels()` appends each tied
  cluster's dominant region (already computed for the region-purity report)
  to its displayed label - e.g. "bare dry open ground (kutch)" vs. "...
  (jaisalmer)" vs. "... (dehradun)" - genuinely more informative, not just a
  tie-breaker. A second pass appends the cluster id to any label that still
  collides after the region suffix (two clusters sharing both a top concept
  *and* dominant region - happened for two small Kerala open-water clusters
  in the production run). Final production run: **0 duplicate labels among
  19 clusters** (started at 14 collisions before disambiguation).

New/changed pure functions (`geoseek/discovery/cluster.py`):
`cluster_embeddings`/`cluster_sample_and_assign` gained
`cluster_selection_method` and `pca_dim` (+ `cluster_vectors` escape hatch)
kwargs, all backward-compatible (existing callers/tests unaffected);
`disambiguate_cluster_labels` and `save_geographic_cluster_map` are new.
5 new tests in `tests/test_phase7b_followup.py` (270→275 total, 3 skipped,
unchanged pre-existing frozen baselines).
