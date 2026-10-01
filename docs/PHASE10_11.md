# Phase 10 / 11 — judge coverage, constrained retrieval, spectral ranking

Built on the Phase 9 baseline (`docs/PHASE9.md`, `data/eval/baseline_v1.json`). Sections are added block by block.
Every evaluation here states which corpus it ran on (PHASE9 §9): **frozen** = production `faiss_id < 101911`
(101,911 vectors), **live** = everything now in the catalog (105,245 vectors).

---

## Block 0 — remediation

Pre-flight: `git status` clean after committing the earlier cloud-deployment work and Phase 9 (commits `d03b764`,
`0c37262`); `pytest -q` was `4 failed, 495 passed, 3 skipped`. After Block 0: **`521 passed, 3 skipped, 0 failed`**.

### 0.1 The four `test_frontend_offline` failures — what was actually in the vendored files

**No live runtime fetch exists in any vendored library.** Network sinks were traced, not grepped: no non-vendored
file contains an `http(s)` string; Leaflet is built with `attributionControl: false` and used only through
`L.imageOverlay(<same-origin API URL>)` (no `L.tileLayer`, so no tile-server request); every three.js texture is loaded
via `new URL(name, "../vendor/earth/")`; the generic loaders inside the libraries (`fetch(`, `.src =`) only receive URLs
the app hands them.

12 flagged occurrences (8 distinct URLs), plus 2 relative `sourceMappingURL` comments and one regex-source fragment:

| file | URL string | class |
|---|---|---|
| `chart.umd.js` | `https://www.chartjs.org`, `https://github.com/kurkle/color#readme` | attribution comment |
| `chart.umd.js` | `sourceMappingURL=chart.umd.js.map` | sourcemap comment (relative; `.map` not shipped) |
| `leaflet.css` | `bugs.chromium.org/…id=600120`, `bugzilla.mozilla.org/…id=888319` | other — browser-bug citations in CSS comments |
| `leaflet.js` | `https://leafletjs.com` (header) | attribution comment |
| `leaflet.js` | `https://leafletjs.com` (string) | other — `<a href>` in the attribution-control HTML, never rendered here |
| `leaflet.js` | `http://www.w3.org/2000/svg` ×3 | other — XML namespace identifier |
| `leaflet.js` | `sourceMappingURL=leaflet.js.map` | sourcemap comment |
| `three.module.min.js` | `http://www.w3.org/1999/xhtml` | other — XML namespace identifier |
| `three.module.min.js` | `https://discourse.threejs.org/…/53733` ×2 | other — text in a `console.warn` |
| `three.module.min.js` | `https?://` regex source ×2 | other — URL-detection regex text |

(The count was stated as "eleven" in the request; the guard flags 12 occurrences of 8 distinct URLs.)

**Root cause of three of the four failures was line endings, not content.** With `core.autocrlf=true` the working
tree held CRLF versions of the vendored JavaScript: `three.module.min.js` hashed `8acd07f8…` on disk against `3e690ac7…`
for the committed (upstream r160) bytes, which is exactly what the manifest had recorded. A hash allowlist recorded from
such a tree breaks on every LF checkout (Linux CI, a fresh clone, a Docker build). `OrbitControls.js` had the same
drift (`61d15e0b…` on disk, `5a44a9e8…` committed) and passed only because it contains no URLs, so its hash was never
checked.

### 0.2 Fixes

* **`.gitattributes`:** `src/geoseek/analyst/web/vendor/** -text`. The four affected files were restored from their
  blobs (conversion verified lossless first: disk with CRLF→LF == blob). Hashes:

  | file | before (working tree) | after = blob |
  |---|---|---|
  | `chart.umd.js` | `2e3da592…` | `fed6a739…` |
  | `leaflet/leaflet.js` | `3104b526…` | `db49d009…` |
  | `leaflet/leaflet.css` | `a7837102…` | `a7837102…` (natively CRLF; unchanged) |
  | `three.module.min.js` | `8acd07f8…` | `3e690ac7…` (= the manifest's existing pin, unchanged) |
  | `OrbitControls.js` | `61d15e0b…` | `5a44a9e8…` (the committed blob, = the manifest's existing pin; used) |

* **Provenance allowlist.** The manifest (`data/provenance_manifest.json`) is **git-ignored** and had no entries at all for
  `chart.umd.js` or any Leaflet file (and `stage_leaflet.py` named `leaflet.js` and `leaflet.css` identically, so a restage
  would have let one overwrite the other — fixed). The classified URL list is now committed in
  `src/geoseek/staging/vendor_provenance.py` and written into the manifest offline by
  `python -m geoseek.staging.vendor_provenance`; entries were added keyed to the blob hashes listed above, each URL with
  its classification as the note. `OrbitControls.js` and `three.module.min.js` needed no hash change.
* **Earth textures** (found while auditing "every vendored file"): the manifest's recorded hashes for `day/night/specular.jpg`
  matched nothing in git and `clouds.png` had no entry. The textures are PIL re-encodes (not upstream bytes); they were
  re-pinned to the **committed** bytes. Stated plainly: that pins our derivative, and the earlier recorded hashes could
  not be tied to any file.
* **`tests/test_frontend_offline.py`:** new `test_every_vendored_file_is_hash_pinned_in_the_manifest` runs for every file
  under `vendor/` (12 files), URL-bearing or not. Mutation-checked: a wrong pin for `OrbitControls.js` (which the old check
  never examined) and for `clouds.png` both fail it. `tests/test_vendor_provenance.py` (6 tests) pins the allowlist module:
  no entry is a live fetch, the allowlist covers every URL the guard finds, update is idempotent, a drift is detected and
  repaired, Leaflet entry names cannot collide, and `.gitattributes` keeps the `-text` rule.
* **`scripts/rebuild_index.py` is non-destructive.** Without `--force` it builds a **new** index in a new directory
  (`data/index/rebuild_<UTC timestamp>/` or `--out-dir`, which must not exist), leaves production alone and does not
  rewrite the provenance manifest. `--force` is the only way to delete the production index, cannot be combined with
  `--out-dir`, and the docstring says what it destroys. 4 tests (`tests/test_rebuild_index.py`).

### 0.3 Provenance string for the vanilla control

`retrieval_evaluation.systems.vanilla_clip` read "OpenCLIP ViT-B/32 laion2b". The staged file
`…/openclip_vanilla_cache/models--timm--vit_base_patch32_clip_224.openai/…/open_clip_model.safetensors` hashes to
`e6d1bd7789aa45192b3bf90570a789b478bae1b74ebcce7eddd908e83a2b7c31`, which is also what the manifest's own artifact record
says (`pretrained='openai'`). Corrected in the manifest and in the generating script (`scripts/eval_retrieval_score.py`)
so a re-run cannot reintroduce it. No "laion" string remains in the manifest.

### 0.4 – 0.7 Documentation

* **0.4** the four older redesign reports no longer name the browser-automation extension (4 one-line edits). The
  repo-wide attribution scan over `src/`, `docs/`, `scripts/`, `tests/` and root `*.md` (274 files) has 0 hits.
* **0.5** `catalog/embedding_map.py` added to the documented `sqlite3` seam list in `EVALUATION_REPORT.md` §13 and in
  `ARCHITECTURE.md` (which repeats the same list). The grep returns exactly `embedding_map.py`, `migrate.py`,
  `sqlite_repository.py` and `vectorindex/faiss_flat.py`.
* **0.6** `docs/PHASE9.md` §9: the frozen evaluation corpus is `faiss_id < 101911`, why, and what the live corpus now holds
  (105,245 vectors = 104,089 Sentinel-2 + 1,156 Maxar; 108,324 catalog tiles; 81 scenes).
* **0.7** The exact phrase "0 external URLs at query time" is **not present in any document in this repository** (it is in
  the external brief), so there was nothing to rewrite in place; the repo's own statements are already request-framed
  ("no external API at serve time", "zero outbound requests"). `EVALUATION_REPORT.md` gains §16, which states the claim as
  **zero external network requests at runtime**, lists the URL strings above with their classification, records the
  sink-tracing, and describes the CRLF trap. Note the end-to-end runtime check behind the claim is the network-disabled UI
  run in `RUN.md`; it is manual, and the static trace plus the guard test are what is automated.

### 0.8 Open observation (not changed)

`docs/EVALUATION_REPORT.md` §1 still carries the Phase 7b counts (101,911 / 104,990 / 75). They are accurate for that
moment and now explained in PHASE9 §9; the report body was not rewritten.
