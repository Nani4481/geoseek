GeoSeek Frontend Rebuild - Build Log
Started: 2026-09-28

[backend] reopen decision value + gate summary exposure: DONE - 415 passed, 3 skipped
[frontend] all 7 screens built: shell/api-client -> Candidate Detail -> Queue -> Overview(A) -> Search -> Object Detection -> Discovery -> Watch Areas
[tests] full suite: 418 passed, 3 skipped; offline scan: 40/40 passed

Per-screen live-backend verification (Playwright, production catalog - 841 candidates, 105,245 vectors):
  shell + api-client   PASS - top bar real regions/counters, rail nav, keyboard nav wired
  Candidate Detail     PASS - real before/after+overlay imagery, real 5-gate trace, confirm/reject/reopen round-tripped against the live DB
  Review Queue         PASS - real filters/sort/counters, gate squares from new /candidates summary field
  Overview (direction A) PASS - real markers/imagery/counters (fixed: active-class bug, imagery date-format bug, readout-strip clipping bug)
  Search               PASS - real RemoteCLIP results, derived reason text, no raw score
  Object Detection     PASS - real Maxar detections + oriented boxes, real per-class metrics (fixed: response-unwrapping bug)
  Discovery            PASS - real 3-tier grouping derived from region+distance
  Watch Areas          PASS - real watch area + notification history (fixed: response-unwrapping bug)

Incident: overwrote src/geoseek/analyst/web/index.html with Write before checking git status closely;
its prior uncommitted redesign content (never staged) is not recoverable. All other uncommitted
frontend work (app.js, style.css, globe.js, tokens.*, components.*, mosaic/, vendor/) was intact and
was committed as a checkpoint (7845ba8) before being retired in the rebuild commit (2b95175).
