"""Phase 6 Step B - print the analyst_decisions schema and demonstrate an
append-only write + read-back through the MetadataRepository seam.

    python scripts/demo_audit_trail.py

Uses a throwaway temp DB - it does not touch the production catalog.
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

from geoseek.catalog.entities import AnalystDecision
from geoseek.catalog.schema import SCHEMA_SQL
from geoseek.catalog.sqlite_repository import SQLiteMetadataRepository


def _print_schema() -> None:
    print("=" * 78)
    print("analyst_decisions - DDL (verbatim from geoseek.catalog.schema.SCHEMA_SQL)")
    print("=" * 78)
    marker = "-- PS 2.2.5 audit trail."
    tail = SCHEMA_SQL[SCHEMA_SQL.index(marker):]
    for ln in tail.splitlines():
        s = ln.strip()
        keep = (
            s.startswith("--")
            or "analyst_decisions" in ln
            or (ln.startswith("    ") and s)          # column / trigger body lines
            or s in ("BEGIN", "END;", ");")
        )
        if keep:
            print(ln.rstrip())
    print("\n(columns: decision_id, candidate_id, decision[confirm|reject], analyst_note, analyst,")
    print(" created_at, model_version, weights_sha256, git_commit, pipeline_version,")
    print(" confidence_at_decision, evidence_snapshot_json)")


def main() -> None:
    _print_schema()

    tmp = Path(tempfile.mkdtemp(prefix="geoseek_audit_demo_")) / "audit.sqlite"
    repo = SQLiteMetadataRepository(tmp)

    print("\n" + "=" * 78)
    print(f"demo DB: {tmp}")
    print("=" * 78)

    ev = {
        "change_type": "water_gain",
        "confidence_breakdown": [
            "model: mean p 0.93 over footprint (rescaled 0.92)",
            "persistence: persistent (confidence 0.95)",
            "spectral: index deltas strongly support 'water_gain' (1.00)",
            "SAR corroboration (VV anomaly -6.2 dB vs expected drop): x1.10",
        ],
        "earliest_supported": ["2019-03-30", "2021-03-04"],
        "provenance": {"observation_ids": ["S2B_44RPQ_20190330_1_L2A_scaled",
                                           "S2A_44RPQ_20240308_0_L2A_scaled"]},
    }

    print("\n--- WRITE #1: confirm 2019_2024_011912 ---")
    d1 = repo.record_analyst_decision(AnalystDecision(
        decision_id="", candidate_id="2019_2024_011912", decision="confirm",
        analyst_note="Reservoir fill is unambiguous; SAR VV drop corroborates.",
        analyst="demo-analyst", model_version="FCSiamDiff@0.80",
        weights_sha256="452ac062e3b3d56f7997b9a5bbf81bab41d148b525fc28621e964cbbd0a3bcab",
        git_commit="ecf0943", pipeline_version="0.1.0",
        confidence_at_decision=0.9873, evidence_snapshot=ev))
    print(f"    stored decision_id={d1.decision_id}  created_at={d1.created_at}")

    print("\n--- WRITE #2: reject 2019_2024_008471 ---")
    d2 = repo.record_analyst_decision(AnalystDecision(
        decision_id="", candidate_id="2019_2024_008471", decision="reject",
        analyst_note="Looks like a phenology edge, not structural. Down-weight radiometric.",
        analyst="demo-analyst", model_version="FCSiamDiff@0.80",
        weights_sha256="452ac062e3b3d56f7997b9a5bbf81bab41d148b525fc28621e964cbbd0a3bcab",
        git_commit="ecf0943", pipeline_version="0.1.0",
        confidence_at_decision=0.71, evidence_snapshot={"change_type": "water_gain"}))
    print(f"    stored decision_id={d2.decision_id}  created_at={d2.created_at}")

    print("\n--- WRITE #3: re-decide 2019_2024_008471 -> confirm (append, not overwrite) ---")
    d3 = repo.record_analyst_decision(AnalystDecision(
        decision_id="", candidate_id="2019_2024_008471", decision="confirm",
        analyst_note="Re-checked against the 2021 frame - it IS a real small tank. Reversing my reject.",
        analyst="demo-analyst", model_version="FCSiamDiff@0.80", git_commit="ecf0943",
        pipeline_version="0.1.0", confidence_at_decision=0.71))
    print(f"    stored decision_id={d3.decision_id}")

    print("\n--- READ BACK: full audit log (write order) ---")
    for d in repo.list_analyst_decisions():
        print(f"    {d.created_at}  {d.decision_id}  {d.candidate_id:>18}  {d.decision:>7}  "
              f"conf@{d.confidence_at_decision}  note={d.analyst_note[:52]!r}")

    print("\n--- READ BACK: history for 2019_2024_008471 (append-only proof) ---")
    hist = repo.list_analyst_decisions(candidate_id="2019_2024_008471")
    for i, d in enumerate(hist, 1):
        print(f"    v{i}: {d.decision:>7}  {d.decision_id}  {d.analyst_note[:60]!r}")
    print(f"    => {len(hist)} rows kept; current verdict = "
          f"{repo.latest_decision_by_candidate()['2019_2024_008471'].decision!r} (most recent)")

    print("\n--- APPEND-ONLY ENFORCED AT THE STORAGE LAYER ---")
    conn: sqlite3.Connection = repo.connection
    for sql in ("UPDATE analyst_decisions SET decision='reject'",
                "DELETE FROM analyst_decisions"):
        try:
            conn.execute(sql)
            print(f"    !! {sql!r} SUCCEEDED - append-only NOT enforced")
        except sqlite3.IntegrityError as e:
            print(f"    OK  {sql[:34]!r:<38} rejected: {e}")

    n = conn.execute("SELECT COUNT(*) FROM analyst_decisions").fetchone()[0]
    print(f"\n    row count unchanged after the blocked UPDATE/DELETE: {n}")
    repo.close()


if __name__ == "__main__":
    main()
