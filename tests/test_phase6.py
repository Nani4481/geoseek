"""Phase 6 - analyst interface: audit trail seam + backend API.

Step B (audit trail, PS 2.2.5): the append-only ``analyst_decisions`` table
reached ONLY through the MetadataRepository seam - never raw sqlite3 in
application code. Tests here prove: the seam round-trips a decision, the log is
ordered and append-only (a second decision on the same candidate adds a row,
never overwrites), and the storage layer itself rejects UPDATE / DELETE.

Step A/C/D (endpoints): a TestClient drives the FastAPI app against whatever
production catalog + change report exist, skipping if they haven't been built.
"""

from __future__ import annotations

import sqlite3

import pytest

from geoseek.catalog.entities import AnalystDecision
from geoseek.catalog.repository import MetadataRepository
from geoseek.catalog.sqlite_repository import CatalogError, SQLiteMetadataRepository


# --------------------------------------------------------------------------
# Step B - audit trail through the repository seam
# --------------------------------------------------------------------------


@pytest.fixture()
def repo(tmp_path):
    r = SQLiteMetadataRepository(tmp_path / "audit.sqlite")
    yield r
    r.close()


def test_seam_still_complete_after_audit_methods_added():
    assert issubclass(SQLiteMetadataRepository, MetadataRepository)
    assert not getattr(SQLiteMetadataRepository, "__abstractmethods__", set())


def test_record_and_read_back_a_decision(repo):
    out = repo.record_analyst_decision(AnalystDecision(
        decision_id="", candidate_id="2019_2024_011912", decision="confirm",
        analyst_note="clear reservoir fill, matches SAR VV drop", analyst="tester",
        model_version="FCSiamDiff@0.80", weights_sha256="452ac0", git_commit="ecf0943",
        pipeline_version="0.1.0", confidence_at_decision=0.9873,
        evidence_snapshot={"change_type": "water_gain", "persistence": "persistent"},
    ))
    assert out.decision_id.startswith("dec_")
    assert out.created_at  # filled in on write

    back = repo.get_analyst_decision(out.decision_id)
    assert back == out
    assert back.evidence_snapshot["change_type"] == "water_gain"
    assert back.confidence_at_decision == pytest.approx(0.9873)


def test_decision_value_is_validated(repo):
    with pytest.raises(CatalogError):
        repo.record_analyst_decision(AnalystDecision(
            decision_id="", candidate_id="c1", decision="maybe"))


def test_log_is_append_only_second_decision_adds_a_row(repo):
    repo.record_analyst_decision(AnalystDecision(
        decision_id="", candidate_id="c1", decision="reject", analyst_note="cloud edge"))
    repo.record_analyst_decision(AnalystDecision(
        decision_id="", candidate_id="c1", decision="confirm", analyst_note="re-checked, real"))

    hist = repo.list_analyst_decisions(candidate_id="c1")
    assert [d.decision for d in hist] == ["reject", "confirm"]        # both kept, in write order
    assert hist[0].decision_id != hist[1].decision_id

    latest = repo.latest_decision_by_candidate()
    assert latest["c1"].decision == "confirm"                         # current verdict = most recent


def test_storage_layer_rejects_update_and_delete(repo):
    d = repo.record_analyst_decision(AnalystDecision(
        decision_id="", candidate_id="c9", decision="confirm"))
    conn = repo.connection  # the migration escape hatch - used here only to prove the trigger fires
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE analyst_decisions SET decision='reject' WHERE decision_id=?",
                     (d.decision_id,))
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("DELETE FROM analyst_decisions WHERE decision_id=?", (d.decision_id,))
    # row is untouched
    assert repo.get_analyst_decision(d.decision_id).decision == "confirm"


def test_global_audit_listing_is_time_ordered(repo):
    for cid in ("a", "b", "c"):
        repo.record_analyst_decision(AnalystDecision(
            decision_id="", candidate_id=cid, decision="confirm"))
    allrows = repo.list_analyst_decisions()
    assert [d.candidate_id for d in allrows] == ["a", "b", "c"]
    assert [d.candidate_id for d in repo.list_analyst_decisions(limit=2)] == ["a", "b"]
