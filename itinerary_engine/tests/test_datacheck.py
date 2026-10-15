"""Dataset quality report."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.datacheck import misfiled_sights, run_checks, unreachable_sights
from app.main import app


def test_misfiled_sight_is_found_with_a_suggested_home(dataset):
    findings = {finding.key: finding for finding in misfiled_sights(dataset)}
    pangong = findings.get("Leh / Pangong Lake")
    assert pangong is not None, "the lake filed under Leh sits 141 km away"
    assert pangong.severity == "error"
    assert "Pangong Lake" in pangong.suggestion


def test_sight_that_cannot_fit_any_day_is_reported(dataset):
    keys = {finding.key for finding in unreachable_sights(dataset)}
    assert "Leh / Pangong Lake" in keys


def test_every_finding_carries_evidence_and_a_fix(dataset):
    findings = run_checks(dataset)
    assert findings
    assert all(finding.detail and finding.suggestion for finding in findings)
    assert all(finding.severity in {"error", "warning"} for finding in findings)


def test_route_gaps_only_cover_pairs_that_are_sold_together(dataset):
    """Ajmer and Alleppey are never sold together, so their missing row is not a finding."""
    keys = {finding.key for finding in run_checks(dataset) if finding.kind == "no_route_data"}
    assert not any("Alleppey" in key and "Ajmer" in key for key in keys)
    assert len(keys) < 40, "the report must stay actionable, not list every city pair"


def test_data_quality_endpoint():
    response = TestClient(app).get("/data-quality")
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == len(body["findings"]) and body["by_kind"]
