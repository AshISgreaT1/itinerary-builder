"""Dataset quality report."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.datacheck import misfiled_sights, run_checks, unreachable_sights
from app.main import app


def test_misfiled_sight_is_found_with_a_suggested_home(dataset):
    findings = {finding.key: finding for finding in misfiled_sights(dataset)}
    khardung = findings.get("Nubra Valley / khardung-la-pass")
    assert khardung is not None, "the pass filed under Nubra Valley sits 100+ km away"
    assert khardung.severity == "error"
    assert "re-home to" in khardung.suggestion

    # The lake filed under Leh used to be this test's specimen. data/corrections.csv drops
    # that row as a duplicate of the Pangong Lake one, so its absence is now the assertion.
    assert "Leh / Pangong Lake" not in findings


def test_sight_that_cannot_fit_any_day_is_reported(dataset):
    keys = {finding.key for finding in unreachable_sights(dataset)}
    assert "Leh / Tanglang La Pass" in keys, "59 km from the nearest city, no pace reaches it"
    assert "Leh / Pangong Lake" not in keys, "the reviewed corrections removed that row"


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


def test_data_quality_endpoint(api_headers):
    response = TestClient(app).get("/data-quality", headers=api_headers)
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == len(body["findings"]) and body["by_kind"]
