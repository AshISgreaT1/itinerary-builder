"""The public REST surface: envelope, authentication, validation, and what it promises.

Two things these tests care about beyond "does it work". First, that the API does not
quietly change the existing endpoints the UI depends on -- the envelope and the error
shape are scoped to `/api/v1` and nothing else. Second, that the fields this engine cannot
act on are reported as unused rather than silently ignored, because an integrator who
believes `budget` did something is a support ticket waiting to happen.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import plancache
from app.api_v1 import API_KEYS_ENV, ALLOW_ANONYMOUS_ENV, API_VERSION
from app.main import app

KEY = "test-key-123"
HEADERS = {"X-API-Key": KEY}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv(API_KEYS_ENV, f"{KEY},another-key")
    monkeypatch.delenv(ALLOW_ANONYMOUS_ENV, raising=False)
    return TestClient(app)


@pytest.fixture
def trip():
    return {"destination": "Jaipur, Udaipur", "days": 6, "travellers": 2,
            "budget": 60000, "currency": "INR", "traveller_type": "couple", "pace": "moderate"}


# --- health -------------------------------------------------------------------------------------
def test_health_is_open_and_keeps_its_existing_shape(client):
    """Probes must not need a key, and existing callers must not break."""
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    for existing in ["status", "dataset", "cities", "sights", "duplicate_names_merged"]:
        assert existing in body, f"{existing} disappeared from /health"
    assert body["status"] == "ok"
    assert body["api_version"] == API_VERSION
    assert body["api_keys_configured"] == 2
    assert body["uptime_seconds"] >= 0


# --- authentication ---------------------------------------------------------------------------------
def test_a_key_is_required(client, trip):
    for method, path, payload in [("get", "/api/v1/info", None),
                                  ("post", "/api/v1/itinerary", trip)]:
        response = getattr(client, method)(path, json=payload) if payload else getattr(client, method)(path)
        assert response.status_code == 401, path
        body = response.json()
        assert body["success"] is False
        assert body["error"]["code"] == "unauthorized"


def test_a_wrong_key_is_rejected(client):
    response = client.get("/api/v1/info", headers={"X-API-Key": "not-a-key"})
    assert response.status_code == 401
    assert response.json()["error"]["message"] == "Invalid API key."


def test_with_no_keys_configured_the_api_refuses_rather_than_opens(monkeypatch):
    """Fail closed: a missing setting must never mean "let everyone in"."""
    monkeypatch.delenv(API_KEYS_ENV, raising=False)
    monkeypatch.delenv(ALLOW_ANONYMOUS_ENV, raising=False)
    response = TestClient(app).get("/api/v1/info", headers=HEADERS)
    assert response.status_code == 503
    assert API_KEYS_ENV in response.json()["error"]["message"]


def test_anonymous_can_be_switched_on_for_local_development(monkeypatch):
    monkeypatch.delenv(API_KEYS_ENV, raising=False)
    monkeypatch.setenv(ALLOW_ANONYMOUS_ENV, "1")
    assert TestClient(app).get("/api/v1/info").status_code == 200


# --- info ---------------------------------------------------------------------------------------------
def test_info_describes_what_the_engine_does_and_does_not_do(client):
    body = client.get("/api/v1/info", headers=HEADERS).json()
    assert body["success"] is True
    data = body["data"]
    assert data["api_version"] == API_VERSION
    assert data["dataset"]["sights"] > 0 and data["dataset"]["cities_plannable"] > 0
    assert "pricing and budgets" in " ".join(data["capabilities"]["does_not_model"])
    assert data["limits"]["max_days"] == 30
    assert set(data["endpoints"]) >= {"health", "info", "itinerary", "docs"}


# --- planning -------------------------------------------------------------------------------------------
def test_a_trip_comes_back_in_the_envelope(client, trip):
    response = client.post("/api/v1/itinerary", headers=HEADERS, json=trip)
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["meta"]["api_version"] == API_VERSION

    data = body["data"]
    assert data["destination"] == ["Jaipur", "Udaipur"]
    assert data["days"] == 6 and len(data["itinerary"]) == 6
    assert data["total_sights"] > 0
    assert data["feasible"] is True
    for day in data["itinerary"]:
        assert day["city"] and day["day"] >= 1
        for item in day["items"]:
            assert item["start"] and item["end"] and item["title"]
            assert isinstance(item["description"], str)


@pytest.mark.parametrize("destination, expected", [
    ("Jaipur", ["Jaipur"]),
    ("Jaipur, Udaipur", ["Jaipur", "Udaipur"]),
    (["Jaipur", "Udaipur"], ["Jaipur", "Udaipur"]),
    ("  Jaipur ,  Udaipur  ", ["Jaipur", "Udaipur"]),
])
def test_destination_accepts_the_shapes_callers_actually_send(client, destination, expected):
    response = client.post("/api/v1/itinerary", headers=HEADERS,
                           json={"destination": destination, "days": 4})
    assert response.status_code == 200, response.json()
    assert response.json()["data"]["destination"] == expected


def test_the_day_schedule_is_carried_through(client):
    """The point of the API is the plan, so the clock has to survive the projection."""
    body = client.post("/api/v1/itinerary", headers=HEADERS,
                       json={"destination": "Udaipur", "days": 3}).json()
    days = body["data"]["itinerary"]
    assert any(day["starts"] and day["ends"] for day in days)
    sights = [item for day in days for item in day["items"] if item["kind"] == "sight"]
    assert sights, "a plan with no sightseeing is not a plan"
    assert any(item["must_visit"] for item in sights)


def test_a_transfer_day_reports_its_journey(client):
    body = client.post("/api/v1/itinerary", headers=HEADERS,
                       json={"destination": "Jaipur, Udaipur", "days": 6}).json()
    transfers = [day["transfer"] for day in body["data"]["itinerary"] if day["transfer"]]
    assert transfers, "a two-city trip has a transfer"
    for transfer in transfers:
        assert transfer["from_city"] and transfer["to_city"]
        assert transfer["drive_hours"] is None or transfer["drive_hours"] > 0


# --- the fields the engine cannot act on ------------------------------------------------------------------
def test_unused_inputs_are_reported_not_silently_ignored(client, trip):
    meta = client.post("/api/v1/itinerary", headers=HEADERS, json=trip).json()["meta"]
    assert set(meta["unused_inputs"]) == {"budget", "travellers", "traveller_type"}
    assert "no pricing" in meta["unused_inputs"]["budget"]
    assert "p=0.87" in meta["unused_inputs"]["traveller_type"]
    assert meta["echo"] == {"travellers": 2, "budget": 60000.0,
                            "currency": "INR", "traveller_type": "couple"}


def test_nothing_is_reported_as_unused_when_nothing_unusable_was_sent(client):
    meta = client.post("/api/v1/itinerary", headers=HEADERS,
                       json={"destination": "Udaipur", "days": 3}).json()["meta"]
    assert meta["unused_inputs"] == {}


def test_traveller_type_really_does_not_change_the_plan(client):
    """If this ever starts failing, the docs are lying and must change with the code."""
    plans = []
    for kind in ["solo", "couple", "family", "group"]:
        body = client.post("/api/v1/itinerary", headers=HEADERS,
                           json={"destination": "Udaipur", "days": 4, "traveller_type": kind}).json()
        plans.append([[item["title"] for item in day["items"]] for day in body["data"]["itinerary"]])
    assert all(plan == plans[0] for plan in plans)


def test_pace_does_change_the_plan(client):
    counts = {}
    for pace in ["relaxed", "moderate", "active"]:
        body = client.post("/api/v1/itinerary", headers=HEADERS,
                           json={"destination": "Jaipur", "days": 4, "pace": pace}).json()
        counts[pace] = body["data"]["total_sights"]
    assert counts["active"] > counts["relaxed"], counts


# --- validation and errors ----------------------------------------------------------------------------------
def test_a_bad_body_comes_back_as_field_level_problems(client):
    response = client.post("/api/v1/itinerary", headers=HEADERS, json={"destination": "", "days": 0})
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    fields = {item["field"] for item in error["details"]}
    assert {"destination", "days"} <= fields, error["details"]


@pytest.mark.parametrize("payload, field", [
    ({"days": 3}, "destination"),
    ({"destination": "Jaipur"}, "days"),
    ({"destination": "Jaipur", "days": 99}, "days"),
    ({"destination": "Jaipur", "days": 3, "pace": "frantic"}, "pace"),
    ({"destination": "Jaipur", "days": 3, "travellers": 0}, "travellers"),
    ({"destination": "Jaipur", "days": 3, "start_month": 13}, "start_month"),
    ({"destination": ["A"] * 11, "days": 3}, "destination"),
])
def test_each_input_is_validated(client, payload, field):
    response = client.post("/api/v1/itinerary", headers=HEADERS, json=payload)
    assert response.status_code == 422
    assert any(item["field"] == field for item in response.json()["error"]["details"]), \
        response.json()["error"]["details"]


def test_an_unknown_city_says_which_ones_are_known(client):
    response = client.post("/api/v1/itinerary", headers=HEADERS,
                           json={"destination": "Atlantis", "days": 3})
    assert response.status_code == 422
    message = response.json()["error"]["message"]
    assert "Atlantis" in message and "Known cities" in message


def test_an_impossible_trip_is_refused_with_the_reason(client):
    response = client.post("/api/v1/itinerary", headers=HEADERS,
                           json={"destination": ["Jaipur", "Udaipur", "Goa"], "days": 2})
    assert response.status_code == 422
    assert response.json()["success"] is False
    assert "day" in response.json()["error"]["message"].lower()


# --- the existing API is untouched ------------------------------------------------------------------------------
def test_the_ui_endpoints_keep_their_own_shapes(client):
    """The envelope and its error format must not leak outside /api/v1."""
    plan = client.post("/generate-itinerary", json={"cities": ["Udaipur"], "duration_days": 3})
    assert plan.status_code == 200
    body = plan.json()
    assert "success" not in body and "days" in body and "validation" in body

    bad = client.post("/generate-itinerary", json={"cities": [], "duration_days": 3})
    assert bad.status_code == 422
    assert "detail" in bad.json() and "error" not in bad.json()

    unknown = client.post("/generate-itinerary", json={"cities": ["Atlantis"], "duration_days": 3})
    assert unknown.status_code == 422
    assert isinstance(unknown.json()["detail"], str)


def test_the_ui_still_loads(client):
    assert client.get("/").status_code == 200


# --- docs ------------------------------------------------------------------------------------------------------
def test_the_docs_page_and_schema_describe_the_api(client):
    assert client.get("/docs").status_code == 200
    schema = client.get("/openapi.json").json()
    assert "/api/v1/itinerary" in schema["paths"]
    assert "/api/v1/info" in schema["paths"]
    schemes = schema.get("components", {}).get("securitySchemes", {})
    assert any(scheme.get("name") == "X-API-Key" for scheme in schemes.values()), schemes
    example = schema["components"]["schemas"]["ItineraryRequestV1"]["example"]
    assert example["destination"] and example["days"]


# --- caching -------------------------------------------------------------------------------------------------------
def test_an_identical_request_is_served_from_memory(client):
    plancache.clear()
    body = {"destination": "Udaipur", "days": 3, "pace": "relaxed"}
    first = client.post("/api/v1/itinerary", headers=HEADERS, json=body).json()
    second = client.post("/api/v1/itinerary", headers=HEADERS, json=body).json()
    assert first["meta"]["cached"] is False
    assert second["meta"]["cached"] is True
    assert first["data"] == second["data"], "a cached plan must be the same plan"


def test_the_cache_keys_on_everything_that_changes_a_plan(client):
    plancache.clear()
    base = {"destination": "Udaipur", "days": 3}
    client.post("/api/v1/itinerary", headers=HEADERS, json=base)
    for change in [{"days": 4}, {"pace": "active"}, {"start_month": 1}, {"destination": "Jaipur"}]:
        response = client.post("/api/v1/itinerary", headers=HEADERS, json={**base, **change})
        assert response.json()["meta"]["cached"] is False, change
