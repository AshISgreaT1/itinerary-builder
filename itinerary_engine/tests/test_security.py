"""Who may call what.

The finding these tests exist for: every endpoint except ``/api/v1`` was open, so anyone
who could reach the server could list customers, read their names and preferences, and
overwrite them. These assertions are the thing that stops that coming back.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import security
from app.main import app

KEY = "test-suite-key"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("ITINERARY_CUSTOMERS", str(Path(tempfile.mkdtemp()) / "customers.json"))
    monkeypatch.setenv("ITINERARY_HTTPS_ONLY", "0")
    monkeypatch.setattr(security, "STORE_PATH", None, raising=False)
    from app import preferences

    monkeypatch.setattr(preferences, "STORE_PATH",
                        Path(tempfile.mkdtemp()) / "customers.json")
    return TestClient(app)


@pytest.fixture
def locked(client, monkeypatch):
    """A deployment with both a key and a UI password set."""
    monkeypatch.setenv(security.UI_PASSWORD_ENV, "planner-pass")
    return client


# --- customer data ------------------------------------------------------------------------
CUSTOMER_ROUTES = [
    ("get", "/customers", None),
    ("get", "/customers/anyone/preferences", None),
    ("put", "/customers/anyone/preferences", {"group_type": "family"}),
    ("get", "/data-quality", None),
]


@pytest.mark.parametrize("method, path, body", CUSTOMER_ROUTES)
def test_customer_data_needs_a_key(client, method, path, body):
    call = getattr(client, method)
    response = call(path, json=body) if body else call(path)
    assert response.status_code == 401, f"{path} answered {response.status_code} with no key"


@pytest.mark.parametrize("method, path, body", CUSTOMER_ROUTES)
def test_a_wrong_key_is_no_better_than_none(client, method, path, body):
    call = getattr(client, method)
    headers = {"X-API-Key": "not-the-key"}
    response = call(path, json=body, headers=headers) if body else call(path, headers=headers)
    assert response.status_code == 401


def test_the_right_key_gets_in(client):
    headers = {"X-API-Key": KEY}
    assert client.get("/customers", headers=headers).status_code == 200
    saved = client.put("/customers/c1/preferences", headers=headers,
                       json={"name": "R. Mehta", "group_type": "family"})
    assert saved.status_code == 200
    assert client.get("/customers/c1/preferences", headers=headers).json()["name"] == "R. Mehta"


def test_a_customer_record_is_never_cached_by_anything_in_between(client):
    response = client.get("/customers", headers={"X-API-Key": KEY})
    assert response.headers.get("cache-control") == "no-store"


def test_with_no_keys_configured_customer_data_is_refused(client, monkeypatch):
    """Fail closed: a setting nobody remembered must not mean "let everyone in"."""
    monkeypatch.delenv(security.API_KEYS_ENV, raising=False)
    monkeypatch.delenv(security.ALLOW_ANONYMOUS_ENV, raising=False)
    assert client.get("/customers").status_code == 503
    assert client.get("/customers", headers={"X-API-Key": KEY}).status_code == 503


# --- the planner ----------------------------------------------------------------------------------
PLANNER_ROUTES = [
    ("post", "/generate-itinerary", {"cities": ["Udaipur"], "duration_days": 3}),
    ("post", "/itinerary/suggestions", {"base": {"cities": ["Udaipur"], "duration_days": 3}}),
    ("post", "/itinerary/customize", {"base": {"cities": ["Udaipur"], "duration_days": 3},
                                      "operations": [{"operation": "set_pace", "pace": "active"}]}),
    ("get", "/cities", None),
]


@pytest.mark.parametrize("method, path, body", PLANNER_ROUTES)
def test_the_planner_is_open_when_no_password_is_set(client, method, path, body):
    """The one place this does not fail closed -- with no password there is no way in."""
    call = getattr(client, method)
    response = call(path, json=body) if body else call(path)
    assert response.status_code == 200


@pytest.mark.parametrize("method, path, body", PLANNER_ROUTES)
def test_the_planner_needs_a_session_once_a_password_is_set(locked, method, path, body):
    call = getattr(locked, method)
    response = call(path, json=body) if body else call(path)
    assert response.status_code == 401


def test_the_page_sends_you_to_sign_in(locked):
    response = locked.get("/", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/login"
    assert locked.get("/login").status_code == 200


def test_signing_in_and_out(locked):
    assert locked.post("/login", json={"password": "wrong"}).status_code == 401
    assert locked.post("/generate-itinerary",
                       json={"cities": ["Udaipur"], "duration_days": 3}).status_code == 401

    opened = locked.post("/login", json={"password": "planner-pass"})
    assert opened.status_code == 200
    assert security.SESSION_COOKIE in opened.cookies

    assert locked.get("/", follow_redirects=False).status_code == 200
    assert locked.post("/generate-itinerary",
                       json={"cities": ["Udaipur"], "duration_days": 3}).status_code == 200

    locked.post("/logout")
    assert locked.get("/", follow_redirects=False).status_code == 303


def test_a_key_also_opens_the_planner(locked):
    """So a script can drive it without pretending to be a browser."""
    assert locked.post("/generate-itinerary", headers={"X-API-Key": KEY},
                       json={"cities": ["Udaipur"], "duration_days": 3}).status_code == 200


def test_the_session_cookie_cannot_be_forged(locked):
    locked.cookies.set(security.SESSION_COOKIE, "9999999999.notarealsignature")
    assert locked.get("/", follow_redirects=False).status_code == 303


def test_an_expired_session_is_refused():
    assert security.session_is_valid(security.issue_session(hours=-1)) is False
    assert security.session_is_valid(security.issue_session(hours=1)) is True
    assert security.session_is_valid(None) is False
    assert security.session_is_valid("nonsense") is False


def test_a_session_from_a_different_secret_is_refused(monkeypatch):
    monkeypatch.setenv(security.SESSION_SECRET_ENV, "first-secret")
    token = security.issue_session()
    assert security.session_is_valid(token)
    monkeypatch.setenv(security.SESSION_SECRET_ENV, "second-secret")
    assert not security.session_is_valid(token)


# --- load ---------------------------------------------------------------------------------------------
def test_planning_is_rate_limited(client, monkeypatch):
    """Planning costs about a second of CPU, so an open endpoint is a cheap way to take
    the service down."""
    limiter = security.RateLimiter(per_minute=4)
    monkeypatch.setattr(security, "_planning_limiter", limiter)
    body = {"base": {"cities": ["Udaipur"], "duration_days": 3}}
    codes = [client.post("/itinerary/suggestions", json=body).status_code for _ in range(6)]
    assert codes[:4] == [200, 200, 200, 200]
    assert codes[4:] == [429, 429]

    refused = client.post("/itinerary/suggestions", json=body)
    assert refused.headers.get("retry-after") == "60"


def test_the_limiter_forgets_old_requests():
    limiter = security.RateLimiter(per_minute=2)
    assert limiter.check("a") and limiter.check("a") and not limiter.check("a")
    assert limiter.check("b"), "one client must not exhaust another's window"
    limiter._seen["a"].clear()
    assert limiter.check("a")


def test_a_limit_of_zero_means_no_limit():
    limiter = security.RateLimiter(per_minute=0)
    assert all(limiter.check("a") for _ in range(50))


# --- headers and visibility -----------------------------------------------------------------------------
def test_standard_headers_are_set(client):
    response = client.get("/health")
    assert response.headers.get("x-content-type-options") == "nosniff"
    assert response.headers.get("x-frame-options") == "SAMEORIGIN"
    assert response.headers.get("referrer-policy") == "same-origin"


def test_health_says_what_is_protected(client, monkeypatch):
    open_state = client.get("/health").json()
    assert open_state["ui_protected"] is False
    assert open_state["api_keys_configured"] == 1
    assert open_state["rate_limit_per_minute"] >= 0

    monkeypatch.setenv(security.UI_PASSWORD_ENV, "x")
    assert client.get("/health").json()["ui_protected"] is True


def test_health_itself_stays_open(locked):
    """A probe that needs a password is a probe that reports the service as down."""
    assert locked.get("/health").status_code == 200
