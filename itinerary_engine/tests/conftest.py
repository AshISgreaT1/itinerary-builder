"""Shared test fixtures: the sample dataset, loaded once."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "sample_dataset_v4_clean.xlsx"
sys.path.insert(0, str(ROOT))
os.environ.setdefault("ITINERARY_DATASET", str(DATASET))


@pytest.fixture(scope="session")
def dataset():
    from app.data_loader import load_dataset

    return load_dataset(DATASET)


@pytest.fixture(autouse=True)
def _isolate_plan_cache():
    """Give every test a cold cache.

    Plans are cached process-wide, which is right in production and wrong in a test suite:
    a test that patches the planner and rebuilds the same request would otherwise be handed
    the plan built *before* the patch, and quietly prove nothing.
    """
    from app import plancache

    plancache.clear()
    yield
    plancache.clear()


TEST_API_KEY = "test-suite-key"


@pytest.fixture(autouse=True)
def _security_defaults(monkeypatch):
    """Give every test a known key and a fresh rate-limit window.

    The limiter is per process and keyed by client, so without this one test exhausts the
    window for the next and the failures land somewhere unrelated. Tests that care about
    the unconfigured case override these afterwards, which works because an autouse fixture
    runs first.
    """
    from app import security

    monkeypatch.setenv(security.API_KEYS_ENV, TEST_API_KEY)
    monkeypatch.delenv(security.UI_PASSWORD_ENV, raising=False)
    security.reset_rate_limit()
    yield
    security.reset_rate_limit()


@pytest.fixture
def api_headers():
    """Headers for an endpoint that holds customer data or dataset internals."""
    return {"X-API-Key": TEST_API_KEY}
