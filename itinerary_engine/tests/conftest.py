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
