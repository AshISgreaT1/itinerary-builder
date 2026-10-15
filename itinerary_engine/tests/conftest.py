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
