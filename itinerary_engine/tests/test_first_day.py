"""Day 1 carries the city's main sights -- the way the agent itineraries do.

The rule is measured, not chosen: in the workbook's 1 241 agent-written days, 85.3% of
day-1 stops are must-visits against roughly 47% on later days, and day 1 stays close to
the centre (median 4.0 km). These tests pin both the behaviour and the evidence, so if the
dataset changes the claim is re-checked rather than inherited.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.builder import PlanRequest, build_itinerary
from app.config import ARRIVAL_DAY_MAX_KM

CASES = [
    ("Udaipur", 4, "moderate"),
    ("Jaipur", 4, "active"),
    ("Jodhpur", 3, "moderate"),
    ("Goa", 5, "moderate"),
    ("Jaisalmer", 4, "moderate"),
    ("Cochin", 4, "relaxed"),
    ("Munnar", 4, "moderate"),
    ("Leh", 4, "moderate"),
]


@pytest.fixture(scope="module")
def plans(dataset):
    return [build_itinerary(PlanRequest(cities=[city], duration_days=days, pace=pace), dataset)
            for city, days, pace in CASES]


def test_day_one_leans_on_must_visits(plans):
    """Across the sample, day 1 is mostly headline sights -- not the leftovers."""
    first = [sight for plan in plans for sight in plan["days"][0]["sights"]]
    later = [sight for plan in plans for day in plan["days"][1:] for sight in day["sights"]]
    assert first, "day 1 should hold sights"
    share_first = sum(1 for sight in first if sight.get("must_visit")) / len(first)
    share_later = sum(1 for sight in later if sight.get("must_visit")) / len(later)
    assert share_first > 0.6, f"only {share_first:.0%} of day-1 stops are must-visits"
    assert share_first > share_later, "day 1 should be richer in must-visits than later days"


def test_day_one_still_stays_in_town(plans):
    """Front-loading the main sights must not drag the arrival day out of the city."""
    for plan in plans:
        for sight in plan["days"][0]["sights"]:
            assert sight["distance_from_centre_km"] <= ARRIVAL_DAY_MAX_KM, \
                f"{sight['sight']} is {sight['distance_from_centre_km']} km out on the arrival day"


def test_the_headline_sight_of_a_city_is_seen_on_day_one(dataset):
    """Udaipur's City Palace and Lake Pichola are the reason people go: day 1 holds them."""
    plan = build_itinerary(PlanRequest(cities=["Udaipur"], duration_days=4, pace="moderate"), dataset)
    first_day = {sight["sight"] for sight in plan["days"][0]["sights"]}
    assert "City Palace" in first_day, f"day 1 holds {sorted(first_day)}"


def test_every_plan_still_validates(plans):
    for plan in plans:
        assert plan["validation"]["passed"], plan["validation"]["issues"]


def test_the_rule_matches_what_the_agents_actually_did(dataset):
    """Re-derive the 85% / 47% split from the workbook, so the claim cannot go stale."""
    from app.patterns import _parse_day_sequences

    rows = {str(row["sight_key"]): row for _, row in dataset.sights.iterrows()}
    trips = pd.read_excel(dataset.source, sheet_name="itineraries")
    first, later = [], []
    for value in trips["day_wise_sightseeing"].dropna():
        for index, names in enumerate(_parse_day_sequences(value), start=1):
            for name in names:
                row = rows.get(name)
                if row is None:
                    continue
                (first if index == 1 else later).append(bool(row.get("must_visit_flag")))

    assert len(first) > 200 and len(later) > 1000
    share_first, share_later = sum(first) / len(first), sum(later) / len(later)
    assert share_first > 0.75, f"day-1 must-visit share is {share_first:.1%}"
    assert share_first > share_later + 0.2, "day 1 should be clearly the headline day"
