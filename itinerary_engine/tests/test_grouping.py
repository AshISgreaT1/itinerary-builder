"""Nearby stops belong on the same day, and in a sensible order.

Clustering decides which stops go together before any day has a clock, so the time-of-day
groups inside a day can stretch the route afterwards. :func:`refine_grouping` sweeps the
days again and trades stops between them where that shortens the driving -- these tests
hold it to shortening the driving *without* loosening any of the rules it is allowed to
move within.
"""

from __future__ import annotations

import pandas as pd
import pytest

import app.solver as solver
from app.builder import PinnedSight, PlanRequest, build_itinerary
from app.config import ARRIVAL_DAY_MAX_KM, HARD_TIME_SLOTS, MAX_DAY_ROUTE_KM, MAX_LEG_KM, TIME_WINDOWS
from app.geo import order_by_route, road_km

CASES = [
    ("Udaipur", 3, "moderate"),
    ("Jaipur", 4, "active"),
    ("Goa", 5, "moderate"),
    ("Leh", 4, "moderate"),
    ("Munnar", 4, "moderate"),
    ("Jodhpur", 3, "moderate"),
]


def city_stops(day: dict) -> list[dict]:
    return [sight for sight in day["sights"] if not sight.get("en_route")]


def route_km(stops: list[dict]) -> float:
    return sum(road_km(float(a["lat"]), float(a["lon"]), float(b["lat"]), float(b["lon"]))
               for a, b in zip(stops, stops[1:]))


def best_route_km(stops: list[dict]) -> float:
    """The shortest order of these same stops, ignoring time of day."""
    frame = pd.DataFrame({"lat": [float(s["lat"]) for s in stops], "lon": [float(s["lon"]) for s in stops]})
    order = order_by_route(frame, float(stops[0]["lat"]), float(stops[0]["lon"]))
    return sum(road_km(float(stops[order[i]]["lat"]), float(stops[order[i]]["lon"]),
                       float(stops[order[i + 1]]["lat"]), float(stops[order[i + 1]]["lon"]))
               for i in range(len(order) - 1))


def plan_without_refinement(dataset, city, days, pace, monkeypatch):
    monkeypatch.setattr(solver, "refine_grouping", lambda *args, **kwargs: None)
    return build_itinerary(PlanRequest(cities=[city], duration_days=days, pace=pace), dataset)


# --- the days stay legal ---------------------------------------------------------------------
@pytest.mark.parametrize("city, days, pace", CASES)
def test_refined_days_still_obey_every_limit(dataset, city, days, pace):
    plan = build_itinerary(PlanRequest(cities=[city], duration_days=days, pace=pace), dataset)
    assert plan["validation"]["passed"], plan["validation"]["issues"]
    for day in plan["days"]:
        stops = city_stops(day)
        assert day["used_hours"] <= day["available_hours"] + 1e-6, f"day {day['day']} overruns its budget"
        if len(stops) > 1:
            legs = [road_km(float(a["lat"]), float(a["lon"]), float(b["lat"]), float(b["lon"]))
                    for a, b in zip(stops, stops[1:])]
            assert max(legs) <= MAX_LEG_KM + 1e-6
            assert sum(legs) <= MAX_DAY_ROUTE_KM + 1e-6
        if day["is_arrival_day"]:
            for sight in stops:
                assert sight["distance_from_centre_km"] <= ARRIVAL_DAY_MAX_KM


@pytest.mark.parametrize("city, days, pace", CASES)
def test_refining_never_moves_a_hard_slot_out_of_its_window(dataset, city, days, pace):
    plan = build_itinerary(PlanRequest(cities=[city], duration_days=days, pace=pace), dataset)
    for day in plan["days"]:
        for sight in day["sights"]:
            slot = str(sight.get("best_time", "")).strip().lower()
            if slot not in HARD_TIME_SLOTS:
                continue
            block = next(b for b in day["schedule"]["blocks"]
                         if b["kind"] == "sight" and b["sight"] == sight["sight"])
            hours, mins = block["start"].split(":")
            assert int(hours) + int(mins) / 60 >= TIME_WINDOWS[slot][0] - 1e-9


# --- it actually groups things better ------------------------------------------------------------
def test_refining_shortens_the_driving_across_the_dataset(dataset, monkeypatch):
    """Measured, not asserted by eye: fewer driving hours for at least as many sights."""
    after_hours = after_sights = 0.0
    for city, days, pace in CASES:
        plan = build_itinerary(PlanRequest(cities=[city], duration_days=days, pace=pace), dataset)
        after_sights += plan["summary"]["total_sights"]
        after_hours += sum(day["local_travel_hours"] for day in plan["days"])

    before_hours = before_sights = 0.0
    for city, days, pace in CASES:
        plan = plan_without_refinement(dataset, city, days, pace, monkeypatch)
        before_sights += plan["summary"]["total_sights"]
        before_hours += sum(day["local_travel_hours"] for day in plan["days"])

    assert after_sights >= before_sights, "refining must not cost sights"
    assert after_hours <= before_hours + 1e-6, (
        f"refining should not add driving: {before_hours:.1f}h -> {after_hours:.1f}h")


def test_refining_moves_days_closer_to_their_best_order(dataset, monkeypatch):
    """Compared against the same plan unrefined, not against an absolute bar.

    ``best_route_km`` is itself a heuristic -- it routes from whichever stop happens to be
    first -- so a fixed "within N km of optimal" threshold would be measuring the baseline's
    noise as much as the planner. What matters is the direction: refining must not leave the
    days further from the best order their own stops allow.
    """
    def total_excess(refined: bool) -> float:
        if not refined:
            monkeypatch.setattr(solver, "refine_grouping", lambda *args, **kwargs: None)
        excess = 0.0
        for city, days, pace in CASES:
            plan = build_itinerary(PlanRequest(cities=[city], duration_days=days, pace=pace), dataset)
            for day in plan["days"]:
                stops = city_stops(day)
                if len(stops) > 1:
                    excess += max(0.0, route_km(stops) - best_route_km(stops))
        return excess

    after = total_excess(refined=True)
    before = total_excess(refined=False)
    assert after <= before + 1e-6, f"refining left the days worse ordered: {before:.1f} km -> {after:.1f} km"


def test_a_pinned_sight_is_never_regrouped(dataset):
    """The refinement optimises; it does not overrule the user."""
    plan = build_itinerary(
        PlanRequest(cities=["Udaipur"], duration_days=4, pace="moderate",
                    pinned_sights=[PinnedSight("Sajjangarh fort", "Udaipur", 2)]),
        dataset,
    )
    day = next(d for d in plan["days"] if any(s["sight"] == "Sajjangarh fort" for s in d["sights"]))
    assert day["day"] == 2, "a pinned sight must stay on the day it was pinned to"


def test_a_must_visit_is_not_moved_off_the_arrival_day(dataset, monkeypatch):
    """Day 1 carries the headline sights; a shorter drive is not worth trading that away."""
    shares = []
    for city, days, pace in CASES:
        plan = build_itinerary(PlanRequest(cities=[city], duration_days=days, pace=pace), dataset)
        stops = city_stops(plan["days"][0])
        if stops:
            shares.append(sum(1 for s in stops if s.get("must_visit")) / len(stops))
    assert sum(shares) / len(shares) > 0.6, "the arrival day lost its must-visits to the refinement"


def test_refinement_is_cheap_enough_to_run_on_every_plan(dataset):
    """It runs inside every build, including the re-plans the slot resolver makes."""
    import time

    start = time.monotonic()
    build_itinerary(PlanRequest(cities=["Jaipur"], duration_days=5, pace="active"), dataset)
    assert time.monotonic() - start < 15, "a single plan should not take this long"
