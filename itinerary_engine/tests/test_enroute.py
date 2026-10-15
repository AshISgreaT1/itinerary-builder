"""Stops made on the way, and days given over entirely to travel."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.builder import PlanRequest, build_itinerary
from app.config import EN_ROUTE_MAX_DETOUR_KM, EXCURSION_KM, TRAVEL_DAY_MAX_DRIVE_HOURS
from app.enroute import plan_en_route
from app.geo import point_to_segment_km
from app.main import app
from app.normalization import normalize_key


@pytest.fixture(scope="module")
def leh_pangong(dataset):
    return build_itinerary(PlanRequest(cities=["Leh", "Pangong Lake"], duration_days=5, pace="active"), dataset)


def en_route_stops(plan) -> list[dict]:
    return [sight for day in plan["days"] for sight in day["sights"] if sight.get("en_route")]


# --- en-route stops -------------------------------------------------------------------
def test_chang_la_is_scheduled_on_the_drive(leh_pangong):
    """1.4 km off the Leh-Pangong road, 77 km from Leh: only the transfer day can hold it."""
    names = {stop["sight"] for stop in en_route_stops(leh_pangong)}
    assert "Changla Pass" in names


def test_en_route_stops_sit_on_a_transfer_day(leh_pangong):
    for day in leh_pangong["days"]:
        if any(sight.get("en_route") for sight in day["sights"]):
            assert day["transfer"], f"day {day['day']} has en-route stops but no journey"


def test_en_route_stop_is_close_to_the_road(dataset, leh_pangong):
    leh, pangong = dataset.city_row("Leh"), dataset.city_row("Pangong Lake")
    for stop in en_route_stops(leh_pangong):
        detour_km, progress = point_to_segment_km(
            stop["lat"], stop["lon"], leh["lat"], leh["lon"], pangong["lat"], pangong["lon"]
        )
        assert detour_km <= EN_ROUTE_MAX_DETOUR_KM
        assert 0 < progress < 1
        assert stop["detour_km"] == pytest.approx(detour_km, abs=0.2)


def test_en_route_day_stays_within_its_hours(leh_pangong):
    for day in leh_pangong["days"]:
        assert day["used_hours"] <= day["available_hours"] + 1e-6
    assert leh_pangong["validation"]["passed"], leh_pangong["validation"]["issues"]


def test_city_sights_are_not_stolen_by_the_road(dataset, leh_pangong):
    """Thiksey is 19 km from Leh, so it stays a Leh day trip rather than a drive-by."""
    en_route = {stop["sight"] for stop in en_route_stops(leh_pangong)}
    assert "Thiksey Monastery" not in en_route
    scheduled = {sight["sight"] for day in leh_pangong["days"] for sight in day["sights"]}
    assert "Thiksey Monastery" in scheduled


def test_en_route_sight_is_not_also_planned_as_a_city_sight(leh_pangong):
    names = [sight["sight"] for day in leh_pangong["days"] for sight in day["sights"]]
    assert len(names) == len(set(names))


def test_nothing_is_taken_en_route_when_a_city_could_host_it(dataset):
    """Every en-route candidate must be too far from all trip cities to be a day trip."""
    plan = build_itinerary(PlanRequest(cities=["Leh", "Pangong Lake"], duration_days=5, pace="active"), dataset)
    for stop in en_route_stops(plan):
        for city in {day["city"] for day in plan["days"]}:
            row = dataset.city_row(city)
            from app.geo import road_km

            assert road_km(row["lat"], row["lon"], stop["lat"], stop["lon"]) > EXCURSION_KM


def test_plan_en_route_respects_the_budget(dataset):
    leg = dataset.graph.leg("Leh", "Pangong Lake", normalize_key("Leh"), normalize_key("Pangong Lake"))
    generous = plan_en_route(dataset, leg, 6.0, {"leh", "pangong lake"})
    none_at_all = plan_en_route(dataset, leg, 0.2, {"leh", "pangong lake"})
    assert generous and sum(stop.cost_hours for stop in generous) <= 6.0
    assert not none_at_all, "nothing fits 12 minutes"


# --- travel-only days -----------------------------------------------------------------
def test_travel_day_may_run_past_the_pace_allowance(dataset):
    """12h Delhi to Manali: no sightseeing to pace, so the day simply starts early."""
    plan = build_itinerary(PlanRequest(cities=["Manali"], duration_days=4, origin_city="Delhi", pace="relaxed"), dataset)
    first = plan["days"][0]
    assert first["time_budget"]["pace_allowance"] >= 13.0
    assert first["time_budget"]["intercity_travel_hours"] == 12.0
    assert plan["validation"]["passed"]


def test_journey_longer_than_a_travel_day_still_fails(dataset):
    """Beyond TRAVEL_DAY_MAX_DRIVE_HOURS there is no early start that makes it work."""
    plan = build_itinerary(PlanRequest(cities=["Goa"], duration_days=4, origin_city="Delhi"), dataset)
    assert not plan["validation"]["passed"]
    assert plan["validation"]["issues"]["travel_feasibility"]


def test_sightseeing_day_keeps_the_normal_pace_ceiling(dataset):
    plan = build_itinerary(PlanRequest(cities=["Jaipur"], duration_days=4, pace="moderate"), dataset)
    for day in plan["days"]:
        assert day["time_budget"]["pace_allowance"] <= 7.0 + 1e-6


def test_travel_day_cap_is_respected(dataset):
    plan = build_itinerary(PlanRequest(cities=["Manali"], duration_days=4, origin_city="Delhi"), dataset)
    leg = plan["route"][0]
    assert leg["drive_hours"] <= TRAVEL_DAY_MAX_DRIVE_HOURS
    assert leg["is_long_travel_day"]


# --- API ---------------------------------------------------------------------------------
def test_api_exposes_en_route_stops():
    response = TestClient(app).post(
        "/generate-itinerary",
        json={"cities": ["Leh", "Pangong Lake"], "duration_days": 5, "pace": "active"},
    )
    assert response.status_code == 200
    payload = response.json()
    stops = [sight for day in payload["days"] for sight in day["sights"] if sight["en_route"]]
    assert stops and stops[0]["detour_km"] is not None
    assert any("on the way" in note for day in payload["days"] for note in day["notes"])
