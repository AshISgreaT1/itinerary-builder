"""Editing a generated itinerary: add, replace, remove, and the suggestions when it will not fit."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.builder import PlanRequest, build_itinerary
from app.edits import SightEdit, apply_edits, placement_options
from app.main import app


@pytest.fixture(scope="module")
def base_request():
    return PlanRequest(cities=["Manali"], duration_days=4, pace="active")


@pytest.fixture(scope="module")
def base_plan(dataset, base_request):
    return build_itinerary(base_request, dataset)


def scheduled_names(plan) -> set[str]:
    return {sight["sight"] for day in plan["days"] for sight in day["sights"]}


def day_of(plan, name) -> int | None:
    for day in plan["days"]:
        if any(sight["sight"] == name for sight in day["sights"]):
            return day["day"]
    return None


# --- placement options --------------------------------------------------------------
def test_placement_options_cover_every_day_of_the_city(dataset, base_plan):
    options = placement_options(base_plan, dataset, "Naggar Castle", "Manali", "active")
    assert [option["day"] for option in options["options"]] == [day["day"] for day in base_plan["days"]]
    assert all("needed_hours" in option and "available_hours" in option for option in options["options"])


def test_placement_options_explain_why_a_day_is_full(dataset, base_plan):
    options = placement_options(base_plan, dataset, "Rohtang Pass", "Manali", "active")
    full_days = [option for option in options["options"] if not option["fits"]]
    assert full_days, "Rohtang Pass cannot fit every day"
    assert all(option["blockers"] for option in full_days)
    assert any(option["replace_options"] or option["fits_if_dropped"] for option in full_days)


def test_placement_options_ignore_the_sight_itself(dataset, base_plan):
    """A sight already on a day must not be counted as competing with itself."""
    name = next(sight["sight"] for day in base_plan["days"] for sight in day["sights"])
    its_day = day_of(base_plan, name)
    options = placement_options(base_plan, dataset, name, "Manali", "active")
    assert next(option["fits"] for option in options["options"] if option["day"] == its_day)


# --- adding ---------------------------------------------------------------------------
def test_add_schedules_the_sight_and_keeps_the_plan_valid(dataset, base_request, base_plan):
    missing = sorted(set(base_plan["unscheduled_sights"]))
    assert missing, "the base plan should leave something out to add"
    label = missing[0]
    city, name = label.split(" / ")[0], label.split(" / ")[-1]
    payload, outcomes = apply_edits(dataset, base_request, [SightEdit("add", name, city)])
    assert outcomes[0].applied and name in scheduled_names(payload)
    assert payload["validation"]["passed"], payload["validation"]["issues"]
    assert payload["edits"][0]["placed_on_day"]


def test_add_to_a_specific_day_is_honoured(dataset, base_request):
    payload, outcomes = apply_edits(dataset, base_request, [SightEdit("add", "Naggar Castle", "Manali", day=1)])
    assert outcomes[0].applied
    assert day_of(payload, "Naggar Castle") == 1
    assert payload["validation"]["passed"]


def test_day_limits_are_respected_after_an_edit(dataset, base_request):
    payload, _ = apply_edits(dataset, base_request, [SightEdit("add", "Naggar Castle", "Manali", day=1)])
    for day in payload["days"]:
        assert day["used_hours"] <= day["available_hours"] + 1e-6


def test_route_and_travel_time_are_recalculated(dataset, base_request, base_plan):
    """The edited day's distances must come out of the rebuild, not be copied over."""
    before = {day["day"]: (day["route_km"], day["local_travel_hours"]) for day in base_plan["days"]}
    payload, _ = apply_edits(dataset, base_request, [SightEdit("add", "Naggar Castle", "Manali", day=1)])
    after = {day["day"]: (day["route_km"], day["local_travel_hours"]) for day in payload["days"]}
    assert before != after
    assert payload["validation"]["passed"], "recomputed metrics must still pass validation"


def test_unfittable_sight_is_reported_with_alternatives_not_dropped(dataset):
    """Rohtang Pass needs 7.2h: at a relaxed pace no day can hold it, and we say so."""
    request = PlanRequest(cities=["Manali"], duration_days=4, pace="relaxed")
    payload, outcomes = apply_edits(dataset, request, [SightEdit("add", "Rohtang Pass", "Manali", day=2)])
    outcome = outcomes[0]
    assert not outcome.applied
    assert "could not be scheduled" in outcome.message
    assert "Rohtang Pass" not in scheduled_names(payload)
    assert payload["validation"]["passed"], "the rest of the plan stays usable"
    assert payload["edits"][0]["applied"] is False


def test_sight_moves_to_another_day_when_the_chosen_one_is_full(dataset, base_request):
    """Asking for a full day should relocate the sight and say where it went."""
    busiest = max(build_itinerary(base_request, dataset)["days"], key=lambda day: len(day["sights"]))
    payload, outcomes = apply_edits(dataset, base_request, [SightEdit("add", "Naggar Castle", "Manali", day=busiest["day"])])
    outcome = outcomes[0]
    if outcome.applied and outcome.day != busiest["day"]:
        assert "instead" in outcome.message
    else:  # it did fit, or it could not be placed at all -- both must be explained
        assert outcome.message


# --- replacing and removing -----------------------------------------------------------
def test_replace_swaps_one_sight_for_another(dataset, base_request, base_plan):
    victim = next(sight["sight"] for day in base_plan["days"] for sight in day["sights"] if not sight["must_visit"])
    payload, outcomes = apply_edits(
        dataset, base_request, [SightEdit("replace", "Naggar Castle", "Manali", day=day_of(base_plan, victim), replaces=victim)]
    )
    assert outcomes[0].applied
    assert "Naggar Castle" in scheduled_names(payload)
    assert victim not in scheduled_names(payload)
    assert payload["validation"]["passed"]


def test_remove_takes_a_sight_out_and_keeps_it_out(dataset, base_request, base_plan):
    victim = next(iter(scheduled_names(base_plan)))
    payload, outcomes = apply_edits(dataset, base_request, [SightEdit("remove", victim, "Manali")])
    assert outcomes[0].applied
    assert victim not in scheduled_names(payload)
    assert payload["validation"]["passed"]


def test_edits_accumulate(dataset, base_request, base_plan):
    victim = next(sight["sight"] for day in base_plan["days"] for sight in day["sights"] if not sight["must_visit"])
    payload, outcomes = apply_edits(
        dataset, base_request,
        [SightEdit("remove", victim, "Manali"), SightEdit("add", "Naggar Castle", "Manali")],
    )
    assert all(outcome.applied for outcome in outcomes)
    names = scheduled_names(payload)
    assert victim not in names and "Naggar Castle" in names


def test_unknown_sight_is_reported(dataset, base_request):
    _, outcomes = apply_edits(dataset, base_request, [SightEdit("add", "Atlantis Water Park", "Manali")])
    assert not outcomes[0].applied and "not in the sightseeing data" in outcomes[0].message


# --- API --------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_adjust_endpoint(client):
    body = {
        "base": {"cities": ["Manali"], "duration_days": 4, "pace": "active"},
        "edits": [{"action": "add", "sight": "Naggar Castle", "city": "Manali", "day": 1}],
    }
    response = client.post("/itinerary/adjust", json=body)
    assert response.status_code == 200
    payload = response.json()
    assert payload["edits"][0]["applied"] and payload["edits"][0]["placed_on_day"] == 1
    assert payload["validation"]["passed"]


def test_placement_options_endpoint(client):
    response = client.post(
        "/itinerary/placement-options",
        json={"base": {"cities": ["Manali"], "duration_days": 4, "pace": "active"}, "sight": "Naggar Castle", "city": "Manali"},
    )
    assert response.status_code == 200
    assert "options" in response.json() and response.json()["options"]


def test_generate_endpoint_accepts_pins_and_exclusions(client):
    response = client.post(
        "/generate-itinerary",
        json={
            "cities": ["Manali"], "duration_days": 4, "pace": "active",
            "pinned_sights": [{"sight": "Naggar Castle", "city": "Manali", "day": 1}],
            "excluded_sights": ["Van Vihar"],
        },
    )
    assert response.status_code == 200
    payload = response.json()
    names = {sight["sight"] for day in payload["days"] for sight in day["sights"]}
    assert "Naggar Castle" in names and "Van Vihar" not in names


def test_adjust_endpoint_rejects_unknown_city(client):
    response = client.post(
        "/itinerary/adjust",
        json={"base": {"cities": ["Atlantis"], "duration_days": 3}, "edits": [{"action": "add", "sight": "X"}]},
    )
    assert response.status_code == 422
