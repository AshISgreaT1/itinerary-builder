"""Changing a plan after it has been generated, and the advice that comes with it.

The contract these tests hold the code to: a customisation either does what it said or is
rolled back and explained -- a stop never quietly leaves the trip -- and a slot clash the
user chose to accept is advice, not a locked door.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.builder import PinnedSight, PlanRequest, PlanningError, build_itinerary
from app.customize import Operation, apply_operations
from app.main import app
from app.suggest import suggestions_for


def names(payload: dict, day_number: int) -> list[str]:
    day = next(item for item in payload["days"] if item["day"] == day_number)
    return [sight["sight"] for sight in day["sights"]]


def day_of(payload: dict, sight: str) -> int | None:
    for day in payload["days"]:
        if any(item["sight"] == sight for item in day["sights"]):
            return day["day"]
    return None


@pytest.fixture
def udaipur():
    return PlanRequest(cities=["Udaipur"], duration_days=3, pace="moderate")


# --- multi-city allocation (rule 3) -----------------------------------------------------------
def test_the_user_can_split_days_between_cities(dataset):
    for split in ({"Jaipur": 4, "Udaipur": 2}, {"Jaipur": 3, "Udaipur": 3}, {"Jaipur": 2, "Udaipur": 4}):
        plan = build_itinerary(
            PlanRequest(cities=["Jaipur", "Udaipur"], duration_days=6, pace="moderate", city_days=split),
            dataset,
        )
        assert plan["summary"]["days_per_city"] == split
        assert plan["validation"]["passed"], plan["validation"]["issues"]


def test_a_split_that_does_not_add_up_is_refused_with_the_arithmetic(dataset):
    with pytest.raises(PlanningError) as error:
        build_itinerary(
            PlanRequest(cities=["Jaipur", "Udaipur"], duration_days=6, city_days={"Jaipur": 4, "Udaipur": 3}),
            dataset,
        )
    assert "adds up to 7" in str(error.value) and "is 6" in str(error.value)

    with pytest.raises(PlanningError) as missing:
        build_itinerary(
            PlanRequest(cities=["Jaipur", "Udaipur"], duration_days=6, city_days={"Jaipur": 6}),
            dataset,
        )
    assert "every city" in str(missing.value)


def test_reallocating_changes_what_fits(dataset):
    even, _ = apply_operations(
        dataset,
        PlanRequest(cities=["Jaipur", "Udaipur"], duration_days=6, pace="moderate"),
        [Operation(operation="reallocate", city_days={"Jaipur": 3, "Udaipur": 3})],
    )
    assert even["summary"]["days_per_city"] == {"Jaipur": 3, "Udaipur": 3}
    assert even["validation"]["passed"]


# --- longer days (rule 4) ----------------------------------------------------------------------
def test_a_day_can_be_lengthened_on_demand(dataset, udaipur):
    base = build_itinerary(udaipur, dataset)
    longer, outcomes = apply_operations(dataset, udaipur, [Operation(operation="extend_day", day=2, hours=2.0)])
    assert outcomes[0].applied
    before = next(d for d in base["days"] if d["day"] == 2)
    after = next(d for d in longer["days"] if d["day"] == 2)
    assert after["available_hours"] == pytest.approx(before["available_hours"] + 2.0)
    assert after["time_budget"]["extra_hours"] == 2.0
    assert longer["validation"]["passed"], longer["validation"]["issues"]
    assert any("by your choice" in note for note in longer["validation"]["notes"])


def test_a_longer_day_may_hold_one_more_stop(dataset, udaipur):
    base = build_itinerary(udaipur, dataset)
    longer, _ = apply_operations(dataset, udaipur, [Operation(operation="extend_day", day=2, hours=2.0)])
    assert len(names(longer, 2)) >= len(names(base, 2))


def test_extending_by_zero_undoes_it(dataset, udaipur):
    plan, outcomes = apply_operations(dataset, udaipur, [
        Operation(operation="extend_day", day=2, hours=2.0),
        Operation(operation="extend_day", day=2, hours=0),
    ])
    assert all(outcome.applied for outcome in outcomes)
    assert next(d for d in plan["days"] if d["day"] == 2)["time_budget"]["extra_hours"] == 0


# --- manual moves, swaps and order (rule 5) -----------------------------------------------------
def test_a_move_that_does_not_fit_is_undone_and_explained(dataset, udaipur):
    """The one thing that must never happen: a stop vanishing because a drag failed."""
    base = build_itinerary(udaipur, dataset)
    crowded = min((day for day in base["days"] if day["sights"]),
                  key=lambda day: day["available_hours"] - day["used_hours"])
    elsewhere = next(day for day in base["days"] if day["day"] != crowded["day"] and day["sights"])
    victim = elsewhere["sights"][0]["sight"]

    plan, outcomes = apply_operations(
        dataset, udaipur,
        [Operation(operation="move", sight=victim, from_day=elsewhere["day"], to_day=crowded["day"])],
    )
    everywhere = {sight["sight"] for day in plan["days"] for sight in day["sights"]}
    if outcomes[0].applied:
        assert day_of(plan, victim) == crowded["day"]
    else:
        assert victim in everywhere, "a rolled-back move must leave the sight where it was"
        assert "undone" in outcomes[0].message
        assert "extend_day" in outcomes[0].message or "swap" in outcomes[0].message


def test_extending_first_makes_the_move_work(dataset, udaipur):
    plan, outcomes = apply_operations(dataset, udaipur, [
        Operation(operation="extend_day", day=1, hours=1.5),
        Operation(operation="move", sight="Jagdish temple", to_day=1),
    ])
    assert [outcome.applied for outcome in outcomes] == [True, True], [o.message for o in outcomes]
    assert day_of(plan, "Jagdish temple") == 1
    assert plan["validation"]["passed"], plan["validation"]["issues"]


def test_a_manual_move_moves_one_thing(dataset, udaipur):
    """Re-planning around a drag must not rearrange the days the user did not touch."""
    base = build_itinerary(udaipur, dataset)
    plan, outcomes = apply_operations(dataset, udaipur, [
        Operation(operation="extend_day", day=1, hours=1.5),
        Operation(operation="move", sight="Jagdish temple", to_day=1),
    ])
    assert outcomes[-1].applied
    untouched = next(day for day in base["days"] if day["day"] == 2)
    after = next(day for day in plan["days"] if day["day"] == 2)
    assert [s["sight"] for s in after["sights"]] == [s["sight"] for s in untouched["sights"]]


def test_two_sights_can_be_swapped(dataset, udaipur):
    base = build_itinerary(udaipur, dataset)
    first, second = base["days"][0]["sights"][0]["sight"], base["days"][1]["sights"][0]["sight"]
    plan, outcomes = apply_operations(
        dataset, udaipur, [Operation(operation="swap", sight=first, with_sight=second)])
    if outcomes[0].applied:
        assert day_of(plan, first) == 2 and day_of(plan, second) == 1
    else:
        assert {first, second} <= {s["sight"] for day in plan["days"] for s in day["sights"]}


def test_a_day_can_be_reordered(dataset, udaipur):
    base = build_itinerary(udaipur, dataset)
    day = next(d for d in base["days"] if len(d["sights"]) > 1)
    reversed_order = [sight["sight"] for sight in day["sights"]][::-1]
    plan, outcomes = apply_operations(
        dataset, udaipur, [Operation(operation="reorder", day=day["day"], order=reversed_order)])
    assert outcomes[0].applied
    assert set(names(plan, day["day"])) == set(reversed_order)


def test_your_own_activity_lands_on_the_clock(dataset, udaipur):
    plan, outcomes = apply_operations(dataset, udaipur, [
        Operation(operation="add_activity", day=2, name="Dinner at Ambrai", duration_hours=1.5, at="20:00"),
    ])
    assert outcomes[0].applied
    day = next(d for d in plan["days"] if d["day"] == 2)
    block = next(b for b in day["schedule"]["blocks"] if b["kind"] == "custom")
    assert block["label"] == "Dinner at Ambrai"
    assert block["start"] == "20:00" and block["hours"] == pytest.approx(1.5)
    assert day["schedule"]["totals"]["your_own_plans"] == pytest.approx(1.5)
    assert day["schedule"]["accounted_hours"] == pytest.approx(day["schedule"]["span_hours"], abs=0.02)
    assert plan["validation"]["passed"], plan["validation"]["issues"]


def test_an_unknown_operation_is_refused(dataset, udaipur):
    with pytest.raises(PlanningError) as error:
        apply_operations(dataset, udaipur, [Operation(operation="teleport", sight="City Palace")])
    assert "Unknown operation" in str(error.value)


# --- non-blocking validation and suggestions (rules 1, 2, 6) --------------------------------------
def test_a_slot_clash_the_user_chose_does_not_block_the_plan(dataset):
    """Three evening sights on one day is the user's call: advised against, not refused."""
    pins = [PinnedSight(name, "Udaipur", 2)
            for name in ["Bagore ki Haveli", "Lake Pichola", "Fatehsagar Lake"]]
    plan = build_itinerary(
        PlanRequest(cities=["Udaipur"], duration_days=3, pace="moderate", pinned_sights=pins), dataset)
    assert plan["validation"]["passed"], "the user must not be locked out of their own plan"
    assert plan["validation"]["critical_count"] == 0
    conflicts = [c for day in plan["days"] for c in day["schedule"]["conflicts"]]
    assert conflicts, "the clash should still be reported"
    for conflict in conflicts:
        assert conflict["suggested_at"] and conflict["window"]
        assert conflict["sight"] in conflict["message"]
    scheduled = {s["sight"] for day in plan["days"] for s in day["sights"]}
    assert {pin.sight for pin in pins} <= scheduled, "every pinned sight is kept"


def test_the_advised_time_matches_where_the_sight_actually_is(dataset):
    """A conflict that reported the wrong clock time would be worse than none."""
    pins = [PinnedSight(name, "Udaipur", 2)
            for name in ["Bagore ki Haveli", "Lake Pichola", "Fatehsagar Lake"]]
    plan = build_itinerary(
        PlanRequest(cities=["Udaipur"], duration_days=3, pace="moderate", pinned_sights=pins), dataset)
    for day in plan["days"]:
        for conflict in day["schedule"]["conflicts"]:
            block = next(b for b in day["schedule"]["blocks"]
                         if b["kind"] == "sight" and b["sight"] == conflict["sight"])
            assert block["start"] == conflict["scheduled_at"]


def test_suggestions_carry_an_action_that_can_be_applied(dataset):
    pins = [PinnedSight(name, "Udaipur", 2)
            for name in ["Bagore ki Haveli", "Lake Pichola", "Fatehsagar Lake"]]
    request = PlanRequest(cities=["Udaipur"], duration_days=3, pace="moderate", pinned_sights=pins)
    plan = build_itinerary(request, dataset)
    advice = suggestions_for(plan, dataset, "moderate")
    assert advice["count"] > 0
    for item in advice["suggestions"]:
        assert item["message"] and item["fix"]
        assert item["action"]["operation"] in {"retime", "move", "to_morning", "extend_day",
                                               "reallocate", "pin"}

    action = dict(advice["suggestions"][0]["action"])
    action.pop("endpoint", None)
    applied, outcomes = apply_operations(dataset, request, [Operation(**action)])
    assert outcomes[0].applied, outcomes[0].message
    assert applied["validation"]["passed"]


def test_a_retime_suggestion_is_only_made_when_the_day_has_room(dataset):
    """Suggesting 16:30 for two sights at once would be advice that cannot be taken."""
    pins = [PinnedSight(name, "Udaipur", 2)
            for name in ["Bagore ki Haveli", "Lake Pichola", "Fatehsagar Lake"]]
    plan = build_itinerary(
        PlanRequest(cities=["Udaipur"], duration_days=3, pace="moderate", pinned_sights=pins), dataset)
    advice = suggestions_for(plan, dataset, "moderate")
    retimes = [item for item in advice["suggestions"] if item["kind"] == "retime"]
    starts = [item["action"]["at"] for item in retimes]
    assert len(starts) == len(set(starts)), f"two stops cannot both start at {starts}"


def test_suggestions_are_empty_for_a_clean_plan(dataset):
    plan = build_itinerary(PlanRequest(cities=["Udaipur"], duration_days=3, pace="moderate"), dataset)
    advice = suggestions_for(plan, dataset, "moderate")
    conflicts = [item for item in advice["suggestions"] if item["group"] == "conflict"]
    assert not conflicts, conflicts


def test_the_generator_itself_is_still_strict(dataset):
    """Non-blocking applies to the user's choices -- not to what the engine produces alone."""
    for city, days, pace in [("Udaipur", 5, "relaxed"), ("Jaipur", 4, "active"), ("Goa", 5, "moderate")]:
        plan = build_itinerary(PlanRequest(cities=[city], duration_days=days, pace=pace), dataset)
        assert plan["validation"]["passed"]
        assert not [c for day in plan["days"] for c in day["schedule"]["conflicts"]], \
            "an unassisted plan should have no slot conflicts at all"


# --- the API ---------------------------------------------------------------------------------------
def test_customize_endpoint_rebuilds_and_advises():
    client = TestClient(app)
    response = client.post("/itinerary/customize", json={
        "base": {"cities": ["Jaipur", "Udaipur"], "duration_days": 6, "pace": "moderate"},
        "operations": [
            {"operation": "reallocate", "city_days": {"Jaipur": 3, "Udaipur": 3}},
            {"operation": "extend_day", "day": 2, "hours": 1.5},
            {"operation": "add_activity", "name": "Cooking class", "day": 3,
             "duration_hours": 2.0, "at": "18:00"},
        ],
    })
    assert response.status_code == 200
    payload = response.json()
    assert [op["applied"] for op in payload["operations"]] == [True, True, True]
    assert payload["summary"]["days_per_city"] == {"Jaipur": 3, "Udaipur": 3}
    assert "suggestions" in payload
    assert payload["validation"]["passed"], payload["validation"]["issues"]


def test_suggestions_endpoint_changes_nothing():
    client = TestClient(app)
    body = {"cities": ["Udaipur"], "duration_days": 3, "pace": "moderate"}
    before = client.post("/generate-itinerary", json=body).json()
    advice = client.post("/itinerary/suggestions", json={"base": body})
    assert advice.status_code == 200
    after = client.post("/generate-itinerary", json=body).json()
    assert [[s["sight"] for s in day["sights"]] for day in before["days"]] == \
           [[s["sight"] for s in day["sights"]] for day in after["days"]]


def test_a_bad_operation_is_a_422_not_a_crash():
    client = TestClient(app)
    response = client.post("/itinerary/customize", json={
        "base": {"cities": ["Udaipur"], "duration_days": 3},
        "operations": [{"operation": "extend_day", "day": 2}],  # no hours
    })
    assert response.status_code == 422
    assert "hours" in response.json()["detail"].lower()


def test_a_rollback_hint_fits_the_day_it_describes(dataset):
    """A transfer day cannot be lengthened or swapped with: say what it is instead."""
    plan, outcomes = apply_operations(
        dataset,
        PlanRequest(cities=["Jaipur", "Udaipur"], duration_days=6, pace="moderate"),
        [Operation(operation="move", sight="City Palace", from_day=6, to_day=5)],
    )
    message = outcomes[0].message
    if not outcomes[0].applied:
        transfer_day = next(d for d in plan["days"] if d["day"] == 5)
        if transfer_day.get("transfer") and transfer_day["available_hours"] <= 0:
            assert "taken up by the journey" in message
            assert "swap with one of ." not in message
