"""Evening and sunset sights are constraints, not preferences.

The rules under test, in order: a hard-slot sight is only ever visited inside its window;
an idle wait before one is filled with a flexible stop rather than wasted; a day that
cannot give the evening hands the sight to a day that can; and the validator rejects any
schedule that breaks the first rule, so none of this can regress quietly.
"""

from __future__ import annotations

import copy

import pytest

from app.builder import PlanRequest, build_itinerary
from app.config import HARD_TIME_SLOTS, TIME_WINDOWS
from app.validation import validate_itinerary

CITY_CASES = [
    ("Udaipur", 5, "relaxed"),
    ("Udaipur", 4, "moderate"),
    ("Jaipur", 4, "active"),
    ("Goa", 5, "moderate"),
    ("Jaisalmer", 4, "moderate"),
    ("Jodhpur", 3, "moderate"),
    ("Cochin", 4, "relaxed"),
    ("Leh", 4, "moderate"),
]


def minutes(value: str) -> int:
    hours, mins = value.split(":")
    return int(hours) * 60 + int(mins)


def plan_for(dataset, city, days, pace):
    return build_itinerary(PlanRequest(cities=[city], duration_days=days, pace=pace), dataset)


# --- rule 1: the window is absolute ---------------------------------------------------------
@pytest.mark.parametrize("city, days, pace", CITY_CASES)
def test_hard_slot_sights_are_never_scheduled_early(dataset, city, days, pace):
    plan = plan_for(dataset, city, days, pace)
    for day in plan["days"]:
        for sight in day["sights"]:
            slot = str(sight.get("best_time", "")).strip().lower()
            if slot not in HARD_TIME_SLOTS:
                continue
            block = next((b for b in day["schedule"]["blocks"]
                          if b["kind"] == "sight" and b["sight"] == sight["sight"]), None)
            assert block, f"{sight['sight']} is listed on day {day['day']} but not on the clock"
            window = TIME_WINDOWS[slot]
            assert minutes(block["start"]) >= window[0] * 60, \
                f"{city}: {sight['sight']} ({slot}) starts at {block['start']}"
            assert minutes(block["end"]) <= window[1] * 60, \
                f"{city}: {sight['sight']} ({slot}) runs to {block['end']}"
    assert plan["validation"]["passed"], plan["validation"]["issues"]


@pytest.mark.parametrize("city, days, pace", CITY_CASES)
def test_nothing_is_left_hanging(dataset, city, days, pace):
    """Every deferred sight is re-homed or dropped with a reason, and still offered."""
    plan = plan_for(dataset, city, days, pace)
    for day in plan["days"]:
        assert not day["schedule"]["deferred"], f"day {day['day']} still defers {day['schedule']['deferred']}"
    for label, reason in (plan.get("unplaceable_reasons") or {}).items():
        assert "slot" in reason and "rather than scheduled outside" in reason, f"{label}: {reason}"
        assert label in plan["unscheduled_sights"], f"{label} was dropped without being offered"


# --- rule 2: swap a flexible sight into the wait ---------------------------------------------
def test_only_a_flexible_sight_is_pulled_into_a_wait(dataset):
    from app.schedule import _pull_flexible_forward

    sights = dataset.city_sights("udaipur")
    flexible = sights[sights["best_time"].str.lower() == "any"].iloc[0].to_dict()
    morning = sights[sights["best_time"].str.lower() == "morning"].iloc[0].to_dict()

    found = _pull_flexible_forward([morning, flexible], 8.0, dataset, "Udaipur", None)
    assert found is not None and found[1]["sight"] == flexible["sight"], "the flexible stop should move"
    assert _pull_flexible_forward([morning], 8.0, dataset, "Udaipur", None) is None, \
        "a morning sight must not be dragged into a late-afternoon gap"
    assert _pull_flexible_forward([morning, flexible], 0.1, dataset, "Udaipur", None) is None, \
        "nothing is moved into a gap it does not fit"


def test_a_day_with_an_evening_sight_either_fills_the_wait_or_starts_later(dataset):
    plan = plan_for(dataset, "Udaipur", 4, "moderate")
    notes = " ".join(note for day in plan["days"] for note in day["schedule"]["notes"])
    assert "brought forward" in notes or "starts at" in notes


# --- rule 3: cross-day fallback ---------------------------------------------------------------
def test_an_evening_sight_is_re_homed_or_explained(dataset):
    plans = [plan_for(dataset, *case) for case in CITY_CASES]
    moved = [note for plan in plans for day in plan["days"] for note in day.get("notes", [])
             if "was moved here from day" in note]
    dropped = [reason for plan in plans for reason in (plan.get("unplaceable_reasons") or {}).values()]
    assert moved or dropped, "with this many evening sights, some must be re-homed or explained"
    for note in moved:
        assert "window" in note


def test_a_departure_day_is_refused_an_evening_it_cannot_give(dataset):
    """The packer declines the booking instead of leaving the clock to throw it out."""
    from app.solver import hard_slot_blocker

    sights = dataset.city_sights("udaipur")
    evening = sights[sights["best_time"].str.lower() == "evening"].head(3)

    class _LongDeparture:
        is_departure = True
        departure_cost_hours = 4.0

    class _OrdinaryDay:
        is_departure = False
        departure_cost_hours = 0.0

    assert hard_slot_blocker(evening, _LongDeparture()) is not None, "a long departure leaves no evening"
    assert hard_slot_blocker(evening.head(1), _OrdinaryDay()) is None, "one evening sight fits a normal day"
    assert hard_slot_blocker(evening, _OrdinaryDay()) is not None, "three do not fit a 3.5h window"
    assert hard_slot_blocker(sights[sights["best_time"].str.lower() == "morning"].head(3),
                             _LongDeparture()) is None, "the rule is about hard slots only"


# --- rule 4: the validator blocks it ------------------------------------------------------------
def test_validation_rejects_an_evening_sight_in_the_morning(dataset):
    plan = plan_for(dataset, "Udaipur", 4, "moderate")
    assert plan["validation"]["passed"]

    for index, day in enumerate(plan["days"]):
        evening = next((s for s in day["sights"]
                        if str(s.get("best_time", "")).lower() in HARD_TIME_SLOTS), None)
        if not evening:
            continue
        broken = copy.deepcopy(plan)
        for block in broken["days"][index]["schedule"]["blocks"]:
            if block["sight"] == evening["sight"]:
                block["start"], block["end"] = "10:00", "11:00"
        report = validate_itinerary(broken, dataset, "moderate", None)
        assert not report.passed, "a morning-scheduled evening sight must fail the gate"
        assert any("before its 16:30 window" in issue.message for issue in report.entries), \
            [issue.message for issue in report.entries]
        return
    pytest.fail("no evening sight in the plan to break")


def test_validation_rejects_a_sight_left_off_the_clock(dataset):
    plan = plan_for(dataset, "Udaipur", 4, "moderate")
    broken = copy.deepcopy(plan)
    broken["days"][0]["schedule"]["deferred"] = [{"sight": "Lake Pichola", "reason": "testing"}]
    report = validate_itinerary(broken, dataset, "moderate", None)
    assert not report.passed
    assert any("left off the clock" in issue.message for issue in report.entries)


# --- lunch, and the bug that cost a sunset ------------------------------------------------------
@pytest.mark.parametrize("city, days, pace", CITY_CASES)
def test_a_day_that_starts_late_does_not_stop_for_lunch(dataset, city, days, pace):
    """A day beginning at 16:00 has already eaten; booking lunch pushed a sunset off the clock."""
    plan = plan_for(dataset, city, days, pace)
    for day in plan["days"]:
        start = day["schedule"]["starts"]
        if not start or minutes(start) <= 14 * 60:
            continue
        kinds = [block["kind"] for block in day["schedule"]["blocks"]]
        assert "lunch" not in kinds, f"{city} day {day['day']} starts {start} and still books lunch"


def test_the_api_exposes_the_slot_outcome():
    """The response model must not quietly drop what the clock decided."""
    from fastapi.testclient import TestClient

    from app.main import app

    response = TestClient(app).post(
        "/generate-itinerary", json={"cities": ["Udaipur"], "duration_days": 5, "pace": "relaxed"}
    )
    assert response.status_code == 200
    payload = response.json()
    assert "unplaceable_reasons" in payload
    for day in payload["days"]:
        assert day["schedule"]["deferred"] == []
    for label, reason in payload["unplaceable_reasons"].items():
        assert reason and label
