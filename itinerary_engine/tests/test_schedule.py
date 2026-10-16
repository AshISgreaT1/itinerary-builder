"""Clock schedules, preferred visiting times, and what the past itineraries teach."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.builder import PlanRequest, build_itinerary
from app.config import (
    HARD_TIME_SLOTS,
    LATEST_END_HOUR,
    LUNCH_HOURS,
    MAX_WAIT_HOURS,
    MORNING_LATEST_START,
    TIME_WINDOWS,
)
from app.main import app
from app.normalization import normalize_key
from app.patterns import learn_patterns, patterns_for
from app.schedule import clock, window_for, window_violation


def minutes(value: str) -> int:
    hours, mins = value.split(":")
    return int(hours) * 60 + int(mins)


@pytest.fixture(scope="module")
def jaipur(dataset):
    return build_itinerary(PlanRequest(cities=["Jaipur"], duration_days=3, pace="moderate"), dataset)


# --- the clock ------------------------------------------------------------------------
def test_every_day_has_a_schedule(jaipur):
    for day in jaipur["days"]:
        assert day["schedule"] is not None
        if day["sights"]:
            assert day["schedule"]["blocks"], f"day {day['day']} has sights but no blocks"


def test_blocks_run_in_order_without_overlap(jaipur):
    for day in jaipur["days"]:
        blocks = day["schedule"]["blocks"]
        for first, second in zip(blocks, blocks[1:]):
            assert minutes(first["end"]) <= minutes(second["start"]), f"day {day['day']} overlaps"
        for block in blocks:
            assert minutes(block["end"]) >= minutes(block["start"])


def test_schedule_covers_exactly_the_planned_sights(jaipur):
    for day in jaipur["days"]:
        scheduled = [block["sight"] for block in day["schedule"]["blocks"] if block["kind"] == "sight"]
        assert scheduled == [sight["sight"] for sight in day["sights"]]
        assert len(scheduled) == len(set(scheduled)), "a sight is scheduled twice"


def test_days_include_travel_and_a_lunch_break(jaipur):
    busy = max(jaipur["days"], key=lambda day: len(day["sights"]))
    kinds = {block["kind"] for block in busy["schedule"]["blocks"]}
    assert "travel" in kinds and "sight" in kinds
    assert "lunch" in kinds, "a full day should break for lunch"
    lunch = next(block for block in busy["schedule"]["blocks"] if block["kind"] == "lunch")
    assert lunch["hours"] == pytest.approx(LUNCH_HOURS)


def test_days_finish_within_the_realistic_window(dataset):
    for city, pace in [("Jaipur", "moderate"), ("Munnar", "active"), ("Leh", "relaxed")]:
        plan = build_itinerary(PlanRequest(cities=[city], duration_days=4, pace=pace), dataset)
        for day in plan["days"]:
            end = day["schedule"].get("ends")
            if end:
                assert minutes(end) <= LATEST_END_HOUR * 60, f"{city} day {day['day']} ends {end}"


def test_activity_window_matches_the_pace(dataset):
    """Sightseeing hours stay inside the pace; lunch and waiting sit on top of them."""
    plan = build_itinerary(PlanRequest(cities=["Jaipur"], duration_days=3, pace="moderate"), dataset)
    for day in plan["days"]:
        active = sum(block["hours"] for block in day["schedule"]["blocks"]
                     if block["kind"] in {"sight", "travel"})
        assert active <= day["available_hours"] + 0.3


# --- preferred times --------------------------------------------------------------------
def test_sunset_sights_are_not_scheduled_in_the_morning(dataset):
    """Hard slots are honoured, or the clash is reported -- never silently ignored."""
    for city in ["Jaipur", "Munnar", "Leh"]:
        plan = build_itinerary(PlanRequest(cities=[city], duration_days=4, pace="moderate"), dataset)
        for day in plan["days"]:
            flags = " ".join(day["schedule"]["flags"])
            for block in day["schedule"]["blocks"]:
                if block["kind"] != "sight":
                    continue
                sight = next((s for s in day["sights"] if s["sight"] == block["sight"]), None)
                if not sight or str(sight.get("best_time")).lower() not in HARD_TIME_SLOTS:
                    continue
                window = window_for(sight["best_time"])
                inside = minutes(block["start"]) >= (window[0] - 0.5) * 60
                assert inside or block["sight"] in flags, f"{block['sight']} at {block['start']} unflagged"


def test_waiting_for_a_sunset_slot_is_bounded(dataset):
    plan = build_itinerary(PlanRequest(cities=["Jaipur"], duration_days=4, pace="moderate"), dataset)
    for day in plan["days"]:
        for block in day["schedule"]["blocks"]:
            if block["kind"] == "wait":
                assert block["hours"] <= MAX_WAIT_HOURS + 1e-6


@pytest.mark.parametrize(
    "slot, start, end, expected",
    [
        ("sunset", 10.0, 11.0, True),       # a sunset point in the morning is wrong
        ("sunset", 17.0, 18.0, False),
        ("evening", 9.0, 10.0, True),
        ("morning", 10.0, 11.0, False),
        ("morning", 15.0, 16.0, True),      # past MORNING_LATEST_START
        ("any", 19.0, 20.0, False),
        ("", 9.0, 10.0, False),             # no preference recorded: nothing to violate
    ],
)
def test_window_violation_rules(slot, start, end, expected):
    assert bool(window_violation(slot, start, end)) is expected


def test_morning_is_soft_because_half_the_dataset_says_morning(dataset):
    """A full day of 'morning' sights must not produce a wall of blocking errors."""
    plan = build_itinerary(PlanRequest(cities=["Jaipur"], duration_days=4, pace="active"), dataset)
    assert plan["validation"]["passed"]
    assert not plan["validation"]["issues"]["schedule"], "time-of-day clashes are warnings, not blockers"
    assert MORNING_LATEST_START > TIME_WINDOWS["morning"][1]


def test_missing_best_time_is_flagged_not_invented(dataset):
    """A sight with no best_time is placed in day order and said to be unplaced."""
    plan = build_itinerary(PlanRequest(cities=["Jaipur"], duration_days=3), dataset)
    day = next(day for day in plan["days"] if day["sights"])
    broken = {**day, "sights": [{**day["sights"][0], "best_time": ""}]}
    from app.schedule import build_day_schedule

    schedule = build_day_schedule(broken, dataset)
    assert any("no best_time" in flag for flag in schedule.flags)


def test_clock_formatting():
    assert clock(0.5) == "00:30" and clock(13.25) == "13:15" and clock(9) == "09:00"


# --- learning from past itineraries -------------------------------------------------------
def test_patterns_are_mined_from_the_workbook(dataset):
    patterns = patterns_for(dataset)
    assert patterns.days_seen > 1000 and patterns.pair_counts
    assert patterns.typical_sights_per_day > 1


def test_past_itineraries_have_no_clock_times(dataset):
    """They record order only, so no time of day is ever taken from them."""
    patterns = patterns_for(dataset)
    assert patterns.has_clock_times is False
    plan = build_itinerary(PlanRequest(cities=["Leh"], duration_days=3, pace="moderate"), dataset)
    assert any("carry no clock times" in note for note in plan["validation"]["notes"])


def test_strong_ordering_pattern_is_detected(dataset):
    """Shanti Stupa before Leh Palace appears 27 times against 6."""
    patterns = patterns_for(dataset)
    assert patterns.prefers(normalize_key("Shanti Stupa"), normalize_key("Leh Palace"))
    assert not patterns.prefers(normalize_key("Leh Palace"), normalize_key("Shanti Stupa"))


def test_pattern_order_only_breaks_ties_within_a_slot(dataset):
    """Shanti Stupa is a sunset sight, so it stays late despite the agents' habit."""
    plan = build_itinerary(PlanRequest(cities=["Leh"], duration_days=4, pace="active"), dataset)
    for day in plan["days"]:
        names = [sight["sight"] for sight in day["sights"]]
        if "Shanti Stupa" in names and "Leh Palace" in names:
            assert names.index("Shanti Stupa") > names.index("Leh Palace")
    assert not any("Shanti Stupa" in note and "past itineraries" in note
                   for note in plan["validation"]["notes"]), "cross-slot pairs must not be reported"


def test_same_slot_disagreement_is_reported_as_a_note(dataset):
    plan = build_itinerary(PlanRequest(cities=["Munnar"], duration_days=4, pace="active"), dataset)
    notes = " ".join(plan["validation"]["notes"])
    assert "past itineraries usually visit" in notes
    assert plan["validation"]["passed"], "a different-but-shorter order is not a failure"


def test_learn_patterns_handles_a_workbook_without_itineraries(tmp_path):
    import pandas as pd

    path = tmp_path / "empty.xlsx"
    pd.DataFrame({"a": [1]}).to_excel(path, sheet_name="something", index=False)
    patterns = learn_patterns(path)
    assert patterns.days_seen == 0 and not patterns.pair_counts


# --- API ------------------------------------------------------------------------------------
def test_api_returns_the_schedule_and_what_it_learned():
    response = TestClient(app).post(
        "/generate-itinerary", json={"cities": ["Jaipur"], "duration_days": 3, "pace": "moderate"}
    )
    assert response.status_code == 200
    payload = response.json()
    schedule = payload["days"][1]["schedule"]
    assert schedule["starts"] and schedule["ends"] and schedule["blocks"]
    assert payload["summary"]["learned_from_past_itineraries"]["days_read"] > 0
    assert "best_time" in payload["summary"]["schedule_assumptions"]


# --- the day's totals are the schedule's totals -------------------------------------------
@pytest.mark.parametrize("city, days, pace", [
    ("Udaipur", 3, "moderate"),
    ("Udaipur", 5, "relaxed"),
    ("Jaipur", 4, "active"),
    ("Leh", 4, "moderate"),
])
def test_day_totals_match_the_clock(dataset, city, days, pace):
    """Every hour the header reports has to be an hour the timeline contains."""
    plan = build_itinerary(PlanRequest(cities=[city], duration_days=days, pace=pace), dataset)
    for day in plan["days"]:
        schedule = day["schedule"]
        totals = schedule["totals"]
        assert schedule["accounted_hours"] == pytest.approx(schedule["span_hours"], abs=0.02), \
            f"day {day['day']} has an unexplained gap"
        assert schedule["activity_hours"] == pytest.approx(
            totals["sightseeing"] + totals["local_travel"] + totals["stop_buffers"], abs=0.02)
        assert day["used_hours"] == pytest.approx(schedule["activity_hours"], abs=0.02)
        assert day["sightseeing_hours"] == pytest.approx(totals["sightseeing"], abs=0.02)
        assert day["buffer_hours"] == pytest.approx(totals["stop_buffers"], abs=0.02)
        assert day["local_travel_hours"] == pytest.approx(totals["local_travel"], abs=0.02)
    assert plan["validation"]["passed"], plan["validation"]["issues"]


def test_meals_are_counted_outside_the_activity_budget(dataset):
    """Lunch is part of the day's span, never part of the sightseeing allowance."""
    plan = build_itinerary(PlanRequest(cities=["Udaipur"], duration_days=3, pace="moderate"), dataset)
    busy = max(plan["days"], key=lambda day: len(day["sights"]))
    totals = busy["schedule"]["totals"]
    assert totals["meals_and_breaks"] >= LUNCH_HOURS
    assert busy["schedule"]["activity_hours"] <= busy["available_hours"] + 1e-6
    assert busy["schedule"]["span_hours"] >= busy["schedule"]["activity_hours"] + totals["meals_and_breaks"] - 0.02


def test_en_route_detour_is_the_days_own_travel_not_the_drive(dataset):
    """A stop taken during a transfer costs the day its detour, not the intercity hours."""
    plan = build_itinerary(PlanRequest(cities=["Leh", "Pangong Lake"], duration_days=5, pace="active"), dataset)
    day = next(day for day in plan["days"]
               if any(sight.get("en_route") for sight in day["sights"]))
    drive = day["transfer"]["drive_hours"]
    assert day["schedule"]["totals"]["intercity_travel"] == pytest.approx(drive, abs=0.02)
    assert day["schedule"]["totals"]["local_travel"] > 0, "the detour has to be charged somewhere"
    assert plan["validation"]["passed"], plan["validation"]["issues"]


# --- Udaipur: evening sights land in the evening ------------------------------------------
def test_a_day_of_evening_sights_starts_late(dataset):
    """Udaipur's lakes are evening sights: the day waits rather than seeing them at noon."""
    plan = build_itinerary(PlanRequest(cities=["Udaipur"], duration_days=5, pace="relaxed"), dataset)
    for day in plan["days"]:
        evening = [sight for sight in day["sights"]
                   if str(sight.get("best_time", "")).lower() == "evening"]
        if not evening:
            continue
        shifted = any("starts at" in note for note in day["schedule"]["notes"])
        offered = {entry["sight"] for entry in day["schedule"]["alternatives"]}
        for sight in evening:
            block = next(b for b in day["schedule"]["blocks"]
                         if b["kind"] == "sight" and b["sight"] == sight["sight"])
            in_window = minutes(block["start"]) >= TIME_WINDOWS["evening"][0] * 60 - 30
            assert in_window or shifted or sight["sight"] in offered, \
                f"{sight['sight']} at {block['start']} with nothing said about it"


def test_adding_an_evening_sight_schedules_it_in_the_evening(dataset):
    """The reported Udaipur defect: Fatehsagar Lake was added, then shown at 12:37."""
    from app.edits import SightEdit, apply_edits

    payload, outcomes = apply_edits(
        dataset,
        PlanRequest(cities=["Udaipur"], duration_days=3, pace="moderate"),
        [SightEdit(action="add", sight="Fatehsagar Lake", city="Udaipur")],
    )
    assert outcomes[0].applied
    day = next(day for day in payload["days"]
               if any(sight["sight"] == "Fatehsagar Lake" for sight in day["sights"]))
    block = next(b for b in day["schedule"]["blocks"] if b["sight"] == "Fatehsagar Lake")
    assert minutes(block["start"]) >= TIME_WINDOWS["evening"][0] * 60 - 30, block["start"]
    assert payload["validation"]["passed"], payload["validation"]["issues"]


def test_a_missed_hard_slot_always_comes_with_an_alternative(dataset):
    """A sight outside a hard slot is allowed; silence about it is not."""
    for city, days, pace in [("Udaipur", 5, "relaxed"), ("Leh", 4, "moderate"), ("Jaipur", 3, "relaxed")]:
        plan = build_itinerary(PlanRequest(cities=[city], duration_days=days, pace=pace), dataset)
        assert plan["validation"]["passed"], plan["validation"]["issues"]
        for day in plan["days"]:
            for entry in day["schedule"]["alternatives"]:
                assert entry["suggestion"], entry
                assert entry["sight"] and entry["kind"] == "slot"
                # Either a day is named, or the reason no day can take it.
                assert "move" in entry["suggestion"] or "no other day" in entry["suggestion"]


def test_an_unknown_pinned_sight_is_refused_not_dropped(dataset):
    from app.builder import PinnedSight, PlanningError

    with pytest.raises(PlanningError) as error:
        build_itinerary(
            PlanRequest(cities=["Udaipur"], duration_days=3,
                        pinned_sights=[PinnedSight("Rope Way/doodh talai", "Udaipur", 1)]),
            dataset,
        )
    assert "not a sight in Udaipur" in str(error.value)
    assert "Rope Way" in str(error.value), "the near miss should be suggested"


def test_api_exposes_the_reconciled_totals():
    """The response model must not quietly drop the numbers the UI reports."""
    response = TestClient(app).post(
        "/generate-itinerary", json={"cities": ["Udaipur"], "duration_days": 5, "pace": "relaxed"}
    )
    assert response.status_code == 200
    for day in response.json()["days"]:
        schedule = day["schedule"]
        assert schedule["totals"] is not None
        assert schedule["span_hours"] > 0 and schedule["accounted_hours"] > 0
        assert schedule["accounted_hours"] == pytest.approx(schedule["span_hours"], abs=0.02)
        assert day["day_totals"]["starts"] == schedule["starts"]
        for entry in schedule["alternatives"]:
            assert entry["suggestion"] and entry["sight"]


# --- the arrival day stays in town ---------------------------------------------------------
@pytest.mark.parametrize("city, days, pace", [
    ("Udaipur", 3, "moderate"),
    ("Jaipur", 4, "active"),
    ("Leh", 4, "moderate"),
    ("Jaisalmer", 4, "moderate"),
    ("Goa", 4, "active"),
    ("Jodhpur", 3, "moderate"),
])
def test_arrival_day_holds_nothing_distant(dataset, city, days, pace):
    from app.config import ARRIVAL_DAY_MAX_KM

    plan = build_itinerary(PlanRequest(cities=[city], duration_days=days, pace=pace), dataset)
    first = plan["days"][0]
    assert first["is_arrival_day"]
    for sight in first["sights"]:
        assert sight["distance_from_centre_km"] <= ARRIVAL_DAY_MAX_KM, \
            f"{sight['sight']} is {sight['distance_from_centre_km']} km out on the arrival day"
    assert plan["validation"]["passed"], plan["validation"]["issues"]


def test_a_sight_moved_off_day_one_is_not_lost(dataset):
    """A sight too far out for day 1 lands on a later day, or is offered -- never dropped.

    Jaisalmer's Kuldhara is the case: 30 km by straight-line estimate, 36 km by measured
    road, which makes it a day trip that a 4-day trip may have no free day for. Either way
    the traveller can still see it is there.
    """
    for city, days, pace in [("Jaisalmer", 4, "moderate"), ("Leh", 4, "moderate"), ("Jaipur", 4, "active")]:
        plan = build_itinerary(PlanRequest(cities=[city], duration_days=days, pace=pace), dataset)
        scheduled = {sight["sight"] for day in plan["days"] for sight in day["sights"]}
        offered = {label.split(" / ")[-1] for label in plan["unscheduled_sights"]}
        known = {str(name) for name in dataset.city_sights(normalize_key(city))["sight"]}
        missing = known - scheduled - offered
        assert not missing, f"{city}: {sorted(missing)} vanished from the plan entirely"
        first_day = {sight["sight"] for sight in plan["days"][0]["sights"]}
        assert not (first_day & offered), "a sight cannot be both on day 1 and unscheduled"


def test_the_arrival_day_rule_follows_the_agent_itineraries(dataset):
    """The limit is measured, not chosen: day 1 is close in, later days are not."""
    import pandas as pd

    from app.config import ARRIVAL_DAY_MAX_KM
    from app.geo import road_km
    from app.patterns import _parse_day_sequences

    centres = {str(row["city_key"]): (float(row["lat"]), float(row["lon"]))
               for _, row in dataset.cities.iterrows() if pd.notna(row.get("lat"))}
    km_by_key = {}
    for _, row in dataset.sights.iterrows():
        centre = centres.get(str(row["city_key"]))
        if centre is not None and pd.notna(row.get("lat")):
            km_by_key.setdefault(str(row["sight_key"]),
                                 road_km(centre[0], centre[1], float(row["lat"]), float(row["lon"])))

    trips = pd.read_excel(dataset.source, sheet_name="itineraries")
    first_day, later = [], []
    for value in trips["day_wise_sightseeing"].dropna():
        for index, names in enumerate(_parse_day_sequences(value), start=1):
            for name in names:
                if name in km_by_key:
                    (first_day if index == 1 else later).append(km_by_key[name])

    assert len(first_day) > 200 and len(later) > 1000
    within = sum(1 for km in first_day if km <= ARRIVAL_DAY_MAX_KM) / len(first_day)
    later_within = sum(1 for km in later if km <= ARRIVAL_DAY_MAX_KM) / len(later)
    assert within > 0.9, f"only {within:.0%} of day-1 stops are within {ARRIVAL_DAY_MAX_KM} km"
    assert within > later_within + 0.1, "day 1 should be clearly closer in than later days"


def test_a_pinned_distant_sight_beats_the_arrival_day_rule(dataset):
    """An explicit request wins over a learned habit -- but the plan says so."""
    from app.builder import PinnedSight

    plan = build_itinerary(
        PlanRequest(cities=["Jaisalmer"], duration_days=4, pace="moderate",
                    pinned_sights=[PinnedSight("Kuldhara", "Jaisalmer", 1)]),
        dataset,
    )
    first = plan["days"][0]
    assert any(sight["sight"] == "Kuldhara" for sight in first["sights"]), "the pin must be honoured"
    assert any("Kuldhara" in warning and "arrival day" in warning
               for warning in plan["validation"]["warnings"]), "the exception should be reported"
    assert plan["validation"]["passed"], "a pinned sight is a choice, not a failure"
