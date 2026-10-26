"""Finding a sight from what someone actually typed, and what to suggest unprompted.

The rule these tests protect: matching may be generous, but it may never invent. Every
result is a row in the dataset or one of the named generic activities, and each says how it
matched so the caller shows a list instead of guessing.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.builder import PlanRequest, build_itinerary
from app.lookup import GENERIC_ACTIVITIES, search, suggestions
from app.main import app


def names(results: list[dict]) -> list[str]:
    return [item["name"] for item in results]


# --- matching what was typed -------------------------------------------------------------------
@pytest.mark.parametrize("typed, expected", [
    ("City Palace", "City Palace"),          # exact
    ("city palace", "City Palace"),          # case
    ("bagore haveli", "Bagore ki Haveli"),   # words out of a longer name
    ("bagore havelli", "Bagore ki Haveli"),  # and a typo on top
    ("sajangarh", "Sajjangarh fort"),        # dropped letter
    ("city plce", "City Palace"),            # dropped vowel
    ("jagdsh", "Jagdish temple"),            # several dropped letters
    ("pichola", "Lake Pichola"),             # a word from the middle
    ("saheliyon", "Saheliyon Ki Bari"),      # first word only
])
def test_a_sight_is_found_from_what_people_type(dataset, typed, expected):
    found = search(dataset, typed, ["Udaipur"])
    assert expected in names(found), f"{typed!r} -> {names(found)}"


def test_the_best_match_comes_first(dataset):
    assert names(search(dataset, "city palace", ["Udaipur"]))[0] == "City Palace"
    assert names(search(dataset, "lake pichola", ["Udaipur"]))[0] == "Lake Pichola"


def test_nonsense_matches_nothing(dataset):
    """Generosity has a floor: a wrong guess is worse than an empty list."""
    for typed in ["xyzzy", "qqqqqq", "zzzzzzzz"]:
        assert search(dataset, typed, ["Udaipur"]) == [], typed


def test_a_short_word_does_not_match_by_sound(dataset):
    """'lake' once matched 'Bhartiya Lok Kala Mandal' on a two-consonant skeleton."""
    found = search(dataset, "lake", ["Udaipur"])
    assert "Lake Pichola" in names(found)
    assert not [item for item in found if item["how"] == "sounds-like"], names(found)


def test_every_result_says_how_it_matched(dataset):
    for item in search(dataset, "sajangarh", ["Udaipur"]):
        assert item["how"] in {"exact", "contains", "fuzzy", "sounds-like", "suggested"}
        assert 0 < item["score"] <= 1


def test_results_stay_inside_the_cities_asked_for(dataset):
    for item in search(dataset, "fort", ["Udaipur"]):
        if item["kind"] == "sight":
            assert item["city"] == "Udaipur", item


def test_generic_activities_are_offered_but_marked_as_such(dataset):
    found = search(dataset, "dinner", ["Udaipur"])
    dinner = next(item for item in found if item["name"] == "Dinner")
    assert dinner["kind"] == "activity"
    assert dinner["city"] is None, "an activity belongs to no city"


def test_a_sight_already_in_the_plan_says_which_day(dataset):
    plan = build_itinerary(PlanRequest(cities=["Udaipur"], duration_days=3, pace="moderate"), dataset)
    scheduled = next(sight["sight"] for day in plan["days"] for sight in day["sights"])
    found = search(dataset, scheduled, ["Udaipur"], plan)
    entry = next(item for item in found if item["name"] == scheduled)
    assert entry["scheduled_on"] in {day["day"] for day in plan["days"]}


def test_unscheduled_sights_rank_above_scheduled_ones(dataset):
    """The box is usually used to add something new."""
    plan = build_itinerary(PlanRequest(cities=["Udaipur"], duration_days=3, pace="moderate"), dataset)
    found = search(dataset, "lake", ["Udaipur"], plan, limit=10)
    days = [item["scheduled_on"] for item in found]
    assert days == sorted(days, key=lambda value: value is not None)


# --- suggesting unprompted -----------------------------------------------------------------------
def test_suggestions_lead_with_must_visits_that_did_not_fit(dataset):
    plan = build_itinerary(PlanRequest(cities=["Udaipur"], duration_days=3, pace="moderate"), dataset)
    advice = suggestions(dataset, ["Udaipur"], plan, day=2)
    assert advice["sights"], "there should be something left to suggest"
    scheduled = {sight["sight"] for day in plan["days"] for sight in day["sights"]}
    for item in advice["sights"]:
        assert item["name"] not in scheduled, "suggesting something already planned is noise"
    musts = [item["must_visit"] for item in advice["sights"]]
    assert musts == sorted(musts, reverse=True), "must-visits should come first"


def test_suggestions_include_the_generic_activities(dataset):
    advice = suggestions(dataset, ["Udaipur"], None, day=1)
    assert {item["name"] for item in advice["activities"]} == {a["name"] for a in GENERIC_ACTIVITIES}
    for item in advice["activities"]:
        assert item["duration_hours"] > 0


def test_a_held_back_sight_explains_itself_in_the_suggestions(dataset):
    """If the engine left something out, say why where the user is choosing."""
    plan = build_itinerary(PlanRequest(cities=["Udaipur"], duration_days=5, pace="relaxed"), dataset)
    held = plan.get("unplaceable_reasons") or {}
    if not held:
        pytest.skip("this plan placed everything")
    advice = suggestions(dataset, ["Udaipur"], plan)
    explained = {item["name"]: item["detail"] for item in advice["sights"] if item["detail"]}
    assert any(label.split(" / ")[-1] in explained for label in held), explained


# --- the endpoints ----------------------------------------------------------------------------------
def test_search_endpoint_tolerates_typos():
    response = TestClient(app).post("/itinerary/activity-search",
                                    json={"text": "sajangarh", "cities": ["Udaipur"]})
    assert response.status_code == 200
    payload = response.json()
    assert payload["query"] == "sajangarh"
    assert "Sajjangarh fort" in [m["name"] for m in payload["matches"]]


def test_search_endpoint_ranks_against_the_trip():
    response = TestClient(app).post("/itinerary/activity-search", json={
        "text": "lake",
        "base": {"cities": ["Udaipur"], "duration_days": 3, "pace": "moderate"},
    })
    assert response.status_code == 200
    for match in response.json()["matches"]:
        assert "scheduled_on" in match


def test_suggestions_endpoint_returns_both_kinds():
    response = TestClient(app).post("/itinerary/activity-suggestions", json={
        "base": {"cities": ["Udaipur"], "duration_days": 3, "pace": "moderate"},
        "day": 2,
    })
    assert response.status_code == 200
    payload = response.json()
    assert payload["sights"] and payload["activities"]
    assert payload["day"] == 2


def test_an_empty_query_is_refused_by_the_schema():
    response = TestClient(app).post("/itinerary/activity-search", json={"text": "", "cities": ["Udaipur"]})
    assert response.status_code == 422
