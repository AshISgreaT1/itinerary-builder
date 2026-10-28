"""Who is travelling, what they came for, and the free time left over.

The line these tests hold: personalisation chooses between sights the planner already
considers equal, and optional activities stay outside the itinerary in every sense -- not
scheduled, not counted, not priced. A plan with no preferences must be byte-identical to
one built before any of this existed.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app import personalise, preferences
from app.builder import PlanRequest, build_itinerary
from app.config import EXPLORE_MAX_OPTIONS, EXPLORE_MIN_WINDOW_HOURS
from app.customize import Operation, apply_operations
from app.freetime import free_windows
from app.main import app
from app.normalization import normalize_key


@pytest.fixture
def store(tmp_path, monkeypatch):
    path = tmp_path / "customers.json"
    monkeypatch.setattr(preferences, "STORE_PATH", path)
    return path


# --- the customer store ---------------------------------------------------------------------
def test_preferences_round_trip(store):
    saved = preferences.save(preferences.CustomerPreferences(
        customer_id="c1", name="R. Mehta", group_type="family",
        interests=["cultural", "nature"], pace="relaxed"), store)
    assert saved.updated_at
    found = preferences.get("c1", store)
    assert found.name == "R. Mehta" and found.group_type == "family"
    assert found.interests == ["cultural", "nature"]


def test_a_new_customer_is_simply_absent(store):
    assert preferences.get("nobody", store) is None


def test_values_the_planner_cannot_act_on_are_dropped(store):
    """A preference that looks stored but does nothing is worse than none."""
    saved = preferences.save(preferences.CustomerPreferences(
        customer_id="c2", group_type="astronauts",
        interests=["cultural", "unicorns", "CULTURAL"], pace="frantic"), store)
    assert saved.group_type == ""
    assert saved.interests == ["cultural"], "unknown dropped, duplicates collapsed, case folded"
    assert saved.pace == ""


def test_preferences_can_be_updated_without_losing_the_rest(store):
    preferences.save(preferences.CustomerPreferences(
        customer_id="c3", name="A. Shah", group_type="couple", interests=["nature"]), store)
    changed = preferences.update("c3", store, interests=["adventure"])
    assert changed.interests == ["adventure"]
    assert changed.name == "A. Shah" and changed.group_type == "couple"


def test_updating_someone_unknown_creates_them(store):
    created = preferences.update("new-person", store, group_type="solo")
    assert created.group_type == "solo" and created.customer_id == "new-person"


def test_trips_are_counted_so_reuse_is_visible(store):
    preferences.save(preferences.CustomerPreferences(customer_id="c4", group_type="solo"), store)
    preferences.record_trip("c4", store)
    preferences.record_trip("c4", store)
    assert preferences.get("c4", store).trips == 2


def test_a_corrupt_store_does_not_take_the_engine_down(store):
    store.write_text("{not json", encoding="utf-8")
    assert preferences.get("anyone", store) is None
    preferences.save(preferences.CustomerPreferences(customer_id="c5"), store)
    assert json.loads(store.read_text(encoding="utf-8"))["c5"]["customer_id"] == "c5"


# --- the matching ----------------------------------------------------------------------------------
def test_interests_map_onto_the_datasets_own_categories(dataset):
    """Nothing here may be vocabulary the sightseeing sheet does not use."""
    known = {str(value).strip().lower()
             for value in dataset.sights["category"].dropna().unique() if str(value).strip()}
    for interest, categories in personalise.INTEREST_CATEGORIES.items():
        assert categories, interest
        for category in categories:
            assert category in known, f"{interest} -> '{category}' is not a dataset category"


def test_group_leanings_only_name_real_categories(dataset):
    known = {str(v).strip().lower() for v in dataset.sights["category"].dropna().unique()}
    for group, leanings in personalise.GROUP_LEANINGS.items():
        for category in leanings:
            assert category in known, f"{group} -> '{category}'"


def test_a_stated_interest_outweighs_a_group_tendency():
    """One is what the customer said; the other is what people like them tend to do."""
    stated = personalise.score_category("adventure", ["adventure"], "family")
    inferred = personalise.score_category("heritage", [], "family")
    assert stated > inferred > 0


def test_nothing_known_means_no_opinion():
    assert personalise.score_category("heritage", [], "") == 0.0
    assert personalise.score_category("", ["cultural"], "family") == 0.0


def test_the_reason_is_in_the_customers_words():
    why = personalise.why_it_suits("adventure", ["adventure"], "friends")
    assert "interest in adventure" in why and "friends" in why


# --- the plan ------------------------------------------------------------------------------------------
def test_no_preferences_means_the_plan_is_unchanged(dataset):
    """The whole feature has to be invisible when nobody has asked for anything."""
    plain = build_itinerary(PlanRequest(cities=["Jaipur"], duration_days=3, pace="moderate"), dataset)
    empty = build_itinerary(PlanRequest(cities=["Jaipur"], duration_days=3, pace="moderate",
                                        group_type="", interests=[]), dataset)
    assert [[s["sight"] for s in day["sights"]] for day in plain["days"]] == \
           [[s["sight"] for s in day["sights"]] for day in empty["days"]]


def test_interests_change_which_sights_are_chosen(dataset):
    """Where sights compete, the ones that suit the customer win."""
    base = build_itinerary(PlanRequest(cities=["Jaipur"], duration_days=3, pace="moderate"), dataset)
    tuned = build_itinerary(PlanRequest(cities=["Jaipur"], duration_days=3, pace="moderate",
                                        group_type="friends",
                                        interests=["adventure", "nature"]), dataset)
    before = {s["sight"] for day in base["days"] for s in day["sights"]}
    after = {s["sight"] for day in tuned["days"] for s in day["sights"]}
    assert after != before, "nothing changed for a customer who asked for something"
    assert tuned["validation"]["passed"], tuned["validation"]["issues"]


def test_personalising_never_drops_a_must_visit(dataset):
    for group, interests in [("family", ["cultural"]), ("friends", ["adventure"]),
                             ("solo", ["shopping"]), ("couple", ["nature", "water"])]:
        base = build_itinerary(PlanRequest(cities=["Udaipur"], duration_days=4), dataset)
        tuned = build_itinerary(PlanRequest(cities=["Udaipur"], duration_days=4,
                                            group_type=group, interests=interests), dataset)
        musts = {s["sight"] for day in base["days"] for s in day["sights"] if s.get("must_visit")}
        after = {s["sight"] for day in tuned["days"] for s in day["sights"]}
        assert musts <= after, f"{group}/{interests} lost {musts - after}"
        assert tuned["validation"]["passed"]


def test_group_type_still_does_not_change_the_pace(dataset):
    """It predicts what people visit, not how fast -- and the code must keep that straight."""
    counts = set()
    for group in ["solo", "couple", "friends", "family"]:
        plan = build_itinerary(PlanRequest(cities=["Udaipur"], duration_days=4,
                                           group_type=group), dataset)
        counts.add(plan["days"][1]["available_hours"])
    assert len(counts) == 1, f"group type moved the day's budget: {counts}"


# --- free time -------------------------------------------------------------------------------------------
@pytest.mark.parametrize("city, days, pace", [
    ("Udaipur", 4, "moderate"), ("Jaipur", 3, "relaxed"), ("Goa", 4, "moderate"),
])
def test_free_windows_are_what_is_genuinely_left(dataset, city, days, pace):
    """A window may not overlap anything the day has already committed to."""
    plan = build_itinerary(PlanRequest(cities=[city], duration_days=days, pace=pace), dataset)
    for day in plan["days"]:
        busy = [b for b in (day["schedule"] or {}).get("blocks", [])
                if b["kind"] not in {"free", "wait"}]
        for window in day["free_windows"]:
            assert window["hours"] >= EXPLORE_MIN_WINDOW_HOURS
            start, end = _mins(window["starts"]), _mins(window["ends"])
            assert end > start
            for block in busy:
                assert not (_mins(block["start"]) < end and _mins(block["end"]) > start), \
                    f"{window} overlaps {block['label']}"


def _mins(text: str) -> int:
    hours, _, minutes = str(text).split(" ")[0].partition(":")
    return int(hours) * 60 + int(minutes or 0)


def test_a_departure_day_is_not_offered_an_evening(dataset):
    plan = build_itinerary(PlanRequest(cities=["Udaipur"], duration_days=4), dataset)
    last = plan["days"][-1]
    assert last["is_departure_day"]
    assert not [w for w in last["free_windows"] if w["when"] == "after the day's plan"]


def test_free_windows_survive_a_day_with_nothing_on_it(dataset):
    plan = build_itinerary(PlanRequest(cities=["Leh", "Pangong Lake"], duration_days=5,
                                       pace="active"), dataset)
    for day in plan["days"]:
        assert isinstance(day["free_windows"], list)
        assert isinstance(day["explore_on_your_own"], list)


# --- optional activities -------------------------------------------------------------------------------------
@pytest.mark.parametrize("city, days", [("Udaipur", 4), ("Jaipur", 4), ("Goa", 4)])
def test_optional_activities_are_offered_and_bounded(dataset, city, days):
    plan = build_itinerary(PlanRequest(cities=[city], duration_days=days,
                                       group_type="couple", interests=["nature"]), dataset)
    for day in plan["days"]:
        assert len(day["explore_on_your_own"]) <= EXPLORE_MAX_OPTIONS


def test_an_option_is_never_something_already_in_the_trip(dataset):
    plan = build_itinerary(PlanRequest(cities=["Udaipur"], duration_days=4,
                                       group_type="family", interests=["cultural"]), dataset)
    planned = {normalize_key(s["sight"]) for day in plan["days"] for s in day["sights"]}
    offered = [o["sight"] for day in plan["days"] for o in day["explore_on_your_own"]]
    assert not (planned & {normalize_key(name) for name in offered})
    assert len(offered) == len(set(offered)), f"the same thing offered twice: {offered}"


def test_an_option_fits_its_window_including_the_travel(dataset):
    """An offer the day cannot keep is worse than no offer."""
    for city in ["Udaipur", "Jaipur", "Goa"]:
        plan = build_itinerary(PlanRequest(cities=[city], duration_days=4,
                                           interests=["cultural", "nature"]), dataset)
        for day in plan["days"]:
            for option in day["explore_on_your_own"]:
                assert option["total_hours"] <= option["window_hours"] + 1e-6, option
                assert option["travel_hours"] <= option["window_hours"] * 0.5 + 1e-6, option


def test_options_are_outside_the_itinerary_in_every_sense(dataset):
    """Not in the stops, not in the hours, not in the validator's budgets, not priced."""
    plan = build_itinerary(PlanRequest(cities=["Udaipur"], duration_days=4,
                                       group_type="couple", interests=["nature"]), dataset)
    assert plan["validation"]["passed"], plan["validation"]["issues"]
    for day in plan["days"]:
        names = {s["sight"] for s in day["sights"]}
        for option in day["explore_on_your_own"]:
            assert option["sight"] not in names
            assert option["in_itinerary"] is False
            assert option["price"] is None and "not priced" in option["price_note"]
        # The day's own numbers must not have moved because something was offered.
        assert day["used_hours"] == pytest.approx(day["schedule"]["activity_hours"], abs=0.02)


def test_the_summary_says_what_was_personalised(dataset):
    plan = build_itinerary(PlanRequest(cities=["Udaipur"], duration_days=4,
                                       group_type="family", interests=["cultural"]), dataset)
    note = plan["summary"]["personalised_for"]
    assert note["group_type"] == "family" and note["interests"] == ["cultural"]
    assert "not priced" in note["note"]


def test_taking_an_option_moves_it_into_the_plan(dataset):
    plan = build_itinerary(PlanRequest(cities=["Udaipur"], duration_days=4,
                                       group_type="couple", interests=["nature"]), dataset)
    day = next(d for d in plan["days"] if d["explore_on_your_own"])
    option = day["explore_on_your_own"][0]
    taken, outcomes = apply_operations(
        dataset,
        PlanRequest(cities=["Udaipur"], duration_days=4, group_type="couple", interests=["nature"]),
        [Operation(operation="pin", sight=option["sight"], city=option["city"], day=day["day"])])
    assert outcomes[0].applied, outcomes[0].message
    assert any(s["sight"] == option["sight"] for d in taken["days"] for s in d["sights"])
    assert not any(o["sight"] == option["sight"]
                   for d in taken["days"] for o in d["explore_on_your_own"]), \
        "something already taken must stop being offered"


# --- the API -------------------------------------------------------------------------------------------------
def test_preference_endpoints(store):
    client = TestClient(app)
    assert client.get("/customers/ghost/preferences").status_code == 404

    saved = client.put("/customers/c9/preferences",
                       json={"name": "T", "group_type": "friends",
                             "interests": ["adventure", "nonsense"]})
    assert saved.status_code == 200
    assert saved.json()["interests"] == ["adventure"]

    assert client.get("/customers/c9/preferences").json()["group_type"] == "friends"
    assert any(row["customer_id"] == "c9" for row in client.get("/customers").json()["customers"])

    options = client.get("/preferences/options").json()
    assert set(options["group_types"]) == set(preferences.GROUP_TYPES)
    assert {item["interest"] for item in options["interests"]} == set(personalise.INTEREST_CATEGORIES)


def test_a_plan_reuses_stored_preferences(store):
    client = TestClient(app)
    client.put("/customers/c10/preferences", json={"group_type": "family", "interests": ["cultural"]})
    payload = client.post("/generate-itinerary",
                          json={"cities": ["Jaipur"], "duration_days": 3,
                                "customer_id": "c10"}).json()
    assert payload["summary"]["personalised_for"]["group_type"] == "family"
    assert payload["summary"]["personalised_for"]["interests"] == ["cultural"]
    assert client.get("/customers/c10/preferences").json()["trips"] >= 1


def test_the_request_wins_over_what_was_stored(store):
    """This trip may not be like the last one."""
    client = TestClient(app)
    client.put("/customers/c11/preferences", json={"group_type": "family", "interests": ["cultural"]})
    payload = client.post("/generate-itinerary",
                          json={"cities": ["Jaipur"], "duration_days": 3, "customer_id": "c11",
                                "group_type": "friends", "interests": ["adventure"]}).json()
    assert payload["summary"]["personalised_for"]["group_type"] == "friends"
    assert payload["summary"]["personalised_for"]["interests"] == ["adventure"]


def test_the_api_does_not_strip_the_new_keys():
    payload = TestClient(app).post(
        "/generate-itinerary",
        json={"cities": ["Udaipur"], "duration_days": 4, "group_type": "couple",
              "interests": ["nature"]}).json()
    for day in payload["days"]:
        assert "free_windows" in day and "explore_on_your_own" in day
