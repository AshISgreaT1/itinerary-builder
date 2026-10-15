"""End-to-end and unit tests for the itinerary engine."""

from __future__ import annotations

import copy

import pytest
from fastapi.testclient import TestClient

from app.builder import PlanningError, PlanRequest, build_itinerary
from app.clustering import cluster_city_sights
from app.config import MAX_DAY_ROUTE_KM, MAX_LEG_KM, MAX_SIGHTS_PER_DAY, PACE_ACTIVE_HOURS, TRANSFER_OVERHEAD_HOURS
from app.data_loader import parse_open_months
from app.main import app
from app.normalization import normalize_key
from app.validation import validate_itinerary

PACES = sorted(PACE_ACTIVE_HOURS)


# --- dataset ------------------------------------------------------------------------
def test_dataset_loads_and_is_deduplicated(dataset):
    assert len(dataset.cities) > 0 and len(dataset.sights) > 0
    keys = dataset.sights[["city_key", "sight_key"]].apply(tuple, axis=1)
    assert keys.is_unique, "canonical sights must be unique per city"


def test_alias_column_resolves_to_canonical_name(dataset):
    # The dataset lists "Ajmer-e-Sharief Dargah" as an alias of the Dargah row.
    assert dataset.dedup.resolve("Ajmer", "Ajmer-e-Sharief Dargah").lower().startswith("dargah")


@pytest.mark.parametrize(
    "value, expected_months",
    [
        ("all", set(range(1, 13))),
        ("5,6,7,8,9,10", {5, 6, 7, 8, 9, 10}),
        ("Apr-Oct", {4, 5, 6, 7, 8, 9, 10}),
        ("Oct-Mar", {10, 11, 12, 1, 2, 3}),  # wraps the new year
        ("nonsense", set(range(1, 13))),  # never hides a sight
    ],
)
def test_open_months_parsing(value, expected_months):
    assert set(parse_open_months(value)) == expected_months


# --- clustering ---------------------------------------------------------------------
def test_clustering_is_deterministic_and_bounded(dataset):
    sights = dataset.city_sights(normalize_key("Jaipur"))
    centre = dataset.city_row("Jaipur")
    first = cluster_city_sights(sights, 3, centre["lat"], centre["lon"])
    second = cluster_city_sights(sights, 3, centre["lat"], centre["lon"])
    assert first.count <= 3
    assert [c["sight"].tolist() for c in first.clusters] == [c["sight"].tolist() for c in second.clusters]


def test_clustering_handles_fewer_sights_than_days(dataset):
    sights = dataset.city_sights(normalize_key("Manikaran"))
    centre = dataset.city_row("Manikaran")
    result = cluster_city_sights(sights, 5, centre["lat"], centre["lon"])
    assert result.count <= max(1, len(sights))


# --- planning -----------------------------------------------------------------------
@pytest.fixture(scope="session")
def plan(dataset):
    return build_itinerary(
        PlanRequest(cities=["Manali", "Kasol"], duration_days=5, pace="moderate", origin_city="Delhi"), dataset
    )


def test_plan_shape(plan):
    assert len(plan["days"]) == 5
    assert [day["day"] for day in plan["days"]] == [1, 2, 3, 4, 5]
    assert plan["validation"]["passed"], plan["validation"]["issues"]


@pytest.mark.parametrize("pace", PACES)
def test_days_never_exceed_pace_budget(dataset, pace):
    plan = build_itinerary(PlanRequest(cities=["Manali", "Kasol"], duration_days=5, pace=pace), dataset)
    for day in plan["days"]:
        assert day["used_hours"] <= day["available_hours"] + 1e-6
        assert len(day["sights"]) <= MAX_SIGHTS_PER_DAY[pace]


def test_no_sight_is_scheduled_twice(plan):
    names = [sight["sight"] for day in plan["days"] for sight in day["sights"]]
    assert len(names) == len(set(names))


def test_days_stay_spatially_tight(plan):
    for day in plan["days"]:
        assert day["max_leg_km"] <= MAX_LEG_KM
        assert day["route_km"] <= MAX_DAY_ROUTE_KM


def test_transfer_day_subtracts_drive_hours(plan):
    transfer_days = [day for day in plan["days"] if day["transfer"]]
    assert transfer_days, "Delhi -> Manali should produce at least one transfer day"
    for day in transfer_days:
        expected = PACE_ACTIVE_HOURS["moderate"] - day["transfer"]["drive_hours"] - TRANSFER_OVERHEAD_HOURS
        assert day["available_hours"] <= max(0.0, expected) + 1e-6


def test_requires_one_day_per_city(dataset):
    with pytest.raises(PlanningError):
        build_itinerary(PlanRequest(cities=["Manali", "Kasol", "Manikaran"], duration_days=2), dataset)


def test_unknown_city_is_rejected(dataset):
    with pytest.raises(PlanningError):
        build_itinerary(PlanRequest(cities=["Atlantis"], duration_days=3), dataset)


def test_seasonal_closure_removes_sights(dataset):
    """Rohtang Pass is open May-Oct only, so a January trip must not schedule it."""
    winter = build_itinerary(PlanRequest(cities=["Manali"], duration_days=4, pace="active", start_month=1), dataset)
    scheduled = {sight["sight"] for day in winter["days"] for sight in day["sights"]}
    assert "Rohtang Pass" not in scheduled
    assert any("Rohtang" in entry for entry in winter["closed_this_month"])


def test_descriptions_are_joined(plan):
    described = [s for day in plan["days"] for s in day["sights"] if s.get("description")]
    assert described, "at least some sights should carry their description text"


# --- validator (fault injection: it must fail when the plan is broken) ---------------
def _first_day_with_sights(payload):
    return next(day for day in payload["days"] if day["sights"])


def test_validator_flags_duplicate_sight(dataset, plan):
    broken = copy.deepcopy(plan)
    day = _first_day_with_sights(broken)
    other = next(d for d in broken["days"] if d is not day and d["city"] == day["city"])
    other["sights"].append(copy.deepcopy(day["sights"][0]))
    report = validate_itinerary(broken, dataset, "moderate")
    assert report.issues["duplicates_and_normalisation"]


def test_validator_flags_far_apart_sights(dataset, plan):
    broken = copy.deepcopy(plan)
    day = _first_day_with_sights(broken)
    far = copy.deepcopy(day["sights"][0])
    far["sight"] = "Far Away Viewpoint"
    far["lat"] += 1.5  # ~165 km north
    day["sights"].append(far)
    report = validate_itinerary(broken, dataset, "moderate")
    assert report.issues["geographic_distance"]
    assert report.issues["destination_and_logic"], "a sight 165 km out is also a wrong-city signal"


def test_validator_flags_overloaded_day(dataset, plan):
    """Pile real sights onto a day until it cannot fit, and the validator must say so."""
    broken = copy.deepcopy(plan)
    day = _first_day_with_sights(broken)
    template = day["sights"][0]
    for index in range(6):
        extra = copy.deepcopy(template)
        extra["sight"] = f"Extra Stop {index}"
        extra["duration_hours"] = 2.0
        extra["lat"] += 0.002 * (index + 1)
        day["sights"].append(extra)
    report = validate_itinerary(broken, dataset, "moderate")
    assert report.issues["time_and_activity_load"]
    assert not report.passed


def test_validator_recomputes_reported_metrics(dataset, plan):
    """Reported distances are not trusted: tampering with them is caught."""
    broken = copy.deepcopy(plan)
    day = _first_day_with_sights(broken)
    day["route_km"] = day["route_km"] + 25
    report = validate_itinerary(broken, dataset, "moderate")
    assert report.issues["route_data"]
    assert not report.passed


def test_validator_flags_sight_from_another_city(dataset, plan):
    broken = copy.deepcopy(plan)
    day = _first_day_with_sights(broken)
    foreign = dataset.sights[dataset.sights["city_key"] == normalize_key("Jaipur")].iloc[0]
    day["sights"].append(
        {
            "sight": foreign["sight"], "visit_type": "", "duration_hours": 1.0, "buffer_hours": 0.3,
            "must_see_tier": 1, "must_visit": True, "priority": 1, "best_time": "any", "zone": "",
            "lat": float(foreign["lat"]), "lon": float(foreign["lon"]), "is_excursion": False,
            "distance_from_centre_km": 0.0,
        }
    )
    report = validate_itinerary(broken, dataset, "moderate")
    assert report.issues["destination_and_logic"]


# --- API ----------------------------------------------------------------------------
@pytest.fixture(scope="session")
def client():
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["cities"] > 0


def test_generate_endpoint(client):
    response = client.post(
        "/generate-itinerary",
        json={"cities": ["Manali", "Kasol"], "duration_days": 5, "pace": "moderate", "origin_city": "Delhi"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["validation"]["passed"]
    assert len(body["days"]) == 5


def test_generate_endpoint_rejects_unknown_city(client):
    response = client.post("/generate-itinerary", json={"cities": ["Atlantis"], "duration_days": 3})
    assert response.status_code == 422


def test_generate_endpoint_validates_pace(client):
    response = client.post("/generate-itinerary", json={"cities": ["Manali"], "duration_days": 3, "pace": "sprint"})
    assert response.status_code == 422


def test_city_sights_endpoint(client):
    assert client.get("/cities/Manali/sights").status_code == 200
    assert client.get("/cities/Atlantis/sights").status_code == 404


def test_every_city_gets_usable_time(dataset):
    """A city must never end up with a single day that travel swallows whole."""
    plan = build_itinerary(
        PlanRequest(cities=["Cochin", "Munnar", "Thekkady"], duration_days=6, pace="moderate"), dataset
    )
    hours_by_city: dict[str, float] = {}
    for day in plan["days"]:
        hours_by_city[day["city"]] = hours_by_city.get(day["city"], 0.0) + day["available_hours"]
    assert all(hours > 1.0 for hours in hours_by_city.values()), hours_by_city
    assert all(any(d["sights"] for d in plan["days"] if d["city"] == city) for city in hours_by_city)


def test_cli_runs(capsys, dataset):
    from app.cli import main

    assert main(["--cities", "Manali", "--days", "3", "--dataset", str(dataset.source)]) == 0
    assert "Feasibility:" in capsys.readouterr().out


# --- travel-aware scheduling, departure, nights, seasonality (the six fixes) ---------
def test_missing_route_data_fails_instead_of_inventing_a_drive(dataset):
    """Delhi to Goa has no row in city_distances, so no journey time may be invented."""
    plan = build_itinerary(
        PlanRequest(cities=["Goa"], duration_days=4, start_month=12, origin_city="Delhi"), dataset
    )
    leg = plan["route"][0]
    assert leg["drive_hours"] is None and leg["data_gap"]
    assert plan["days"][0]["available_hours"] == 0.0 and not plan["days"][0]["sights"]
    assert not plan["validation"]["passed"]
    assert plan["validation"]["issues"]["travel_feasibility"]
    assert any("city_distances" in entry["fix"] for entry in plan["validation"]["entries"])


def test_overnight_leg_is_a_warning_not_a_failure(dataset):
    """Delhi to Manali is 12h: impossible as a day drive, fine as an overnight bus."""
    plan = build_itinerary(PlanRequest(cities=["Manali"], duration_days=4, origin_city="Delhi"), dataset)
    assert plan["route"][0]["is_overnight"]
    assert plan["days"][0]["available_hours"] == 0.0
    assert plan["validation"]["passed"]
    assert any("overnight" in warning for warning in plan["validation"]["warnings"])


def test_long_journey_never_leaves_a_day_looking_free(dataset):
    plan = build_itinerary(PlanRequest(cities=["Goa"], duration_days=4, origin_city="Delhi"), dataset)
    travel_day = plan["days"][0]
    assert travel_day["is_travel_only"] and travel_day["used_hours"] == 0.0
    assert travel_day["available_hours"] == 0.0, "a day consumed by travel must offer no hours"


def test_departure_day_reserves_checkout_and_transfer(dataset):
    """Goa's airport distance is in the dataset, so the last day pays for the transfer."""
    plan = build_itinerary(PlanRequest(cities=["Goa"], duration_days=4, start_month=12), dataset)
    last = plan["days"][-1]
    assert last["is_departure_day"]
    assert last["time_budget"]["departure_cost_hours"] >= 2.5
    assert last["available_hours"] < last["time_budget"]["pace_allowance"]
    assert last["used_hours"] <= last["available_hours"] + 1e-6
    assert "airport" in plan["summary"]["departure_day_reserved_for"]


def test_departure_day_overrun_is_rejected(dataset):
    plan = build_itinerary(PlanRequest(cities=["Goa"], duration_days=4, start_month=12), dataset)
    broken = copy.deepcopy(plan)
    broken["days"][-1]["used_hours"] = broken["days"][-1]["available_hours"] + 2
    report = validate_itinerary(broken, dataset, "moderate")
    assert not report.passed
    assert report.issues["departure_feasibility"] or report.issues["route_data"]


def test_hotel_nights_match_trip_length(dataset):
    plan = build_itinerary(PlanRequest(cities=["Manali", "Kasol"], duration_days=5), dataset)
    assert plan["summary"]["total_nights"] == 4, plan["summary"]["nights_per_city"]
    assert sum(plan["summary"]["days_per_city"].values()) == 5
    broken = copy.deepcopy(plan)
    broken["summary"]["nights_per_city"]["Kasol"] += 2
    assert not validate_itinerary(broken, dataset, "moderate").passed


@pytest.mark.parametrize("month", [6, 9])
def test_goa_monsoon_months_are_a_note_not_a_failure(dataset, month):
    """Goa's Oct-Mar window is its peak season, not a closure."""
    plan = build_itinerary(PlanRequest(cities=["Goa"], duration_days=4, start_month=month), dataset)
    assert plan["validation"]["passed"]
    assert not plan["validation"]["issues"]["seasonality"]
    assert any("season window" in note for note in plan["validation"]["notes"])
    assert plan["summary"]["total_sights"] > 0


def test_goa_december_itinerary_is_feasible(dataset):
    plan = build_itinerary(PlanRequest(cities=["Goa"], duration_days=4, start_month=12), dataset)
    assert plan["validation"]["passed"]
    assert plan["summary"]["total_sights"] >= 8
    assert all(day["used_hours"] <= day["available_hours"] + 1e-6 for day in plan["days"])


def test_attraction_closed_in_month_still_fails(dataset):
    """Sight-level closures remain blocking: Rohtang Pass is shut in January."""
    plan = build_itinerary(PlanRequest(cities=["Manali"], duration_days=4, pace="active", start_month=1), dataset)
    broken = copy.deepcopy(plan)
    rohtang = dataset.sights[dataset.sights["sight_key"] == normalize_key("Rohtang Pass")].iloc[0]
    broken["days"][1]["sights"].append(
        {
            "sight": rohtang["sight"], "visit_type": "", "duration_hours": 5.0, "buffer_hours": 0.3,
            "must_see_tier": 2, "must_visit": True, "priority": 3, "best_time": "morning", "zone": "",
            "lat": float(rohtang["lat"]), "lon": float(rohtang["lon"]), "is_excursion": False,
            "distance_from_centre_km": 0.0,
        }
    )
    report = validate_itinerary(broken, dataset, "active", month=1)
    assert report.issues["seasonality"]
    assert not report.passed


def test_strict_gate_passes_only_without_critical_issues(dataset, plan):
    assert plan["validation"]["passed"]
    broken = copy.deepcopy(plan)
    day = _first_day_with_sights(broken)
    day["sights"].append(copy.deepcopy(day["sights"][0]))  # duplicate = critical
    report = validate_itinerary(broken, dataset, "moderate")
    assert not report.passed and report.as_dict()["critical_count"] >= 1
    assert all(entry["fix"] for entry in report.as_dict()["entries"] if entry["severity"] == "critical")
