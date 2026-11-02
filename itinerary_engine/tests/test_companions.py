"""Cities suggested alongside the one being searched.

The point of these tests: the suggestion has to come from what the agents actually sold, the
count attached to it has to be true, and nothing unplannable may be offered.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fastapi.testclient import TestClient

from app import companions
from app.companions import CityPairings, learn_pairings, pairings_for, suggest
from app.normalization import normalize_key


@pytest.fixture
def client():
    from app.main import app

    return TestClient(app)


# --- what the sheet says ------------------------------------------------------------------
def test_pairings_are_mined_from_the_destination_column(dataset):
    pairings = pairings_for(dataset)
    assert pairings.trips_seen > 150, "the workbook holds 196 itineraries"
    assert pairings.sold_with_all(["leh", "pangong lake"]) > 30


def test_a_city_the_cities_sheet_does_not_carry_is_ignored():
    """Agra appears in four destinations and has no city row, so it cannot be suggested."""
    keys = companions._destination_keys("Delhi + Agra + Jaipur", {"delhi", "jaipur"})
    assert keys == ["delhi", "jaipur"]


def test_a_repeated_city_is_counted_once():
    keys = companions._destination_keys("Cochin + Munnar + Cochin", {"cochin", "munnar"})
    assert keys == ["cochin", "munnar"]


def test_a_workbook_without_the_sheet_has_no_opinion(tmp_path):
    book = tmp_path / "empty.xlsx"
    pd.DataFrame({"city": ["Leh"]}).to_excel(book, sheet_name="cities", index=False)
    pairings = learn_pairings(book, frozenset({"leh"}))
    assert pairings.trips_seen == 0 and pairings.partners(["leh"]) == []


# --- the counts have to be true -----------------------------------------------------------
def test_the_count_is_the_real_combination_not_a_sum_of_pairs():
    """Summing pairwise counts would claim more evidence than the sheet holds."""
    pairings = CityPairings(trip_sets=[
        frozenset({"leh", "pangong lake"}),
        frozenset({"leh", "nubra valley"}),
        frozenset({"leh", "pangong lake", "nubra valley"}),
    ], trips_seen=3)
    # Nubra shares 2 trips with Leh and 1 with Pangong Lake, but only 1 with both.
    assert pairings.sold_with_all(["leh", "nubra valley"]) == 2
    assert pairings.sold_with_all(["leh", "pangong lake", "nubra valley"]) == 1
    assert dict(pairings.partners(["leh", "pangong lake"])) == {"nubra valley": 1}


def test_the_reported_count_matches_the_sheet(dataset):
    pairings = pairings_for(dataset)
    for item in suggest(dataset, ["Leh"]):
        if item.basis != "sold_together":
            continue
        real = pairings.sold_with_all(["leh", normalize_key(item.city)])
        assert item.trips == real, f"{item.city} claims {item.trips}, the sheet says {real}"
        assert str(item.trips) in item.detail


# --- the suggestions themselves -----------------------------------------------------------
def test_searching_leh_offers_pangong_lake_first(dataset):
    found = suggest(dataset, ["Leh"])
    assert found[0].city == "Pangong Lake"
    assert found[0].basis == "sold_together"
    assert {item.city for item in found[:3]} == {"Pangong Lake", "Nubra Valley", "Alchi"}


def test_a_chosen_city_is_never_offered_again(dataset):
    found = suggest(dataset, ["Leh", "Pangong Lake"])
    names = {item.city for item in found}
    assert "Leh" not in names and "Pangong Lake" not in names
    assert "Nubra Valley" in names, "the obvious third city should rise once two are picked"


def test_everything_offered_can_actually_be_planned(dataset):
    """Agents sell Leh with Shey 16 times, but Shey has no sightseeing rows: an empty day."""
    with_sights = set(dataset.sights["city_key"])
    for city in ["Leh", "Jaipur", "Cochin", "Manali", "Udaipur", "Jodhpur"]:
        for item in suggest(dataset, [city]):
            assert normalize_key(item.city) in with_sights, f"{item.city} has no sights"


def test_proximity_fills_in_where_no_itinerary_pairs_them(dataset):
    """Manali appears in three packages, too few to pair; Kasol is 3h away by road."""
    found = suggest(dataset, ["Manali"])
    assert found, "a city with no pairings should still suggest a reachable neighbour"
    assert any(item.city == "Kasol" and item.basis == "nearby" for item in found)
    assert all(item.drive_hours <= companions.NEARBY_MAX_HOURS
               for item in found if item.basis == "nearby")


def test_a_city_with_no_evidence_and_no_road_suggests_nothing(dataset):
    """Goa is always sold alone and has no row in city_distances, so there is no answer.

    Inventing one is the thing this engine does not do.
    """
    assert suggest(dataset, ["Goa"]) == []


def test_an_unsold_combination_falls_back_to_the_last_city_and_says_so(dataset):
    """Nobody sells Jaipur with Cochin; the suggestions should follow Cochin, not go blank."""
    found = suggest(dataset, ["Jaipur", "Cochin"])
    assert found and found[0].city == "Munnar"
    assert "Cochin" in found[0].detail and "Jaipur" not in found[0].detail


def test_nothing_chosen_gives_the_most_sold_destinations(dataset):
    found = suggest(dataset, [])
    assert found and all(item.basis == "popular" for item in found)
    assert "sold with" not in found[0].detail, "there is no pairing to report yet"
    assert found[0].trips >= found[-1].trips


def test_the_limit_is_honoured(dataset):
    assert len(suggest(dataset, ["Leh"], limit=2)) == 2
    assert suggest(dataset, ["Leh"], limit=0) == []


# --- the endpoint -------------------------------------------------------------------------
def test_the_endpoint_answers_for_one_city(client):
    body = client.get("/cities/companions?city=Leh").json()
    assert body["chosen"] == ["Leh"]
    assert body["itineraries_searched"] > 150
    first = body["companions"][0]
    assert first["city"] == "Pangong Lake" and first["trips"] > 30


def test_the_endpoint_takes_the_whole_selection(client):
    body = client.get("/cities/companions?city=Leh&city=Pangong+Lake&limit=3").json()
    assert len(body["companions"]) <= 3
    assert body["companions"][0]["city"] == "Nubra Valley"


def test_an_unknown_city_is_a_404_not_an_empty_list(client):
    """So a typo in the picker is visible rather than looking like "nothing goes with this"."""
    assert client.get("/cities/companions?city=Atlantis").status_code == 404


def test_the_sights_route_still_matches(client):
    """/cities/companions is a literal path next to /cities/{city}/sights."""
    assert client.get("/cities/Leh/sights").status_code == 200
    assert client.get("/cities/companions").status_code == 200


@pytest.mark.parametrize("limit", ["0", "99"])
def test_a_silly_limit_is_refused(client, limit):
    assert client.get(f"/cities/companions?city=Leh&limit={limit}").status_code == 422
