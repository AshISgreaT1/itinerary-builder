"""Reviewed corrections to the dataset.

The design these tests protect: the engine proposes, a person decides, and the workbook is
never rewritten. The two failure modes worth guarding are a correction applied that nobody
reviewed, and a proposal run that quietly destroys the reviewed file.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest

from app import corrections
from app.datacheck import propose_corrections, run_checks
from app.normalization import normalize_key

REAL_FILE = Path(__file__).resolve().parents[1] / "data" / "corrections.csv"


@pytest.fixture
def book(tmp_path):
    cities = pd.DataFrame({"city": ["Leh", "Nubra Valley", "Alchi"],
                           "city_key": ["leh", "nubra valley", "alchi"]})
    sights = pd.DataFrame({
        "city": ["Leh", "Leh", "Alchi"],
        "sight": ["Disket monastery", "Alchi Monastery", "Alchi Monastery"],
    })
    return sights, cities, tmp_path / "corrections.csv"


def write(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=corrections.FIELDS)
        writer.writeheader()
        writer.writerows(rows)


# --- applying -------------------------------------------------------------------------------
def test_no_file_means_no_corrections(book):
    sights, cities, path = book
    result, notes = corrections.apply(sights, cities, path)
    assert len(result) == len(sights) and notes == []


def test_a_sight_can_be_re_homed(book):
    sights, cities, path = book
    write(path, [{"action": "rehome", "city": "Leh", "sight": "Disket monastery",
                  "value": "Nubra Valley", "reason": "115 km from Leh"}])
    result, notes = corrections.apply(sights, cities, path)
    moved = result[result["sight"] == "Disket monastery"].iloc[0]
    assert moved["city"] == "Nubra Valley"
    assert "re-homed" in notes[0] and "115 km from Leh" in notes[0]


def test_a_duplicate_row_can_be_dropped(book):
    sights, cities, path = book
    write(path, [{"action": "drop", "city": "Leh", "sight": "Alchi Monastery",
                  "value": "", "reason": "duplicate"}])
    result, notes = corrections.apply(sights, cities, path)
    assert len(result) == len(sights) - 1
    assert not ((result["city"] == "Leh") & (result["sight"] == "Alchi Monastery")).any()
    assert ((result["city"] == "Alchi")).any(), "the kept copy must survive"


def test_a_correction_pointing_at_nothing_is_reported_not_ignored(book):
    """A stale file is a problem to be told about, not a silent no-op."""
    sights, cities, path = book
    write(path, [{"action": "rehome", "city": "Leh", "sight": "Nowhere",
                  "value": "Alchi", "reason": ""}])
    result, notes = corrections.apply(sights, cities, path)
    assert len(result) == len(sights)
    assert "skipped" in notes[0] and "Nowhere" in notes[0]


def test_an_unknown_target_city_is_refused(book):
    sights, cities, path = book
    write(path, [{"action": "rehome", "city": "Leh", "sight": "Disket monastery",
                  "value": "Atlantis", "reason": ""}])
    result, notes = corrections.apply(sights, cities, path)
    assert result[result["sight"] == "Disket monastery"].iloc[0]["city"] == "Leh"
    assert "not a city in the dataset" in notes[0]


def test_unknown_actions_are_ignored(book):
    sights, cities, path = book
    write(path, [{"action": "delete-everything", "city": "Leh", "sight": "Disket monastery",
                  "value": "", "reason": ""}])
    assert corrections.load(path) == []


def test_a_directory_is_not_a_corrections_file(tmp_path):
    """Path("") resolves to the working directory, which exists and is not a file."""
    assert corrections.load(tmp_path) == []
    assert corrections.load(Path("")) == []


# --- proposing ----------------------------------------------------------------------------------
def test_proposals_are_made_against_the_shipped_workbook(dataset, monkeypatch, tmp_path):
    """Reading the corrected view instead would propose nothing on the second run."""
    monkeypatch.setattr(corrections, "CORRECTIONS_PATH", tmp_path / "none.csv")
    first = propose_corrections(dataset)
    assert first, "there are known problems in this workbook"

    corrections.write(first, tmp_path / "written.csv")
    again = propose_corrections(dataset)
    assert len(again) == len(first), "proposing twice must give the same answer"


def test_every_proposal_is_an_action_the_loader_understands(dataset):
    for proposal in propose_corrections(dataset):
        assert proposal.action in corrections.ACTIONS
        assert proposal.city and proposal.sight
        assert proposal.reason, "a correction without a reason cannot be reviewed"
        if proposal.action == "rehome":
            assert proposal.value


# --- the file this project actually ships -----------------------------------------------------------
def test_the_committed_corrections_are_valid(dataset):
    """Every line has to name a row that exists and a city that exists."""
    applied = corrections.load(REAL_FILE)
    assert applied, "the reviewed corrections file should not be empty"

    sight_keys = {(normalize_key(str(row["city"])), normalize_key(str(row["sight"])))
                  for _, row in dataset.sights.iterrows()}
    city_keys = {normalize_key(str(row["city"])) for _, row in dataset.cities.iterrows()}
    for correction in applied:
        if correction.action == "rehome":
            assert normalize_key(correction.value) in city_keys, correction.value
            # After loading it is either under its new city, or merged into a row that was
            # already there -- which is what happens when the move reveals a duplicate:
            # Leh's "sand dunes" turned out to be Nubra Valley's "Hundar sand dunes".
            moved = (normalize_key(correction.value),
                     normalize_key(correction.sight)) in sight_keys
            merged = any(normalize_key(entry["merged"]) == normalize_key(correction.sight)
                         and normalize_key(entry["city"]) == normalize_key(correction.value)
                         for entry in dataset.dedup.merges)
            assert moved or merged, f"{correction.sight} went nowhere"
        assert correction.reason, f"{correction.sight} has no reason recorded"


def test_the_corrections_fix_what_they_claim_to(dataset):
    """They exist to remove duplicates and unplannable days; check they did."""
    kinds: dict[str, int] = {}
    for finding in run_checks(dataset):
        kinds[finding.kind] = kinds.get(finding.kind, 0) + 1

    assert kinds.get("same_place_two_cities", 0) == 0, "the duplicate rows should be gone"
    assert kinds.get("cannot_fit_any_day", 0) <= 10, (
        "sights 150+ km from their city made the planner propose impossible days")


def test_the_dataset_still_holds_the_destinations_people_book(dataset):
    """A correction that empties a bookable region is a regression, not a fix."""
    for city in ["Leh", "Nubra Valley", "Pangong Lake", "Udaipur", "Jaipur", "Jaisalmer"]:
        assert len(dataset.city_sights(normalize_key(city))) > 0, f"{city} has no sights left"


def test_a_re_home_can_reveal_a_duplicate(dataset):
    """Moving a sight to its real city sometimes lands it next to itself.

    Leh listed "sand dunes" 125 km away; Nubra Valley already had "Hundar sand dunes" with
    that name as a declared alias. Re-homing the first put the two side by side and the
    loader's own de-duplication merged them -- one fewer phantom sight, found by accident.
    """
    merged = {entry["merged"] for entry in dataset.dedup.merges}
    assert "sand dunes" in merged
