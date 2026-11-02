"""Which cities are sold together, so searching one offers the rest.

An agent typing "Leh" is almost never selling Leh alone: 41 of the 69 Leh packages in the
``itineraries`` sheet also sell Pangong Lake, and 37 sell Nubra Valley. That pairing is a
fact the dataset already records, so it is read rather than guessed.

The evidence is the ``destination`` column ("Leh + Nubra Valley + Pangong Lake"), which is
the agents' own statement of what belongs in one trip. ``day_wise_route`` is deliberately
not used: it names the gateway the travellers flew into, so Delhi would end up suggested
alongside every destination in the workbook on the strength of being passed through.

Two kinds of suggestion come back, and they are labelled so the agent can tell them apart:

* ``sold_together`` -- counted from past itineraries, strongest first.
* ``nearby`` -- a fallback for the 10 cities that appear in no package and for one-city
  destinations like Goa. Only cities the dataset can actually route to are offered, because
  suggesting a leg with no journey time would just produce a plan that fails validation.

Nothing here invents a pairing. A city with no shared itinerary and no routable neighbour
simply returns nothing.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import pandas as pd

from .data_loader import Dataset
from .normalization import normalize_key

#: A pairing is only offered as evidence when at least this many itineraries sold it.
MIN_PAIRED_TRIPS = 2
#: A fallback neighbour has to be reachable inside a travel day.
NEARBY_MAX_HOURS = 6.0


@dataclass
class Companion:
    """One suggested city, with the reason it is being suggested."""

    city: str
    basis: str                      # "sold_together" or "nearby"
    trips: int = 0                  # itineraries that sold it with the chosen cities
    drive_hours: float | None = None
    detail: str = ""

    def as_dict(self) -> dict:
        return {
            "city": self.city,
            "basis": self.basis,
            "trips": self.trips,
            "drive_hours": self.drive_hours,
            "detail": self.detail,
        }


@dataclass
class CityPairings:
    """Which cities each past itinerary sold together."""

    #: one frozen set of city keys per itinerary
    trip_sets: list[frozenset[str]] = field(default_factory=list)
    #: city_key -> itineraries containing it at all
    trip_counts: Counter = field(default_factory=Counter)
    trips_seen: int = 0

    def sold_with_all(self, keys: list[str]) -> int:
        """Itineraries that sold every one of these cities together.

        Counting the real combination matters: Nubra Valley shares 37 itineraries with Leh
        and 37 with Pangong Lake, but summing those to 74 would claim more evidence than
        the sheet holds -- there are only 194 itineraries in it.
        """
        if not keys:
            return 0
        wanted = set(keys)
        return sum(1 for trip in self.trip_sets if wanted <= trip)

    def partners(self, keys: list[str]) -> list[tuple[str, int]]:
        """Cities sold alongside all of ``keys``, most frequent first."""
        wanted = set(keys)
        counts: Counter = Counter()
        for trip in self.trip_sets:
            if wanted <= trip:
                for key in trip - wanted:
                    counts[key] += 1
        return sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))


def _destination_keys(value: object, known: set[str]) -> list[str]:
    """``"Leh + Nubra Valley"`` -> ``["leh", "nubra valley"]``, keeping only real cities.

    Names the cities sheet does not carry (Agra, Srinagar) are dropped: the engine cannot
    plan them, so suggesting them would waste the agent's click.
    """
    keys: list[str] = []
    for part in re.split(r"\+", str(value or "")):
        key = normalize_key(part.strip())
        if key and key in known and key not in keys:
            keys.append(key)
    return keys


def learn_pairings(source: str | Path, known: frozenset[str]) -> CityPairings:
    """Count co-occurrence in a workbook's ``itineraries`` sheet.

    A workbook without that sheet or column yields empty pairings, which every caller
    treats as "no opinion" rather than an error.
    """
    pairings = CityPairings()
    try:
        trips = pd.read_excel(source, sheet_name="itineraries")
    except Exception:
        return pairings
    if "destination" not in trips.columns:
        return pairings

    for value in trips["destination"].dropna():
        keys = _destination_keys(value, set(known))
        if not keys:
            continue
        pairings.trips_seen += 1
        pairings.trip_sets.append(frozenset(keys))
        for key in keys:
            pairings.trip_counts[key] += 1
    return pairings


@lru_cache(maxsize=4)
def _cached_pairings(source: str, known: frozenset[str]) -> CityPairings:
    return learn_pairings(source, known)


def pairings_for(dataset: Dataset) -> CityPairings:
    """Pairings for the workbook this dataset came from (cached per process)."""
    known = frozenset(normalize_key(str(city)) for city in dataset.cities["city"])
    return _cached_pairings(str(dataset.source), known)


def _plannable(dataset: Dataset) -> dict[str, str]:
    """city_key -> display name, for cities that have sightseeing rows."""
    with_sights = set(dataset.sights["city_key"])
    return {row["city_key"]: str(row["city"]) for _, row in dataset.cities.iterrows()
            if row["city_key"] in with_sights}


def _nearby(dataset: Dataset, chosen_keys: list[str], exclude: set[str],
            limit: int) -> list[Companion]:
    """Cities within a travel day of the last chosen one, by the dataset's own roads."""
    if not chosen_keys or limit <= 0:
        return []
    anchor_key = chosen_keys[-1]
    anchor = dataset.city_row(anchor_key)
    if anchor is None:
        return []

    found: list[Companion] = []
    for key, name in _plannable(dataset).items():
        if key in exclude or key == anchor_key:
            continue
        leg = dataset.graph.leg(str(anchor["city"]), name, anchor_key, key)
        # No journey time means the planner would report a data gap and fail validation,
        # so the suggestion would be a dead end. A leg that *has* a time still counts even
        # when it carries a note -- Manali to Kasol is flagged because the row's 75 km is a
        # kilometre under the straight-line distance, which does not make the 3h wrong.
        if leg.drive_hours is None or leg.drive_hours > NEARBY_MAX_HOURS:
            continue
        found.append(Companion(
            city=name,
            basis="nearby",
            drive_hours=round(float(leg.drive_hours), 1),
            detail=f"{leg.drive_hours:.1f}h from {anchor['city']} by {leg.mode}",
        ))
    found.sort(key=lambda item: (item.drive_hours, item.city))
    return found[:limit]


def suggest(dataset: Dataset, chosen: list[str], limit: int = 6) -> list[Companion]:
    """Cities to offer alongside the ones already picked.

    Itinerary evidence comes first and is summed across everything chosen, so picking Leh
    then Pangong Lake pushes Nubra Valley up rather than re-offering the obvious. Proximity
    only fills the slots left over, and only from cities the dataset can route to.
    """
    if limit <= 0:
        return []
    plannable = _plannable(dataset)
    chosen_keys = [normalize_key(city) for city in chosen if city]
    pairings = pairings_for(dataset)

    # Measure against everything chosen, so picking Leh then Pangong Lake pushes Nubra
    # Valley up rather than re-offering the obvious. An unusual combination nobody has sold
    # has no evidence at all, so fall back to the city chosen last and say so.
    against = chosen_keys
    partners = [(key, count) for key, count in pairings.partners(against)
                if key in plannable and count >= MIN_PAIRED_TRIPS]
    if not partners and len(chosen_keys) > 1:
        against = chosen_keys[-1:]
        partners = [(key, count) for key, count in pairings.partners(against)
                    if key in plannable and count >= MIN_PAIRED_TRIPS]

    with_what = " + ".join(str(dataset.city_row(key)["city"]) for key in against
                           if dataset.city_row(key) is not None)
    sold = [
        Companion(
            city=plannable[key],
            # With nothing chosen yet there is no pairing to report, only how often each
            # city is sold at all -- which is what the picker wants before anyone has typed.
            basis="sold_together" if with_what else "popular",
            trips=count,
            detail=(f"sold with {with_what} in {count} past itinerar{'y' if count == 1 else 'ies'}"
                    if with_what else
                    f"in {count} of {pairings.trips_seen} past itineraries"),
        )
        for key, count in partners[:limit]
    ]

    exclude = set(chosen_keys) | {normalize_key(item.city) for item in sold}
    return sold + _nearby(dataset, chosen_keys, exclude, limit - len(sold))
