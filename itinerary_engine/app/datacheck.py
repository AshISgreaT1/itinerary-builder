"""Dataset quality report.

The planner can only be as sensible as the rows it reads. This module looks for the kinds
of dataset problems that quietly produce odd itineraries -- an attraction filed under a
city it is nowhere near, a city pair with no journey time, a sight nobody can reach in the
time the dataset claims -- and reports them with the evidence and a suggested fix.

Nothing is changed and nothing is guessed: every suggestion is derived from coordinates and
rows already in the workbook, and the team decides what to apply.

    python -m app.datacheck                 # readable report
    python -m app.datacheck --csv out.csv   # one row per finding
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict, dataclass

from .config import EXCURSION_KM, MAX_SINGLE_DAY_TRAVEL_HOURS, PACE_ACTIVE_HOURS, SIGHT_BUFFER_HOURS, WRONG_CITY_KM
from .data_loader import Dataset, get_dataset
from .geo import haversine_km, parse_access_distance_km, road_km
from .normalization import normalize_key


@dataclass
class Finding:
    """One dataset problem, with the evidence that proves it."""

    severity: str      # "error" | "warning"
    kind: str
    key: str           # "City / Sight" or "City -> City"
    detail: str
    suggestion: str


def _nearest_city(dataset: Dataset, lat: float, lon: float, exclude_key: str) -> tuple[str, float] | None:
    """Closest city in the cities sheet to a point (used to suggest a sight's real home)."""
    best: tuple[str, float] | None = None
    for _, city in dataset.cities.iterrows():
        if city["city_key"] == exclude_key:
            continue
        distance = haversine_km(lat, lon, city["lat"], city["lon"])
        if best is None or distance < best[1]:
            best = (str(city["city"]), distance)
    return best


def misfiled_sights(dataset: Dataset) -> list[Finding]:
    """Sights that sit far from the city they are filed under."""
    findings: list[Finding] = []
    for _, sight in dataset.sights.iterrows():
        city = dataset.city_row(sight["city"])
        if city is None:
            findings.append(Finding("error", "unknown_city", f"{sight['city']} / {sight['sight']}",
                                    f"city '{sight['city']}' is not in the cities sheet",
                                    "add the city with coordinates, or re-home the sight"))
            continue
        distance = road_km(city["lat"], city["lon"], sight["lat"], sight["lon"])
        if distance < EXCURSION_KM:
            continue
        nearest = _nearest_city(dataset, sight["lat"], sight["lon"], city["city_key"])
        closer = nearest and nearest[1] * 1.3 < distance
        severity = "error" if distance > WRONG_CITY_KM else "warning"
        suggestion = (
            f"re-home to {nearest[0]} ({nearest[1] * 1.3:.0f} km away)" if closer
            else "keep as a day trip, or split it into its own city row"
        )
        findings.append(
            Finding(severity, "far_from_city", f"{sight['city']} / {sight['sight']}",
                    f"{distance:.0f} km from the {sight['city']} centre", suggestion)
        )
    return findings


def unreachable_sights(dataset: Dataset) -> list[Finding]:
    """Sights that cannot fit any day, however the trip is planned."""
    findings: list[Finding] = []
    longest_day = max(PACE_ACTIVE_HOURS.values())
    for _, sight in dataset.sights.iterrows():
        city = dataset.city_row(sight["city"])
        if city is None:
            continue
        distance = road_km(city["lat"], city["lon"], sight["lat"], sight["lon"])
        drive = 2 * distance / 25.0  # there and back at a generous hill speed
        needed = float(sight["duration_hours"]) + SIGHT_BUFFER_HOURS + drive
        if needed > longest_day:
            findings.append(
                Finding("error", "cannot_fit_any_day", f"{sight['city']} / {sight['sight']}",
                        f"needs {needed:.1f}h ({sight['duration_hours']}h visit + {drive:.1f}h driving) "
                        f"but the longest possible day is {longest_day:.0f}h",
                        "re-home it to a nearer city, or give it its own city row with a hotel")
            )
    return findings


def duplicate_city_sight_names(dataset: Dataset) -> list[Finding]:
    """A sight carrying its city's name reads as 'Pangong Lake / Pangong Lake'."""
    return [
        Finding("warning", "sight_named_after_city", f"{sight['city']} / {sight['sight']}",
                "the sight and its city have the same name",
                "harmless, but rename the sight (e.g. 'Pangong Tso lakefront') if the duplication bothers you")
        for _, sight in dataset.sights.iterrows()
        if sight["sight_key"] == sight["city_key"]
    ]


def _pairs_actually_sold(dataset: Dataset) -> set[tuple[str, str]]:
    """City pairs that appear together in the workbook's own ``itineraries`` sheet.

    Those are the journeys the business really sells, so a missing route row there matters;
    pairing every gateway with every city would just produce noise (Ajmer to Alleppey).
    """
    import re

    import pandas as pd

    try:
        trips = pd.read_excel(dataset.source, sheet_name="itineraries")
    except Exception:
        return set()

    pairs: set[tuple[str, str]] = set()
    keys = set(dataset.cities["city_key"])
    for _, trip in trips.iterrows():
        route: list[str] = []
        for chunk in re.split(r"\|", str(trip.get("day_wise_route", "") or "")):
            for part in re.split(r"->|→", re.sub(r"^\s*Day\s*\d+\s*", "", chunk.strip())):
                key = normalize_key(re.sub(r"\(.*?\)", "", part))
                if key in keys and (not route or route[-1] != key):
                    route.append(key)
        if not route:
            route = [normalize_key(part) for part in str(trip.get("destination", "") or "").split("+")]
            route = [key for key in route if key in keys]
        for first, second in zip(route, route[1:]):
            if first != second:
                pairs.add(tuple(sorted([first, second])))
    return pairs


def cross_city_duplicates(dataset: Dataset) -> list[Finding]:
    """The same place filed under two cities, e.g. Chang La under both Leh and Pangong Lake.

    Each copy is then planned as a day trip from its own city, so the place competes with
    itself and is never treated as what it is: a stop on the road between the two.
    """
    findings: list[Finding] = []
    for _, group in dataset.sights.groupby("sight_key"):
        if group["city_key"].nunique() < 2:
            continue
        rows = [row for _, row in group.iterrows()]
        for i, first in enumerate(rows):
            for second in rows[i + 1:]:
                if first["city_key"] == second["city_key"]:
                    continue
                if haversine_km(first["lat"], first["lon"], second["lat"], second["lon"]) > 2:
                    continue
                nearer, farther = sorted(
                    (first, second),
                    key=lambda row: road_km(
                        dataset.city_row(row["city"])["lat"], dataset.city_row(row["city"])["lon"],
                        row["lat"], row["lon"],
                    ),
                )
                findings.append(
                    Finding("error", "same_place_two_cities",
                            f"{first['city']} / {first['sight']}  ==  {second['city']} / {second['sight']}",
                            "identical coordinates, filed under both cities",
                            f"keep the row under {nearer['city']} and delete the {farther['city']} copy")
                )
    return findings


def route_gaps(dataset: Dataset) -> list[Finding]:
    """Journeys the dataset's own packages contain but cannot cost in time."""
    findings: list[Finding] = []
    names = dict(zip(dataset.cities["city_key"], dataset.cities["city"]))
    for first, second in sorted(_pairs_actually_sold(dataset)):
        leg = dataset.graph.leg(str(names[first]), str(names[second]), first, second)
        label = f"{names[first]} -> {names[second]}"
        if leg.drive_hours is None:
            findings.append(
                Finding("error", "no_route_data", label,
                        f"sold together in the itineraries sheet, but there is no journey time"
                        + (f" ({leg.straight_line_km:.0f} km apart)" if leg.straight_line_km else ""),
                        "add a city_distances row (mode=flight/train/road) with the real duration")
            )
        elif leg.drive_hours > MAX_SINGLE_DAY_TRAVEL_HOURS and leg.mode == "road":
            findings.append(
                Finding("warning", "road_only_long_haul", label,
                        f"{leg.drive_hours:.1f}h by road only",
                        "add a flight or train row so the trip does not lose a whole day")
            )
    return findings


def missing_transfer_distances(dataset: Dataset) -> list[Finding]:
    """Cities whose airport/railhead cell has no distance, so departure time is assumed."""
    return [
        Finding("warning", "no_airport_distance", str(city["city"]),
                f"nearest_airport='{city.get('nearest_airport')}' has no distance",
                "write it as 'CODE (Name, 45km)' so the departure day is budgeted from real data")
        for _, city in dataset.cities.iterrows()
        if city["city_key"] in set(dataset.sights["city_key"])
        and parse_access_distance_km(city.get("nearest_airport")) is None
        and parse_access_distance_km(city.get("nearest_railhead")) is None
    ]


def run_checks(dataset: Dataset) -> list[Finding]:
    """Every check, worst first."""
    findings = (
        misfiled_sights(dataset)
        + unreachable_sights(dataset)
        + cross_city_duplicates(dataset)
        + duplicate_city_sight_names(dataset)
        + route_gaps(dataset)
        + missing_transfer_distances(dataset)
    )
    order = {"error": 0, "warning": 1}
    return sorted(findings, key=lambda f: (order[f.severity], f.kind, f.key))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report dataset problems that distort itineraries.")
    parser.add_argument("--dataset", default=None, help="Path to the Excel dataset")
    parser.add_argument("--csv", default=None, help="Write the findings to this CSV")
    parser.add_argument("--kind", default=None, help="Only this finding kind")
    args = parser.parse_args(argv)

    findings = run_checks(get_dataset(args.dataset))
    if args.kind:
        findings = [finding for finding in findings if finding.kind == args.kind]

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["severity", "kind", "key", "detail", "suggestion"])
            writer.writeheader()
            writer.writerows(asdict(finding) for finding in findings)
        print(f"{len(findings)} finding(s) written to {args.csv}")
        return 0

    counts: dict[str, int] = {}
    for finding in findings:
        counts[finding.kind] = counts.get(finding.kind, 0) + 1
    print(f"{len(findings)} finding(s): " + ", ".join(f"{kind} {count}" for kind, count in sorted(counts.items())))
    current = None
    for finding in findings:
        if finding.kind != current:
            current = finding.kind
            print(f"\n--- {current} ---")
        print(f"  [{finding.severity}] {finding.key}: {finding.detail}\n        -> {finding.suggestion}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
