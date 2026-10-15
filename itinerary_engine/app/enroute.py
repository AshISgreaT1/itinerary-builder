"""Stops made on the way between two cities.

The planner otherwise treats every sight as orbiting a city centre, charging a round trip
from the hotel. That is wrong for a place you simply drive past: Chang La sits 1.4 km off
the Leh to Pangong Lake road, a third of the way along, yet it is 77 km from Leh and 95 km
from the lake, so neither city can host it and it was never scheduled at all.

Here such a sight is attached to the transfer day instead, and charged only the detour off
the road plus the stop itself.

A sight is only taken en route when no city on the trip could reasonably hold it; Thiksey
(19 km from Leh) stays a Leh day trip, Chang La does not.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import (
    EN_ROUTE_MAX_DETOUR_KM,
    EN_ROUTE_MAX_PROGRESS,
    EN_ROUTE_MAX_STOPS,
    EN_ROUTE_MIN_PROGRESS,
    EXCURSION_KM,
    SIGHT_BUFFER_HOURS,
)
from .data_loader import Dataset
from .geo import RouteLeg, local_travel_hours, point_to_segment_km, road_km
from .solver import sort_by_priority


@dataclass
class EnRouteStop:
    """A sight scheduled on a transfer day, with what the detour costs."""

    sight: pd.Series
    detour_km: float
    progress: float
    detour_hours: float

    @property
    def cost_hours(self) -> float:
        """Detour driving + the visit + the usual per-stop buffer."""
        return round(self.detour_hours + float(self.sight["duration_hours"]) + SIGHT_BUFFER_HOURS, 2)

    def payload(self) -> dict:
        row = self.sight
        return {
            "sight": str(row["sight"]),
            "visit_type": str(row.get("visit_type", "")),
            "duration_hours": round(float(row["duration_hours"]), 2),
            "buffer_hours": SIGHT_BUFFER_HOURS,
            "must_see_tier": int(row["must_see_tier"]),
            "must_visit": bool(row["must_visit_flag"]),
            "priority": int(row["priority"]),
            "best_time": str(row.get("best_time", "any")),
            "zone": str(row.get("zone", "")),
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            "is_excursion": False,
            "distance_from_centre_km": 0.0,
            "en_route": True,
            "detour_km": round(self.detour_km, 1),
            "detour_hours": round(self.detour_hours, 2),
            "along_route_percent": round(self.progress * 100),
            "home_city": str(row["city"]),
        }


def corridor_candidates(
    dataset: Dataset,
    leg: RouteLeg,
    from_row: pd.Series,
    to_row: pd.Series,
    trip_city_keys: set[str],
    month: int | None = None,
) -> list[EnRouteStop]:
    """Sights near the straight line between two cities that no trip city can host.

    Ordered by importance, not by position: the caller decides how many fit, then puts the
    chosen ones back into road order.
    """
    if dataset.sights.empty:
        return []
    speed_reference = to_row.get("altitude_m")
    city_points = [
        (row["lat"], row["lon"])
        for _, row in dataset.cities.iterrows()
        if row["city_key"] in trip_city_keys
    ]

    rows = []
    for _, sight in dataset.sights.iterrows():
        if month and month not in sight["open_months_set"]:
            continue
        detour_km, progress = point_to_segment_km(
            sight["lat"], sight["lon"], from_row["lat"], from_row["lon"], to_row["lat"], to_row["lon"]
        )
        if pd.isna(detour_km) or detour_km > EN_ROUTE_MAX_DETOUR_KM:
            continue
        if not EN_ROUTE_MIN_PROGRESS < progress < EN_ROUTE_MAX_PROGRESS:
            continue  # near a city end of the leg: that city should host it
        # Skip anything a city on this trip could visit as a normal day trip.
        if any(road_km(lat, lon, sight["lat"], sight["lon"]) <= EXCURSION_KM for lat, lon in city_points):
            continue
        rows.append((sight, detour_km, progress))

    if not rows:
        return []
    # One row per place: the same pass is often filed under both endpoint cities.
    unique: dict[str, tuple] = {}
    for sight, detour_km, progress in rows:
        unique.setdefault(sight["sight_key"], (sight, detour_km, progress))

    frame = pd.DataFrame([item[0] for item in unique.values()])
    ordered = sort_by_priority(frame)
    by_key = {key: value for key, value in unique.items()}
    stops = []
    for _, sight in ordered.iterrows():
        _, detour_km, progress = by_key[sight["sight_key"]]
        detour_hours = local_travel_hours(2 * detour_km * 1.3, speed_reference)  # off the road and back
        stops.append(EnRouteStop(sight=sight, detour_km=detour_km, progress=progress, detour_hours=detour_hours))
    return stops


def select_en_route(stops: list[EnRouteStop], budget_hours: float, max_stops: int = EN_ROUTE_MAX_STOPS) -> list[EnRouteStop]:
    """Take the most important stops that fit the hours left on the transfer day."""
    chosen: list[EnRouteStop] = []
    spent = 0.0
    for stop in stops:
        if len(chosen) >= max_stops:
            break
        if spent + stop.cost_hours > budget_hours + 1e-9:
            continue
        chosen.append(stop)
        spent += stop.cost_hours
    chosen.sort(key=lambda stop: stop.progress)  # drive order, not importance order
    return chosen


def plan_en_route(
    dataset: Dataset,
    leg: RouteLeg,
    budget_hours: float,
    trip_city_keys: set[str],
    month: int | None = None,
    already_used: set[str] | None = None,
) -> list[EnRouteStop]:
    """En-route stops for one transfer, within the hours that day has left."""
    if leg.drive_hours is None or budget_hours <= 0:
        return []
    from_row = dataset.city_row(leg.from_city)
    to_row = dataset.city_row(leg.to_city)
    if from_row is None or to_row is None:
        return []
    stops = corridor_candidates(dataset, leg, from_row, to_row, trip_city_keys, month)
    if already_used:
        stops = [stop for stop in stops if stop.sight["sight_key"] not in already_used]
    return select_en_route(stops, budget_hours)
