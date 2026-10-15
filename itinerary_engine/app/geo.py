"""Distances, local travel time and the intercity road graph."""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import pandas as pd

import math
import re

from .config import (
    HILL_ALTITUDE_M,
    INTERCITY_SPEED_KMPH,
    LOCAL_SPEED_KMPH,
    LOCAL_SPEED_KMPH_HILLS,
    MAX_OVERNIGHT_TRAVEL_HOURS,
    MAX_ROAD_SPEED_KMPH,
    MAX_SINGLE_DAY_TRAVEL_HOURS,
    MIN_ROAD_SPEED_KMPH,
    RECOMPUTE_ROAD_SPEED_KMPH,
    ROAD_DISTANCE_FACTOR,
    SPEED_CHECK_MIN_KM,
    TRANSFER_SPEED_KMPH,
)

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two points."""
    if any(pd.isna(value) for value in (lat1, lon1, lat2, lon2)):
        return float("nan")
    phi1, phi2 = math.radians(float(lat1)), math.radians(float(lat2))
    d_phi = phi2 - phi1
    d_lambda = math.radians(float(lon2) - float(lon1))
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def road_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Approximate road distance: straight line inflated by a detour factor."""
    return haversine_km(lat1, lon1, lat2, lon2) * ROAD_DISTANCE_FACTOR


def local_speed_kmph(altitude_m: float | None) -> float:
    """Hill towns drive slower than plains cities."""
    if altitude_m is None or pd.isna(altitude_m):
        return LOCAL_SPEED_KMPH
    return LOCAL_SPEED_KMPH_HILLS if float(altitude_m) >= HILL_ALTITUDE_M else LOCAL_SPEED_KMPH


def local_travel_hours(distance_km: float, altitude_m: float | None = None) -> float:
    """Driving time for an intra-city hop."""
    if pd.isna(distance_km):
        return 0.0
    return float(distance_km) / local_speed_kmph(altitude_m)


@dataclass
class RouteLeg:
    """One intercity hop of the trip.

    ``drive_hours`` is None when the dataset has no usable journey time: the engine never
    invents one, it reports the gap instead (see ``data_gap``).
    """

    from_city: str
    to_city: str
    road_km: float | None
    drive_hours: float | None
    mode: str
    #: "dataset", "dataset (recomputed)", "via <city>" or "no route data".
    source: str
    #: Straight-line distance, always available from the city coordinates.
    straight_line_km: float | None = None
    #: True when the journey time is missing or had to be corrected.
    data_gap: bool = False
    #: Why this leg is flagged (plain English, shown to the user).
    notes: list[str] = field(default_factory=list)

    @property
    def travel_days_needed(self) -> int:
        """Whole days this journey consumes (1 unless it is longer than a travel day)."""
        if self.drive_hours is None:
            return 1
        return max(1, math.ceil(self.drive_hours / MAX_SINGLE_DAY_TRAVEL_HOURS))

    @property
    def fits_one_day(self) -> bool:
        return self.drive_hours is not None and self.drive_hours <= MAX_SINGLE_DAY_TRAVEL_HOURS

    @property
    def is_overnight(self) -> bool:
        """Doable as an overnight bus/train: too long for a day, short enough for a night."""
        return (
            self.drive_hours is not None
            and MAX_SINGLE_DAY_TRAVEL_HOURS < self.drive_hours <= MAX_OVERNIGHT_TRAVEL_HOURS
        )

    @property
    def hours_charged_to_day(self) -> float:
        """Hours this leg takes out of the arrival day's sightseeing budget."""
        if self.drive_hours is None:
            return float("inf")  # unknown journey: the day cannot be planned
        return min(self.drive_hours, MAX_SINGLE_DAY_TRAVEL_HOURS)

    def as_dict(self) -> dict:
        return {
            "from_city": self.from_city,
            "to_city": self.to_city,
            "road_km": round(self.road_km, 1) if self.road_km is not None else None,
            "straight_line_km": round(self.straight_line_km, 1) if self.straight_line_km is not None else None,
            "drive_hours": round(self.drive_hours, 2) if self.drive_hours is not None else None,
            "mode": self.mode,
            "source": self.source,
            "data_gap": self.data_gap,
            "travel_days_needed": self.travel_days_needed,
            "is_overnight": self.is_overnight,
            "notes": self.notes,
        }


def parse_access_distance_km(value: object) -> float | None:
    """Pull the distance out of an airport/railhead cell such as ``KUU (Bhuntar, 50km)``.

    ``DEL`` (no distance given) returns None, which the caller reports as a data gap
    rather than filling in with a guess.
    """
    match = re.search(r"(\d+(?:\.\d+)?)\s*km", str(value or ""), flags=re.IGNORECASE)
    return float(match.group(1)) if match else None


def transfer_hours_for_distance(distance_km: float) -> float:
    """Hotel -> airport/station transfer time, never less than half an hour."""
    return max(0.5, distance_km / TRANSFER_SPEED_KMPH)


class CityGraph:
    """Intercity road network built from the ``city_distances`` sheet.

    Distances are stored one way in the dataset, so every edge is treated as
    bidirectional. When two cities have no direct row, the shortest path through other
    cities is used; if they are not connected at all, a straight-line estimate is used so
    the planner degrades gracefully instead of failing.
    """

    def __init__(self, distances: pd.DataFrame, cities: pd.DataFrame) -> None:
        self._graph = nx.Graph()
        self._cities = cities.set_index("city_key")
        for _, row in distances.iterrows():
            a, b = row["from_city_key"], row["to_city_key"]
            hours = float(row["drive_hours"])
            # Keep the quicker of duplicated rows.
            if self._graph.has_edge(a, b) and self._graph[a][b]["drive_hours"] <= hours:
                continue
            self._graph.add_edge(
                a, b, drive_hours=hours, road_km=float(row["road_km"]), mode=str(row["mode"])
            )

    @property
    def graph(self) -> nx.Graph:
        return self._graph

    def _city(self, key: str):
        try:
            return self._cities.loc[key]
        except KeyError:
            return None

    def _straight_line_km(self, from_key: str, to_key: str) -> float | None:
        origin, destination = self._city(from_key), self._city(to_key)
        if origin is None or destination is None:
            return None
        return road_km(origin["lat"], origin["lon"], destination["lat"], destination["lon"])

    def _checked(self, leg: RouteLeg) -> RouteLeg:
        """Sanity-check a dataset journey time against the distance; recompute if absurd.

        A row claiming 300 km in 2 hours (150 km/h on Indian roads) is a data error, not a
        fast car, so the time is recomputed and the correction is reported.
        """
        if leg.drive_hours is None or leg.road_km is None or leg.road_km < SPEED_CHECK_MIN_KM:
            return leg
        if leg.drive_hours <= 0:
            leg.notes.append(f"{leg.road_km:.0f} km with no journey time in the dataset.")
            leg.drive_hours, leg.data_gap = None, True
            return leg
        speed = leg.road_km / leg.drive_hours
        if not MIN_ROAD_SPEED_KMPH <= speed <= MAX_ROAD_SPEED_KMPH:
            corrected = leg.road_km / RECOMPUTE_ROAD_SPEED_KMPH
            leg.notes.append(
                f"dataset says {leg.road_km:.0f} km in {leg.drive_hours:.1f}h ({speed:.0f} km/h), which is not "
                f"plausible by road; recomputed as {corrected:.1f}h at {RECOMPUTE_ROAD_SPEED_KMPH:.0f} km/h."
            )
            leg.drive_hours, leg.data_gap, leg.source = corrected, True, "dataset (recomputed)"
        straight = leg.straight_line_km
        if straight and leg.road_km + 1 < straight:
            leg.notes.append(
                f"dataset road distance ({leg.road_km:.0f} km) is shorter than the straight-line distance "
                f"({straight:.0f} km); the coordinates or the row are wrong."
            )
            leg.data_gap = True
        return leg

    def leg(self, from_city: str, to_city: str, from_key: str, to_key: str) -> RouteLeg:
        """Travel between two cities from the dataset, never invented.

        Order of preference: a direct row, then the shortest path over known roads. With
        neither, the leg is returned with no journey time and a note saying what is missing,
        so the planner and the validator can treat it as a blocking data gap.
        """
        straight = self._straight_line_km(from_key, to_key)
        if self._graph.has_edge(from_key, to_key):
            edge = self._graph[from_key][to_key]
            return self._checked(
                RouteLeg(from_city, to_city, edge["road_km"], edge["drive_hours"], edge["mode"], "dataset", straight)
            )
        if from_key in self._graph and to_key in self._graph and nx.has_path(self._graph, from_key, to_key):
            path = nx.shortest_path(self._graph, from_key, to_key, weight="drive_hours")
            hours = sum(self._graph[a][b]["drive_hours"] for a, b in zip(path, path[1:]))
            km = sum(self._graph[a][b]["road_km"] for a, b in zip(path, path[1:]))
            via = " -> ".join(str(self._cities.loc[key, "city"]) for key in path[1:-1])
            leg = RouteLeg(from_city, to_city, km, hours, "road", f"via {via}", straight)
            leg.notes.append(f"no direct row; routed over known roads via {via}.")
            return self._checked(leg)

        leg = RouteLeg(from_city, to_city, None, None, "unknown", "no route data", straight, data_gap=True)
        if straight:
            leg.notes.append(
                f"no row in city_distances for {from_city} to {to_city}; the cities are about "
                f"{straight:.0f} km apart, so a journey time cannot be assumed."
            )
            if straight / INTERCITY_SPEED_KMPH > MAX_OVERNIGHT_TRAVEL_HOURS:
                leg.mode = "air/rail (not in dataset)"
                leg.notes.append(
                    "at this distance the journey is normally flown; add a city_distances row with the real "
                    "flight or train duration to plan it."
                )
        else:
            leg.notes.append(f"no route data and no coordinates for {from_city} or {to_city}.")
        return leg


def order_by_route(sights: pd.DataFrame, start_lat: float, start_lon: float) -> list[int]:
    """Nearest-neighbour visiting order from the city centre, refined with 2-opt.

    Returns positional indices into ``sights``. Deterministic: ties break on the row order,
    which is already the dataset's priority order.
    """
    count = len(sights)
    if count <= 2:
        return list(range(count))

    lats = sights["lat"].to_numpy(dtype=float)
    lons = sights["lon"].to_numpy(dtype=float)
    distances = [[road_km(lats[i], lons[i], lats[j], lons[j]) for j in range(count)] for i in range(count)]
    from_start = [road_km(start_lat, start_lon, lats[i], lons[i]) for i in range(count)]

    def tour_length(order: list[int]) -> float:
        legs = sum(distances[a][b] for a, b in zip(order, order[1:]))
        return from_start[order[0]] + legs + from_start[order[-1]]

    remaining = list(range(count))
    order = [min(remaining, key=lambda i: (from_start[i], i))]
    remaining.remove(order[0])
    while remaining:
        last = order[-1]
        nxt = min(remaining, key=lambda i: (distances[last][i], i))
        order.append(nxt)
        remaining.remove(nxt)

    improved = True
    while improved:  # 2-opt: reverse a segment whenever it shortens the tour
        improved = False
        for i in range(1, count - 1):
            for j in range(i + 1, count):
                candidate = order[:i] + order[i : j + 1][::-1] + order[j + 1 :]
                if tour_length(candidate) + 1e-9 < tour_length(order):
                    order, improved = candidate, True
    return order
