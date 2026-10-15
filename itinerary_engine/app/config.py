"""Tunable constants for the itinerary engine.

Everything the planner and the validator argue about lives here, so the rules can be
changed without touching the logic.
"""

from __future__ import annotations

from typing import Final

# --- Pace: total active sightseeing hours available per day -------------------------
PACE_ACTIVE_HOURS: Final[dict[str, float]] = {
    "relaxed": 5.0,
    "moderate": 7.0,
    "active": 9.0,
}

#: Mandatory per-sight buffer (entry queue, parking, local hop) added to duration_hours.
SIGHT_BUFFER_HOURS: Final[float] = 0.3

#: Most stops a day may contain, per pace. A day of six 20-minute stops is not a real day.
MAX_SIGHTS_PER_DAY: Final[dict[str, int]] = {"relaxed": 4, "moderate": 5, "active": 6}

# --- Day overheads ------------------------------------------------------------------
#: Hours lost on the first day to arrival formalities / hotel check-in.
ARRIVAL_OVERHEAD_HOURS: Final[float] = 2.0
#: Hours lost on the last day to check-out and departure.
DEPARTURE_OVERHEAD_HOURS: Final[float] = 2.0
#: A city must reach at least this share of a daily allowance in usable hours before any
#: other city is given extra days; otherwise a city can end up with one unusable day.
MIN_CITY_USABLE_FRACTION: Final[float] = 0.5

#: Hours lost on an intercity transfer day on top of the drive itself (pack, check in/out).
TRANSFER_OVERHEAD_HOURS: Final[float] = 1.0

# --- Intercity travel realism -------------------------------------------------------
#: Longest journey that can be done inside one travel day (driver hours + stops).
MAX_SINGLE_DAY_TRAVEL_HOURS: Final[float] = 10.0
#: Journeys up to this length are possible as an overnight bus/train, which still costs
#: the day but is not impossible. Anything longer needs several days or a flight.
MAX_OVERNIGHT_TRAVEL_HOURS: Final[float] = 14.0
#: Plausible average speed band for a road leg; outside this the dataset row is suspect.
MIN_ROAD_SPEED_KMPH: Final[float] = 15.0
MAX_ROAD_SPEED_KMPH: Final[float] = 90.0
#: Speed checks are only meaningful on longer legs; short hops are dominated by town traffic.
SPEED_CHECK_MIN_KM: Final[float] = 50.0
#: Speed used when a dataset leg's own drive_hours is implausible and has to be recomputed.
RECOMPUTE_ROAD_SPEED_KMPH: Final[float] = 45.0

# --- Arrival / departure logistics --------------------------------------------------
#: Hotel checkout on the departure day.
CHECKOUT_HOURS: Final[float] = 0.5
#: Reporting time at the airport before a domestic flight.
AIRPORT_BUFFER_HOURS: Final[float] = 2.0
#: Reporting time at the railway station.
RAILWAY_BUFFER_HOURS: Final[float] = 1.0
#: Speed of the hotel -> airport/station transfer.
TRANSFER_SPEED_KMPH: Final[float] = 30.0
#: Used only when the dataset gives no distance to the airport/railhead. It is a labelled
#: assumption, reported as a data gap rather than treated as a fact.
ASSUMED_AIRPORT_TRANSFER_HOURS: Final[float] = 1.0
#: A departure day must keep at least this much slack, or the schedule is rejected.
MIN_DEPARTURE_BUFFER_HOURS: Final[float] = 0.5

# --- Local (intra-city) travel ------------------------------------------------------
#: Straight-line km are multiplied by this to approximate road distance.
ROAD_DISTANCE_FACTOR: Final[float] = 1.3
#: Average local driving speed (km/h); hill roads are slower.
LOCAL_SPEED_KMPH: Final[float] = 28.0
LOCAL_SPEED_KMPH_HILLS: Final[float] = 22.0
#: A city above this altitude uses the hill speed.
HILL_ALTITUDE_M: Final[float] = 1200.0
#: Fallback intercity speed when city_distances has no path between two cities.
INTERCITY_SPEED_KMPH: Final[float] = 45.0

# --- Geographic feasibility ---------------------------------------------------------
#: Max road km between two consecutive sights on the same day.
MAX_LEG_KM: Final[float] = 30.0
#: Max total sight-to-sight road km in one day (excluding the hotel legs).
MAX_DAY_ROUTE_KM: Final[float] = 70.0
#: A sight this far from the city centre is a day-trip excursion, not a city stop.
EXCURSION_KM: Final[float] = 35.0
#: How many ordinary stops may share a day with an excursion.
MAX_STOPS_WITH_EXCURSION: Final[int] = 2
#: A sight farther than this from its city centre is almost certainly filed under the
#: wrong city (e.g. "sunset point" sitting 315 km from Jaisalmer).
WRONG_CITY_KM: Final[float] = 120.0
#: Two sights closer than this with related names are the same place under two names.
SAME_PLACE_KM: Final[float] = 0.05
#: ...or this close when one name contains the other.
SAME_PLACE_NAME_KM: Final[float] = 1.0

# --- Ordering -----------------------------------------------------------------------
#: Sort order inside a day: morning sights first, sunset/evening last.
BEST_TIME_ORDER: Final[dict[str, int]] = {
    "morning": 0,
    "any": 1,
    "afternoon": 2,
    "evening": 3,
    "sunset": 4,
}

#: Words that are too generic to identify an attraction on their own.
GENERIC_SIGHT_WORDS: Final[frozenset[str]] = frozenset(
    {
        "museum", "fort", "palace", "temple", "lake", "garden", "gardens", "market",
        "local market", "park", "zoo", "church", "monastery", "gompa", "beach",
        "waterfall", "falls", "dam", "city tour", "view point", "viewpoint", "point",
        "mall", "bazaar", "village", "river", "hill", "hills", "valley", "pass",
        "island", "cave", "caves", "other",
    }
)

#: Words that only describe how a place is visited, not which place it is.
DESCRIPTOR_WORDS: Final[frozenset[str]] = frozenset(
    {"the", "sunrise", "sunset", "point", "view", "outer", "ride", "tour", "visit",
     "complex", "area", "and", "trip"}
)

#: Deterministic seed for KMeans so the same request always yields the same clusters.
RANDOM_SEED: Final[int] = 42
