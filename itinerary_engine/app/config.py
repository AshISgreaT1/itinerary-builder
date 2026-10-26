"""Tunable constants for the itinerary engine.

Everything the planner and the validator argue about lives here, so the rules can be
changed without touching the logic.
"""

from __future__ import annotations

import os
from pathlib import Path
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

#: A day given over entirely to travel has no sightseeing to fit around, so it can start
#: early and run long: this is the most driving such a day can absorb.
TRAVEL_DAY_MAX_DRIVE_HOURS: Final[float] = 13.0

# --- En-route stops (sights passed on the way between two cities) -------------------
#: How far off the road a stop may be before it stops being "on the way".
EN_ROUTE_MAX_DETOUR_KM: Final[float] = 15.0
#: Most en-route stops on one transfer day: the drive is still the point of the day.
EN_ROUTE_MAX_STOPS: Final[int] = 3
#: Ignore the first and last stretch of the leg; those sights belong to the cities.
EN_ROUTE_MIN_PROGRESS: Final[float] = 0.05
EN_ROUTE_MAX_PROGRESS: Final[float] = 0.95

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

#: Where measured road distances live, when they have been built (see ``app/osrm.py``).
#: Absent by default: without it every distance is the straight-line estimate above.
ROAD_CACHE_PATH: Final[Path] = Path(os.environ.get("ITINERARY_ROAD_CACHE", "data/road_distances.json"))
#: OSRM's default ``max-table-size``; a larger matrix needs the server started with more.
OSRM_MAX_TABLE_LOCATIONS: Final[int] = 100
#: A coordinate this far from the nearest road is reported: the measurement is suspect.
OSRM_SNAP_WARN_KM: Final[float] = 2.0
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
#: How far from the city centre a sight may be on the arrival day.
#:
#: Measured from the 1 241 agent-written days in the workbook: day 1 has a median of 4.0 km
#: from the centre and a 90th percentile of 23.2 km, and only 1.0% of its stops are 35 km or
#: more out (against 18.3% on later days). Arrival days are short and the traveller has just
#: travelled, so the agents keep day 1 in town; 25 km keeps 93% of what they actually did.
ARRIVAL_DAY_MAX_KM: Final[float] = 25.0
#: How many ordinary stops may share a day with an excursion.
MAX_STOPS_WITH_EXCURSION: Final[int] = 2
#: Hours a user must add to a day before it may hold one more stop than the pace allows.
EXTRA_HOURS_PER_STOP: Final[float] = 2.0
#: How many times the grouping refinement sweeps a city's days looking for a shorter
#: arrangement. Each sweep is O(days^2 x stops^2) on a handful of stops, and in practice
#: nothing improves after the second.
REFINE_PASSES: Final[int] = 2
#: How much further from the other day's stops a stop may be and still be worth testing.
#: Zero considers only clear-cut cases and misses re-orderings; too large and every pair of
#: stops is evaluated against every day, which costs seconds per plan.
REFINE_MARGIN_KM: Final[float] = 2.0
#: A sight farther than this from its city centre is almost certainly filed under the
#: wrong city (e.g. "sunset point" sitting 315 km from Jaisalmer).
WRONG_CITY_KM: Final[float] = 120.0
#: Two sights closer than this with related names are the same place under two names.
SAME_PLACE_KM: Final[float] = 0.05
#: ...or this close when one name contains the other.
SAME_PLACE_NAME_KM: Final[float] = 1.0

# --- Clock schedule -----------------------------------------------------------------
# The dataset has no opening_time/closing_time, and the agent itineraries carry no clock
# times, so these windows are the engine's own stated assumption -- reported with every
# schedule rather than passed off as data.
DAY_START_HOUR: Final[float] = 9.0          # a normal sightseeing day begins
TRANSFER_START_HOUR: Final[float] = 8.0     # driving days leave earlier
LATEST_END_HOUR: Final[float] = 20.0        # past this a day is no longer realistic
#: When each ``best_time`` value can be visited (24h clock).
TIME_WINDOWS: Final[dict[str, tuple[float, float]]] = {
    "morning": (8.0, 12.5),
    "afternoon": (12.5, 16.5),
    "evening": (16.5, 20.0),
    "sunset": (16.5, 19.5),
    "any": (8.0, 20.0),
}
#: Slots that are genuinely time-critical: a sunset point at 10:00 is simply wrong.
HARD_TIME_SLOTS: Final[frozenset[str]] = frozenset({"sunset", "evening"})
#: Soft slots only complain when a visit lands in the wrong half of the day. Nearly half
#: the dataset is marked "morning", so no day could honour them all literally.
MORNING_LATEST_START: Final[float] = 14.0
AFTERNOON_EARLIEST_START: Final[float] = 11.0
AFTERNOON_LATEST_START: Final[float] = 18.0
#: Longest the plan will have someone wait for a sunset/evening slot. Beyond this the wait
#: is not worth it: the sight is visited early and the clash is reported instead.
MAX_WAIT_HOURS: Final[float] = 2.5
#: How many times one plan may be re-planned to rescue a sight that missed its slot.
MAX_SLOT_REPAIRS: Final[int] = 2
#: How many sights one plan may re-home or drop before it stops trying. Each round is a
#: full re-plan, and every round either places a sight or removes one, so this terminates.
MAX_SLOT_ROUNDS: Final[int] = 6

#: Slack allowed before a hard slot counts as missed.
TIME_WINDOW_GRACE_HOURS: Final[float] = 0.5

#: Lunch: taken once, when the day crosses this window.
LUNCH_WINDOW: Final[tuple[float, float]] = (12.5, 14.0)
LUNCH_HOURS: Final[float] = 1.0
#: A short rest once the day has been going this long, and how long it lasts.
BREAK_AFTER_HOURS: Final[float] = 4.0
BREAK_HOURS: Final[float] = 0.25
#: Hotel check-in after arriving in a new city.
CHECK_IN_HOURS: Final[float] = 0.5

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
