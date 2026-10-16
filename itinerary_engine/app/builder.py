"""Orchestration: request -> route -> day slots -> clusters -> packed days -> JSON."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace

import pandas as pd

from .clustering import ClusterResult, cluster_city_sights
from .config import (
    AIRPORT_BUFFER_HOURS,
    ASSUMED_AIRPORT_TRANSFER_HOURS,
    CHECKOUT_HOURS,
    DAY_START_HOUR,
    LATEST_END_HOUR,
    HARD_TIME_SLOTS,
    MAX_SLOT_REPAIRS,
    MAX_SLOT_ROUNDS,
    MIN_CITY_USABLE_FRACTION,
    RAILWAY_BUFFER_HOURS,
    SIGHT_BUFFER_HOURS,
)
from .data_loader import Dataset, get_dataset
from .enroute import EnRouteStop, plan_en_route
from .patterns import patterns_for
from .schedule import SCHEDULE_ASSUMPTIONS, build_day_schedule, clock
from .geo import (RouteLeg, local_travel_hours, parse_access_distance_km, road_km,
                  road_km_is_measured, transfer_hours_for_distance)
from .normalization import normalize_key
from .solver import DaySlot, PackedDay, pace_hours, plan_city, sort_by_priority
from .validation import validate_itinerary

logger = logging.getLogger(__name__)


class PlanningError(ValueError):
    """Raised when a request cannot be planned (unknown city, too few days...)."""


@dataclass
class PinnedSight:
    """A sight the user asked for, optionally on a particular day."""

    sight: str
    city: str | None = None
    day: int | None = None


@dataclass
class CustomActivity:
    """Something the user added that the dataset knows nothing about.

    It has no coordinates, so it costs no travel: it is simply time the day owes to
    something, placed where the user asked for it.
    """

    name: str
    day: int
    duration_hours: float = 1.0
    #: Optional clock time ("19:30"). Without one it goes where it fits in the day.
    at: str | None = None
    note: str = ""

    def start_hour(self) -> float | None:
        if not self.at:
            return None
        hours, _, minutes = str(self.at).partition(":")
        try:
            return int(hours) + int(minutes or 0) / 60.0
        except ValueError:
            return None


@dataclass
class PlanRequest:
    """A validated planning request."""

    cities: list[str]
    duration_days: int
    pace: str = "moderate"
    origin_city: str | None = None
    start_month: int | None = None
    include_descriptions: bool = True
    #: Sights the user added by hand; placed before anything else.
    pinned_sights: list[PinnedSight] = field(default_factory=list)
    #: Sights the user removed or replaced; never scheduled.
    excluded_sights: list[str] = field(default_factory=list)
    #: Sights the engine itself could not place at the right time of day. Held apart from
    #: ``excluded_sights`` because the user did not ask for these: the plan has to explain
    #: them (see ``unplaceable_reasons``), not present them as a choice.
    unplaceable_sights: list[str] = field(default_factory=list)
    #: The user's own split of the trip across cities, by city name: ``{"Jaipur": 4,
    #: "Udaipur": 2}``. Overrides the automatic allocation when given.
    city_days: dict[str, int] = field(default_factory=dict)
    #: Extra hours the user is willing to spend on a given day, by day number. The pace
    #: stays what it is -- this is the traveller saying "I will do a longer day here".
    extra_day_hours: dict[int, float] = field(default_factory=dict)
    #: The user's own order for a day's stops, by day number, as a list of sight names.
    day_order: dict[int, list[str]] = field(default_factory=dict)
    #: Things the user added that are not in the dataset (a dinner, a show, a nap).
    custom_activities: list[CustomActivity] = field(default_factory=list)
    #: Times the user asked for, by day number then sight name: ``{2: {"Lake Pichola": "16:30"}}``.
    sight_times: dict[int, dict[str, str]] = field(default_factory=dict)
    #: Start times the user chose, by day number: ``{2: "07:30"}``. An early start buys
    #: daylight the pace allowance cannot.
    day_starts: dict[int, str] = field(default_factory=dict)

    def pins_for_city(self, city_key: str) -> list[PinnedSight]:
        return [pin for pin in self.pinned_sights if pin.city is None or normalize_key(pin.city) == city_key]

    def is_user_choice(self, sight: str) -> bool:
        """Did the user put this sight here by hand? Then the engine does not overrule it."""
        key = normalize_key(sight)
        return any(normalize_key(pin.sight) == key for pin in self.pinned_sights)

    def activities_for_day(self, day: int) -> list[CustomActivity]:
        return [item for item in self.custom_activities if item.day == day]


def departure_cost(city_row) -> tuple[float, str, bool]:
    """Hours the departure day loses, from the dataset's own airport/railhead distance.

    Returns (hours, explanation, data_gap). When the dataset gives no distance the engine
    uses a clearly labelled assumption and reports it as a gap instead of a fact.
    """
    airport_km = parse_access_distance_km(city_row.get("nearest_airport"))
    rail_km = parse_access_distance_km(city_row.get("nearest_railhead"))
    if airport_km is not None and (rail_km is None or airport_km <= rail_km):
        transfer = transfer_hours_for_distance(airport_km)
        total = CHECKOUT_HOURS + transfer + AIRPORT_BUFFER_HOURS
        return total, (
            f"checkout {CHECKOUT_HOURS}h + {transfer:.1f}h to {city_row.get('nearest_airport')} "
            f"+ {AIRPORT_BUFFER_HOURS}h airport reporting"
        ), False
    if rail_km is not None:
        transfer = transfer_hours_for_distance(rail_km)
        total = CHECKOUT_HOURS + transfer + RAILWAY_BUFFER_HOURS
        return total, (
            f"checkout {CHECKOUT_HOURS}h + {transfer:.1f}h to {city_row.get('nearest_railhead')} "
            f"+ {RAILWAY_BUFFER_HOURS}h station reporting"
        ), False
    total = CHECKOUT_HOURS + ASSUMED_AIRPORT_TRANSFER_HOURS + AIRPORT_BUFFER_HOURS
    return total, (
        f"checkout {CHECKOUT_HOURS}h + an assumed {ASSUMED_AIRPORT_TRANSFER_HOURS}h transfer "
        f"+ {AIRPORT_BUFFER_HOURS}h reporting (no airport/railhead distance in the dataset)"
    ), True


def _content_hours(dataset: Dataset, city_key: str) -> float:
    """Total buffered sightseeing hours a city offers."""
    sights = dataset.city_sights(city_key)
    return float((sights["duration_hours"] + SIGHT_BUFFER_HOURS).sum()) if not sights.empty else 0.0


def _usable_hours_by_city(
    dataset: Dataset, request: PlanRequest, city_keys: list[str], allocation: dict[str, int], pace: str
) -> dict[str, float]:
    """Sightseeing hours each city actually gets under an allocation.

    Days are not interchangeable: the day a traveller drives in is mostly gone, and so is
    the day they fly home. Allocating on day counts alone is what leaves a last city with
    a single day that is both, and therefore empty.

    The candidate is measured as if it were the whole trip. Otherwise a part-filled
    allocation never reaches the departure day, its final day looks like a free one, and a
    city that is really all travel (Leh to Pangong Lake: 5.5h in, a 140 km airport transfer
    out) is judged "already covered" and never gets a usable day.
    """
    probe = replace(request, duration_days=max(1, sum(allocation.values())))
    slots, _ = build_day_slots(dataset, probe, city_keys, allocation)
    usable = {key: 0.0 for key in city_keys}
    for slot in slots:
        usable[slot.city_key] += slot.available_hours(pace)
    return usable


def _requested_allocation(dataset: Dataset, request: PlanRequest, city_keys: list[str]) -> dict[str, int] | None:
    """The user's own split of the days, checked before it is used.

    A split that does not add up, or that starves a city, is refused with the arithmetic
    rather than quietly corrected -- the user asked for something specific.
    """
    if not request.city_days:
        return None
    wanted: dict[str, int] = {}
    for name, days in request.city_days.items():
        row = dataset.city_row(name)
        if row is None:
            raise PlanningError(f"Unknown city '{name}' in city_days.")
        key = str(row["city_key"])
        if key not in city_keys:
            raise PlanningError(f"'{name}' is in city_days but not in the trip's cities.")
        if int(days) < 1:
            raise PlanningError(f"'{name}' needs at least one day; {days} were given.")
        wanted[key] = int(days)

    missing = [key for key in city_keys if key not in wanted]
    if missing:
        names = ", ".join(str(dataset.cities[dataset.cities["city_key"] == key].iloc[0]["city"]) for key in missing)
        raise PlanningError(f"city_days must cover every city in the trip; {names} missing.")
    total = sum(wanted.values())
    if total != request.duration_days:
        raise PlanningError(
            f"city_days adds up to {total} day(s) but the trip is {request.duration_days}. "
            f"Change the split or the trip length."
        )
    return {key: wanted[key] for key in city_keys}


def allocate_city_days(
    dataset: Dataset, city_keys: list[str], duration_days: int, pace: str, request: PlanRequest
) -> dict[str, int]:
    """Split the trip's days between cities.

    Every city gets at least one day (and its ``min_days`` where the dataset has them).
    Each remaining day is given to whichever city gains the most usable sightseeing time
    from it, so cities whose day is eaten by a long drive get a second day before a city
    that is already well covered gets a fourth.
    """
    if duration_days < len(city_keys):
        raise PlanningError(
            f"{duration_days} day(s) cannot cover {len(city_keys)} cities; allow at least one day per city."
        )

    allocation: dict[str, int] = {}
    for key in city_keys:
        row = dataset.cities[dataset.cities["city_key"] == key].iloc[0]
        allocation[key] = max(1, int(row.get("min_days", 1) or 1))
    if sum(allocation.values()) > duration_days:  # min_days cannot exceed the trip length
        allocation = {key: 1 for key in city_keys}

    content = {key: _content_hours(dataset, key) for key in city_keys}
    remaining = duration_days - sum(allocation.values())

    # Phase 1 (fairness): no city should be left with a day that travel swallows whole.
    floor_hours = MIN_CITY_USABLE_FRACTION * pace_hours(pace)
    while remaining > 0:
        usable = _usable_hours_by_city(dataset, request, city_keys, allocation, pace)
        shortfalls = {
            key: min(content[key], floor_hours) - usable[key]
            for key in city_keys
            if content[key] > 0 and usable[key] < min(content[key], floor_hours) - 1e-9
        }
        if not shortfalls:
            break
        neediest = max(shortfalls, key=lambda key: (shortfalls[key], -city_keys.index(key)))
        allocation[neediest] += 1
        remaining -= 1

    # Phase 2 (efficiency): spend what is left where it buys the most sightseeing time.
    while remaining > 0:
        current = _usable_hours_by_city(dataset, request, city_keys, allocation, pace)
        best_key, best_gain = None, None
        for key in city_keys:
            trial = dict(allocation)
            trial[key] += 1
            gained = _usable_hours_by_city(dataset, request, city_keys, trial, pace)
            # Hours beyond what the city can fill are worthless, hence the min().
            gain = min(content[key], gained[key]) - min(content[key], current[key])
            if best_gain is None or gain > best_gain + 1e-9:
                best_key, best_gain = key, gain
        if best_gain is not None and best_gain <= 1e-9:
            # Everything is already covered: give the day to the richest city.
            best_key = max(city_keys, key=lambda c: (content[c], -city_keys.index(c)))
        allocation[best_key] += 1
        remaining -= 1
    return allocation


def build_day_slots(dataset: Dataset, request: PlanRequest, city_keys: list[str], allocation: dict[str, int]) -> tuple[list[DaySlot], list[RouteLeg]]:
    """Lay the trip out day by day, charging intercity drives to the day they happen on."""
    slots: list[DaySlot] = []
    legs: list[RouteLeg] = []
    day_number = 1
    previous_key = normalize_key(request.origin_city) if request.origin_city else None
    previous_city = request.origin_city

    for key in city_keys:
        row = dataset.cities[dataset.cities["city_key"] == key].iloc[0]
        for index in range(allocation[key]):
            transfer = None
            if index == 0 and previous_key and previous_key != key:
                transfer = dataset.graph.leg(str(previous_city), str(row["city"]), previous_key, key)
                legs.append(transfer)
            is_departure = day_number == request.duration_days
            cost_hours, _, _ = departure_cost(row) if is_departure else (0.0, "", False)
            slots.append(
                DaySlot(
                    day=day_number,
                    city=str(row["city"]),
                    city_key=key,
                    centre_lat=float(row["lat"]),
                    centre_lon=float(row["lon"]),
                    altitude_m=row.get("altitude_m"),
                    transfer=transfer,
                    is_arrival=(day_number == 1),
                    is_departure=is_departure,
                    departure_cost_hours=cost_hours,
                    # An unknown or multi-day journey leaves no room for sightseeing.
                    travel_only=bool(transfer is not None and not transfer.fits_one_day),
                    order_patterns=patterns_for(dataset),
                    extra_hours=float(request.extra_day_hours.get(day_number, 0.0)),
                )
            )
            day_number += 1
        previous_key, previous_city = key, str(row["city"])
    return slots, legs


def _eligible_sights(
    dataset: Dataset, city_key: str, days_in_city: int, month: int | None, request: PlanRequest | None = None
) -> pd.DataFrame:
    """City sights that are open in the travel month and worth the length of stay.

    User edits are applied here: excluded sights disappear, and pinned sights are kept even
    if the length-of-stay filter would normally drop them, then flagged so the packer puts
    them in first.
    """
    sights = dataset.city_sights(city_key)
    if sights.empty:
        return sights
    pins = request.pins_for_city(city_key) if request else []
    pinned_keys = {normalize_key(pin.sight) for pin in pins}
    excluded_keys = {normalize_key(name.split(" / ")[-1])
                     for name in ((request.excluded_sights + request.unplaceable_sights) if request else [])}

    eligible = sights[(sights["min_days"] <= days_in_city) | sights["sight_key"].isin(pinned_keys)]
    if month:
        eligible = eligible[eligible["open_months_set"].map(lambda months: month in months)]
    eligible = eligible[~eligible["sight_key"].isin(excluded_keys)].reset_index(drop=True)
    if pinned_keys:
        eligible["_forced"] = eligible["sight_key"].isin(pinned_keys)
        day_by_key = {normalize_key(pin.sight): pin.day for pin in pins if pin.day is not None}
        eligible["_forced_day"] = eligible["sight_key"].map(day_by_key)
    return eligible


def build_itinerary(request: PlanRequest, dataset: Dataset | None = None, repair: bool = True) -> dict:
    """Build the full itinerary payload for a request.

    ``repair`` lets the builder re-plan once to rescue a sight that cannot reach its
    preferred slot where it landed; the retry itself runs with ``repair=False``.
    """
    dataset = dataset or get_dataset()
    pace = request.pace.strip().lower()
    pace_hours(pace)  # validates the pace name

    if not request.cities:
        raise PlanningError("At least one city is required.")
    if request.duration_days < 1:
        raise PlanningError("duration_days must be 1 or more.")

    city_keys: list[str] = []
    for name in request.cities:
        row = dataset.city_row(name)
        if row is None:
            raise PlanningError(f"Unknown city '{name}'. Known cities: {', '.join(dataset.city_names[:10])}...")
        if row["city_key"] not in city_keys:
            city_keys.append(row["city_key"])
    if request.origin_city and dataset.city_row(request.origin_city) is None:
        raise PlanningError(f"Unknown origin city '{request.origin_city}'.")

    _check_pins(dataset, request, city_keys)

    allocation = (_requested_allocation(dataset, request, city_keys)
                  or allocate_city_days(dataset, city_keys, request.duration_days, pace, request))
    slots, legs = build_day_slots(dataset, request, city_keys, allocation)

    # En-route stops first: they belong to the drive, not to either city, and the hours
    # they take must be reserved before the destination city packs its own day.
    trip_city_keys = set(city_keys)
    en_route_by_day: dict[int, list[EnRouteStop]] = {}
    en_route_keys: set[str] = set()
    for slot in slots:
        if slot.transfer is None:
            continue
        budget = slot.available_hours(pace)
        stops = plan_en_route(dataset, slot.transfer, budget, trip_city_keys, request.start_month, en_route_keys)
        if not stops:
            continue
        en_route_by_day[slot.day] = stops
        en_route_keys |= {str(stop.sight["sight_key"]) for stop in stops}
        # Whatever the stops cost is no longer available to the destination city.
        slot.available_override = max(0.0, round(budget - sum(stop.cost_hours for stop in stops), 2))

    packed_days: list[PackedDay] = []
    cluster_methods: dict[str, str] = {}
    skipped_by_season: list[str] = []

    for key in city_keys:
        city_slots = [slot for slot in slots if slot.city_key == key]
        usable_slots = [slot for slot in city_slots if slot.available_hours(pace) > 0]
        days_in_city = len(city_slots)
        eligible = _eligible_sights(dataset, key, days_in_city, request.start_month, request)
        if en_route_keys and not eligible.empty:
            eligible = eligible[~eligible["sight_key"].isin(en_route_keys)].reset_index(drop=True)
        all_city_sights = dataset.city_sights(key)
        if request.start_month is not None:
            closed = all_city_sights[~all_city_sights["sight"].isin(eligible["sight"])]
            skipped_by_season += [f"{row['city']} / {row['sight']}" for _, row in closed.iterrows()]

        centre = dataset.cities[dataset.cities["city_key"] == key].iloc[0]
        clustered: ClusterResult = cluster_city_sights(
            eligible, max(1, len(usable_slots)), float(centre["lat"]), float(centre["lon"])
        )
        cluster_methods[str(centre["city"])] = clustered.method
        packed_days += plan_city(clustered.clusters, clustered.excursions, city_slots, pace)

    packed_days.sort(key=lambda day: day.slot.day)
    payload = _assemble_payload(dataset, request, pace, packed_days, legs, allocation, cluster_methods,
                                skipped_by_season, en_route_by_day)
    payload["unscheduled_reasons"] = _unscheduled_reasons(dataset, payload["unscheduled_must_visits"], slots, pace)
    patterns = patterns_for(dataset)
    _attach_customisations(payload, request)
    _attach_schedules(payload, dataset, patterns)
    payload["summary"]["schedule_assumptions"] = SCHEDULE_ASSUMPTIONS
    payload["summary"]["road_distances"] = _distance_provenance(payload)
    payload["summary"]["learned_from_past_itineraries"] = {
        "days_read": patterns.days_seen,
        "sights_read": patterns.sights_seen,
        "typical_sights_per_day": patterns.typical_sights_per_day,
        "ordering_pairs": len(patterns.pair_counts),
        "source_has_clock_times": patterns.has_clock_times,
    }
    if repair:
        payload = _resolve_hard_slots(request, dataset, payload, pace) or payload
    report = validate_itinerary(payload, dataset, pace, request.start_month)
    payload["validation"] = report.as_dict()
    if not report.passed and repair:
        logger.warning("Itinerary for %s failed validation: %s", request.cities, report.as_dict()["issues"])
    return payload


def _assemble_payload(
    dataset: Dataset,
    request: PlanRequest,
    pace: str,
    packed_days: list[PackedDay],
    legs: list[RouteLeg],
    allocation: dict[str, int],
    cluster_methods: dict[str, str],
    skipped_by_season: list[str],
    en_route_by_day: dict[int, list[EnRouteStop]] | None = None,
) -> dict:
    """Turn packed days into the response JSON, joining descriptions on (city, sight)."""
    days_payload = []
    scheduled: set[tuple[str, str]] = set()

    en_route_by_day = en_route_by_day or {}
    for packed in packed_days:
        slot = packed.slot
        sights_payload = []
        stops = en_route_by_day.get(slot.day, [])
        for stop in stops:  # visited during the drive, in road order
            entry = stop.payload()
            if request.include_descriptions:
                entry["description"] = dataset.description(entry["home_city"], entry["sight"])
            sights_payload.append(entry)
            scheduled.add((normalize_key(entry["home_city"]), normalize_key(entry["sight"])))
        en_route_hours = round(sum(stop.cost_hours for stop in stops), 2)
        for sight in packed.sights:
            enriched = dict(sight)
            # A stop the user put here by hand is theirs: the engine advises, never overrules.
            enriched["user_pinned"] = request.is_user_choice(sight["sight"])
            if request.include_descriptions:
                enriched["description"] = dataset.description(slot.city, sight["sight"])
            sights_payload.append(enriched)
            scheduled.add((slot.city_key, normalize_key(sight["sight"])))

        costs = slot.cost_breakdown(pace)
        days_payload.append(
            {
                "day": slot.day,
                "city": slot.city,
                "transfer": slot.transfer.as_dict() if slot.transfer else None,
                "is_arrival_day": slot.is_arrival,
                "is_departure_day": slot.is_departure,
                "is_travel_only": slot.travel_only,
                "time_budget": {
                    "pace_allowance": costs["pace_allowance"],
                    "intercity_travel_hours": round(costs["intercity_travel"], 2) if costs["intercity_travel"] != float("inf") else None,
                    "day_overhead_hours": round(costs["day_overheads"], 2),
                    "departure_cost_hours": round(slot.departure_cost_hours, 2) if slot.is_departure else 0.0,
                    "extra_hours": round(slot.extra_hours, 2),
                },
                "available_hours": round(slot.available_hours(pace) + en_route_hours, 2),
                "used_hours": round(packed.used_hours + en_route_hours, 2),
                "spare_hours": round(slot.available_hours(pace) - packed.used_hours, 2),
                "en_route_hours": en_route_hours,
                "sightseeing_hours": round(packed.sightseeing_hours + sum(float(s.sight["duration_hours"]) for s in stops), 2),
                "buffer_hours": round(packed.buffer_hours + SIGHT_BUFFER_HOURS * len(stops), 2),
                "local_travel_hours": round(packed.local_travel_hours + sum(s.detour_hours for s in stops), 2),
                "route_km": packed.route_km,
                "max_leg_km": packed.max_leg_km,
                "sights": sights_payload,
                "notes": _day_notes(slot, packed, stops),
            }
        )

    unscheduled, unscheduled_must = [], []
    for key in allocation:
        for _, row in sort_by_priority(dataset.city_sights(key)).iterrows():
            if (key, row["sight_key"]) in scheduled:
                continue
            label = f"{row['city']} / {row['sight']}"
            unscheduled.append(label)
            if row["must_visit_flag"]:
                unscheduled_must.append(label)

    total_sights = sum(len(day["sights"]) for day in days_payload)
    # Nights, not days: the last day of the trip is a checkout day, so it has no night.
    days_by_city: dict[str, int] = {}
    for day in days_payload:
        days_by_city[day["city"]] = days_by_city.get(day["city"], 0) + 1
    nights_by_city = dict(days_by_city)
    if days_payload:
        last_city = days_payload[-1]["city"]
        nights_by_city[last_city] = max(0, nights_by_city[last_city] - 1)
    data_gaps = [
        f"{leg.from_city} to {leg.to_city}: " + " ".join(leg.notes)
        for leg in legs
        if leg.data_gap
    ]
    departure_note = ""
    if days_payload:
        last_row = dataset.city_row(days_payload[-1]["city"])
        if last_row is not None:
            _, departure_note, departure_gap = departure_cost(last_row)
            if departure_gap:
                data_gaps.append(f"{days_payload[-1]['city']}: {departure_note}")
    return {
        "request": {
            "cities": [day for day in dict.fromkeys(d["city"] for d in days_payload)],
            "duration_days": request.duration_days,
            "pace": pace,
            "origin_city": request.origin_city,
            "start_month": request.start_month,
        },
        "summary": {
            "days": len(days_payload),
            "cities": len(allocation),
            "days_per_city": days_by_city,
            "nights_per_city": nights_by_city,
            "total_nights": sum(nights_by_city.values()),
            "departure_day_reserved_for": departure_note,
            "total_sights": total_sights,
            "total_sightseeing_hours": round(sum(d["sightseeing_hours"] for d in days_payload), 2),
            "total_local_travel_hours": round(sum(d["local_travel_hours"] for d in days_payload), 2),
            "total_intercity_drive_hours": round(sum(leg.drive_hours or 0.0 for leg in legs), 2),
            "legs_without_journey_time": [f"{leg.from_city} to {leg.to_city}" for leg in legs if leg.drive_hours is None],
            "daily_hour_cap": pace_hours(pace),
            "clustering_method_by_city": cluster_methods,
        },
        "route": [leg.as_dict() for leg in legs],
        "data_gaps": data_gaps,
        "days": days_payload,
        "unscheduled_sights": unscheduled,
        "unscheduled_must_visits": unscheduled_must,
        "closed_this_month": sorted(set(skipped_by_season)),
    }


def _attach_customisations(payload: dict, request: PlanRequest) -> None:
    """Put the user's own order, times and extra activities on the days they belong to."""
    for day in payload.get("days", []):
        number = day.get("day")
        times = {name: value for name, value in (request.sight_times.get(number) or {}).items()}
        if times:
            day["requested_times"] = times
        order = request.day_order.get(number)
        if order:
            day["requested_order"] = list(order)
            position = {normalize_key(name): index for index, name in enumerate(order)}
            day["sights"].sort(key=lambda sight: position.get(normalize_key(sight["sight"]), len(position)))
            # An order the user set by hand is a choice like any other: the clock keeps it
            # and advises, instead of re-homing a stop that now misses its slot.
            for sight in day["sights"]:
                if normalize_key(sight["sight"]) in position:
                    sight["user_pinned"] = True
        start = request.day_starts.get(number)
        if start:
            day["requested_start"] = start
        activities = request.activities_for_day(number)
        if activities:
            day["custom_activities"] = [
                {"name": item.name, "duration_hours": item.duration_hours, "at": item.at, "note": item.note}
                for item in activities
            ]


def _attach_schedules(payload: dict, dataset: Dataset, patterns) -> None:
    """Put a clock against every day and make the day's totals match it.

    The schedule is the single source of truth for time: once it exists, the day reports
    what it actually contains -- sightseeing, local driving, the intercity leg, meals,
    waiting and check-in -- instead of a separate running total that can drift from it.
    """
    for day in payload["days"]:
        schedule = build_day_schedule(day, dataset, patterns).as_dict()
        day["schedule"] = schedule
        totals = schedule["totals"]
        day["used_hours"] = schedule["activity_hours"]
        day["sightseeing_hours"] = totals["sightseeing"]
        day["local_travel_hours"] = totals["local_travel"]
        day["buffer_hours"] = totals["stop_buffers"]
        day["spare_hours"] = round(day["available_hours"] - schedule["activity_hours"], 2)
        day["day_totals"] = {
            "activity_hours": schedule["activity_hours"],
            "meal_and_break_hours": totals["meals_and_breaks"],
            "waiting_hours": totals["waiting"],
            "intercity_travel_hours": totals["intercity_travel"],
            "check_in_out_hours": totals["check_in_out"],
            "your_own_plans_hours": totals["your_own_plans"],
            "span_hours": schedule["span_hours"],
            "starts": schedule["starts"],
            "ends": schedule["ends"],
        }


def _check_pins(dataset: Dataset, request: PlanRequest, city_keys: list[str]) -> None:
    """Refuse a pin that names a sight the dataset does not have.

    Silently dropping it was worse than failing: the user asked for a stop, the plan came
    back without it, and nothing said why.
    """
    import difflib

    known: dict[str, set[str]] = {}
    for key in city_keys:
        known[key] = set(dataset.city_sights(key)["sight_key"])
    for pin in request.pinned_sights:
        pin_key = normalize_key(pin.sight)
        targets = [normalize_key(pin.city)] if pin.city else list(city_keys)
        if any(pin_key in known.get(key, set()) for key in targets):
            continue
        pool = [name for key in targets for name in dataset.city_sights(key)["sight"]]
        close = difflib.get_close_matches(pin.sight, pool, n=3, cutoff=0.6)
        hint = f" Did you mean: {', '.join(close)}?" if close else ""
        where = pin.city or ", ".join(request.cities)
        raise PlanningError(f"'{pin.sight}' is not a sight in {where}.{hint}")


def _clock_hours(text: str | None) -> float:
    """``"19:19"`` -> 19.32. Missing times sort as the start of the day."""
    if not text:
        return DAY_START_HOUR
    hours, _, minutes = str(text).partition(":")
    return int(hours) + int(minutes or 0) / 60.0


def _move_candidates(payload: dict, source: dict, sight_name: str, window_start: float) -> list[int]:
    """Days in the same city that could plausibly host ``sight_name`` in its slot.

    Plausible, not proven: the rebuild is what decides. This only avoids re-planning
    against days that obviously cannot take the stop, and keeps the roomiest first.
    """
    entry = next((item for item in source.get("sights", []) if item["sight"] == sight_name), None)
    visit = float(entry.get("duration_hours", 0.0)) if entry else 0.0
    ranked: list[tuple[float, int]] = []
    for day in payload.get("days", []):
        if day["day"] == source["day"] or day.get("city") != source.get("city"):
            continue
        if day.get("is_travel_only"):
            continue
        schedule = day.get("schedule") or {}
        spare = float(day.get("available_hours", 0.0)) - float(schedule.get("activity_hours", 0.0))
        if spare < visit + SIGHT_BUFFER_HOURS:
            continue
        if max(_clock_hours(schedule.get("ends")), window_start) + visit > LATEST_END_HOUR:
            continue
        ranked.append((spare, day["day"]))
    return [day for _, day in sorted(ranked, reverse=True)]


def _why_no_move(payload: dict, source: dict, window_start: float) -> str:
    """Day by day, the reason none of them can take the stop."""
    reasons = []
    for day in payload.get("days", []):
        if day["day"] == source["day"] or day.get("city") != source.get("city"):
            continue
        schedule = day.get("schedule") or {}
        ends = schedule.get("ends")
        spare = float(day.get("available_hours", 0.0)) - float(schedule.get("activity_hours", 0.0))
        if day.get("is_travel_only"):
            reasons.append(f"day {day['day']} is a travel day")
        elif day.get("is_departure_day"):
            reasons.append(f"day {day['day']} is the departure day")
        elif spare < SIGHT_BUFFER_HOURS:
            reasons.append(f"day {day['day']} has {spare:.1f}h left of its {day.get('available_hours')}h budget")
        elif ends and _clock_hours(ends) > window_start:
            reasons.append(f"day {day['day']} is already busy until {ends}")
        else:
            reasons.append(f"day {day['day']} has only {spare:.1f}h spare")
    inside = f" ({'; '.join(reasons)})" if reasons else ""
    return (f"no other day in {source.get('city')} can host it after {clock(window_start)}{inside}; "
            f"add a day in {source.get('city')}, or accept the earlier visit")


def _unresolved(payload: dict, sight_name: str) -> bool:
    return any(entry.get("sight") == sight_name
               for day in payload.get("days", [])
               for entry in (day.get("schedule") or {}).get("alternatives", []))


def _deferred_everywhere(payload: dict) -> list[tuple[dict, dict]]:
    """Every hard-slot sight the clock could not place, with the day it came from."""
    return [(day, entry) for day in payload.get("days", [])
            for entry in (day.get("schedule") or {}).get("deferred", [])]


def _evening_candidates(payload: dict, source: dict, entry: dict) -> list[int]:
    """Days in the same city that might host this sight inside its window (rule 3).

    Ordered by how free their evening already is: a day with no hard-slot sight of its own
    can simply start later, while one that already ends at a sunset point cannot stretch
    much further. The re-plan is what decides; this only picks the order to try.
    """
    window_start = float(entry.get("window_start", DAY_START_HOUR))
    visit = float(entry.get("visit_hours", 0.0))
    ranked: list[tuple[int, float, int]] = []
    for day in payload.get("days", []):
        if day["day"] == source["day"] or day.get("city") != source.get("city"):
            continue
        if day.get("is_travel_only") or day.get("is_departure_day"):
            continue  # a travel day has no evening, and the last day owes the onward journey
        schedule = day.get("schedule") or {}
        hard_already = sum(1 for sight in day.get("sights", [])
                           if str(sight.get("best_time", "")).lower() in HARD_TIME_SLOTS)
        spare = float(day.get("available_hours", 0.0)) - float(schedule.get("activity_hours", 0.0))
        if spare < visit:
            continue
        # A day that cannot be stretched to the window at all is not worth a re-plan.
        if window_start + visit > LATEST_END_HOUR + 1e-9:
            continue
        ranked.append((hard_already, -spare, day["day"]))
    return [day for _, _, day in sorted(ranked)]


def _resolve_hard_slots(request: PlanRequest, dataset: Dataset, payload: dict, pace: str) -> dict | None:
    """Re-home every sight the clock could not fit inside its hard slot.

    The scheduler refuses to put an evening sight in the morning, so each one it could not
    place arrives here. In order:

    1. move it to another day in the same city whose evening is free (rule 3), by re-planning
       with it pinned there -- the whole pipeline then re-costs the route and the clock;
    2. failing that, take it out of the trip and say why, so the day is packed with something
       that does fit and the sight is still offered in ``unscheduled_sights``.

    Never by putting it on at the wrong time of day.
    """
    if not _deferred_everywhere(payload) and not _slot_alternatives(payload):
        return None

    current = payload
    reasons: dict[str, str] = dict(payload.get("unplaceable_reasons") or {})
    dropped = list(request.unplaceable_sights)

    for _ in range(MAX_SLOT_ROUNDS):
        pending = _deferred_everywhere(current)
        if not pending:
            break
        source, entry = pending[0]
        name = entry["sight"]
        moved = False
        for target in _evening_candidates(current, source, entry)[:MAX_SLOT_REPAIRS]:
            trial = _try_plan(request, dataset, pinned=[PinnedSight(name, source.get("city"), target)],
                              unplaceable=dropped)
            if trial is None or _is_deferred(trial, name):
                continue
            if len(_deferred_everywhere(trial)) >= len(_deferred_everywhere(current)):
                continue  # it only moved the problem to another day
            if len(trial.get("unscheduled_must_visits", [])) > len(current.get("unscheduled_must_visits", [])):
                continue  # re-homing one sight must not push a must-visit out
            for day in trial["days"]:
                if day["day"] == target:
                    day.setdefault("notes", []).append(
                        f"'{name}' was moved here from day {source['day']} so it falls inside its "
                        f"{entry.get('best_time', 'preferred')} window."
                    )
            current, moved = trial, True
            break
        if moved:
            continue
        # No day can hold it at the right time: leave it out rather than visit it wrongly.
        dropped.append(f"{source.get('city')} / {name}")
        reasons[f"{source.get('city')} / {name}"] = (
            f"{entry['reason']}, and no other day in {source.get('city')} has a free "
            f"{entry.get('best_time', 'evening')} slot. Left out rather than scheduled outside "
            f"{clock(float(entry.get('window_start', 0)))}-{clock(float(entry.get('window_end', 0)))}; "
            f"add a day, or accept seeing it at the wrong time by pinning it."
        )
        trial = _try_plan(request, dataset, unplaceable=dropped)
        if trial is None:
            break
        current = trial

    current["unplaceable_reasons"] = reasons
    _sharpen_suggestions(current)
    return current


def _try_plan(request: PlanRequest, dataset: Dataset, pinned: list[PinnedSight] | None = None,
              unplaceable: list[str] | None = None) -> dict | None:
    """One re-plan attempt, or None when it cannot be built at all.

    A trial is *not* judged by the validation gate: while sights are still waiting to be
    re-homed the plan is mid-flight and cannot pass. What matters is whether the trial
    places more of them than the plan it would replace; the gate runs on the result.
    """
    trial_request = replace(
        request,
        pinned_sights=list(request.pinned_sights) + list(pinned or []),
        unplaceable_sights=list(unplaceable if unplaceable is not None else request.unplaceable_sights),
    )
    try:
        return build_itinerary(trial_request, dataset, repair=False)
    except PlanningError:
        return None


def _is_deferred(payload: dict, name: str) -> bool:
    return any(entry["sight"] == name for _, entry in _deferred_everywhere(payload))


def _slot_alternatives(payload: dict) -> list[tuple[dict, dict]]:
    return [(day, item) for day in payload.get("days", [])
            for item in (day.get("schedule") or {}).get("alternatives", [])
            if item.get("kind") == "slot"]


def _sharpen_suggestions(payload: dict) -> None:
    """Replace "move it somewhere" with a real day number, or the reason there is none."""
    for day, item in _slot_alternatives(payload):
        start = float(item.get("window_start", DAY_START_HOUR))
        options = _move_candidates(payload, day, item.get("sight"), start)
        item["suggestion"] = (
            f"move '{item.get('sight')}' to day {options[0]}, which still has time after {clock(start)}"
            if options else _why_no_move(payload, day, start)
        )


def _distance_provenance(payload: dict) -> dict:
    """Where this plan's distances came from, pair by pair.

    A plan built half on measured roads and half on straight-line estimates is still a
    usable plan, but it must not look like one measured throughout -- so the count of each
    is reported, and so are the legs that are still estimates.
    """
    from .roadcache import info as cache_info

    detail = cache_info()
    measured = estimated = 0
    estimated_legs: list[str] = []
    for day in payload.get("days", []):
        city_row = get_dataset().city_row(day.get("city", ""))
        if city_row is None:
            continue
        points = [(float(city_row["lat"]), float(city_row["lon"]))]
        points += [(float(sight["lat"]), float(sight["lon"])) for sight in day.get("sights", [])
                   if sight.get("lat") is not None]
        points.append((float(city_row["lat"]), float(city_row["lon"])))
        names = ["the hotel"] + [sight["sight"] for sight in day.get("sights", [])] + ["the hotel"]
        for index in range(len(points) - 1):
            (lat1, lon1), (lat2, lon2) = points[index], points[index + 1]
            if road_km_is_measured(lat1, lon1, lat2, lon2):
                measured += 1
            else:
                estimated += 1
                leg = f"day {day.get('day')}: {names[index]} to {names[index + 1]}"
                if leg not in estimated_legs:
                    estimated_legs.append(leg)
    return {
        "source": detail["source"],
        "measured_legs": measured,
        "estimated_legs": estimated,
        "estimated_detail": estimated_legs[:10],
        "note": detail["detail"],
    }


def _unscheduled_reasons(dataset: Dataset, unscheduled: list[str], slots: list[DaySlot], pace: str) -> dict[str, str]:
    """Explain why a must-visit sight did not make it into the plan.

    The usual reason is simple arithmetic: the sight plus its buffer plus the drive out and
    back is longer than the longest day the trip has in that city.
    """
    reasons: dict[str, str] = {}
    for label in unscheduled:
        city, _, sight = label.partition(" / ")
        city_key = normalize_key(city)
        row = dataset.sights[(dataset.sights["city_key"] == city_key) & (dataset.sights["sight_key"] == normalize_key(sight))]
        if row.empty:
            continue
        row = row.iloc[0]
        city_slots = [slot for slot in slots if slot.city_key == city_key]
        if not city_slots:
            reasons[label] = f"{city} has no day in this trip."
            continue
        best_slot = max(city_slots, key=lambda slot: slot.available_hours(pace))
        centre = dataset.cities[dataset.cities["city_key"] == city_key].iloc[0]
        drive_hours = 2 * local_travel_hours(
            road_km(float(centre["lat"]), float(centre["lon"]), float(row["lat"]), float(row["lon"])),
            centre.get("altitude_m"),
        )
        needed = float(row["duration_hours"]) + SIGHT_BUFFER_HOURS + drive_hours
        longest = best_slot.available_hours(pace)
        if needed > longest:
            reasons[label] = (
                f"needs about {needed:.1f}h on its own (visit {row['duration_hours']:.1f}h + "
                f"{drive_hours:.1f}h driving), but the longest {city} day has {longest:.1f}h at {pace} pace."
            )
        else:
            reasons[label] = f"no day had room left; the {city} days were filled by higher-priority sights."
    return reasons


def _day_notes(slot: DaySlot, packed: PackedDay, stops: list[EnRouteStop] | None = None) -> list[str]:
    """Human-readable reasons a day looks the way it does."""
    notes: list[str] = []
    if slot.transfer:
        leg = slot.transfer
        if leg.drive_hours is None:
            notes.append(
                f"Travel {leg.from_city} to {leg.to_city}: no journey time in the dataset, so the whole day is "
                f"reserved for travel and nothing is scheduled."
            )
        elif not leg.fits_one_day:
            notes.append(
                f"Travel {leg.from_city} to {leg.to_city}: {leg.drive_hours:.1f}h by {leg.mode} -- longer than one "
                f"travel day, so the day holds no sightseeing."
            )
        else:
            notes.append(f"Intercity transfer {leg.from_city} to {leg.to_city}: {leg.drive_hours:.1f}h by {leg.mode}.")
        notes += [f"Route data: {note}" for note in leg.notes]
    if slot.is_arrival:
        notes.append("Arrival day: time reserved for reaching the hotel and checking in.")
    if slot.is_departure:
        notes.append(f"Departure day: {slot.departure_cost_hours:.1f}h reserved for check-out and the onward transfer.")
    if stops:
        names = ", ".join(f"{stop.sight['sight']} ({stop.detour_km:.0f} km off the road)" for stop in stops)
        notes.append(f"Stops made on the way: {names}.")
    if not packed.sights and not stops:
        notes.append("No sightseeing fits this day once travel and check-in are counted.")
    elif any(s["is_excursion"] for s in packed.sights):
        notes.append("Day trip outside the city; the rest of the day is kept light.")
    return notes
