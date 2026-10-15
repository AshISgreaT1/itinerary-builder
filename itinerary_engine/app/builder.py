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
    MIN_CITY_USABLE_FRACTION,
    RAILWAY_BUFFER_HOURS,
    SIGHT_BUFFER_HOURS,
)
from .data_loader import Dataset, get_dataset
from .enroute import EnRouteStop, plan_en_route
from .geo import RouteLeg, local_travel_hours, parse_access_distance_km, road_km, transfer_hours_for_distance
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

    def pins_for_city(self, city_key: str) -> list[PinnedSight]:
        return [pin for pin in self.pinned_sights if pin.city is None or normalize_key(pin.city) == city_key]


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
    excluded_keys = {normalize_key(name.split(" / ")[-1]) for name in (request.excluded_sights if request else [])}

    eligible = sights[(sights["min_days"] <= days_in_city) | sights["sight_key"].isin(pinned_keys)]
    if month:
        eligible = eligible[eligible["open_months_set"].map(lambda months: month in months)]
    eligible = eligible[~eligible["sight_key"].isin(excluded_keys)].reset_index(drop=True)
    if pinned_keys:
        eligible["_forced"] = eligible["sight_key"].isin(pinned_keys)
        day_by_key = {normalize_key(pin.sight): pin.day for pin in pins if pin.day is not None}
        eligible["_forced_day"] = eligible["sight_key"].map(day_by_key)
    return eligible


def build_itinerary(request: PlanRequest, dataset: Dataset | None = None) -> dict:
    """Build the full itinerary payload for a request."""
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

    allocation = allocate_city_days(dataset, city_keys, request.duration_days, pace, request)
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
    report = validate_itinerary(payload, dataset, pace, request.start_month)
    payload["validation"] = report.as_dict()
    if not report.passed:
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
