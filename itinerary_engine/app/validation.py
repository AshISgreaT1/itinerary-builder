"""Feasibility validation of a generated itinerary.

Ported from the Qwen itinerary validation layer and hardened afterwards. An itinerary can
be structurally perfect and still be impossible, so the checks are grouped by the ways that
happens. Two rules matter most here:

* **Nothing is trusted.** Day distances, driving time and hour totals are recomputed from
  the coordinates and the dataset, then compared with what the planner claimed.
* **Strict gate.** ``passed`` is true only when no *critical* check fails. Warnings and
  notes never flip the gate, but criticals always do, and each one carries a fix.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import (
    BEST_TIME_ORDER,
    MAX_DAY_ROUTE_KM,
    MAX_LEG_KM,
    MAX_OVERNIGHT_TRAVEL_HOURS,
    MAX_SIGHTS_PER_DAY,
    MAX_SINGLE_DAY_TRAVEL_HOURS,
    MAX_STOPS_WITH_EXCURSION,
    MIN_DEPARTURE_BUFFER_HOURS,
    PACE_ACTIVE_HOURS,
    SIGHT_BUFFER_HOURS,
    WRONG_CITY_KM,
)
from .data_loader import Dataset
from .geo import local_travel_hours, order_by_route, road_km
from .normalization import names_contain, normalize_key

CATEGORIES = (
    "travel_feasibility",
    "route_data",
    "time_and_activity_load",
    "departure_feasibility",
    "hotel_nights",
    "duplicates_and_normalisation",
    "geographic_distance",
    "destination_and_logic",
    "seasonality",
)

#: Hour/km tolerance when comparing the planner's numbers with recomputed ones.
TOLERANCE_HOURS = 0.05
TOLERANCE_KM = 0.5
#: A day's route may exceed the best possible order by this much before it is reported.
ROUTE_ORDER_SLACK = 1.25


@dataclass
class Issue:
    """One validation finding."""

    severity: str  # "critical" | "warning" | "info"
    category: str
    message: str
    fix: str = ""
    day: int | None = None

    def as_dict(self) -> dict:
        return {
            "severity": self.severity,
            "category": self.category,
            "day": self.day,
            "message": self.message,
            "fix": self.fix,
        }


@dataclass
class ValidationReport:
    """Findings for one itinerary. ``passed`` is false if anything critical failed."""

    entries: list[Issue] = field(default_factory=list)

    def add(self, severity: str, category: str, message: str, fix: str = "", day: int | None = None) -> None:
        self.entries.append(Issue(severity, category, message, fix, day))

    def critical(self, category: str, message: str, fix: str, day: int | None = None) -> None:
        self.add("critical", category, message, fix, day)

    def warn(self, category: str, message: str, fix: str = "", day: int | None = None) -> None:
        self.add("warning", category, message, fix, day)

    def note(self, category: str, message: str, day: int | None = None) -> None:
        self.add("info", category, message, "", day)

    def of(self, severity: str) -> list[Issue]:
        return [entry for entry in self.entries if entry.severity == severity]

    @property
    def passed(self) -> bool:
        return not self.of("critical")

    @property
    def issues(self) -> dict[str, list[str]]:
        """Blocking problems per category (kept for callers that want a quick dict)."""
        grouped: dict[str, list[str]] = {category: [] for category in CATEGORIES}
        for entry in self.of("critical"):
            grouped[entry.category].append(entry.message)
        return grouped

    @property
    def warnings(self) -> list[str]:
        return [entry.message for entry in self.of("warning")]

    def as_dict(self) -> dict:
        return {
            "passed": self.passed,
            "critical_count": len(self.of("critical")),
            "warning_count": len(self.of("warning")),
            "info_count": len(self.of("info")),
            "issues": self.issues,
            "entries": [entry.as_dict() for entry in self.entries],
            "warnings": self.warnings,
            "notes": [entry.message for entry in self.of("info")],
        }


# --------------------------------------------------------------------------------------
# 1. Intercity travel: can the journeys actually be made on the days they are placed on?
# --------------------------------------------------------------------------------------
def _check_travel(payload: dict, report: ValidationReport) -> None:
    for day in payload.get("days", []):
        leg = day.get("transfer")
        if not leg:
            continue
        number = day.get("day")
        hours, route = leg.get("drive_hours"), f"{leg['from_city']} to {leg['to_city']}"
        if hours is None:
            report.critical(
                "travel_feasibility",
                f"Day {number}: {route} has no journey time in the dataset"
                + (f" (about {leg['straight_line_km']:.0f} km apart)." if leg.get("straight_line_km") else "."),
                f"Add a city_distances row for {route} with the real duration and mode "
                f"(flight, train or road). The day is held empty until then.",
                number,
            )
            continue
        if hours > MAX_OVERNIGHT_TRAVEL_HOURS:
            report.critical(
                "travel_feasibility",
                f"Day {number}: {route} takes {hours:.1f}h by {leg.get('mode', 'road')}, which cannot be done in "
                f"one day (limit {MAX_SINGLE_DAY_TRAVEL_HOURS:.0f}h, or {MAX_OVERNIGHT_TRAVEL_HOURS:.0f}h overnight).",
                f"Give this leg {leg.get('travel_days_needed', 2)} travel days, add a faster mode "
                f"(flight/train) to city_distances, or drop one of the two cities.",
                number,
            )
        elif leg.get("is_overnight"):
            report.warn(
                "travel_feasibility",
                f"Day {number}: {route} is {hours:.1f}h, which only works as an overnight journey; "
                f"the day holds no sightseeing.",
                "Confirm an overnight bus or train exists on this route, or fly.",
                number,
            )
        if day.get("sights") and (hours > MAX_SINGLE_DAY_TRAVEL_HOURS or day.get("is_travel_only")):
            report.critical(
                "travel_feasibility",
                f"Day {number}: sightseeing is scheduled on a day that is entirely taken by travel.",
                "Move these sights to a day without a long transfer.",
                number,
            )
        if leg.get("data_gap") and leg.get("source") != "no route data":
            report.warn(
                "route_data",
                f"Day {number}: route data for {route} was corrected or is incomplete. "
                + " ".join(leg.get("notes", [])),
                "Check this row in city_distances.",
                number,
            )
        if leg.get("source", "").startswith("via"):
            report.note(
                "route_data",
                f"Day {number}: {route} has no direct row; the time comes from connecting roads "
                f"({leg['source']}).",
                number,
            )


# --------------------------------------------------------------------------------------
# 2. Route data: recompute every day's geometry instead of trusting the planner.
# --------------------------------------------------------------------------------------
def _recompute_day(day: dict, dataset: Dataset) -> dict | None:
    """Recalculate a day's distances and hours from the sight coordinates."""
    city_row = dataset.city_row(day.get("city", ""))
    if city_row is None:
        return None
    sights = day.get("sights", [])
    if not sights:
        return {"route_km": 0.0, "max_leg_km": 0.0, "local_travel_hours": 0.0,
                "sightseeing_hours": 0.0, "buffer_hours": 0.0, "best_route_km": 0.0}
    lats = [float(s["lat"]) for s in sights]
    lons = [float(s["lon"]) for s in sights]
    legs = [road_km(lats[i], lons[i], lats[i + 1], lons[i + 1]) for i in range(len(sights) - 1)]
    hotel_legs = [
        road_km(float(city_row["lat"]), float(city_row["lon"]), lats[0], lons[0]),
        road_km(float(city_row["lat"]), float(city_row["lon"]), lats[-1], lons[-1]),
    ]
    # The best order the planner could legitimately use: time-of-day groups in order, each
    # group routed optimally. Comparing against an unconstrained optimum would penalise a
    # day for putting its morning sights first, which is correct behaviour.
    import pandas as pd  # local import keeps the module import-light

    ranked: dict[int, list[int]] = {}
    for index, sight in enumerate(sights):
        rank = BEST_TIME_ORDER.get(str(sight.get("best_time", "any")).lower(), 1)
        ranked.setdefault(rank, []).append(index)
    best_sequence: list[int] = []
    cursor_lat, cursor_lon = float(city_row["lat"]), float(city_row["lon"])
    for rank in sorted(ranked):
        members = ranked[rank]
        frame = pd.DataFrame({"lat": [lats[i] for i in members], "lon": [lons[i] for i in members]})
        for position in order_by_route(frame, cursor_lat, cursor_lon):
            best_sequence.append(members[position])
        cursor_lat, cursor_lon = lats[best_sequence[-1]], lons[best_sequence[-1]]
    best_legs = [
        road_km(lats[best_sequence[i]], lons[best_sequence[i]], lats[best_sequence[i + 1]], lons[best_sequence[i + 1]])
        for i in range(len(best_sequence) - 1)
    ]
    return {
        "route_km": sum(legs),
        "max_leg_km": max(legs) if legs else 0.0,
        "local_travel_hours": local_travel_hours(sum(legs) + sum(hotel_legs), city_row.get("altitude_m")),
        "sightseeing_hours": sum(float(s["duration_hours"]) for s in sights),
        "buffer_hours": SIGHT_BUFFER_HOURS * len(sights),
        "best_route_km": sum(best_legs),
    }


def _check_route_and_load(payload: dict, dataset: Dataset, pace: str, report: ValidationReport) -> None:
    cap = MAX_SIGHTS_PER_DAY[pace.lower()]
    allowance = PACE_ACTIVE_HOURS[pace.lower()]

    for day in payload.get("days", []):
        number = day.get("day")
        city = day.get("city", "")
        sights = day.get("sights", [])
        city_row = dataset.city_row(city)
        if city_row is None:
            report.critical("route_data", f"Day {number}: '{city}' is not in the cities sheet.",
                            "Add the city (with coordinates) or remove it from the request.", number)
            continue

        for sight in sights:
            if sight.get("lat") is None or sight.get("lon") is None:
                report.critical("route_data", f"Day {number}: '{sight['sight']}' has no coordinates.",
                                "Fill lat/lon for this row in the sightseeing sheet.", number)
                continue
            distance = road_km(float(city_row["lat"]), float(city_row["lon"]), float(sight["lat"]), float(sight["lon"]))
            if distance > WRONG_CITY_KM:
                report.critical(
                    "destination_and_logic",
                    f"Day {number}: '{sight['sight']}' is {distance:.0f} km from {city}, so it is filed under the "
                    f"wrong city or its coordinates are wrong.",
                    f"Correct the city or the coordinates of '{sight['sight']}' in the sightseeing sheet.",
                    number,
                )

        recomputed = _recompute_day(day, dataset)
        if recomputed is None:
            continue

        # The planner's own numbers must match a recomputation from the raw coordinates.
        for key, tolerance, unit in (("route_km", TOLERANCE_KM, "km"), ("max_leg_km", TOLERANCE_KM, "km"),
                                     ("local_travel_hours", TOLERANCE_HOURS, "h"),
                                     ("sightseeing_hours", TOLERANCE_HOURS, "h")):
            claimed = float(day.get(key, 0.0) or 0.0)
            actual = recomputed[key]
            if abs(claimed - actual) > tolerance:
                report.critical(
                    "route_data",
                    f"Day {number}: {key} says {claimed:.2f}{unit} but recalculating from the coordinates gives "
                    f"{actual:.2f}{unit}.",
                    "The planner and the dataset disagree; re-run the plan and report this as a bug.",
                    number,
                )

        used = recomputed["sightseeing_hours"] + recomputed["buffer_hours"] + recomputed["local_travel_hours"]
        available = float(day.get("available_hours", 0.0))
        if used > available + TOLERANCE_HOURS:
            report.critical(
                "time_and_activity_load",
                f"Day {number}: the day needs {used:.1f}h (sightseeing {recomputed['sightseeing_hours']:.1f}h + "
                f"buffers {recomputed['buffer_hours']:.1f}h + local travel {recomputed['local_travel_hours']:.1f}h) "
                f"but only {available:.1f}h are available after travel and overheads.",
                "Drop a sight from this day, move it to a day with spare hours, or choose a faster pace.",
                number,
            )
        if abs(used - float(day.get("used_hours", 0.0))) > TOLERANCE_HOURS:
            report.critical(
                "route_data",
                f"Day {number}: used_hours says {day.get('used_hours')}h but the parts add up to {used:.2f}h.",
                "Re-run the plan; the reported total does not match its own components.",
                number,
            )
        if available > allowance + TOLERANCE_HOURS:
            report.critical(
                "time_and_activity_load",
                f"Day {number}: {available:.1f}h available exceeds the {allowance:.1f}h {pace} allowance.",
                "The day budget is wrong; check the pace and the day overheads.",
                number,
            )
        if len(sights) > cap:
            report.critical(
                "time_and_activity_load",
                f"Day {number}: {len(sights)} stops is more than the {cap} a {pace} day allows.",
                "Remove the lowest-priority stops or use a faster pace.",
                number,
            )

        # Geometry limits.
        for first, second in zip(sights, sights[1:]):
            leg_km = road_km(float(first["lat"]), float(first["lon"]), float(second["lat"]), float(second["lon"]))
            if leg_km > MAX_LEG_KM:
                report.critical(
                    "geographic_distance",
                    f"Day {number}: '{first['sight']}' and '{second['sight']}' are {leg_km:.0f} km apart "
                    f"(limit {MAX_LEG_KM:.0f} km between consecutive stops).",
                    "Split these stops across different days or cluster the day differently.",
                    number,
                )
        if recomputed["route_km"] > MAX_DAY_ROUTE_KM:
            report.critical(
                "geographic_distance",
                f"Day {number}: the day covers {recomputed['route_km']:.0f} km between stops "
                f"(limit {MAX_DAY_ROUTE_KM:.0f} km).",
                "Move the outlying stop to its own day.",
                number,
            )
        if recomputed["best_route_km"] > 0 and recomputed["route_km"] > ROUTE_ORDER_SLACK * recomputed["best_route_km"] + 1:
            report.warn(
                "route_data",
                f"Day {number}: the visiting order drives {recomputed['route_km']:.0f} km where "
                f"{recomputed['best_route_km']:.0f} km is possible.",
                "Reorder the day's stops.",
                number,
            )


# --------------------------------------------------------------------------------------
# 3. Departure day, 4. hotel nights, 5. seasonality, 6. duplicates and day logic
# --------------------------------------------------------------------------------------
def _check_departure(payload: dict, pace: str, report: ValidationReport) -> None:
    days = payload.get("days", [])
    if not days:
        return
    last = days[-1]
    number = last.get("day")
    budget = last.get("time_budget", {})
    allowance = float(budget.get("pace_allowance", PACE_ACTIVE_HOURS[pace.lower()]))
    departure_cost = float(budget.get("departure_cost_hours", 0.0))
    if not last.get("is_departure_day"):
        report.warn("departure_feasibility", f"Day {number} is the last day but is not marked as a departure day.",
                    "Check the day layout.", number)
        return
    if departure_cost <= 0:
        report.critical(
            "departure_feasibility",
            f"Day {number}: no time is reserved for checkout and the transfer to the airport or station.",
            "Reserve checkout, transfer and reporting time on the departure day.",
            number,
        )
        return
    spare = float(last.get("available_hours", 0.0)) - float(last.get("used_hours", 0.0))
    if departure_cost >= allowance:
        report.warn(
            "departure_feasibility",
            f"Day {number}: checkout and the onward transfer need {departure_cost:.1f}h of a {allowance:.1f}h day, "
            f"so no sightseeing is possible.",
            "Plan the departure as a travel-only day, or leave later in the day.",
            number,
        )
    if spare < -TOLERANCE_HOURS:
        report.critical(
            "departure_feasibility",
            f"Day {number}: the plan overruns the departure day by {abs(spare):.1f}h once checkout and the "
            f"transfer are counted.",
            "Remove a stop from the last day.",
            number,
        )
    else:
        report.note(
            "departure_feasibility",
            f"Day {number}: {departure_cost:.1f}h is reserved for checkout and the onward transfer, "
            f"plus a {MIN_DEPARTURE_BUFFER_HOURS:.1f}h delay buffer.",
            number,
        )
    for sight in last.get("sights", []):
        if sight.get("is_excursion"):
            report.critical(
                "departure_feasibility",
                f"Day {number}: '{sight['sight']}' is a day trip out of town, scheduled on the departure day.",
                "Move the day trip earlier in the trip.",
                number,
            )


def _check_nights(payload: dict, report: ValidationReport) -> None:
    summary = payload.get("summary", {})
    days = payload.get("days", [])
    duration = int(payload.get("request", {}).get("duration_days", len(days)))
    nights = summary.get("nights_per_city", {})
    days_per_city = summary.get("days_per_city", {})
    expected_nights = max(0, duration - 1)
    total_nights = sum(nights.values()) if nights else 0

    if len(days) != duration:
        report.critical(
            "hotel_nights",
            f"The plan has {len(days)} days but the request asked for {duration}.",
            "Re-run the plan; the day layout does not match the request.",
        )
    if total_nights != expected_nights:
        report.critical(
            "hotel_nights",
            f"Hotel nights add up to {total_nights} but a {duration}-day trip has {expected_nights} nights.",
            "Check the per-city night split: the last day is a checkout day and has no night.",
        )
    for city, city_days in days_per_city.items():
        city_nights = nights.get(city, 0)
        if city_nights not in (city_days, city_days - 1):
            report.critical(
                "hotel_nights",
                f"{city} has {city_days} day(s) but {city_nights} night(s).",
                "Each city needs one night per day, minus the final checkout day.",
            )
        if city_days > 0 and city_nights == 0 and city != (days[-1]["city"] if days else None):
            report.critical(
                "hotel_nights",
                f"{city} is visited but has no hotel night.",
                "Give the city a night or remove it from the route.",
            )


def _check_seasonality(payload: dict, dataset: Dataset, month: int | None, report: ValidationReport) -> None:
    """Only attraction-level closures block a trip.

    A city's ``season_open`` window is the *peak* season, not a shutdown: Goa in June is wet
    and quieter, but it is open. That is reported as a note, never as a failure.
    """
    if not month:
        return
    for day in payload.get("days", []):
        number, city = day.get("day"), day.get("city", "")
        for sight in day.get("sights", []):
            row = _sight_row(dataset, normalize_key(city), sight["sight"])
            if row is not None and month not in row["open_months_set"]:
                report.critical(
                    "seasonality",
                    f"Day {number}: '{sight['sight']}' is closed in month {month} "
                    f"(open_months = {row.get('open_months', 'unknown')}).",
                    "Remove this sight for this month or correct open_months in the dataset.",
                    number,
                )

    for city in dict.fromkeys(day.get("city", "") for day in payload.get("days", [])):
        row = dataset.city_row(city)
        if row is not None and month not in row["season_months"]:
            report.note(
                "seasonality",
                f"Month {month} is outside {city}'s usual season window ({row.get('season_open')}). "
                f"All its scheduled attractions are open; expect off-season weather and fewer crowds.",
            )


def _check_duplicates_and_logic(payload: dict, dataset: Dataset, report: ValidationReport) -> None:
    seen: list[tuple[str, str, float, float, int]] = []
    for day in payload.get("days", []):
        number, city = day.get("day"), day.get("city", "")
        city_key = normalize_key(city)
        sights = day.get("sights", [])

        for sight in sights:
            key = normalize_key(sight["sight"])
            canonical = normalize_key(dataset.dedup.resolve(city, sight["sight"]))
            for prev_city, prev_key, prev_lat, prev_lon, prev_day in seen:
                same_name = prev_key == key or normalize_key(dataset.dedup.resolve(prev_city, prev_key)) == canonical
                same_spot = (
                    prev_city == city_key
                    and road_km(prev_lat, prev_lon, float(sight["lat"]), float(sight["lon"])) <= 0.05
                    and (names_contain(prev_key, key) or prev_key == key)
                )
                if same_name or same_spot:
                    report.critical(
                        "duplicates_and_normalisation",
                        f"Day {number}: '{sight['sight']}' is already scheduled on day {prev_day}.",
                        "Remove the repeat; the two names are the same place.",
                        number,
                    )
                    break
            seen.append((city_key, key, float(sight["lat"]), float(sight["lon"]), number))

            row = _sight_row(dataset, city_key, sight["sight"])
            if row is not None and row["city_key"] != city_key:
                report.critical(
                    "destination_and_logic",
                    f"Day {number}: '{sight['sight']}' belongs to {row['city']}, not {city}.",
                    "Schedule it on a day in its own city, or correct the dataset.",
                    number,
                )

        excursions = [s["sight"] for s in sights if s.get("is_excursion")]
        if len(excursions) > 1:
            report.critical(
                "destination_and_logic",
                f"Day {number}: two day trips on one day ({', '.join(excursions)}).",
                "Keep one day trip per day.",
                number,
            )
        elif excursions and len(sights) - 1 > MAX_STOPS_WITH_EXCURSION:
            report.critical(
                "destination_and_logic",
                f"Day {number}: '{excursions[0]}' is a day trip, so at most {MAX_STOPS_WITH_EXCURSION} other stops "
                f"fit that day ({len(sights) - 1} are scheduled).",
                "Move the extra stops to another day.",
                number,
            )
        if excursions and day.get("transfer"):
            report.critical(
                "destination_and_logic",
                f"Day {number}: day trip '{excursions[0]}' is scheduled on an intercity travel day.",
                "Move the day trip to a day without a transfer.",
                number,
            )


def _sight_row(dataset: Dataset, city_key: str, sight: str):
    """Dataset row for a scheduled sight, searched in its city first, then anywhere."""
    key = normalize_key(sight)
    local = dataset.sights[(dataset.sights["city_key"] == city_key) & (dataset.sights["sight_key"] == key)]
    if not local.empty:
        return local.iloc[0]
    anywhere = dataset.sights[dataset.sights["sight_key"] == key]
    return None if anywhere.empty else anywhere.iloc[0]


def validate_itinerary(payload: dict, dataset: Dataset, pace: str, month: int | None = None) -> ValidationReport:
    """Run every feasibility check. ``passed`` is true only if nothing critical fails."""
    report = ValidationReport()
    if not payload.get("days"):
        report.critical("time_and_activity_load", "The itinerary has no days.", "Check the request.")
        return report

    _check_travel(payload, report)
    _check_route_and_load(payload, dataset, pace, report)
    _check_departure(payload, pace, report)
    _check_nights(payload, report)
    _check_seasonality(payload, dataset, month, report)
    _check_duplicates_and_logic(payload, dataset, report)

    for gap in payload.get("data_gaps", []):
        report.warn("route_data", f"Data gap: {gap}", "Fill the missing row so the plan rests on real data.")
    unscheduled = payload.get("unscheduled_must_visits", [])
    if unscheduled:
        report.warn(
            "time_and_activity_load",
            f"{len(unscheduled)} must-visit sight(s) did not fit: {', '.join(unscheduled[:5])}.",
            "Add a day, choose a faster pace, or accept the omission (see unscheduled_reasons).",
        )
    return report
