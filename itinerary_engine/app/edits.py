"""Editing a generated itinerary: add, replace or remove sightseeing.

The engine is deterministic, so an edit is not patched into the finished plan. Instead the
edit becomes part of the request -- a pin ("this sight, optionally on this day") or an
exclusion -- and the whole itinerary is rebuilt by the same planner. That keeps the route,
travel time and day budgets consistent after every change.

When a pinned sight cannot fit, nothing is dropped silently: the edit is reported as
rejected, with the days it *would* fit on and what would have to go to make room.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .builder import PinnedSight, PlanRequest, PlanningError, build_itinerary
from .clustering import mark_excursions
from .config import SIGHT_BUFFER_HOURS
from .data_loader import Dataset
from .normalization import normalize_key
from .geo import RouteLeg, local_travel_hours, road_km
from .solver import DaySlot, evaluate_day, pace_hours

ACTIONS = ("add", "replace", "remove")


@dataclass
class SightEdit:
    """One user action on the itinerary."""

    action: str                   # "add" | "replace" | "remove"
    sight: str                    # the sight being added, or removed
    city: str | None = None
    day: int | None = None        # preferred day for an add/replace
    replaces: str | None = None   # the sight being swapped out (action="replace")

    def normalised(self) -> "SightEdit":
        if self.action not in ACTIONS:
            raise PlanningError(f"Unknown edit action '{self.action}'. Use one of {ACTIONS}.")
        return self


@dataclass
class EditOutcome:
    """What happened to one edit, in words the UI can show as-is."""

    edit: SightEdit
    applied: bool
    message: str
    day: int | None = None
    suggestions: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "action": self.edit.action,
            "sight": self.edit.sight,
            "replaces": self.edit.replaces,
            "requested_day": self.edit.day,
            "applied": self.applied,
            "placed_on_day": self.day,
            "message": self.message,
            "suggestions": self.suggestions,
        }


def _slot_from_day(day: dict, dataset: Dataset, pace: str) -> DaySlot | None:
    """Rebuild the planner's view of a day from the payload, so fits are judged identically."""
    city_row = dataset.city_row(day.get("city", ""))
    if city_row is None:
        return None
    transfer = day.get("transfer")
    leg = None
    if transfer:
        leg = RouteLeg(
            from_city=transfer.get("from_city", ""),
            to_city=transfer.get("to_city", ""),
            road_km=transfer.get("road_km"),
            drive_hours=transfer.get("drive_hours"),
            mode=transfer.get("mode", "road"),
            source=transfer.get("source", "dataset"),
        )
    return DaySlot(
        day=int(day["day"]),
        city=str(day["city"]),
        city_key=normalize_key(day["city"]),
        centre_lat=float(city_row["lat"]),
        centre_lon=float(city_row["lon"]),
        altitude_m=city_row.get("altitude_m"),
        transfer=leg,
        is_arrival=bool(day.get("is_arrival_day")),
        is_departure=bool(day.get("is_departure_day")),
        travel_only=bool(day.get("is_travel_only")),
        # The plan already knows this day's real budget: reuse it rather than recomputing.
        available_override=float(day.get("available_hours", 0.0)),
    )


def _rows_for_day(dataset: Dataset, day: dict) -> pd.DataFrame:
    """Dataset rows for the sights currently on a day, with the planner's extra columns."""
    city_key = normalize_key(day.get("city", ""))
    keys = [normalize_key(sight["sight"]) for sight in day.get("sights", [])]
    rows = dataset.sights[(dataset.sights["city_key"] == city_key) & (dataset.sights["sight_key"].isin(keys))].copy()
    if rows.empty:
        return rows
    order = {key: index for index, key in enumerate(keys)}
    rows["_order"] = rows["sight_key"].map(order)
    rows = rows.sort_values("_order").drop(columns="_order")
    excursion_by_key = {normalize_key(s["sight"]): bool(s.get("is_excursion")) for s in day.get("sights", [])}
    rows["is_excursion"] = rows["sight_key"].map(excursion_by_key).fillna(False)
    return rows.reset_index(drop=True)


def find_sight(dataset: Dataset, sight: str, city: str | None = None) -> pd.Series | None:
    """Look up a sight by name (canonical or alias), preferring the given city."""
    label = sight.split(" / ")[-1]
    key = normalize_key(label)
    if city:
        city_key = normalize_key(city)
        canonical = normalize_key(dataset.dedup.resolve(city, label))
        match = dataset.sights[
            (dataset.sights["city_key"] == city_key)
            & ((dataset.sights["sight_key"] == key) | (dataset.sights["sight_key"] == canonical))
        ]
        if not match.empty:
            return match.iloc[0]
    match = dataset.sights[dataset.sights["sight_key"] == key]
    return None if match.empty else match.iloc[0]


def placement_options(payload: dict, dataset: Dataset, sight: str, city: str | None, pace: str) -> dict:
    """Where could this sight go? One entry per day of the trip in that sight's city.

    Each entry says whether it fits as-is and, when it does not, which stops could be
    dropped to make room (cheapest first). Nothing is changed by this call.
    """
    row = find_sight(dataset, sight, city)
    if row is None:
        raise PlanningError(f"'{sight}' is not in the sightseeing data.")
    sight_city_key = row["city_key"]
    options: list[dict] = []

    for day in payload.get("days", []):
        if normalize_key(day.get("city", "")) != sight_city_key:
            continue
        slot = _slot_from_day(day, dataset, pace)
        if slot is None:
            continue
        current = _rows_for_day(dataset, day)
        if not current.empty:  # a sight already on this day is not competing with itself
            current = current[current["sight_key"] != row["sight_key"]]
        candidate = mark_excursions(row.to_frame().T.copy(), slot.centre_lat, slot.centre_lon)
        combined = pd.concat([current, candidate], ignore_index=True) if not current.empty else candidate
        verdict = evaluate_day(combined, slot, pace)

        drops: list[dict] = []
        clearing_set: list[str] = []
        if not verdict["fits"] and not current.empty:
            # Single swaps first: the tidiest edit is "drop one stop for this one".
            for _, existing in current.iterrows():
                trimmed = combined[combined["sight_key"] != existing["sight_key"]]
                if evaluate_day(trimmed, slot, pace)["fits"]:
                    drops.append(
                        {
                            "sight": existing["sight"],
                            "frees_hours": round(float(existing["duration_hours"]) + SIGHT_BUFFER_HOURS, 2),
                            "must_visit": bool(existing["must_visit_flag"]),
                        }
                    )
            drops.sort(key=lambda item: (item["must_visit"], -item["frees_hours"]))

            if not drops:
                # No single swap works: drop the least valuable stops until it does.
                order = current.sort_values(
                    ["must_visit_flag", "must_see_tier", "priority"], ascending=[True, False, False]
                )
                remaining = combined
                for _, existing in order.iterrows():
                    remaining = remaining[remaining["sight_key"] != existing["sight_key"]]
                    clearing_set.append(str(existing["sight"]))
                    if evaluate_day(remaining, slot, pace)["fits"]:
                        break
                else:
                    clearing_set = []  # even an empty day cannot hold it

        options.append(
            {
                "day": day["day"],
                "city": day["city"],
                "fits": verdict["fits"],
                "needed_hours": verdict["needed_hours"],
                "available_hours": verdict["available_hours"],
                "spare_hours": round(verdict["available_hours"] - verdict["needed_hours"], 2),
                "blockers": verdict["blockers"],
                "replace_options": drops[:4],
                "fits_if_dropped": clearing_set,
                "is_travel_day": bool(day.get("transfer")) or bool(day.get("is_travel_only")),
            }
        )

    best = next((option["day"] for option in options if option["fits"]), None)
    return {
        "sight": row["sight"],
        "city": row["city"],
        "duration_hours": float(row["duration_hours"]),
        "must_visit": bool(row["must_visit_flag"]),
        "best_day": best,
        "needs_more_than_a_day": _too_big_for_any_day(options, row, dataset, pace),
        "options": options,
    }


def _too_big_for_any_day(options: list[dict], row: pd.Series, dataset: Dataset, pace: str) -> str | None:
    """Say so when the sight cannot fit *any* day, even an empty one.

    Rohtang Pass is 61 km of mountain road out of Manali: 5h at the pass plus 5.6h of
    driving is more than an active day holds, so no amount of dropping other stops helps.
    Without this the caller is told only that every day is full, which reads as a packing
    problem when it is not one.
    """
    if not options or any(option["fits"] or option["replace_options"] or option["fits_if_dropped"]
                          for option in options):
        return None
    city_row = dataset.city_row(str(row["city"]))
    if city_row is None:
        return None
    out = road_km(float(city_row["lat"]), float(city_row["lon"]), float(row["lat"]), float(row["lon"]))
    back = road_km(float(row["lat"]), float(row["lon"]), float(city_row["lat"]), float(city_row["lon"]))
    driving = local_travel_hours(out + back, city_row.get("altitude_m"))
    alone = float(row["duration_hours"]) + SIGHT_BUFFER_HOURS + driving
    longest = max(option["available_hours"] for option in options)
    if alone <= longest:
        return None
    return (f"'{row['sight']}' needs {alone:.1f}h on its own ({float(row['duration_hours']):.1f}h at the sight "
            f"plus {driving:.1f}h driving {out:.0f} km each way), which is more than the longest day on this "
            f"trip ({longest:.1f}h). Dropping other stops will not make room: it needs a faster pace, or a "
            f"trip that gives it a day of its own.")


def _describe_alternatives(options: dict) -> list[dict]:
    """Suggestions for a sight that did not fit where the user asked."""
    suggestions = []
    for option in options.get("options", []):
        if option["fits"]:
            suggestions.append(
                {"day": option["day"], "kind": "fits", "detail": f"Day {option['day']} has {option['spare_hours']:.1f}h spare."}
            )
        elif option["replace_options"]:
            swap = option["replace_options"][0]
            suggestions.append(
                {
                    "day": option["day"],
                    "kind": "replace",
                    "detail": f"Day {option['day']} works if you drop '{swap['sight']}' ({swap['frees_hours']:.1f}h).",
                    "replace": swap["sight"],
                }
            )
        elif option.get("fits_if_dropped"):
            dropped = option["fits_if_dropped"]
            suggestions.append(
                {
                    "day": option["day"],
                    "kind": "clear",
                    "detail": f"Day {option['day']} works if you drop {len(dropped)} stops: " + ", ".join(dropped) + ".",
                    "replace": dropped[0],
                    "drop_all": dropped,
                }
            )
    return suggestions[:4]


def apply_edits(
    dataset: Dataset, request: PlanRequest, edits: list[SightEdit]
) -> tuple[dict, list[EditOutcome]]:
    """Rebuild the itinerary with the user's edits applied, and report what happened.

    Edits are applied together, not one at a time, so the planner can balance them against
    each other. An edit that cannot be honoured leaves the rest of the plan intact and
    comes back with alternative days.
    """
    pins = list(request.pinned_sights)
    excluded = list(request.excluded_sights)

    for edit in (item.normalised() for item in edits):
        row = find_sight(dataset, edit.sight, edit.city)
        if row is None and edit.action != "remove":
            continue
        if edit.action == "remove":
            excluded.append(edit.sight)
            pins = [pin for pin in pins if normalize_key(pin.sight) != normalize_key(edit.sight)]
            continue
        if edit.action == "replace" and edit.replaces:
            excluded.append(edit.replaces)
            pins = [pin for pin in pins if normalize_key(pin.sight) != normalize_key(edit.replaces)]
        excluded = [name for name in excluded if normalize_key(name.split(" / ")[-1]) != normalize_key(edit.sight)]
        pins = [pin for pin in pins if normalize_key(pin.sight) != normalize_key(edit.sight)]
        pins.append(PinnedSight(sight=str(row["sight"]), city=str(row["city"]), day=edit.day))

    edited_request = PlanRequest(
        cities=request.cities,
        duration_days=request.duration_days,
        pace=request.pace,
        origin_city=request.origin_city,
        start_month=request.start_month,
        include_descriptions=request.include_descriptions,
        pinned_sights=pins,
        excluded_sights=excluded,
    )
    payload = build_itinerary(edited_request, dataset)

    scheduled = {
        (normalize_key(day["city"]), normalize_key(sight["sight"])): day["day"]
        for day in payload["days"]
        for sight in day["sights"]
    }
    outcomes: list[EditOutcome] = []
    for edit in edits:
        row = find_sight(dataset, edit.sight, edit.city)
        if edit.action == "remove":
            still_there = row is not None and (row["city_key"], row["sight_key"]) in scheduled
            outcomes.append(
                EditOutcome(edit, not still_there, f"Removed '{edit.sight}' from the itinerary." if not still_there
                            else f"'{edit.sight}' is still scheduled; remove failed.")
            )
            continue
        if row is None:
            outcomes.append(EditOutcome(edit, False, f"'{edit.sight}' is not in the sightseeing data."))
            continue
        placed_on = scheduled.get((row["city_key"], row["sight_key"]))
        if placed_on and (edit.day is None or placed_on == edit.day):
            note = f"Added '{row['sight']}' to day {placed_on}."
            if edit.action == "replace" and edit.replaces:
                note = f"Replaced '{edit.replaces}' with '{row['sight']}' on day {placed_on}."
            outcomes.append(EditOutcome(edit, True, note, placed_on))
        elif placed_on:
            outcomes.append(
                EditOutcome(
                    edit, True,
                    f"'{row['sight']}' did not fit day {edit.day}, so it was scheduled on day {placed_on} instead.",
                    placed_on,
                )
            )
        else:
            options = placement_options(payload, dataset, str(row["sight"]), str(row["city"]), request.pace)
            suggestions = _describe_alternatives(options)
            reason = payload.get("unscheduled_reasons", {}).get(f"{row['city']} / {row['sight']}", "")
            message = f"'{row['sight']}' could not be scheduled."
            if reason:
                message += f" {reason}"
            if not suggestions:
                message += " No day in this trip can hold it; try a faster pace or an extra day."
            outcomes.append(EditOutcome(edit, False, message, None, suggestions))

    payload["edits"] = [outcome.as_dict() for outcome in outcomes]
    payload["request"]["pinned_sights"] = [
        {"sight": pin.sight, "city": pin.city, "day": pin.day} for pin in pins
    ]
    payload["request"]["excluded_sights"] = excluded
    return payload, outcomes
