"""What the traveller could do about a plan, as actions rather than complaints.

A warning that says "this sight is in the wrong slot" leaves the work to the reader. Every
suggestion here carries an ``action`` the UI can apply directly -- retime a stop, move it to
a day whose evening is free, lengthen a day, or shift days between cities -- so a conflict
becomes one click rather than a puzzle.

Nothing here changes a plan. Suggestions are computed from the plan as it stands and the
dataset behind it; applying one goes back through the ordinary planning path, so the route,
the clock and the validator all stay in agreement.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .config import (
    EXTRA_HOURS_PER_STOP,
    HARD_TIME_SLOTS,
    LATEST_END_HOUR,
    SIGHT_BUFFER_HOURS,
    TIME_WINDOWS,
)
from .data_loader import Dataset
from .schedule import clock

#: Suggestions are shown worst-first; this is what "worst" means.
RANK = {"conflict": 0, "unplaceable": 1, "capacity": 2, "balance": 3, "polish": 4}


@dataclass
class Suggestion:
    """One thing the user could do, and the call that would do it."""

    kind: str          # retime | move_day | extend_day | reallocate | free_evening | add
    group: str         # conflict | unplaceable | capacity | balance | polish
    message: str       # what is wrong, in one sentence
    fix: str           # what this suggestion would do about it
    action: dict       # the endpoint call that applies it
    sight: str | None = None
    day: int | None = None
    detail: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def _hours(text: str | None) -> float:
    if not text:
        return 0.0
    hours, _, minutes = str(text).partition(":")
    return int(hours) + int(minutes or 0) / 60.0


def _window_for(sight: dict) -> tuple[float, float] | None:
    slot = str(sight.get("best_time", "")).strip().lower()
    return TIME_WINDOWS.get(slot) if slot in HARD_TIME_SLOTS else None


def _evening_load(day: dict, excluding: str | None = None) -> float:
    """Hours of hard-slot sightseeing already booked into a day's evening."""
    return sum(float(sight.get("duration_hours", 0)) + SIGHT_BUFFER_HOURS
               for sight in day.get("sights", [])
               if str(sight.get("best_time", "")).lower() in HARD_TIME_SLOTS
               and sight.get("sight") != excluding)


def _evening_room(day: dict, excluding: str | None = None) -> float:
    """Hours still free inside the evening window on this day."""
    if day.get("is_travel_only"):
        return 0.0
    window = TIME_WINDOWS["evening"]
    latest = LATEST_END_HOUR - float((day.get("time_budget") or {}).get("departure_cost_hours", 0.0))
    return max(0.0, min(window[1], latest) - window[0] - _evening_load(day, excluding))


def retime_suggestions(payload: dict) -> list[Suggestion]:
    """Rule 1: a hard-slot sight at the wrong hour gets a better hour, not a refusal."""
    found: list[Suggestion] = []
    for day in payload.get("days", []):
        for conflict in (day.get("schedule") or {}).get("conflicts", []):
            name = conflict["sight"]
            sight = next((s for s in day.get("sights", []) if s["sight"] == name), None)
            cost = float((sight or {}).get("duration_hours", 0)) + SIGHT_BUFFER_HOURS
            # Only worth suggesting a time this day can actually give: the evening is 3.5
            # hours long, and whatever else is booked into it has already taken some.
            if _evening_room(day, excluding=name) < cost:
                continue
            start = TIME_WINDOWS["evening"][0] + _evening_load(day, excluding=name)
            found.append(Suggestion(
                kind="retime",
                group="conflict",
                message=conflict["message"],
                fix=f"Start it at {clock(start)} instead (its window is {conflict['window']}).",
                action={"endpoint": "/itinerary/customize", "operation": "retime",
                        "day": day["day"], "sight": name, "at": clock(start)},
                sight=name,
                day=day["day"],
                detail={"scheduled_at": conflict["scheduled_at"], "window": conflict["window"]},
            ))
    return found


def move_day_suggestions(payload: dict) -> list[Suggestion]:
    """Rule 2, second half: an excess evening sight goes to a day whose evening is free."""
    found: list[Suggestion] = []
    rooms = {day["day"]: _evening_room(day) for day in payload.get("days", [])}
    for day in payload.get("days", []):
        conflicted = [c["sight"] for c in (day.get("schedule") or {}).get("conflicts", [])]
        for name in conflicted:
            sight = next((s for s in day.get("sights", []) if s["sight"] == name), None)
            if sight is None:
                continue
            cost = float(sight.get("duration_hours", 0)) + SIGHT_BUFFER_HOURS
            options = sorted(
                (other["day"] for other in payload.get("days", [])
                 if other["day"] != day["day"]
                 and other.get("city") == day.get("city")
                 and not other.get("is_departure_day")
                 and rooms.get(other["day"], 0.0) >= cost),
                key=lambda number: -rooms.get(number, 0.0),
            )
            if not options:
                continue
            target = options[0]
            found.append(Suggestion(
                kind="move_day",
                group="conflict",
                message=f"Day {day['day']} has no room left in its evening for '{name}'.",
                fix=f"Move it to day {target}, whose evening still has "
                    f"{rooms[target]:.1f}h free.",
                action={"endpoint": "/itinerary/customize", "operation": "move",
                        "sight": name, "from_day": day["day"], "to_day": target},
                sight=name,
                day=day["day"],
                detail={"evening_room_hours": round(rooms[target], 2)},
            ))
    return found


def free_the_evening_suggestions(payload: dict) -> list[Suggestion]:
    """Rule 2, first half: a flexible stop sitting in the evening is moved to the morning."""
    found: list[Suggestion] = []
    for day in payload.get("days", []):
        schedule = day.get("schedule") or {}
        waiting = [c["sight"] for c in schedule.get("conflicts", [])]
        if not waiting:
            continue
        for block in schedule.get("blocks", []):
            if block["kind"] != "sight" or _hours(block["start"]) < TIME_WINDOWS["evening"][0]:
                continue
            sight = next((s for s in day.get("sights", []) if s["sight"] == block["sight"]), None)
            if sight is None or str(sight.get("best_time", "")).lower() not in {"any", ""}:
                continue
            found.append(Suggestion(
                kind="free_evening",
                group="conflict",
                message=f"'{block['sight']}' can be seen at any time but is taking evening space "
                        f"on day {day['day']}.",
                fix=f"Move it to the morning so the evening is free for "
                    f"{', '.join(repr(name) for name in waiting)}.",
                action={"endpoint": "/itinerary/customize", "operation": "to_morning",
                        "day": day["day"], "sight": block["sight"]},
                sight=block["sight"],
                day=day["day"],
            ))
    return found


def capacity_suggestions(payload: dict, pace: str) -> list[Suggestion]:
    """Rule 4: a day that is nearly full can be lengthened rather than trimmed."""
    found: list[Suggestion] = []
    pool = [label.split(" / ")[-1] for label in payload.get("unscheduled_must_visits", [])]
    if not pool:
        return found
    for day in payload.get("days", []):
        if day.get("is_travel_only"):
            continue
        spare = float(day.get("available_hours", 0)) - float(day.get("used_hours", 0))
        ends = _hours((day.get("schedule") or {}).get("ends"))
        headroom = LATEST_END_HOUR - ends
        if spare > 1.5 or headroom < 1.0:
            continue
        add = min(EXTRA_HOURS_PER_STOP, round(headroom, 1))
        found.append(Suggestion(
            kind="extend_day",
            group="capacity",
            message=f"Day {day['day']} is full ({day['used_hours']}h of {day['available_hours']}h) while "
                    f"{len(pool)} must-visit sight(s) did not fit.",
            fix=f"Add {add:.1f}h to day {day['day']} -- it currently ends at "
                f"{(day.get('schedule') or {}).get('ends')}, so there is room before "
                f"{clock(LATEST_END_HOUR)}.",
            action={"endpoint": "/itinerary/customize", "operation": "extend_day",
                    "day": day["day"], "hours": add},
            day=day["day"],
            detail={"would_fit": pool[:3]},
        ))
    return found


def allocation_suggestions(payload: dict, dataset: Dataset) -> list[Suggestion]:
    """Rule 3: when one city is starved and another is comfortable, move a day across."""
    days_per_city: dict[str, int] = dict(payload.get("summary", {}).get("days_per_city", {}))
    if len(days_per_city) < 2:
        return []
    missed: dict[str, int] = {}
    for label in payload.get("unscheduled_must_visits", []):
        city = label.split(" / ")[0]
        missed[city] = missed.get(city, 0) + 1
    if not missed:
        return []

    spare_by_city: dict[str, float] = {}
    for day in payload.get("days", []):
        spare = float(day.get("available_hours", 0)) - float(day.get("used_hours", 0))
        spare_by_city[day["city"]] = spare_by_city.get(day["city"], 0.0) + spare

    short = max(missed, key=lambda city: missed[city])
    donors = [city for city, days in days_per_city.items()
              if city != short and days > 1 and spare_by_city.get(city, 0.0) >= 2.0]
    if not donors:
        return []
    donor = max(donors, key=lambda city: spare_by_city[city])
    split = dict(days_per_city)
    split[donor] -= 1
    split[short] += 1
    return [Suggestion(
        kind="reallocate",
        group="balance",
        message=f"{short} has {missed[short]} must-visit sight(s) that did not fit, while {donor} has "
                f"{spare_by_city[donor]:.1f}h spare across its {days_per_city[donor]} days.",
        fix=f"Split the trip {', '.join(f'{city} {days}' for city, days in split.items())} instead.",
        action={"endpoint": "/itinerary/customize", "operation": "reallocate", "city_days": split},
        detail={"current": days_per_city, "proposed": split},
    )]


def unplaceable_suggestions(payload: dict) -> list[Suggestion]:
    """Sights the engine held back: say what it would take to see them."""
    found: list[Suggestion] = []
    for label, reason in (payload.get("unplaceable_reasons") or {}).items():
        name = label.split(" / ")[-1]
        found.append(Suggestion(
            kind="add",
            group="unplaceable",
            message=f"'{name}' is not in the plan: {reason.split(', and')[0]}.",
            fix="Pin it to a day anyway -- it will be scheduled at the best time that day can "
                "give it, with a note saying so.",
            action={"endpoint": "/itinerary/customize", "operation": "pin", "sight": name},
            sight=name,
        ))
    return found


def suggestions_for(payload: dict, dataset: Dataset, pace: str) -> dict:
    """Every suggestion for this plan, worst first."""
    found = (retime_suggestions(payload)
             + move_day_suggestions(payload)
             + free_the_evening_suggestions(payload)
             + unplaceable_suggestions(payload)
             + capacity_suggestions(payload, pace)
             + allocation_suggestions(payload, dataset))
    found.sort(key=lambda item: (RANK.get(item.group, 9), item.day or 0))
    return {
        "count": len(found),
        "suggestions": [item.as_dict() for item in found],
        "evening_room": {day["day"]: round(_evening_room(day), 2) for day in payload.get("days", [])},
    }
