"""Changing a plan after it has been generated.

Every operation here is a change to the *request*, not to the finished payload: the plan is
then rebuilt through the ordinary pipeline. That is slower than patching the JSON in place
and it is the only way the route, the clock, the day budgets and the validator can stay in
agreement -- a hand-patched payload is exactly how a header starts disagreeing with its own
timeline.

The operations are the ones a suggestion can carry (see :mod:`app.suggest`) plus the manual
ones a drag-and-drop UI needs:

``retime``        give a stop a particular time of day
``move``          move a stop to another day
``swap``          exchange two stops between days
``reorder``       set a day's order outright (what a drag ends with)
``to_morning``    move a flexible stop to the front of its day
``extend_day``    lengthen a day beyond the pace
``reallocate``    change the split of days between cities
``pin``           add a sight, optionally to a given day
``drop``          remove a sight
``add_activity``  add something of your own (dinner, a show)
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from .builder import CustomActivity, PinnedSight, PlanningError, PlanRequest, build_itinerary
from .data_loader import Dataset
from .normalization import normalize_key

OPERATIONS = {"retime", "move", "swap", "reorder", "to_morning", "extend_day",
              "reallocate", "pin", "drop", "add_activity"}


@dataclass
class Operation:
    """One customisation, as the UI sends it."""

    operation: str
    sight: str | None = None
    city: str | None = None
    day: int | None = None
    to_day: int | None = None
    from_day: int | None = None
    at: str | None = None
    hours: float | None = None
    order: list[str] = field(default_factory=list)
    city_days: dict[str, int] = field(default_factory=dict)
    name: str | None = None
    duration_hours: float | None = None
    note: str = ""
    with_sight: str | None = None
    with_day: int | None = None


@dataclass
class OperationOutcome:
    """What one operation did, in words the UI can show."""

    operation: str
    applied: bool
    message: str

    def as_dict(self) -> dict:
        return {"operation": self.operation, "applied": self.applied, "message": self.message}


def _day_of(payload: dict, sight: str) -> int | None:
    key = normalize_key(sight)
    for day in payload.get("days", []):
        if any(normalize_key(item["sight"]) == key for item in day.get("sights", [])):
            return day["day"]
    return None


def _city_of(payload: dict, sight: str, fallback: str | None = None) -> str | None:
    key = normalize_key(sight)
    for day in payload.get("days", []):
        if any(normalize_key(item["sight"]) == key for item in day.get("sights", [])):
            return day.get("city")
    return fallback


def _order_of(payload: dict, day_number: int) -> list[str]:
    day = next((item for item in payload.get("days", []) if item["day"] == day_number), None)
    return [sight["sight"] for sight in (day or {}).get("sights", [])]


def _without(names: list[str], sight: str) -> list[str]:
    key = normalize_key(sight)
    return [name for name in names if normalize_key(name) != key]


def _freeze(request: PlanRequest, payload: dict, except_for: set[str]) -> PlanRequest:
    """Pin every stop where it already is, so one manual move moves one thing.

    Without this, re-planning around a dragged sight is free to re-cluster the whole trip,
    and the user who moved one temple watches four other stops rearrange themselves. The
    engine still re-costs the route and the clock -- it just keeps the days it was given.
    """
    pins = list(request.pinned_sights)
    pinned = {normalize_key(pin.sight) for pin in pins}
    for day in payload.get("days", []):
        for sight in day.get("sights", []):
            key = normalize_key(sight["sight"])
            if key in pinned or key in except_for or sight.get("en_route"):
                continue
            pins.append(PinnedSight(sight=sight["sight"], city=day.get("city"), day=day["day"]))
            pinned.add(key)
    return replace(request, pinned_sights=pins)


def _pin(request: PlanRequest, sight: str, city: str | None, day: int | None) -> PlanRequest:
    """Pin a sight, replacing any earlier pin of the same sight."""
    key = normalize_key(sight)
    pins = [pin for pin in request.pinned_sights if normalize_key(pin.sight) != key]
    pins.append(PinnedSight(sight=sight, city=city, day=day))
    # A sight the user asks for is no longer one the engine gets to hold back.
    kept = [label for label in request.unplaceable_sights if normalize_key(label.split(" / ")[-1]) != key]
    excluded = [label for label in request.excluded_sights if normalize_key(label.split(" / ")[-1]) != key]
    return replace(request, pinned_sights=pins, unplaceable_sights=kept, excluded_sights=excluded)


def apply_operations(
    dataset: Dataset, request: PlanRequest, operations: list[Operation]
) -> tuple[dict, list[OperationOutcome]]:
    """Apply the operations one at a time, keeping only the ones that did what they said.

    Each one is rebuilt and then checked against its own intent: a stop asked onto a full
    day does not quietly disappear from the trip. An operation that could not be honoured
    is rolled back and reported with what would make room, so the plan the user gets is
    always one they can see the reasoning for.
    """
    payload = build_itinerary(request, dataset)
    outcomes: list[OperationOutcome] = []
    current = request

    for op in operations:
        if op.operation not in OPERATIONS:
            raise PlanningError(f"Unknown operation '{op.operation}'. Known: {', '.join(sorted(OPERATIONS))}.")
        candidate, message = _apply_one(dataset, current, payload, op)
        rebuilt = build_itinerary(candidate, dataset)
        problem = _unmet_intent(op, rebuilt)
        if problem:
            outcomes.append(OperationOutcome(op.operation, False, problem))
            continue
        current, payload = candidate, rebuilt
        outcomes.append(OperationOutcome(op.operation, True, message))

    return payload, outcomes


def _unmet_intent(op: Operation, payload: dict) -> str | None:
    """Did the plan actually do what the operation asked? If not, what would help."""
    wanted: list[tuple[str, int | None]] = []
    if op.operation in {"move", "retime"} and op.sight:
        wanted.append((op.sight, op.to_day or op.day))
    elif op.operation == "pin" and op.sight:
        wanted.append((op.sight, op.day))
    elif op.operation == "swap" and op.sight and op.with_sight:
        wanted.append((op.sight, op.with_day or _day_of(payload, op.sight)))
        wanted.append((op.with_sight, op.day or _day_of(payload, op.with_sight)))

    for name, day_number in wanted:
        placed = _day_of(payload, name)
        if placed is None:
            return (f"'{name}' could not be placed{_on(day_number)}: nothing was dropped to make room, "
                    f"so the change was undone. {_room_hint(payload, day_number)}")
        if day_number is not None and placed != day_number:
            return (f"'{name}' did not fit on day {day_number} and the change was undone; it is still on "
                    f"day {placed}. {_room_hint(payload, day_number)}")
    return None


def _on(day_number: int | None) -> str:
    return f" on day {day_number}" if day_number else ""


def _room_hint(payload: dict, day_number: int | None) -> str:
    """The fix, named concretely: why the day is short and what to do about it."""
    day = next((item for item in payload.get("days", []) if item["day"] == day_number), None)
    if day is None:
        return "Lengthen a day, or remove a stop first."

    transfer = day.get("transfer") or {}
    if day.get("is_travel_only") or (transfer and float(day.get("available_hours", 0)) <= 0):
        where = transfer.get("to_city") or day.get("city")
        hours = transfer.get("drive_hours")
        drive = f" ({hours:.1f}h)" if isinstance(hours, (int, float)) else ""
        return (f"Day {day_number} is taken up by the journey to {where}{drive}, so it has no sightseeing "
                f"hours at all. Try a day either side of it.")
    if float(day.get("available_hours", 0)) <= 0:
        return f"Day {day_number} has no sightseeing hours. Try another day."

    spare = float(day.get("available_hours", 0)) - float(day.get("used_hours", 0))
    stops = [f"'{sight['sight']}'" for sight in day.get("sights", [])[:3]]
    swap = f", or swap with one of {', '.join(stops)}" if stops else ""
    return (f"Day {day_number} has {spare:.1f}h spare of {day.get('available_hours')}h. Add hours to it "
            f"(extend_day){swap}.")


def _apply_one(dataset: Dataset, request: PlanRequest, payload: dict,
               op: Operation) -> tuple[PlanRequest, str]:
    """One operation against the request. Returns the new request and what it did."""
    if op.operation == "retime":
        if not op.sight or not op.at:
            raise PlanningError("retime needs a sight and a time, e.g. {'sight': 'Lake Pichola', 'at': '16:30'}.")
        day = op.day or _day_of(payload, op.sight)
        if day is None:
            raise PlanningError(f"'{op.sight}' is not in the plan; add it first.")
        times = {number: dict(values) for number, values in request.sight_times.items()}
        times.setdefault(day, {})[op.sight] = op.at
        request = _pin(request, op.sight, _city_of(payload, op.sight, op.city), day)
        return replace(request, sight_times=times), f"'{op.sight}' is asked for at {op.at} on day {day}."

    if op.operation == "move":
        if not op.sight or op.to_day is None:
            raise PlanningError("move needs a sight and a to_day.")
        city = _city_of(payload, op.sight, op.city)
        request = _freeze(request, payload, {normalize_key(op.sight)})
        return _pin(request, op.sight, city, op.to_day), f"'{op.sight}' moved to day {op.to_day}."

    if op.operation == "swap":
        if not op.sight or not op.with_sight:
            raise PlanningError("swap needs two sights.")
        first_day = op.day or _day_of(payload, op.sight)
        second_day = op.with_day or _day_of(payload, op.with_sight)
        if first_day is None or second_day is None:
            raise PlanningError("both sights have to be in the plan to swap them.")
        request = _freeze(request, payload, {normalize_key(op.sight), normalize_key(op.with_sight)})
        request = _pin(request, op.sight, _city_of(payload, op.sight, op.city), second_day)
        request = _pin(request, op.with_sight, _city_of(payload, op.with_sight, op.city), first_day)
        return request, (f"'{op.sight}' and '{op.with_sight}' swapped days "
                         f"({first_day} <-> {second_day}).")

    if op.operation == "reorder":
        if op.day is None or not op.order:
            raise PlanningError("reorder needs a day and an order.")
        orders = {number: list(values) for number, values in request.day_order.items()}
        orders[op.day] = list(op.order)
        return replace(request, day_order=orders), f"Day {op.day} reordered."

    if op.operation == "to_morning":
        if op.day is None or not op.sight:
            raise PlanningError("to_morning needs a day and a sight.")
        order = [op.sight] + _without(_order_of(payload, op.day), op.sight)
        orders = {number: list(values) for number, values in request.day_order.items()}
        orders[op.day] = order
        return replace(request, day_order=orders), f"'{op.sight}' moved to the start of day {op.day}."

    if op.operation == "extend_day":
        if op.day is None or op.hours is None:
            raise PlanningError("extend_day needs a day and hours.")
        if op.hours < 0:
            raise PlanningError("extend_day hours cannot be negative; use 0 to undo.")
        extra = dict(request.extra_day_hours)
        if op.hours == 0:
            extra.pop(op.day, None)
            return replace(request, extra_day_hours=extra), f"Day {op.day} back to the normal pace."
        extra[op.day] = float(op.hours)
        return replace(request, extra_day_hours=extra), f"Day {op.day} lengthened by {op.hours:.1f}h."

    if op.operation == "reallocate":
        if not op.city_days:
            raise PlanningError("reallocate needs city_days, e.g. {'Jaipur': 4, 'Udaipur': 2}.")
        split = {name: int(days) for name, days in op.city_days.items()}
        return (replace(request, city_days=split),
                "Days split " + ", ".join(f"{name} {days}" for name, days in split.items()) + ".")

    if op.operation == "pin":
        if not op.sight:
            raise PlanningError("pin needs a sight.")
        where = f" on day {op.day}" if op.day else ""
        return _pin(request, op.sight, op.city, op.day), f"'{op.sight}' added{where}."

    if op.operation == "drop":
        if not op.sight:
            raise PlanningError("drop needs a sight.")
        key = normalize_key(op.sight)
        city = _city_of(payload, op.sight, op.city) or ""
        excluded = list(request.excluded_sights) + [f"{city} / {op.sight}"]
        pins = [pin for pin in request.pinned_sights if normalize_key(pin.sight) != key]
        return replace(request, excluded_sights=excluded, pinned_sights=pins), f"'{op.sight}' removed."

    # add_activity
    if not op.name or op.day is None:
        raise PlanningError("add_activity needs a name and a day.")
    activity = CustomActivity(
        name=op.name,
        day=op.day,
        duration_hours=float(op.duration_hours or 1.0),
        at=op.at,
        note=op.note,
    )
    at = f" at {op.at}" if op.at else ""
    return (replace(request, custom_activities=list(request.custom_activities) + [activity]),
            f"'{op.name}' added to day {op.day}{at}.")
