"""Turning a day's stops into a clock schedule.

The planner decides *what* happens on a day; this decides *when*. Every block is derived
from something measured -- the visit duration, the driving distance, the intercity leg --
or from a stated assumption in ``config`` (when a day starts, how long lunch takes). The
dataset has no opening or closing times and the agent itineraries carry no clock times, so
nothing here pretends to know a real opening hour: a sight is placed inside the window its
``best_time`` implies, and anything that cannot be is flagged rather than quietly moved.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import (
    AFTERNOON_EARLIEST_START,
    AFTERNOON_LATEST_START,
    BREAK_AFTER_HOURS,
    BREAK_HOURS,
    CHECK_IN_HOURS,
    DAY_START_HOUR,
    LATEST_END_HOUR,
    LUNCH_HOURS,
    HARD_TIME_SLOTS,
    LUNCH_WINDOW,
    MAX_WAIT_HOURS,
    MORNING_LATEST_START,
    SIGHT_BUFFER_HOURS,
    TIME_WINDOW_GRACE_HOURS,
    TIME_WINDOWS,
    TRANSFER_START_HOUR,
)
import pandas as pd

from .data_loader import Dataset
from .geo import local_travel_hours, road_km
from .normalization import normalize_key
from .patterns import ItineraryPatterns

SCHEDULE_ASSUMPTIONS = (
    f"Sightseeing days start at {DAY_START_HOUR:.0f}:00 and travel days at {TRANSFER_START_HOUR:.0f}:00; "
    f"lunch is {LUNCH_HOURS:.0f}h between {LUNCH_WINDOW[0]:.1f} and {LUNCH_WINDOW[1]:.1f}; "
    f"a {BREAK_HOURS * 60:.0f}-minute break follows {BREAK_AFTER_HOURS:.0f}h of activity. "
    "The dataset has no opening or closing times, so each sight is placed inside the window "
    "implied by its best_time rather than a real opening hour."
)


def clock(hour: float) -> str:
    """0.5 -> '00:30', 13.25 -> '13:15', 26.7 -> '02:42 (+1)'.

    A day can legitimately run past midnight once a traveller adds a long evening of their
    own, and printing "26:42" for that is simply wrong.
    """
    hour = max(0.0, hour)
    hours, minutes = int(hour), int(round((hour - int(hour)) * 60))
    if minutes == 60:
        hours, minutes = hours + 1, 0
    next_day, hours = divmod(hours, 24)
    return f"{hours:02d}:{minutes:02d}" + (f" (+{next_day})" if next_day else "")


@dataclass
class Block:
    """One stretch of a day."""

    start: float
    end: float
    kind: str      # travel | sight | transfer | break | lunch | check-in | checkout | free | wait
    label: str
    detail: str = ""
    sight: str | None = None
    #: Of a travel block, the share that is parking/tickets rather than driving, so the
    #: day's reported road time stays comparable with the distance table.
    buffer_hours: float = 0.0

    def as_dict(self) -> dict:
        return {
            "start": clock(self.start),
            "end": clock(self.end),
            "hours": round(self.end - self.start, 2),
            "kind": self.kind,
            "label": self.label,
            "detail": self.detail,
            "sight": self.sight,
            "buffer_hours": round(self.buffer_hours, 2),
        }


@dataclass
class DaySchedule:
    """A day's blocks, plus anything the engine could not honour."""

    blocks: list[Block] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    #: For a slot that could not be honoured: what to do instead.
    alternatives: list[dict] = field(default_factory=list)
    #: Hard-slot sights this day could not fit inside their window. They are left out of
    #: the clock entirely rather than visited at the wrong time; the builder then moves
    #: them to another day or drops them from the trip with a reason.
    deferred: list[dict] = field(default_factory=list)
    #: Slot clashes the user chose to accept, each with the time that would have been
    #: better. These are advice, never a reason to refuse the plan.
    conflicts: list[dict] = field(default_factory=list)

    @property
    def buffer_total(self) -> float:
        return round(sum(block.buffer_hours for block in self.blocks), 2)

    def hours_of(self, *kinds: str) -> float:
        return round(sum(block.end - block.start for block in self.blocks if block.kind in kinds), 2)

    @property
    def totals(self) -> dict:
        """Every hour of the day, split the way the itinerary header reports it."""
        return {
            "sightseeing": self.hours_of("sight"),
            "local_travel": round(self.hours_of("travel") - self.buffer_total, 2),
            "stop_buffers": self.buffer_total,
            "intercity_travel": self.hours_of("transfer"),
            "meals_and_breaks": self.hours_of("lunch", "break"),
            "waiting": self.hours_of("wait"),
            "check_in_out": self.hours_of("check-in", "checkout"),
            #: Time the user added themselves; it is theirs, not the pace's.
            "your_own_plans": self.hours_of("custom"),
            "free": self.hours_of("free"),
        }

    def as_dict(self) -> dict:
        totals = self.totals
        span = round(self.blocks[-1].end - self.blocks[0].start, 2) if self.blocks else 0.0
        return {
            "starts": clock(self.blocks[0].start) if self.blocks else None,
            "ends": clock(self.blocks[-1].end) if self.blocks else None,
            "blocks": [block.as_dict() for block in self.blocks],
            "totals": totals,
            # What the day actually occupies, and the part of it that is sightseeing effort.
            "span_hours": span,
            # What the planner budgets against: sightseeing plus the local driving and buffers
            # around it. The intercity drive, meals and waiting are the day's other hours.
            "activity_hours": round(totals["sightseeing"] + totals["local_travel"]
                                    + totals["stop_buffers"], 2),
            "accounted_hours": round(sum(totals.values()), 2),
            "flags": self.flags,
            "notes": self.notes,
            "alternatives": self.alternatives,
            "deferred": self.deferred,
            "conflicts": self.conflicts,
        }


def _offer_alternative(schedule: DaySchedule, name: str, day: dict, city: str,
                       window_start: float, problem: str) -> None:
    """Record what to do about a slot that could not be honoured (once per sight)."""
    if any(entry["sight"] == name for entry in schedule.alternatives):
        return
    schedule.alternatives.append({
        "kind": "slot",
        "sight": name,
        "day": day.get("day"),
        "city": city,
        "window_start": round(window_start, 2),
        "problem": problem,
        "suggestion": f"move '{name}' to a day in {city} that still has time after {clock(window_start)}",
    })


def window_for(best_time: str) -> tuple[float, float] | None:
    """The clock window a ``best_time`` value implies, or None when it is missing."""
    key = str(best_time or "").strip().lower()
    if not key or key == "nan":
        return None
    return TIME_WINDOWS.get(key)


def opening_window(dataset: Dataset, city: str, sight_name: str) -> tuple[float, float] | None:
    """Real opening hours from the dataset, when it has them."""
    rows = dataset.sights[
        (dataset.sights["city_key"] == normalize_key(city))
        & (dataset.sights["sight_key"] == normalize_key(sight_name))
    ]
    if rows.empty:
        return None
    row = rows.iloc[0]
    opens, closes = row.get("opens_at"), row.get("closes_at")
    if opens is None or closes is None or pd.isna(opens) or pd.isna(closes):
        return None
    return float(opens), float(closes)


def effective_window(dataset: Dataset, city: str, sight: dict) -> tuple[tuple[float, float] | None, bool]:
    """The window a visit must sit in, and whether real opening hours backed it.

    Opening hours win where they exist: a sight marked "evening" that shuts at 17:00 is an
    afternoon visit, whatever the preference says.
    """
    preferred = window_for(sight.get("best_time", ""))
    opening = opening_window(dataset, city, sight.get("sight", ""))
    if opening is None:
        return preferred, False
    if preferred is None:
        return opening, True
    start, end = max(preferred[0], opening[0]), min(preferred[1], opening[1])
    return ((start, end) if start < end else opening), True


def window_violation(best_time: str, start: float, end: float) -> str | None:
    """Is this visit at a time the dataset would object to?

    Sunset and evening are hard: those sights exist for that light. Morning and afternoon
    are preferences -- almost half the dataset says "morning" -- so they only complain when
    the visit lands in the wrong half of the day.
    """
    slot = str(best_time or "").strip().lower()
    window = window_for(slot)
    if window is None or slot == "any":
        return None
    if slot in HARD_TIME_SLOTS:
        if start < window[0] - TIME_WINDOW_GRACE_HOURS or end > window[1] + TIME_WINDOW_GRACE_HOURS:
            article = "an" if slot[0] in "aeiou" else "a"
            return (f"is {article} {slot} sight but is scheduled {clock(start)}-{clock(end)}, outside "
                    f"{clock(window[0])}-{clock(window[1])}")
        return None
    if slot == "morning" and start > MORNING_LATEST_START:
        return f"is marked a morning sight but starts at {clock(start)}"
    if slot == "afternoon" and not AFTERNOON_EARLIEST_START - TIME_WINDOW_GRACE_HOURS <= start <= AFTERNOON_LATEST_START:
        return f"is marked an afternoon sight but starts at {clock(start)}"
    return None


def _travel_hours_between(dataset: Dataset, city: str, first: dict | None, second: dict) -> float:
    """Driving time from the previous stop (or the hotel) to the next one."""
    city_row = dataset.city_row(city)
    if city_row is None:
        return 0.0
    if first is None:
        start_lat, start_lon = float(city_row["lat"]), float(city_row["lon"])
    else:
        start_lat, start_lon = float(first["lat"]), float(first["lon"])
    distance = road_km(start_lat, start_lon, float(second["lat"]), float(second["lon"]))
    return local_travel_hours(distance, city_row.get("altitude_m"))


def _place_activities(schedule: DaySchedule, pending: list[dict], now: float,
                      until: float | None) -> float:
    """Lay down the traveller's own activities that are due by ``until``.

    ``until=None`` drains the list -- used once the sightseeing is done. A timed activity
    waits for its hour if the day reaches it early, and is placed as soon as the day can
    manage if it is already past. Neither case is an error: these are the traveller's own
    plans and the engine's job is to fit around them, not to refuse them.
    """
    while pending:
        activity = pending[0]
        at = _clock_to_hours(activity.get("at"))
        if until is not None and (at is None or at > until + 1e-9):
            break
        pending.pop(0)
        length = max(0.0, float(activity.get("duration_hours", 1.0) or 1.0))
        if at is not None and at > now + 1e-9:
            schedule.blocks.append(Block(now, at, "free", "Free time", f"before {activity['name']}"))
            now = at
        elif at is not None and at < now - 1e-9:
            schedule.notes.append(
                f"'{activity['name']}' was asked for at {clock(at)}, but the day is busy until "
                f"{clock(now)}; it starts there instead."
            )
        schedule.blocks.append(Block(now, now + length, "custom", activity["name"],
                                     activity.get("note", "") or "added by you"))
        now += length
    return now


def _clock_to_hours(text: object) -> float | None:
    """``"16:30"`` -> 16.5, or None when there is no time to read."""
    if not text:
        return None
    hours, _, minutes = str(text).partition(":")
    try:
        return int(hours) + int(minutes or 0) / 60.0
    except ValueError:
        return None


def _requested_hour(day: dict, name: str) -> float | None:
    """The clock time the user asked for this stop on this day, if any."""
    return _clock_to_hours((day.get("requested_times") or {}).get(name))


def _conflict(sight: dict, slot: str, scheduled_at: str, suggested_at: str,
              window: tuple[float, float], why: str) -> dict:
    """A slot clash the user chose to accept, with the time that would have been better."""
    return {
        "sight": sight["sight"],
        "best_time": slot,
        "scheduled_at": scheduled_at,
        "suggested_at": suggested_at,
        "window": f"{clock(window[0])}-{clock(window[1])}",
        "reason": why,
        "message": (f"'{sight['sight']}' is best between {clock(window[0])} and {clock(window[1])}; "
                    f"here it is scheduled at {scheduled_at} because {why}."),
    }


def is_hard_slot(best_time: object) -> bool:
    """True for a time of day that is a constraint, not a preference (evening, sunset)."""
    return str(best_time or "").strip().lower() in HARD_TIME_SLOTS


def hard_window(dataset: Dataset, city: str, sight: dict) -> tuple[float, float] | None:
    """The window a hard-slot sight must be visited in, or None when it is flexible.

    Opening hours narrow it where the dataset has them, so a sunset point that closes at
    19:00 gets 16:30-19:00 rather than 16:30-19:30.
    """
    if not is_hard_slot(sight.get("best_time")):
        return None
    window, _ = effective_window(dataset, city, sight)
    return window


def fits_window(start: float, visit: float, window: tuple[float, float]) -> bool:
    """Does a visit starting at ``start`` sit inside its window?"""
    return start >= window[0] - 1e-9 and start + visit <= window[1] + 1e-9


def _pull_flexible_forward(queue: list[dict], gap: float, dataset: Dataset, city: str,
                           previous: dict | None) -> tuple[int, dict, float] | None:
    """Rule 2: fill a wait with a flexible stop instead of standing still.

    Only a genuinely flexible stop ("any") may be moved -- pulling a morning sight into a
    late-afternoon gap would just trade one misplacement for another -- and only when it
    fits the gap whole, so the hard-slot sight behind it still starts on time.
    """
    for index, candidate in enumerate(queue):
        candidate_window, _ = effective_window(dataset, city, candidate)
        if candidate_window and candidate_window != TIME_WINDOWS.get("any"):
            continue
        hop = _travel_hours_between(dataset, city, previous, candidate) + SIGHT_BUFFER_HOURS
        cost = hop + float(candidate["duration_hours"])
        if cost <= gap + 1e-9:
            return index, candidate, hop
    return None


def build_day_schedule(
    day: dict,
    dataset: Dataset,
    patterns: ItineraryPatterns | None = None,
    pace_hours_available: float | None = None,
) -> DaySchedule:
    """Lay a planned day out on the clock, starting late when its slots ask for it.

    A day of evening sights has no business beginning at 09:00: laying it out from the
    morning pushes a sunset lake to midday and leaves the evening empty. So the day is laid
    out once, and if a hard slot was missed only because the day started too early, it is
    laid out again from later in the morning -- the shift a human agent would make.
    """
    schedule = _lay_out_day(day, dataset, patterns, pace_hours_available)
    if day.get("requested_start"):
        return schedule  # the user set the start; the engine does not second-guess it
    # Smallest shift first: rescuing one evening sight by starting at 13:24 beats a shift
    # big enough for all of them that runs the day past the evening limit.
    for shift in _shifts_worth_trying(schedule):
        later = _lay_out_day(day, dataset, patterns, pace_hours_available, start_shift=shift)
        if _overruns(later):
            continue
        if (len(later.deferred), len(later.alternatives)) >= (len(schedule.deferred), len(schedule.alternatives)):
            continue
        start = later.blocks[0].start if later.blocks else DAY_START_HOUR
        rescued = ({entry["sight"] for entry in schedule.deferred} - {e["sight"] for e in later.deferred}
                   or {entry["sight"] for entry in schedule.alternatives} - {e["sight"] for e in later.alternatives})
        names = sorted(rescued)
        subject = " and ".join([", ".join(names[:-1]), names[-1]]) if len(names) > 1 else names[0]
        later.notes.append(
            f"The day starts at {clock(start)} rather than {clock(DAY_START_HOUR)} so that {subject} "
            f"{'fall' if len(names) > 1 else 'falls'} inside {'their' if len(names) > 1 else 'its'} "
            f"preferred time of day."
        )
        return later
    return schedule


def _overruns(schedule: DaySchedule) -> bool:
    return bool(schedule.blocks) and schedule.blocks[-1].end > LATEST_END_HOUR + 1e-6


def _shifts_worth_trying(schedule: DaySchedule) -> list[float]:
    """Later start times that would put a missed slot back in its window, smallest first.

    Empty when nothing was missed, or when the day is a transfer: a drive cannot be
    postponed to suit a viewpoint.
    """
    if not schedule.blocks or any(block.kind == "transfer" for block in schedule.blocks):
        return []
    needed = set()
    for entry in schedule.deferred:
        shift = float(entry.get("needed_shift", 0.0))
        if shift > 0:
            needed.add(round(shift, 2))
    for entry in schedule.alternatives:
        if entry.get("kind") != "slot":
            continue
        block = next((b for b in schedule.blocks if b.kind == "sight" and b.sight == entry["sight"]), None)
        if block is None:
            continue
        shift = round(float(entry.get("window_start", 0.0)) - block.start, 2)
        if shift > 0:
            needed.add(shift)
    return sorted(needed)


def _lay_out_day(
    day: dict,
    dataset: Dataset,
    patterns: ItineraryPatterns | None = None,
    pace_hours_available: float | None = None,
    start_shift: float = 0.0,
) -> DaySchedule:
    """One pass of the clock over a planned day, optionally starting ``start_shift`` later."""
    schedule = DaySchedule()
    city = day.get("city", "")
    transfer = day.get("transfer")
    city_sights = [sight for sight in day.get("sights", []) if not sight.get("en_route")]
    en_route = [sight for sight in day.get("sights", []) if sight.get("en_route")]

    # The order was already settled by the planner (geography, time of day, then the habits
    # in past itineraries); the schedule only puts a clock against it.
    # A start the user chose wins over both the default and the engine's own late-start
    # retry: they know when they want to get up.
    chosen_start = _clock_to_hours(day.get("requested_start"))
    if chosen_start is not None:
        now = chosen_start
    else:
        now = (TRANSFER_START_HOUR if transfer else DAY_START_HOUR) + start_shift
        # An activity of your own at 06:00 means the day starts at 06:00. The engine's idea
        # of when a day begins is a default, not a rule the traveller has to live with.
        earliest = min((hour for hour in (_clock_to_hours(item.get("at"))
                                          for item in day.get("custom_activities", []) or [])
                        if hour is not None), default=None)
        if earliest is not None and earliest < now:
            now = earliest

    # 1. The drive, with any en-route stops taken along it.
    if transfer:
        drive_hours = transfer.get("drive_hours")
        if drive_hours is None:
            schedule.flags.append(
                f"{transfer.get('from_city')} to {transfer.get('to_city')} has no journey time in the dataset, "
                "so this day cannot be laid out on the clock."
            )
            return schedule
        # The transfer blocks carry the road time alone; the detour out to a stop and back
        # is the day's own travel, which is how the budget reserved it.
        travelled = 0.0
        for stop in en_route:
            share = max(0.0, min(1.0, float(stop.get("along_route_percent", 50)) / 100))
            leg_hours = max(0.0, share * float(drive_hours) - travelled)
            schedule.blocks.append(Block(now, now + leg_hours, "transfer",
                                         f"Drive towards {transfer['to_city']}",
                                         f"{leg_hours:.1f}h, turning off for {stop['sight']}"))
            now += leg_hours
            travelled += leg_hours
            detour = float(stop.get("detour_hours", 0))
            turn_off = detour / 2 + SIGHT_BUFFER_HOURS
            schedule.blocks.append(Block(now, now + turn_off, "travel", f"Turn off for {stop['sight']}",
                                         f"{stop.get('detour_km', 0)} km off the road, including the stop",
                                         buffer_hours=SIGHT_BUFFER_HOURS))
            now += turn_off
            visit = float(stop["duration_hours"])
            schedule.blocks.append(Block(now, now + visit, "sight", stop["sight"],
                                         f"{stop.get('detour_km', 0)} km off the road", stop["sight"]))
            now += visit
            if detour:
                schedule.blocks.append(Block(now, now + detour / 2, "travel", "Back to the main road",
                                             f"{detour / 2:.2f}h"))
                now += detour / 2
        remaining = max(0.0, float(drive_hours) - travelled)
        schedule.blocks.append(Block(now, now + remaining, "transfer",
                                     f"Drive to {transfer['to_city']}", f"{remaining:.1f}h by {transfer.get('mode', 'road')}"))
        now += remaining
        schedule.blocks.append(Block(now, now + CHECK_IN_HOURS, "check-in", f"Hotel check-in, {city}"))
        now += CHECK_IN_HOURS
    elif day.get("is_arrival_day"):
        schedule.blocks.append(Block(now, now + CHECK_IN_HOURS, "check-in", f"Arrival and hotel check-in, {city}"))
        now += CHECK_IN_HOURS

    # 2. The day's stops, each inside the window its best_time implies.
    previous: dict | None = None
    active_since = now
    # A day that only begins in the late afternoon does not stop for lunch: the traveller
    # ate before setting out. Booking one anyway was enough to push a sunset visit past the
    # departure transfer and have the sight thrown off the day.
    lunch_taken = now > LUNCH_WINDOW[1]
    if lunch_taken and city_sights:
        schedule.notes.append(f"The day starts at {clock(now)}, after lunch.")
    break_taken = False

    # Timed activities are appointments: the day works around them rather than appending
    # them at the end, so a 13:00 lunch of your own lands at 13:00 and not after the last
    # fort. Untimed ones fall wherever the day has room, which is after the sightseeing.
    pending_activities = sorted(
        (dict(item) for item in (day.get("custom_activities", []) or [])),
        key=lambda item: (_clock_to_hours(item.get("at")) is None,
                          _clock_to_hours(item.get("at")) or 0.0),
    )

    queue = list(city_sights)
    while queue:
        sight = queue.pop(0)
        travel = _travel_hours_between(dataset, city, previous, sight) + SIGHT_BUFFER_HOURS
        # Anything booked for a time this stop would run through goes first -- otherwise a
        # 13:00 lunch of your own waits for a fort that finishes at 14:42.
        now = _place_activities(schedule, pending_activities, now,
                                until=now + travel + float(sight["duration_hours"]))
        label = "Drive to" if previous else "Set off for"
        window, from_opening_hours = effective_window(dataset, city, sight)
        if window is None:
            schedule.flags.append(f"'{sight['sight']}' has no best_time in the dataset; placed in day order.")
        elif from_opening_hours:
            schedule.notes.append(
                f"'{sight['sight']}' is held to its opening hours "
                f"({clock(window[0])}-{clock(window[1])})."
            )

        # A time the user asked for is kept: the day waits for it, exactly as it would for
        # a sunset. Asking for an hour the day cannot reach is reported, never silently moved.
        requested = _requested_hour(day, sight["sight"])
        if requested is not None and now + travel < requested:
            schedule.blocks.append(Block(now, requested - travel, "wait",
                                         f"Free time before {sight['sight']}",
                                         f"you asked for {clock(requested)}"))
            now = requested - travel
        elif requested is not None and now + travel > requested + 1e-9:
            schedule.conflicts.append(_conflict(
                sight, str(sight.get("best_time", "")).strip().lower(), clock(now + travel),
                clock(requested), (requested, LATEST_END_HOUR),
                f"the day only reaches it at {clock(now + travel)}"))

        # An idle wait is wasted time: if a later flexible stop fits inside it, do that first.
        if window and now + travel < window[0]:
            swap = _pull_flexible_forward(queue, window[0] - (now + travel), dataset, city, previous)
            if swap is not None:
                index, candidate, hop = swap
                queue.pop(index)
                queue.insert(0, sight)
                schedule.notes.append(
                    f"'{candidate['sight']}' was brought forward to use the wait before "
                    f"'{sight['sight']}'."
                )
                sight, travel = candidate, hop
                window, from_opening_hours = effective_window(dataset, city, candidate)

        # Lunch and a breather, whenever the day reaches them.
        if not lunch_taken and now >= LUNCH_WINDOW[0]:
            schedule.blocks.append(Block(now, now + LUNCH_HOURS, "lunch", "Lunch"))
            now += LUNCH_HOURS
            lunch_taken = True
            active_since = now
        if not break_taken and now - active_since >= BREAK_AFTER_HOURS:
            schedule.blocks.append(Block(now, now + BREAK_HOURS, "break", "Short break", "tea, rest, photographs"))
            now += BREAK_HOURS
            break_taken = True

        schedule.blocks.append(Block(now, now + travel, "travel", f"{label} {sight['sight']}",
                                     f"{travel:.1f}h including parking and tickets",
                                     buffer_hours=min(SIGHT_BUFFER_HOURS, travel)))
        now += travel

        visit = float(sight["duration_hours"])
        slot = str(sight.get("best_time", "")).strip().lower()
        if window and slot in HARD_TIME_SLOTS:
            # A hard slot is a constraint, not a preference. The day waits for it where it
            # can; where it cannot, a sight the engine chose is left off the day entirely,
            # while one the user chose is kept and advised on.
            hotel = dataset.city_row(city)
            return_home = (_travel_hours_between(dataset, city, sight, {"lat": hotel["lat"], "lon": hotel["lon"]})
                           if hotel is not None else 0.0)
            tail = visit + return_home + (float((day.get("time_budget") or {}).get("departure_cost_hours", 0.0))
                                          if day.get("is_departure_day") else 0.0)
            chosen_by_user = bool(sight.get("user_pinned"))
            wait = max(0.0, window[0] - now)
            arrival = max(now, window[0])
            # A user who asked for this stop will wait longer for it than the engine would.
            affordable = MAX_WAIT_HOURS if not chosen_by_user else max(MAX_WAIT_HOURS, wait)
            reachable = fits_window(arrival, visit, window) and wait <= affordable + 1e-9
            finishes_in_time = arrival + tail <= LATEST_END_HOUR + 1e-9

            if reachable and (finishes_in_time or chosen_by_user):
                if wait > 0:
                    schedule.blocks.append(Block(now, now + wait, "wait", f"Free time before {sight['sight']}",
                                                 f"it is best seen from {clock(window[0])}"))
                    now += wait
                if not finishes_in_time:
                    schedule.conflicts.append(_conflict(sight, slot, clock(now), clock(window[0]), window,
                                                        f"the day then runs past {clock(LATEST_END_HOUR)}"))
            else:
                why = (f"a {wait:.1f}h wait is longer than the {MAX_WAIT_HOURS:.1f}h this day can absorb"
                       if wait > affordable else
                       f"it would not be back before {clock(LATEST_END_HOUR)}" if not finishes_in_time else
                       f"the day only reaches it at {clock(arrival)}, outside {clock(window[0])}-{clock(window[1])}")
                if chosen_by_user:
                    # Kept where the user put it; the better time is offered as advice.
                    schedule.conflicts.append(_conflict(sight, slot, clock(now), clock(window[0]),
                                                        window, why))
                else:
                    schedule.deferred.append({
                        "sight": sight["sight"],
                        "best_time": slot,
                        "window_start": round(window[0], 2),
                        "window_end": round(window[1], 2),
                        "visit_hours": round(visit, 2),
                        "reason": f"day {day.get('day')} cannot reach its {slot} slot: {why}",
                        "needed_shift": round(wait, 2),
                    })
                    _offer_alternative(schedule, sight["sight"], day, city, window[0],
                                       f"cannot reach its {slot} slot on day {day.get('day')}")
                    continue  # never placed in the morning instead
        violation = window_violation(sight.get("best_time", ""), now, now + visit)
        if window and from_opening_hours and (now < window[0] - 1e-9 or now + visit > window[1] + 1e-9):
            violation = (f"is open {clock(window[0])}-{clock(window[1])} but is scheduled "
                         f"{clock(now)}-{clock(now + visit)}")
        if violation:
            crowded = sum(1 for other in city_sights
                          if str(other.get("best_time", "")).lower() == slot)
            reason = (f" (this day holds {crowded} sights marked {slot}, so they cannot all be)"
                      if crowded > 2 and slot not in HARD_TIME_SLOTS else "")
            schedule.flags.append(f"'{sight['sight']}' {violation}{reason}.")
            if slot in HARD_TIME_SLOTS or from_opening_hours:
                # A hard slot or a closed door is not a matter of taste: say what to do instead.
                _offer_alternative(schedule, sight["sight"], day, city, window[0] if window else DAY_START_HOUR,
                                   f"{violation} on day {day.get('day')}")
        schedule.blocks.append(Block(now, now + visit, "sight", sight["sight"],
                                     f"{sight.get('best_time', 'any')} visit", sight["sight"]))
        now += visit
        previous = sight

    # 3. Anything of the traveller's own that is still waiting its turn.
    now = _place_activities(schedule, pending_activities, now, until=None)

    # 4. Lunch still owed on a day that never reached a stop after noon.
    if city_sights and not lunch_taken and now <= LUNCH_WINDOW[1]:
        schedule.blocks.append(Block(now, now + LUNCH_HOURS, "lunch", "Lunch"))
        now += LUNCH_HOURS

    # 5. Getting back, and what is left of the day.
    if city_sights:
        back = _travel_hours_between(dataset, city, previous, {"lat": dataset.city_row(city)["lat"],
                                                               "lon": dataset.city_row(city)["lon"]})
        # Always booked, however short: the day's reported driving has to be the whole loop,
        # or the header stops matching the distances the validator recomputes.
        if back > 0:
            schedule.blocks.append(Block(now, now + back, "travel", "Return to the hotel", f"{back:.2f}h"))
            now += back

    if day.get("is_departure_day"):
        cost = float((day.get("time_budget") or {}).get("departure_cost_hours", 0.0))
        if cost > 0:
            schedule.blocks.append(Block(now, now + cost, "checkout", "Check-out and onward transfer",
                                         (day.get("notes") or [""])[-1] if day.get("notes") else ""))
            now += cost
    elif schedule.blocks and now < LATEST_END_HOUR:
        # The day ends when the plan ends. The rest of the evening is free, and saying so in
        # a note is honest; turning it into a block would stretch the day to 20:00 on paper.
        schedule.notes.append(
            f"The planned day finishes at {clock(now)}; the evening until {clock(LATEST_END_HOUR)} is free."
        )

    if schedule.blocks and schedule.blocks[-1].end > LATEST_END_HOUR + 1e-6:
        # Running late because of your own evening is a choice, not a planning error.
        if any(block.kind == "custom" for block in schedule.blocks):
            schedule.notes.append(
                f"The day runs to {clock(schedule.blocks[-1].end)} because of what you added."
            )
        else:
            schedule.flags.append(
                f"The day runs to {clock(schedule.blocks[-1].end)}, past the "
                f"{clock(LATEST_END_HOUR)} limit."
            )
    return schedule


def schedule_conflicts(day: dict, patterns: ItineraryPatterns) -> list[str]:
    """Where this day's order differs from past itineraries for a reason worth stating.

    Pairs in different time slots are skipped: Shanti Stupa is a sunset sight, so putting it
    after Leh Palace follows the dataset even though past itineraries do the reverse.
    Within one slot the planner chose the shorter drive, which is worth saying out loud.
    """
    sights = day.get("sights", [])
    slot_of = {normalize_key(sight["sight"]): str(sight.get("best_time", "")).strip().lower() for sight in sights}
    name_of = {normalize_key(sight["sight"]): sight["sight"] for sight in sights}
    keys = [normalize_key(sight["sight"]) for sight in sights]
    messages = []
    for first, second, forward, reverse in patterns.conflicts(keys):
        if slot_of.get(first) != slot_of.get(second):
            continue  # the time of day explains the order, not habit
        messages.append(
            f"Day {day.get('day')}: past itineraries usually visit '{name_of.get(first, first)}' before "
            f"'{name_of.get(second, second)}' ({forward} times against {reverse}); this plan keeps the "
            f"shorter driving order instead."
        )
    return messages
