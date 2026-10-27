"""Daily time-packing engine.

A day's budget is the pace allowance minus whatever the day loses to arrival, departure or
an intercity transfer. Each sight costs its ``duration_hours`` plus a fixed buffer, and the
local driving between consecutive stops is charged from real coordinates, so a scattered
day runs out of hours before a compact one does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import pandas as pd

from .config import (
    ARRIVAL_OVERHEAD_HOURS,
    TRAVEL_DAY_MAX_DRIVE_HOURS,
    BEST_TIME_ORDER,
    DEPARTURE_OVERHEAD_HOURS,
    MIN_DEPARTURE_BUFFER_HOURS,
    MAX_DAY_ROUTE_KM,
    MAX_LEG_KM,
    EXTRA_HOURS_PER_STOP,
    REFINE_MARGIN_KM,
    REFINE_PASSES,
    MAX_SIGHTS_PER_DAY,
    ARRIVAL_DAY_MAX_KM,
    HARD_TIME_SLOTS,
    LATEST_END_HOUR,
    MAX_STOPS_WITH_EXCURSION,
    TIME_WINDOWS,
    PACE_ACTIVE_HOURS,
    SIGHT_BUFFER_HOURS,
    TRANSFER_OVERHEAD_HOURS,
)
from .geo import RouteLeg, local_travel_hours, order_by_route, road_km, route_order


def pace_hours(pace: str) -> float:
    """Active sightseeing hours for a pace name."""
    try:
        return PACE_ACTIVE_HOURS[pace.strip().lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown pace '{pace}'. Use one of {sorted(PACE_ACTIVE_HOURS)}.") from exc


@dataclass
class DaySlot:
    """A day of the trip before its sights are chosen."""

    day: int
    city: str
    city_key: str
    centre_lat: float
    centre_lon: float
    altitude_m: float | None = None
    transfer: RouteLeg | None = None
    #: Ordering habits mined from past itineraries; used only to settle ties.
    order_patterns: object | None = None
    is_arrival: bool = False
    is_departure: bool = False
    #: Checkout + transfer to the airport/station + reporting buffer, set by the builder.
    departure_cost_hours: float = DEPARTURE_OVERHEAD_HOURS
    #: True when travel alone fills the day (unknown journey, or longer than a travel day).
    travel_only: bool = False
    #: Budget taken straight from an existing plan, so edits are judged against the same
    #: hours the day actually has.
    available_override: float | None = None
    #: Hours the user chose to add to this day on top of the pace allowance.
    extra_hours: float = 0.0
    #: Hours this day already owes to the traveller's own plans. A twelve-hour activity
    #: leaves no room for sightseeing, and the packer has to know that before it fills
    #: the day -- otherwise the clock ends up running to four in the morning.
    committed_hours: float = 0.0

    @property
    def is_travel_day(self) -> bool:
        """True when the drive alone fills the day, leaving no sightseeing hours."""
        return bool(self.travel_only or (self.transfer is not None and not self.transfer.fits_one_day))

    def day_allowance(self, pace: str) -> float:
        """Hours the day can run to.

        A day given over entirely to travel is not limited by the sightseeing pace: there
        is nothing to pace. It can start at dawn and run as long as the drive needs, up to
        TRAVEL_DAY_MAX_DRIVE_HOURS, so a 12h haul is one long travel day rather than an
        impossible one.
        """
        allowance = pace_hours(pace)
        if self.transfer is None or self.transfer.drive_hours is None:
            return allowance
        needed = self.transfer.drive_hours + TRANSFER_OVERHEAD_HOURS
        if self.is_arrival:
            needed += ARRIVAL_OVERHEAD_HOURS
        if self.is_departure:
            needed += self.departure_cost_hours
        if needed <= allowance:
            return allowance
        return min(max(allowance, needed), TRAVEL_DAY_MAX_DRIVE_HOURS + TRANSFER_OVERHEAD_HOURS)

    def cost_breakdown(self, pace: str) -> dict[str, float]:
        """Where the day's hours go, for the response and for the validator."""
        transfer_hours = 0.0
        overhead = 0.0
        if self.transfer is not None:
            transfer_hours = self.transfer.hours_charged_to_day
            overhead += TRANSFER_OVERHEAD_HOURS
        if self.is_arrival:
            overhead += ARRIVAL_OVERHEAD_HOURS
        if self.is_departure:
            overhead += self.departure_cost_hours
        return {
            "pace_allowance": self.day_allowance(pace),
            "intercity_travel": transfer_hours,
            "day_overheads": overhead,
        }

    def available_hours(self, pace: str) -> float:
        """Sightseeing hours left after travel and day overheads.

        A day whose journey time is unknown, or longer than a single travel day, has no
        sightseeing hours at all -- the engine will not quietly schedule sights into it.
        """
        if self.available_override is not None:
            return max(0.0, round(self.available_override + self.extra_hours - self.committed_hours, 2))
        if self.is_travel_day:
            return 0.0
        costs = self.cost_breakdown(pace)
        hours = costs["pace_allowance"] - costs["intercity_travel"] - costs["day_overheads"]
        if self.is_departure:
            hours -= MIN_DEPARTURE_BUFFER_HOURS
        # The traveller can choose a longer day than the pace suggests; the clock still has
        # to fit it before LATEST_END_HOUR, so this buys sights, not miracles.
        return max(0.0, round(hours + self.extra_hours - self.committed_hours, 2))

    def max_sights(self, pace: str) -> int:
        """Stop count cap: shortened days get fewer stops, lengthened ones get more.

        The cap exists so a day is not diced into six twenty-minute visits. A traveller who
        asks for two extra hours has said they want a fuller day, so the cap moves with the
        hours -- one more stop for every two hours added.
        """
        cap = MAX_SIGHTS_PER_DAY[pace.strip().lower()]
        if self.transfer is not None or self.is_arrival or self.is_departure:
            cap = max(1, cap - 2)
        return cap + int(self.extra_hours // EXTRA_HOURS_PER_STOP)


@dataclass
class PackedDay:
    """A day after packing: the chosen sights in visiting order, with its time budget."""

    slot: DaySlot
    sights: list[dict] = field(default_factory=list)
    sightseeing_hours: float = 0.0
    buffer_hours: float = 0.0
    local_travel_hours: float = 0.0
    route_km: float = 0.0
    max_leg_km: float = 0.0

    @property
    def used_hours(self) -> float:
        return round(self.sightseeing_hours + self.buffer_hours + self.local_travel_hours, 2)

    @property
    def spare_hours(self) -> float:
        return round(self.slot.available_hours(self._pace) - self.used_hours, 2)

    _pace: str = "moderate"


def sort_by_priority(sights: pd.DataFrame) -> pd.DataFrame:
    """must_visit first, then must_see_tier (1 best), then priority rank.

    A day reserved for a big-ticket sight carries an ``_anchor`` flag, which outranks
    everything else so the day is packed around that sight.
    """
    columns = ["must_visit_flag", "must_see_tier", "priority", "mentions"]
    ascending = [False, True, True, False]
    if "_anchor" in sights.columns:
        columns, ascending = ["_anchor", *columns], [False, *ascending]
    if "_forced" in sights.columns:  # a sight the user pinned goes in before anything else
        columns, ascending = ["_forced", *columns], [False, *ascending]
    return sights.sort_values(columns, ascending=ascending, kind="mergesort")  # stable


def is_big_ticket(row: pd.Series, budget_hours: float) -> bool:
    """A sight that eats at least half a day and deserves a day built around it."""
    cost = float(row["duration_hours"]) + SIGHT_BUFFER_HOURS
    important = bool(row["must_visit_flag"]) or int(row["must_see_tier"]) <= 2
    return important and cost >= 0.5 * budget_hours


def _cost_hours(row: pd.Series) -> float:
    return float(row["duration_hours"]) + SIGHT_BUFFER_HOURS


def _day_metrics(selected: pd.DataFrame, slot: DaySlot) -> tuple[float, float, float]:
    """(local driving hours, sight-to-sight km, longest leg km) for a day's order."""
    if selected.empty:
        return 0.0, 0.0, 0.0
    lats, lons = selected["lat"].tolist(), selected["lon"].tolist()
    legs = [road_km(lats[i], lons[i], lats[i + 1], lons[i + 1]) for i in range(len(selected) - 1)]
    hotel_legs = [
        road_km(slot.centre_lat, slot.centre_lon, lats[0], lons[0]),   # out to the first stop
        road_km(lats[-1], lons[-1], slot.centre_lat, slot.centre_lon),  # and back from the last
    ]
    total_km = sum(legs) + sum(hotel_legs)
    hours = local_travel_hours(total_km, slot.altitude_m)
    return hours, sum(legs), max(legs) if legs else 0.0


def _ordered(candidate: pd.DataFrame, slot: DaySlot) -> pd.DataFrame:
    """Order a day's sights by time of day, and geographically inside each time group.

    Sorting geographically and then by time undoes the routing (a sunset point can be
    dragged back across town), so the time groups come first and each group is routed from
    where the previous one ended.

    The ordering itself is worked out on plain tuples and the frame is sliced once at the
    end. It used to build a new DataFrame per time group, per swap and per concat, which
    for a five-stop day meant dozens of frame constructions -- on a path the packer calls
    hundreds of times a plan.
    """
    if len(candidate) < 2:
        return candidate.reset_index(drop=True)
    frame = candidate.reset_index(drop=True)
    rows = tuple(
        (str(key), float(lat), float(lon), BEST_TIME_ORDER.get(str(best).lower(), 1))
        for key, lat, lon, best in zip(frame["sight_key"], frame["lat"], frame["lon"],
                                       frame["best_time"])
    )
    positions = _order_positions(rows, slot.centre_lat, slot.centre_lon, slot.order_patterns)
    return frame.iloc[list(positions)].reset_index(drop=True)


def _order_positions(rows: tuple[tuple[str, float, float, int], ...],
                     centre_lat: float, centre_lon: float, patterns) -> tuple[int, ...]:
    """Where each stop goes, as positions into ``rows``. No pandas, no copies."""
    order: list[int] = []
    cursor_lat, cursor_lon = centre_lat, centre_lon
    for rank in sorted({row[3] for row in rows}):
        members = [index for index, row in enumerate(rows) if row[3] == rank]
        points = tuple((rows[index][1], rows[index][2]) for index in members)
        routed = [members[position] for position in route_order(points, cursor_lat, cursor_lon)]
        routed = _learned_order(routed, rows, centre_lat, centre_lon, patterns)
        order += routed
        cursor_lat, cursor_lon = rows[routed[-1]][1], rows[routed[-1]][2]
    return tuple(order)


def _learned_order(group: list[int], rows: tuple[tuple[str, float, float, int], ...],
                   centre_lat: float, centre_lon: float, patterns) -> list[int]:
    """Swap neighbouring stops to match past itineraries, when the driving does not suffer.

    Geography and time of day decide the shape of the day; this only settles ties inside a
    single time slot, and a swap is kept only when it costs no extra driving.
    """
    if patterns is None or len(group) < 2:
        return group

    def driving_km(positions: list[int]) -> float:
        points = [(rows[index][1], rows[index][2]) for index in positions]
        legs = sum(road_km(points[i][0], points[i][1], points[i + 1][0], points[i + 1][1])
                   for i in range(len(points) - 1))
        return road_km(centre_lat, centre_lon, points[0][0], points[0][1]) + legs

    ordered = list(group)
    for index in range(len(ordered) - 1):
        first, second = rows[ordered[index]][0], rows[ordered[index + 1]][0]
        if not patterns.prefers(second, first):
            continue
        candidate = list(ordered)
        candidate[index], candidate[index + 1] = candidate[index + 1], candidate[index]
        if driving_km(candidate) <= driving_km(ordered) + 1e-9:
            ordered = candidate
    return ordered


def evaluate_day(sights: pd.DataFrame, slot: DaySlot, pace: str) -> dict:
    """Would this exact set of sights work on this day?

    Returns the ordered sights, the hours they need, and any rule they break. Used by the
    packer and by the edit API, so "can I add this sight here?" is answered by the same
    code that builds the plan.
    """
    budget = slot.available_hours(pace)
    if sights.empty:
        return {"ordered": sights, "needed_hours": 0.0, "available_hours": budget, "fits": budget >= 0,
                "route_km": 0.0, "max_leg_km": 0.0, "blockers": []}

    ordered = _ordered(sights, slot)
    travel_hours, route_km_total, max_leg = _day_metrics(ordered, slot)
    visit_hours = float(ordered["duration_hours"].sum())
    buffer_hours = SIGHT_BUFFER_HOURS * len(ordered)
    needed = visit_hours + buffer_hours + travel_hours
    excursions = int(ordered["is_excursion"].sum()) if "is_excursion" in ordered else 0

    blockers: list[str] = []
    if slot.is_departure and excursions:
        blockers.append("a day trip out of town cannot share the departure day with the onward journey")
    hard_slot_problem = hard_slot_blocker(ordered, slot)
    if hard_slot_problem and not ("_forced" in ordered and ordered["_forced"].any()):
        blockers.append(hard_slot_problem)
    if slot.is_arrival:
        for name in _far_from_centre(ordered):
            blockers.append(
                f"'{name}' is {float(ordered.loc[ordered['sight'] == name, 'distance_from_centre_km'].iloc[0]):.0f} km "
                f"from the centre, too far for the arrival day (limit {ARRIVAL_DAY_MAX_KM:.0f} km)"
            )
    if needed > budget:
        blockers.append(
            f"needs {needed:.1f}h but the day has {budget:.1f}h "
            f"(visits {visit_hours:.1f}h + buffers {buffer_hours:.1f}h + local travel {travel_hours:.1f}h)"
        )
    if len(ordered) > slot.max_sights(pace):
        blockers.append(f"{len(ordered)} stops is over the {slot.max_sights(pace)}-stop limit for this day")
    if max_leg > MAX_LEG_KM:
        blockers.append(f"two stops would be {max_leg:.0f} km apart (limit {MAX_LEG_KM:.0f} km)")
    if route_km_total > MAX_DAY_ROUTE_KM:
        blockers.append(f"the day would cover {route_km_total:.0f} km between stops (limit {MAX_DAY_ROUTE_KM:.0f} km)")
    if excursions > 1:
        blockers.append("two day trips cannot share a day")
    elif excursions == 1 and len(ordered) - 1 > MAX_STOPS_WITH_EXCURSION:
        blockers.append(f"a day trip leaves room for at most {MAX_STOPS_WITH_EXCURSION} other stops")
    return {
        "ordered": ordered,
        "needed_hours": round(needed, 2),
        "available_hours": budget,
        "fits": not blockers,
        "route_km": round(route_km_total, 1),
        "max_leg_km": round(max_leg, 1),
        "blockers": blockers,
    }


def _far_from_centre(sights: pd.DataFrame) -> list[str]:
    """Sights too far out for an arrival day, by name.

    Day 1 is short, and the traveller has just spent it travelling: the agent itineraries
    keep it in town (see ``ARRIVAL_DAY_MAX_KM``). A distant sight is not dropped, only
    deferred -- :func:`plan_city` places it on a later day.
    """
    if sights.empty or "distance_from_centre_km" not in sights:
        return []
    far = sights[sights["distance_from_centre_km"].astype(float) > ARRIVAL_DAY_MAX_KM]
    if "_forced" in far.columns:
        far = far[~far["_forced"].fillna(False).astype(bool)]  # an explicit pin outranks the habit
    return [str(name) for name in far["sight"]]


def _is_pinned(row: pd.Series) -> bool:
    """Did the user put this sight on this day by hand?"""
    return bool(row.get("_forced", False)) and not pd.isna(row.get("_forced_day", float("nan")))


def hard_slot_blocker(sights: pd.DataFrame, slot: DaySlot) -> str | None:
    """Why this day cannot honour the hard slots it is being asked to hold, if it cannot.

    Evening and sunset are constraints: every such sight has to happen between 16:30 and
    the end of its window. That window is only three and a half hours long, so a day can
    hold one or two of them, not four -- and a departure day, which owes its evening to the
    onward journey, can hold none at all. Catching it here keeps the clock from having to
    throw the sight out later.
    """
    if sights.empty or "best_time" not in sights:
        return None
    # Plain iteration: `.astype(str).str.lower().isin(...)` is three passes over a new
    # column, and this runs for every candidate the packer tries.
    hard_rows = [(str(name), str(best).strip().lower(), float(hours))
                 for name, best, hours in zip(sights["sight"], sights["best_time"],
                                              sights["duration_hours"])
                 if str(best).strip().lower() in HARD_TIME_SLOTS]
    if not hard_rows:
        return None
    slots = {row[1] for row in hard_rows}
    window_start = max(TIME_WINDOWS[name][0] for name in slots)
    window_end = min(TIME_WINDOWS[name][1] for name in slots)
    # The onward journey still has to happen, so a departure day's evening is already spoken for.
    latest = LATEST_END_HOUR - (slot.departure_cost_hours if slot.is_departure else 0.0)
    end = min(window_end, latest)
    needed = sum(row[2] for row in hard_rows) + SIGHT_BUFFER_HOURS * len(hard_rows)
    if window_start + needed <= end + 1e-9:
        return None
    names = ", ".join(f"'{row[0]}'" for row in hard_rows)
    if end <= window_start:
        return (f"{names} must be seen after {_clock(window_start)}, which this day owes to the "
                f"onward journey")
    return (f"{names} need {needed:.1f}h inside {_clock(window_start)}-{_clock(end)}, "
            f"which is only {end - window_start:.1f}h long")


def _clock(hours: float) -> str:
    return f"{int(hours):02d}:{int(round((hours % 1) * 60)):02d}"


def pack_day(candidates: pd.DataFrame, slot: DaySlot, pace: str) -> PackedDay:
    """Fill one day from its cluster, respecting time, stop count and distance limits.

    Sights are tried in priority order; each is kept only if the day still fits afterwards.
    A sight that does not fit is left for :func:`plan_city` to try on another day.
    """
    packed = PackedDay(slot=slot)
    packed._pace = pace
    budget = slot.available_hours(pace)
    stop_cap = slot.max_sights(pace)
    if candidates.empty or budget <= 0:
        return packed

    # Candidates are tried in priority order, each kept only if the day still fits. The
    # chosen set is carried as positions into `ranked` and sliced once per trial;
    # concatenating a one-row frame per candidate was most of the planner's runtime.
    ranked = sort_by_priority(candidates).reset_index(drop=True)
    chosen: list[int] = []
    for position in range(len(ranked)):
        if len(chosen) >= stop_cap:
            break
        trial = _ordered(ranked.iloc[chosen + [position]], slot)
        travel_hours, route_km_total, max_leg = _day_metrics(trial, slot)
        visit_hours = float(trial["duration_hours"].sum())
        buffer_hours = SIGHT_BUFFER_HOURS * len(trial)
        excursions = int(trial["is_excursion"].sum()) if "is_excursion" in trial else 0

        if slot.is_arrival and _far_from_centre(trial):
            continue  # the arrival day stays in town
        if slot.is_departure and excursions:
            continue  # you cannot spend the last morning an hour out of town
        if hard_slot_blocker(trial, slot) and not _is_pinned(ranked.iloc[position]):
            continue  # an evening sight belongs on a day that can give it the evening
        if visit_hours + buffer_hours + travel_hours > budget:
            continue
        if max_leg > MAX_LEG_KM or route_km_total > MAX_DAY_ROUTE_KM:
            continue  # would mean criss-crossing the region in one day
        if excursions > 1 or (excursions == 1 and len(trial) - 1 > MAX_STOPS_WITH_EXCURSION):
            continue  # a day trip cannot share the day with a full city programme

        chosen.append(position)
        packed.sightseeing_hours = round(visit_hours, 2)
        packed.buffer_hours = round(buffer_hours, 2)
        packed.local_travel_hours = round(travel_hours, 2)
        packed.route_km = round(route_km_total, 1)
        packed.max_leg_km = round(max_leg, 1)

    if chosen:
        packed.sights = [_sight_payload(row)
                         for _, row in _ordered(ranked.iloc[chosen], slot).iterrows()]
    return packed


def _sight_payload(row: pd.Series) -> dict:
    """The planner's view of one scheduled sight (enriched later in builder.py)."""
    return {
        "sight": row["sight"],
        "visit_type": row.get("visit_type", ""),
        "duration_hours": round(float(row["duration_hours"]), 2),
        "buffer_hours": SIGHT_BUFFER_HOURS,
        "must_see_tier": int(row["must_see_tier"]),
        "must_visit": bool(row["must_visit_flag"]),
        "priority": int(row["priority"]),
        "best_time": row.get("best_time", "any"),
        "zone": row.get("zone", ""),
        "lat": float(row["lat"]),
        "lon": float(row["lon"]),
        "is_excursion": bool(row.get("is_excursion", False)),
        "distance_from_centre_km": round(float(row.get("distance_from_centre_km", 0.0)), 1),
    }


def plan_city(
    clusters: list[pd.DataFrame],
    excursions: pd.DataFrame,
    slots: list[DaySlot],
    pace: str,
) -> list[PackedDay]:
    """Assign clusters to a city's days and pack each day.

    Order of decisions: excursions claim the longest ordinary day (never a transfer,
    arrival or departure day), the heaviest clusters go to the days with the most hours,
    and whatever is left over is offered to every day that still has room.
    """
    if not slots:
        return []

    frames = [c for c in clusters if not c.empty] + ([excursions] if not excursions.empty else [])
    universe = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if universe.empty:
        return [PackedDay(slot=slot, _pace=pace) for slot in sorted(slots, key=lambda s: s.day)]

    by_capacity = sorted(slots, key=lambda s: (-s.available_hours(pace), s.day))
    assignments: dict[int, list[pd.DataFrame]] = {}

    # Sights the user pinned to a particular day are placed there first; the rest of the
    # day is then built around them.
    pinned_names: set[str] = set()
    if "_forced_day" in universe.columns:
        for day_number in sorted({d for d in universe["_forced_day"].dropna().tolist()}):
            rows = universe[universe["_forced_day"] == day_number]
            if any(slot.day == int(day_number) for slot in slots):
                assignments.setdefault(int(day_number), []).insert(0, rows)
                pinned_names |= set(rows["sight"])

    # Day trips get a day of their own, but they must not take the last usable day away from
    # the city's own must-visit stops. In Pangong Lake, four of the five rows are 40-180 km
    # away: without this rule they claim the only free day and the lake itself -- the reason
    # for the stop -- has nowhere to go.
    def rank(row: pd.Series) -> tuple:
        return (not bool(row["must_visit_flag"]), int(row["must_see_tier"]), int(row["priority"]))

    excursion_rows = [row for _, row in sort_by_priority(excursions).iterrows()] if not excursions.empty else []
    free_slots = [
        slot for slot in by_capacity
        if slot.transfer is None and not slot.is_arrival and not slot.is_departure and slot.available_hours(pace) > 0
    ]
    cluster_rank = min(
        (rank(row) for cluster in clusters if not cluster.empty for _, row in cluster.iterrows()),
        default=(True, 9, 99),
    )
    spare_days = max(0, len(free_slots) - len([c for c in clusters if not c.empty]))

    for slot in free_slots:
        if not excursion_rows:
            break
        if spare_days <= 0 and rank(excursion_rows[0]) > cluster_rank:
            break  # the city's own stops need this day more than the day trip does
        spare_days -= 1
        assignments.setdefault(slot.day, []).append(excursion_rows.pop(0).to_frame().T)

    # Big-ticket sights (e.g. Rohtang Pass: 5h plus the drive) get a day built around them,
    # otherwise a cheaper neighbour fills the day first and they never fit anywhere.
    budget = pace_hours(pace)
    anchored: set[str] = set()
    for slot in by_capacity:
        if slot.day in assignments or slot.transfer is not None or slot.is_arrival or slot.is_departure:
            continue
        candidates = universe[~universe["sight"].isin(anchored | pinned_names)]
        big = [row for _, row in sort_by_priority(candidates).iterrows() if is_big_ticket(row, budget)]
        if not big:
            break
        anchor = big[0].to_frame().T
        anchor["_anchor"] = True
        assignments[slot.day] = [anchor]
        anchored.add(big[0]["sight"])

    available_clusters = [i for i, cluster in enumerate(clusters) if not cluster.empty]

    # The arrival day gets the headline cluster. In the agent itineraries 85% of day-1 stops
    # are must-visits against 47% later on: the first day is when the city's famous sights
    # are seen, not what is left over. Capacity order alone did the opposite, because the
    # arrival day has the fewest hours and so chose last.
    arrival = next((slot for slot in by_capacity
                    if slot.is_arrival and slot.day not in assignments and slot.available_hours(pace) > 0), None)
    if arrival is not None and available_clusters:
        def headline(index: int) -> tuple:
            cluster = clusters[index]
            in_town = cluster[cluster["distance_from_centre_km"] <= ARRIVAL_DAY_MAX_KM]                 if "distance_from_centre_km" in cluster else cluster
            return (int(in_town["must_visit_flag"].sum()),
                    -float(in_town["must_see_tier"].mean()) if not in_town.empty else -9.0,
                    -float(in_town["priority"].min()) if not in_town.empty else -99.0)

        best = max(available_clusters, key=headline)
        if headline(best)[0] > 0:  # only worth claiming the day if it really holds must-visits
            available_clusters.remove(best)
            assignments[arrival.day] = [clusters[best]]

    for slot in by_capacity:
        if not available_clusters or slot.day in assignments:
            continue
        index = max(
            available_clusters,
            key=lambda i: (float(clusters[i]["duration_hours"].sum()), -float(clusters[i]["priority"].min())),
        )
        available_clusters.remove(index)
        assignments[slot.day] = [clusters[index]]

    leftovers = [
        cluster[~cluster["sight"].isin(anchored | pinned_names)]
        for cluster in (clusters[i] for i in available_clusters)
    ]
    leftovers += [row.to_frame().T for row in excursion_rows]

    packed_days: list[PackedDay] = []
    scheduled: set[str] = set()
    for slot in sorted(slots, key=lambda s: s.day):
        candidate_frames = assignments.get(slot.day, []) + leftovers
        candidates = pd.concat(candidate_frames, ignore_index=True) if candidate_frames else universe.iloc[0:0]
        candidates = candidates[~candidates["sight"].isin(scheduled)]
        candidates = candidates.drop_duplicates(subset="sight", keep="first")
        if "_forced_day" in candidates.columns and not candidates.empty:
            # A sight pinned to another day must not be consumed by this one.
            candidates = candidates[candidates["_forced_day"].isna() | (candidates["_forced_day"] == slot.day)]
        packed = pack_day(candidates, slot, pace)
        packed._pace = pace
        scheduled |= {sight["sight"] for sight in packed.sights}
        packed_days.append(packed)

    # Top-up pass: a day with spare hours takes anything still unscheduled and nearby.
    for packed in packed_days:
        remaining = universe[~universe["sight"].isin(scheduled)]
        if remaining.empty:
            break
        if packed.spare_hours <= SIGHT_BUFFER_HOURS or len(packed.sights) >= packed.slot.max_sights(pace):
            continue
        current = universe[universe["sight"].isin({sight["sight"] for sight in packed.sights})]
        refilled = pack_day(pd.concat([current, remaining], ignore_index=True), packed.slot, pace)
        if len(refilled.sights) > len(packed.sights):
            packed.sights = refilled.sights
            packed.sightseeing_hours = refilled.sightseeing_hours
            packed.buffer_hours = refilled.buffer_hours
            packed.local_travel_hours = refilled.local_travel_hours
            packed.route_km, packed.max_leg_km = refilled.route_km, refilled.max_leg_km
            scheduled = {sight["sight"] for day in packed_days for sight in day.sights}

    refine_grouping(packed_days, universe, pace)
    return packed_days


def _name_index(universe: pd.DataFrame) -> dict[str, int]:
    """sight name -> row label, built once per frame rather than per lookup."""
    return {str(name): label for label, name in zip(universe.index, universe["sight"])}


def _frame_for(universe: pd.DataFrame, names: list[str],
               by_name: dict[str, int] | None = None) -> pd.DataFrame:
    """The planner's rows for these sight names, in this order."""
    if not names:
        return universe.iloc[0:0]
    by_name = _name_index(universe) if by_name is None else by_name
    wanted = [by_name[name] for name in names if name in by_name]
    return universe.loc[wanted]


def _day_cost(frame: pd.DataFrame, slot: DaySlot, pace: str,
              memo: dict | None = None) -> float | None:
    """Driving hours for this set on this day, or None when the day could not hold it.

    The search tries the same set against the same day many times over, and evaluating one
    is the expensive part, so results are memoised per (day, set of stops).
    """
    if frame.empty:
        return 0.0
    key = (slot.day, frozenset(frame["sight"]))
    if memo is not None and key in memo:
        return memo[key]
    verdict = evaluate_day(frame, slot, pace)
    cost = None if not verdict["fits"] else float(_day_metrics(verdict["ordered"], slot)[0])
    if memo is not None:
        memo[key] = cost
    return cost


def refine_grouping(packed_days: list[PackedDay], universe: pd.DataFrame, pace: str) -> None:
    """Group nearby stops together by moving and swapping between days.

    Clustering decides which stops belong together before any day has a clock, and the
    time-of-day groups inside a day then stretch the route: across the dataset 17% of days
    drove more than two kilometres further than their own stops allow. This pass takes each
    pair of days in a city and tries every single move and every pairwise swap, keeping only
    changes that shorten the driving while both days still pass every rule -- the budget,
    the stop cap, the leg limits, the evening capacity, the arrival-day distance. Feasibility
    is checked with :func:`evaluate_day`, the same function the packer and the edit API use,
    so nothing here can quietly invent a day the rest of the engine would reject.
    """
    movable = [day for day in packed_days
               if day.slot.transfer is None and not day.slot.travel_only and day.sights]
    pinned = set()
    if "_forced_day" in universe.columns:
        pinned = {str(row["sight"]) for _, row in universe.iterrows() if not pd.isna(row.get("_forced_day"))}

    memo: dict = {}
    by_name = _name_index(universe)
    coords = {str(row["sight"]): (float(row["lat"]), float(row["lon"]))
              for _, row in universe.iterrows() if not pd.isna(row.get("lat"))}

    def centre(names: list[str]) -> tuple[float, float] | None:
        points = [coords[name] for name in names if name in coords]
        if not points:
            return None
        return (sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points))

    def nearer_to(name: str, target: tuple[float, float] | None,
                  home: tuple[float, float] | None) -> bool:
        """Is this stop closer to the other day's stops than to its own?

        The whole search is "which day do these belong to", so only a stop that is actually
        nearer the other group is worth evaluating. Without this filter every pair of stops
        on every pair of days gets a full feasibility check, and a plan takes 14 seconds.
        """
        if name not in coords or target is None or home is None:
            return False
        here = road_km(coords[name][0], coords[name][1], home[0], home[1])
        there = road_km(coords[name][0], coords[name][1], target[0], target[1])
        # A small margin lets near-ties through: a stop a kilometre further from the other
        # group can still shorten the route once the order is re-solved.
        return there < here + REFINE_MARGIN_KM

    for _ in range(REFINE_PASSES):
        improved = False
        for first, second in combinations(movable, 2):
            if first.slot.city_key != second.slot.city_key:
                continue
            here = [sight["sight"] for sight in first.sights]
            there = [sight["sight"] for sight in second.sights]
            base = _day_cost(_frame_for(universe, here, by_name), first.slot, pace, memo)
            other = _day_cost(_frame_for(universe, there, by_name), second.slot, pace, memo)
            if base is None or other is None:
                continue
            best = (base + other, None)
            my_centre, their_centre = centre(here), centre(there)

            for name in here:
                if name in pinned or _protected(first, name, universe):
                    continue
                if not nearer_to(name, their_centre, my_centre):
                    continue
                left, right = [x for x in here if x != name], there + [name]
                cost = _pair_cost(universe, left, right, first, second, pace, memo, by_name)
                if cost is not None and cost < best[0] - 1e-6:
                    best = (cost, (left, right))
            for name in there:
                if name in pinned or _protected(second, name, universe):
                    continue
                if not nearer_to(name, my_centre, their_centre):
                    continue
                left, right = here + [name], [x for x in there if x != name]
                cost = _pair_cost(universe, left, right, first, second, pace, memo, by_name)
                if cost is not None and cost < best[0] - 1e-6:
                    best = (cost, (left, right))
            for mine in here:
                for yours in there:
                    if mine in pinned or yours in pinned:
                        continue
                    if _protected(first, mine, universe) or _protected(second, yours, universe):
                        continue
                    if not (nearer_to(mine, their_centre, my_centre)
                            or nearer_to(yours, my_centre, their_centre)):
                        continue
                    left = [x for x in here if x != mine] + [yours]
                    right = [x for x in there if x != yours] + [mine]
                    cost = _pair_cost(universe, left, right, first, second, pace, memo, by_name)
                    if cost is not None and cost < best[0] - 1e-6:
                        best = (cost, (left, right))

            if best[1] is not None:
                left, right = best[1]
                _rebuild(first, _frame_for(universe, left, by_name), pace)
                _rebuild(second, _frame_for(universe, right, by_name), pace)
                improved = True
        if not improved:
            break


def _protected(packed: PackedDay, name: str, universe: pd.DataFrame) -> bool:
    """A must-visit on the arrival day stays there.

    Day 1 carries the city's headline sights on purpose (the agent itineraries put 85% of
    their day-1 stops in that bracket). Shaving a kilometre off the driving is not worth
    trading the first day's reason for being there.
    """
    if not packed.slot.is_arrival:
        return False
    rows = universe[universe["sight"] == name]
    return not rows.empty and bool(rows.iloc[0].get("must_visit_flag", False))


def _pair_cost(universe: pd.DataFrame, left: list[str], right: list[str],
               first: PackedDay, second: PackedDay, pace: str, memo: dict,
               by_name: dict[str, int] | None = None) -> float | None:
    """Combined driving for two days after a proposed change, or None if either breaks."""
    here = _day_cost(_frame_for(universe, left, by_name), first.slot, pace, memo)
    if here is None:
        return None
    there = _day_cost(_frame_for(universe, right, by_name), second.slot, pace, memo)
    return None if there is None else here + there


def _rebuild(packed: PackedDay, frame: pd.DataFrame, pace: str) -> None:
    """Put a new set of sights on a day and recompute everything it reports."""
    if frame.empty:
        packed.sights = []
        packed.sightseeing_hours = packed.buffer_hours = packed.local_travel_hours = 0.0
        packed.route_km = packed.max_leg_km = 0.0
        return
    ordered = _ordered(frame, packed.slot)
    travel, route_km, max_leg = _day_metrics(ordered, packed.slot)
    packed.sights = [_sight_payload(row) for _, row in ordered.iterrows()]
    packed.sightseeing_hours = round(float(ordered["duration_hours"].sum()), 2)
    packed.buffer_hours = round(SIGHT_BUFFER_HOURS * len(ordered), 2)
    packed.local_travel_hours = round(travel, 2)
    packed.route_km, packed.max_leg_km = round(route_km, 1), round(max_leg, 1)
