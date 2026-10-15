"""Daily time-packing engine.

A day's budget is the pace allowance minus whatever the day loses to arrival, departure or
an intercity transfer. Each sight costs its ``duration_hours`` plus a fixed buffer, and the
local driving between consecutive stops is charged from real coordinates, so a scattered
day runs out of hours before a compact one does.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .config import (
    ARRIVAL_OVERHEAD_HOURS,
    TRAVEL_DAY_MAX_DRIVE_HOURS,
    BEST_TIME_ORDER,
    DEPARTURE_OVERHEAD_HOURS,
    MIN_DEPARTURE_BUFFER_HOURS,
    MAX_DAY_ROUTE_KM,
    MAX_LEG_KM,
    MAX_SIGHTS_PER_DAY,
    MAX_STOPS_WITH_EXCURSION,
    PACE_ACTIVE_HOURS,
    SIGHT_BUFFER_HOURS,
    TRANSFER_OVERHEAD_HOURS,
)
from .geo import RouteLeg, local_travel_hours, order_by_route, road_km


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
    is_arrival: bool = False
    is_departure: bool = False
    #: Checkout + transfer to the airport/station + reporting buffer, set by the builder.
    departure_cost_hours: float = DEPARTURE_OVERHEAD_HOURS
    #: True when travel alone fills the day (unknown journey, or longer than a travel day).
    travel_only: bool = False
    #: Budget taken straight from an existing plan, so edits are judged against the same
    #: hours the day actually has.
    available_override: float | None = None

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
            return max(0.0, round(self.available_override, 2))
        if self.is_travel_day:
            return 0.0
        costs = self.cost_breakdown(pace)
        hours = costs["pace_allowance"] - costs["intercity_travel"] - costs["day_overheads"]
        if self.is_departure:
            hours -= MIN_DEPARTURE_BUFFER_HOURS
        return max(0.0, round(hours, 2))

    def max_sights(self, pace: str) -> int:
        """Stop count cap; shortened days also get fewer stops."""
        cap = MAX_SIGHTS_PER_DAY[pace.strip().lower()]
        if self.transfer is not None or self.is_arrival or self.is_departure:
            cap = max(1, cap - 2)
        return cap


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
        road_km(slot.centre_lat, slot.centre_lon, lats[0], lons[0]),
        road_km(slot.centre_lat, slot.centre_lon, lats[-1], lons[-1]),
    ]
    total_km = sum(legs) + sum(hotel_legs)
    hours = local_travel_hours(total_km, slot.altitude_m)
    return hours, sum(legs), max(legs) if legs else 0.0


def _ordered(candidate: pd.DataFrame, slot: DaySlot) -> pd.DataFrame:
    """Order a day's sights by time of day, and geographically inside each time group.

    Sorting geographically and then by time undoes the routing (a sunset point can be
    dragged back across town), so the time groups come first and each group is routed from
    where the previous one ended.
    """
    frame = candidate.reset_index(drop=True).copy()
    frame["_time_rank"] = frame["best_time"].map(lambda t: BEST_TIME_ORDER.get(str(t).lower(), 1))
    ordered_parts: list[pd.DataFrame] = []
    cursor_lat, cursor_lon = slot.centre_lat, slot.centre_lon
    for rank in sorted(frame["_time_rank"].unique()):
        group = frame[frame["_time_rank"] == rank].reset_index(drop=True)
        order = order_by_route(group, cursor_lat, cursor_lon)
        routed = group.iloc[order].reset_index(drop=True)
        ordered_parts.append(routed)
        cursor_lat, cursor_lon = float(routed.iloc[-1]["lat"]), float(routed.iloc[-1]["lon"])
    return pd.concat(ordered_parts, ignore_index=True).drop(columns="_time_rank")


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

    selected = candidates.iloc[0:0]
    for _, row in sort_by_priority(candidates).iterrows():
        if len(selected) >= stop_cap:
            break
        trial = _ordered(pd.concat([selected, row.to_frame().T], ignore_index=True), slot)
        travel_hours, route_km_total, max_leg = _day_metrics(trial, slot)
        visit_hours = float(trial["duration_hours"].sum())
        buffer_hours = SIGHT_BUFFER_HOURS * len(trial)
        excursions = int(trial["is_excursion"].sum()) if "is_excursion" in trial else 0

        if visit_hours + buffer_hours + travel_hours > budget:
            continue
        if max_leg > MAX_LEG_KM or route_km_total > MAX_DAY_ROUTE_KM:
            continue  # would mean criss-crossing the region in one day
        if excursions > 1 or (excursions == 1 and len(trial) - 1 > MAX_STOPS_WITH_EXCURSION):
            continue  # a day trip cannot share the day with a full city programme

        selected = trial
        packed.sightseeing_hours = round(visit_hours, 2)
        packed.buffer_hours = round(buffer_hours, 2)
        packed.local_travel_hours = round(travel_hours, 2)
        packed.route_km = round(route_km_total, 1)
        packed.max_leg_km = round(max_leg, 1)

    packed.sights = [_sight_payload(row) for _, row in selected.iterrows()]
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
    return packed_days
