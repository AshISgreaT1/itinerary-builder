"""The hours a day genuinely has left, and what could fill them.

"Free time" is only meaningful once everything else has been paid for. The clock already
accounts for sightseeing, the driving between stops, the intercity leg, lunch, the
afternoon break, waiting for an evening slot and hotel check-in, so a free window here is
what is actually left over -- not the day's budget minus its sightseeing.

Two kinds of window come out of a day:

* **gaps inside it** -- a wait before an evening sight, or time the schedule set aside
* **the evening after it** -- from the last block until the day's realistic end

Against those windows the engine offers *optional* activities. They are kept out of the
itinerary in every sense that matters: they never enter ``day["sights"]``, never count
towards the day's hours, never reach the validator's budgets, and never appear in the
printed plan unless the customer takes one. Each is checked for travel there and back
inside its own window, so an offer is never one the day cannot keep.
"""

from __future__ import annotations

from .config import (
    EXPLORE_MAX_OPTIONS,
    EXPLORE_MAX_TRAVEL_SHARE,
    EXPLORE_MIN_WINDOW_HOURS,
    LATEST_END_HOUR,
    SIGHT_BUFFER_HOURS,
)
from .data_loader import Dataset
from .geo import local_travel_hours, road_km
from .normalization import normalize_key
from .personalise import score_row, why_it_suits
from .schedule import clock

#: Block kinds that are genuinely free rather than committed to something.
FREE_KINDS = {"free", "wait"}


def _hours(text: str | None) -> float:
    if not text:
        return 0.0
    cleaned = str(text).split(" ")[0]
    hours, _, minutes = cleaned.partition(":")
    try:
        return int(hours) + int(minutes or 0) / 60.0
    except ValueError:
        return 0.0


def _where_at(day: dict, hour: float) -> tuple[float, float] | None:
    """Roughly where the traveller is when a window starts: the stop they last visited."""
    last = None
    for block in (day.get("schedule") or {}).get("blocks", []):
        if block["kind"] == "sight" and _hours(block["start"]) <= hour + 1e-9:
            last = block.get("sight")
    for sight in day.get("sights", []):
        if sight["sight"] == last and sight.get("lat") is not None:
            return float(sight["lat"]), float(sight["lon"])
    return None


def free_windows(day: dict) -> list[dict]:
    """Every genuinely free stretch of a day, in order.

    The day's own blocks are contiguous, so the only gaps are the ones the scheduler put
    there on purpose plus whatever is left after the last block.
    """
    schedule = day.get("schedule") or {}
    blocks = schedule.get("blocks", [])
    if not blocks:
        return []

    windows: list[dict] = []
    for block in blocks:
        if block["kind"] in FREE_KINDS:
            start, end = _hours(block["start"]), _hours(block["end"])
            if end - start >= EXPLORE_MIN_WINDOW_HOURS:
                windows.append({"starts": block["start"], "ends": block["end"],
                                "hours": round(end - start, 2), "when": "during the day"})

    # Whatever is left of the evening, once the day's plan has finished.
    ends = _hours(blocks[-1]["end"])
    if not day.get("is_departure_day") and LATEST_END_HOUR - ends >= EXPLORE_MIN_WINDOW_HOURS:
        windows.append({"starts": clock(ends), "ends": clock(LATEST_END_HOUR),
                        "hours": round(LATEST_END_HOUR - ends, 2), "when": "after the day's plan"})
    return windows


def _reachable(dataset: Dataset, city: str, origin: tuple[float, float] | None,
               row, window_hours: float) -> tuple[bool, float, float]:
    """Does this sight fit the window once the travel there and back is paid for?"""
    city_row = dataset.city_row(city)
    if city_row is None:
        return False, 0.0, 0.0
    start = origin or (float(city_row["lat"]), float(city_row["lon"]))
    out_km = road_km(start[0], start[1], float(row["lat"]), float(row["lon"]))
    back_km = road_km(float(row["lat"]), float(row["lon"]), float(city_row["lat"]),
                      float(city_row["lon"]))
    travel = local_travel_hours(out_km + back_km, city_row.get("altitude_m"))
    visit = float(row.get("duration_hours") or 0.0)
    needed = travel + visit + SIGHT_BUFFER_HOURS
    # Spending most of a free window in the car is not free time well spent.
    if travel > window_hours * EXPLORE_MAX_TRAVEL_SHARE:
        return False, travel, needed
    return needed <= window_hours + 1e-9, travel, needed


def explore_options(day: dict, dataset: Dataset, already_planned: set[str],
                    interests: list[str], group_type: str,
                    limit: int = EXPLORE_MAX_OPTIONS,
                    already_offered: set[str] | None = None) -> list[dict]:
    """Optional nearby things to do in this day's free time.

    Ranked by how well they suit the customer, then by how little of the window the
    travelling eats. Anything already in the trip is skipped, and so is anything offered
    on an earlier day -- the same museum suggested three evenings running reads as a
    shortage of ideas rather than a choice.
    """
    windows = free_windows(day)
    if not windows or not day.get("city"):
        return []

    city_key = normalize_key(day["city"])
    pool = dataset.city_sights(city_key)
    if pool.empty:
        return []

    offers: list[dict] = []
    taken: set[str] = set() if already_offered is None else already_offered
    for window in windows:
        origin = _where_at(day, _hours(window["starts"]))
        candidates = []
        for _, row in pool.iterrows():
            key = str(row["sight_key"])
            if key in already_planned or key in taken:
                continue
            if row.get("lat") is None or row.get("lon") is None:
                continue
            fits, travel, needed = _reachable(dataset, day["city"], origin, row, window["hours"])
            if not fits:
                continue
            candidates.append((score_row(row, interests, group_type), -travel, row, travel, needed))

        candidates.sort(key=lambda item: (-item[0], item[1], str(item[2]["sight"])))
        for score, _, row, travel, needed in candidates:
            if len(offers) >= limit:
                break
            taken.add(str(row["sight_key"]))
            offers.append({
                "sight": str(row["sight"]),
                "city": str(row["city"]),
                "category": str(row.get("category") or ""),
                "visit_type": str(row.get("visit_type") or ""),
                "duration_hours": round(float(row.get("duration_hours") or 0.0), 2),
                "travel_hours": round(travel, 2),
                "total_hours": round(needed, 2),
                "window": f"{window['starts']}-{window['ends']}",
                "window_hours": window["hours"],
                "best_time": str(row.get("best_time") or ""),
                "why": why_it_suits(str(row.get("category") or ""), interests, group_type),
                "score": score,
                # The dataset carries no prices: entry_fee_inr is 0.00 for all 256 rows.
                # Saying "free" would be a claim the data cannot support.
                "price": None,
                "price_note": "not priced in this dataset",
                "in_itinerary": False,
            })
        if len(offers) >= limit:
            break
    return offers


def attach(payload: dict, dataset: Dataset, interests: list[str], group_type: str) -> None:
    """Put free windows and optional activities on every day of a plan.

    Called after the itinerary is final, and deliberately additive: nothing here changes a
    day's stops, hours or validation. Remove these two keys and the plan is what it was.
    """
    planned = {normalize_key(sight["sight"])
               for day in payload.get("days", []) for sight in day.get("sights", [])}
    offered: set[str] = set()
    for day in payload.get("days", []):
        day["free_windows"] = free_windows(day)
        day["explore_on_your_own"] = explore_options(day, dataset, planned, interests,
                                                     group_type, already_offered=offered)
    payload.setdefault("summary", {})["personalised_for"] = {
        "group_type": group_type or None,
        "interests": list(interests),
        "optional_activities_offered": sum(len(day.get("explore_on_your_own", []))
                                           for day in payload.get("days", [])),
        "note": "Optional activities sit outside the itinerary: they are not scheduled, not "
                "counted in any day's hours, and not priced.",
    }
