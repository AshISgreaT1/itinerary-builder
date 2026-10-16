"""What the agent-written itineraries can teach the planner.

The ``itineraries`` sheet holds day-by-day sightseeing written by people, so it is the only
record of how these trips are really run. Two caveats decide how far it can be trusted:

* **It carries no clock times at all.** Nothing in the sheet says when anything happens, so
  times are never taken from here -- they come from ``best_time`` and measured durations.
* **Its order does not follow ``best_time``.** Averaged over the sheet, "morning" sights sit
  at position 0.52 of a day and "evening" ones at 0.54: no signal. So ``best_time`` decides
  the time of day, and these patterns only break ties inside the same slot.

What the sheet *does* give is strong pairwise ordering (Shanti Stupa before Leh Palace 27
times against 6) and a sense of how many stops a day usually holds.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import pandas as pd

from .data_loader import Dataset
from .normalization import normalize_key

#: A pair is only treated as a pattern with at least this much evidence...
MIN_PAIR_COUNT = 4
#: ...and this much agreement (times seen in this order vs the other).
MIN_PAIR_RATIO = 2.0


@dataclass
class ItineraryPatterns:
    """Ordering habits mined from the agent itineraries."""

    #: (first_key, second_key) -> times the first was visited before the second
    pair_counts: Counter = field(default_factory=Counter)
    #: how many sights a day usually holds, ignoring empty days
    sights_per_day: Counter = field(default_factory=Counter)
    days_seen: int = 0
    sights_seen: int = 0
    #: True when the source itineraries contain no times of day (they do not).
    has_clock_times: bool = False

    def precedence(self, first: str, second: str) -> tuple[int, int]:
        """How often each order was used (forward, reverse)."""
        return self.pair_counts.get((first, second), 0), self.pair_counts.get((second, first), 0)

    def prefers(self, first: str, second: str) -> bool:
        """True when the agents clearly put ``first`` before ``second``."""
        forward, reverse = self.precedence(first, second)
        return forward >= MIN_PAIR_COUNT and forward >= MIN_PAIR_RATIO * max(reverse, 1)

    def conflicts(self, sequence: list[str]) -> list[tuple[str, str, int, int]]:
        """Pairs in this order that the agents consistently do the other way round."""
        found = []
        for index, first in enumerate(sequence):
            for second in sequence[index + 1:]:
                if self.prefers(second, first):
                    forward, reverse = self.precedence(second, first)
                    found.append((second, first, forward, reverse))
        return found

    @property
    def typical_sights_per_day(self) -> float:
        total = sum(count * size for size, count in self.sights_per_day.items())
        days = sum(self.sights_per_day.values())
        return round(total / days, 1) if days else 0.0


def _parse_day_sequences(value: object) -> list[list[str]]:
    """``"Day 1: A (1h); B | Day 2: -"`` -> ``[["a"], []]`` (keys, in written order)."""
    days: list[list[str]] = []
    for chunk in str(value or "").split("|"):
        match = re.match(r"\s*Day\s*(\d+)\s*:\s*(.*)$", chunk.strip(), flags=re.IGNORECASE)
        if not match:
            continue
        body = match.group(2).strip()
        if body in ("", "-"):
            days.append([])
            continue
        names = [normalize_key(re.sub(r"\(.*?\)", "", part)) for part in body.split(";")]
        days.append([name for name in names if name])
    return days


def learn_patterns(source: str | Path) -> ItineraryPatterns:
    """Mine ordering habits from a workbook's ``itineraries`` sheet.

    A workbook without that sheet simply yields empty patterns, and every caller treats
    that as "no opinion" rather than an error.
    """
    patterns = ItineraryPatterns()
    try:
        trips = pd.read_excel(source, sheet_name="itineraries")
    except Exception:
        return patterns
    if "day_wise_sightseeing" not in trips.columns:
        return patterns

    clock = re.compile(r"\b\d{1,2}[:.]\d{2}\s*(?:am|pm)?\b", re.IGNORECASE)
    for value in trips["day_wise_sightseeing"].dropna():
        if clock.search(str(value)):
            patterns.has_clock_times = True
        for names in _parse_day_sequences(value):
            patterns.days_seen += 1
            if not names:
                continue
            patterns.sights_per_day[len(names)] += 1
            patterns.sights_seen += len(names)
            for index, first in enumerate(names):
                for second in names[index + 1:]:
                    if first != second:
                        patterns.pair_counts[(first, second)] += 1
    return patterns


@lru_cache(maxsize=4)
def _cached_patterns(source: str) -> ItineraryPatterns:
    return learn_patterns(source)


def patterns_for(dataset: Dataset) -> ItineraryPatterns:
    """Patterns for the workbook this dataset came from (cached per process)."""
    return _cached_patterns(str(dataset.source))


def apply_pattern_order(sights: list[dict], patterns: ItineraryPatterns, swap_cost) -> tuple[list[dict], list[str]]:
    """Reorder adjacent stops to match the agents, but only when travel does not suffer.

    The geographic order and the preferred time of day come first; this only settles ties
    between neighbours, and a swap is kept only if it does not lengthen the day's driving.
    """
    ordered = list(sights)
    notes: list[str] = []
    for index in range(len(ordered) - 1):
        first, second = ordered[index], ordered[index + 1]
        if first.get("best_time") != second.get("best_time"):
            continue  # time of day decides, not habit
        if not patterns.prefers(normalize_key(second["sight"]), normalize_key(first["sight"])):
            continue
        candidate = ordered[:index] + [second, first] + ordered[index + 2:]
        if swap_cost(candidate) <= swap_cost(ordered) + 1e-9:
            forward, reverse = patterns.precedence(normalize_key(second["sight"]), normalize_key(first["sight"]))
            notes.append(
                f"Visited '{second['sight']}' before '{first['sight']}' as in {forward} past itineraries "
                f"(the other order appears {reverse} times)."
            )
            ordered = candidate
    return ordered, notes
