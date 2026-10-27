"""Remembering plans, so the same question is not answered twice.

Planning is deterministic: the same request against the same dataset always produces the
same plan. That makes it cacheable, and caching matters more than it looks, because the
editing API replays work by design.

Every customisation is applied to the *request* and the plan rebuilt, which is what keeps
the route, the clock and the validator in agreement. But the UI sends the whole accumulated
list of operations each time, so the sixth edit rebuilds the plan seven times -- once for
the base and once per operation -- and every one of those builds except the last is
identical to a build made during the previous edit. Measured on a two-city trip, edits cost
4.1s, 3.4s, 5.5s, then 7.3s as the list grew.

With this cache those repeated prefixes are free, and the cost of an edit becomes the cost
of the one build that is genuinely new.

The cache is keyed on everything that can change a plan -- every field of the request, the
dataset file and its modification time -- and hands out deep copies, because callers do
mutate what they are given (the builder itself attaches notes to days). A copy is about a
millisecond against a rebuild of one to four seconds.
"""

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from copy import deepcopy
from dataclasses import fields, is_dataclass
from typing import Any

logger = logging.getLogger(__name__)

#: How many plans to keep. Each is roughly 30 KB, so the default costs a few megabytes.
#: Lower it with ITINERARY_PLAN_CACHE=0 to switch caching off entirely.
CACHE_SIZE = int(os.environ.get("ITINERARY_PLAN_CACHE", "256"))

_cache: "OrderedDict[tuple, dict]" = OrderedDict()
_stats = {"hits": 0, "misses": 0}


def _canonical(value: Any) -> Any:
    """A hashable, order-stable rendering of anything a request can hold."""
    if is_dataclass(value) and not isinstance(value, type):
        return (type(value).__name__,
                tuple((field.name, _canonical(getattr(value, field.name))) for field in fields(value)))
    if isinstance(value, dict):
        return tuple(sorted((str(key), _canonical(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple, set)):
        items = sorted(value, key=str) if isinstance(value, set) else value
        return tuple(_canonical(item) for item in items)
    return value


def key_for(request: Any, dataset: Any, repair: bool) -> tuple:
    """Everything that can change the answer, and nothing that cannot."""
    source = getattr(dataset, "source", None)
    try:
        stamp = os.path.getmtime(source) if source else 0.0
    except OSError:  # pragma: no cover - the file vanished under us
        stamp = 0.0
    return (_canonical(request), str(source), stamp, bool(repair))


def get(key: tuple) -> dict | None:
    """A copy of the remembered plan, or None."""
    if CACHE_SIZE <= 0:
        return None
    found = _cache.get(key)
    if found is None:
        _stats["misses"] += 1
        return None
    _cache.move_to_end(key)
    _stats["hits"] += 1
    return deepcopy(found)


def put(key: tuple, payload: dict) -> None:
    if CACHE_SIZE <= 0:
        return
    _cache[key] = deepcopy(payload)
    _cache.move_to_end(key)
    while len(_cache) > CACHE_SIZE:
        _cache.popitem(last=False)


def clear() -> None:
    _cache.clear()
    _stats.update(hits=0, misses=0)


def info() -> dict:
    total = _stats["hits"] + _stats["misses"]
    return {
        "size": len(_cache),
        "capacity": CACHE_SIZE,
        "hits": _stats["hits"],
        "misses": _stats["misses"],
        "hit_rate": round(_stats["hits"] / total, 3) if total else 0.0,
    }
