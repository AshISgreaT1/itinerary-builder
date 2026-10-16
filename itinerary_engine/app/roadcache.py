"""The measured-distance cache that :func:`app.geo.road_km` reads.

Kept apart from :mod:`app.osrm` so that nothing in the planning path imports a network
client. This module only ever reads a JSON file, and works perfectly well when that file
does not exist -- then every distance is an estimate, exactly as before.
"""

from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from pathlib import Path

from .config import ROAD_CACHE_PATH

logger = logging.getLogger(__name__)


def _key(lat: float, lon: float) -> str:
    return f"{float(lat):.5f},{float(lon):.5f}"


@lru_cache(maxsize=4)
def _load(path: str) -> tuple[dict, dict, dict]:
    """``(distances, durations, meta)`` from the cache file, or empty when there is none."""
    file = Path(path)
    if not file.exists():
        return {}, {}, {}
    try:
        document = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        logger.warning("Road distance cache at %s could not be read (%s); using estimates.", path, error)
        return {}, {}, {}
    distances = document.get("distances_km") or {}
    meta = document.get("meta") or {}
    logger.info("Road distances: %d measured pairs from %s.", len(distances), meta.get("source", path))
    return distances, document.get("durations_hours") or {}, meta


def reload(path: str | Path = ROAD_CACHE_PATH) -> None:
    """Forget the loaded cache -- used by the tests and after rebuilding it."""
    _load.cache_clear()
    _load(str(path))


def measured_km(lat1: float, lon1: float, lat2: float, lon2: float,
                path: str | Path = ROAD_CACHE_PATH) -> float | None:
    """The measured road distance for this pair, or None when it was never measured.

    OSRM distances are directional (one-way streets), so the exact direction is preferred
    and the reverse is only used as a fallback -- better than an estimate, and flagged as
    the same measured source.
    """
    distances, _, _ = _load(str(path))
    if not distances:
        return None
    first, second = _key(lat1, lon1), _key(lat2, lon2)
    value = distances.get(f"{first}>{second}")
    if value is None:
        value = distances.get(f"{second}>{first}")
    return None if value is None else float(value)


def measured_hours(lat1: float, lon1: float, lat2: float, lon2: float,
                   path: str | Path = ROAD_CACHE_PATH) -> float | None:
    """The measured driving time for this pair, or None.

    Not used by the planner yet: the engine derives hours from distance and its own speed
    model, which is altitude-aware. Cached here so the two can be compared.
    """
    _, durations, _ = _load(str(path))
    if not durations:
        return None
    first, second = _key(lat1, lon1), _key(lat2, lon2)
    value = durations.get(f"{first}>{second}")
    if value is None:
        value = durations.get(f"{second}>{first}")
    return None if value is None else float(value)


def info(path: str | Path = ROAD_CACHE_PATH) -> dict:
    """What the plan should say about where its distances came from."""
    distances, _, meta = _load(str(path))
    if not distances:
        return {
            "source": "estimated",
            "measured_pairs": 0,
            "detail": "No road distance cache found; distances are straight lines inflated by a detour "
                      "factor. Build one with `python -m app.osrm --build` for measured roads.",
        }
    return {
        "source": meta.get("source", "cache"),
        "profile": meta.get("profile"),
        "measured_pairs": len(distances),
        "unroutable_pairs": meta.get("unroutable_pairs", 0),
        "detail": f"{len(distances)} point-to-point distances measured with "
                  f"{meta.get('source', 'a routing engine')}; anything outside the cache falls back to the "
                  f"straight-line estimate and is counted as estimated_legs.",
    }


def snap_warnings(path: str | Path = ROAD_CACHE_PATH) -> list[dict]:
    """Points the routing engine had to move a long way to put on a road.

    Parsed back out of the cache's own metadata, so the data-quality report does not need
    a routing engine to tell you a coordinate is wrong.
    """
    _, _, meta = _load(str(path))
    parsed = []
    for line in meta.get("snap_warnings", []):
        # "City (n sights): a point at 33.2000,78.5000 is 6.1 km from the nearest road"
        match = re.match(r"^(?P<key>.+?): a point at (?P<point>[-\d.,]+) is (?P<km>[\d.]+) km", str(line))
        if match:
            parsed.append({"key": match.group("key"), "point": match.group("point"),
                           "km": float(match.group("km"))})
    # One entry per point: the same coordinate appears in its city matrix and the centres one.
    seen, unique = set(), []
    for item in parsed:
        if item["point"] in seen:
            continue
        seen.add(item["point"])
        unique.append(item)
    return unique
