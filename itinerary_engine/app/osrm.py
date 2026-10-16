"""Real road distances from OSRM, fetched once and cached.

The engine estimates road distance as a straight line inflated by
:data:`~app.config.ROAD_DISTANCE_FACTOR` (1.3). That is the largest single source of error
in a plan: it understates switchbacks badly in Ladakh and overstates them on a straight
highway. OSRM replaces the estimate with a measured one.

**Why a cache and not live calls.** Plans are deterministic and the validator re-derives
every distance from the coordinates; calling a network service inside that loop would make
two runs of the same request disagree and put an outage in the middle of validation. So the
matrix is fetched once into ``data/road_distances.json`` and read from there afterwards.
A pair that is not in the cache keeps the old estimate and is *reported* as estimated --
it is never silently mixed in with measured values.

Building the cache::

    # one-off, with a local OSRM (see the README for the Docker commands)
    python -m app.osrm --build --url http://localhost:5000

OSRM's ``table`` service returns metres and seconds, and its distances are directional:
Udaipur's City Palace to Fatehsagar Lake is 3.25 km out and 3.72 km back, because of the
one-way streets. Both directions are kept.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .config import OSRM_MAX_TABLE_LOCATIONS, OSRM_SNAP_WARN_KM, ROAD_CACHE_PATH
from .data_loader import Dataset, get_dataset

logger = logging.getLogger(__name__)

#: OSRM's own demo server. Rate-limited and explicitly not for production use; fine for a
#: one-off build of this dataset (33 requests), but a local instance is the intended path.
DEMO_URL = "https://router.project-osrm.org"


class OsrmError(RuntimeError):
    """OSRM could not be reached, or answered with something unusable."""


def coordinate_key(lat: float, lon: float) -> str:
    """A stable cache key for one point, to about a metre."""
    return f"{float(lat):.5f},{float(lon):.5f}"


def pair_key(lat1: float, lon1: float, lat2: float, lon2: float) -> str:
    return f"{coordinate_key(lat1, lon1)}>{coordinate_key(lat2, lon2)}"


@dataclass
class TableResult:
    """One OSRM ``table`` answer, in the units the engine uses."""

    #: distances[i][j] in km, or None where OSRM could not route the pair
    distances_km: list[list[float | None]]
    #: durations[i][j] in hours, or None
    durations_hours: list[list[float | None]]
    #: how far each coordinate had to be moved to reach a road, in km
    snap_km: list[float] = field(default_factory=list)


class OsrmClient:
    """Minimal OSRM HTTP client -- the ``table`` service is all the engine needs."""

    def __init__(self, base_url: str = DEMO_URL, profile: str = "driving", timeout: float = 60.0,
                 delay: float = 0.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.profile = profile
        self.timeout = timeout
        #: Pause between requests. Zero against your own server; a second against the shared
        #: demo one, which is someone else's machine.
        self.delay = delay
        self._last_call = 0.0

    def table(self, points: list[tuple[float, float]]) -> TableResult:
        """Full distance and duration matrix between ``points`` (lat, lon)."""
        if len(points) > OSRM_MAX_TABLE_LOCATIONS:
            raise OsrmError(
                f"{len(points)} locations in one table request; OSRM's default limit is "
                f"{OSRM_MAX_TABLE_LOCATIONS}. Split the request or raise --max-table-size on the server."
            )
        coordinates = ";".join(f"{float(lon):.6f},{float(lat):.6f}" for lat, lon in points)
        url = (f"{self.base_url}/table/v1/{self.profile}/{coordinates}"
               f"?annotations=distance,duration")
        payload = self._get(url)
        if payload.get("code") != "Ok":
            raise OsrmError(f"OSRM answered {payload.get('code')}: {payload.get('message', 'no message')}")

        def convert(matrix: object, factor: float, places: int) -> list[list[float | None]]:
            rows = matrix if isinstance(matrix, list) else []
            return [[None if cell is None else round(float(cell) * factor, places) for cell in row]
                    for row in rows]

        snap = [round(float(source.get("distance", 0.0)) / 1000, 4)
                for source in payload.get("sources", []) if isinstance(source, dict)]
        return TableResult(
            distances_km=convert(payload.get("distances"), 1 / 1000, 4),      # to the metre
            durations_hours=convert(payload.get("durations"), 1 / 3600, 6),  # to a hundredth of a second
            snap_km=snap,
        )

    def _get(self, url: str) -> dict:
        if self.delay:
            waited = time.monotonic() - self._last_call
            if waited < self.delay:
                time.sleep(self.delay - waited)
            self._last_call = time.monotonic()
        request = urllib.request.Request(url, headers={"User-Agent": "itinerary-engine/1.0"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:  # pragma: no cover - network failure path
            raise OsrmError(f"OSRM returned HTTP {error.code} for {url}") from error
        except (urllib.error.URLError, TimeoutError) as error:  # pragma: no cover
            raise OsrmError(
                f"Could not reach OSRM at {self.base_url}: {error}. Start a local instance, or pass "
                f"--url {DEMO_URL} to use the public demo server."
            ) from error
        except json.JSONDecodeError as error:  # pragma: no cover
            raise OsrmError(f"OSRM sent something that is not JSON: {error}") from error


def _points_to_measure(dataset: Dataset) -> list[tuple[str, list[tuple[float, float]]]]:
    """The groups worth a matrix: each city's sights with its centre, then the centres.

    Only pairs the engine actually asks about are fetched. A day's route never leaves its
    city, so a sight in Leh is never measured against one in Goa.
    """
    groups: list[tuple[str, list[tuple[float, float]]]] = []
    centres = dataset.cities.dropna(subset=["lat", "lon"])
    for _, city in centres.iterrows():
        sights = dataset.city_sights(str(city["city_key"]))
        sights = sights.dropna(subset=["lat", "lon"]) if not sights.empty else sights
        if sights.empty:
            continue
        points = [(float(city["lat"]), float(city["lon"]))]
        points += [(float(row["lat"]), float(row["lon"])) for _, row in sights.iterrows()]
        groups.append((f"{city['city']} ({len(sights)} sights)", points))

    city_points = [(float(row["lat"]), float(row["lon"])) for _, row in centres.iterrows()]
    if len(city_points) > OSRM_MAX_TABLE_LOCATIONS:
        # Chunking would measure within each chunk but not across them, and a half-measured
        # city graph is worse than an honest estimate, so say so rather than hide it.
        logger.warning(
            "%d cities is more than one table request holds (%d); city-to-city distances will not be "
            "measured. Start OSRM with --max-table-size %d to measure them.",
            len(city_points), OSRM_MAX_TABLE_LOCATIONS, len(city_points),
        )
    elif len(city_points) > 1:
        groups.append((f"{len(city_points)} city centres", city_points))
    return groups


def build_cache(dataset: Dataset | None = None, client: OsrmClient | None = None) -> dict:
    """Measure every pair the engine can ask for, and return the cache document."""
    dataset = dataset or get_dataset()
    client = client or OsrmClient()
    distances: dict[str, float] = {}
    durations: dict[str, float] = {}
    far_snaps: list[str] = []
    unroutable = 0

    for label, points in _points_to_measure(dataset):
        logger.info("OSRM table for %s (%d points)", label, len(points))
        result = client.table(points)
        for index, snap in enumerate(result.snap_km):
            if snap > OSRM_SNAP_WARN_KM:
                lat, lon = points[index]
                far_snaps.append(f"{label}: a point at {lat:.4f},{lon:.4f} is {snap:.1f} km from the nearest road")
        for i, (lat1, lon1) in enumerate(points):
            for j, (lat2, lon2) in enumerate(points):
                if i == j:
                    continue
                km = result.distances_km[i][j] if i < len(result.distances_km) else None
                if km is None:
                    unroutable += 1
                    continue  # left out on purpose: the estimate takes over and says so
                key = pair_key(lat1, lon1, lat2, lon2)
                distances[key] = km
                hours = result.durations_hours[i][j] if i < len(result.durations_hours) else None
                if hours is not None:
                    durations[key] = hours

    return {
        "meta": {
            "source": "OSRM",
            "profile": client.profile,
            "server": client.base_url,
            "dataset": Path(str(dataset.source)).name,
            "pairs": len(distances),
            "unroutable_pairs": unroutable,
            "snap_warnings": far_snaps,
        },
        "distances_km": distances,
        "durations_hours": durations,
    }


def write_cache(document: dict, path: Path | str = ROAD_CACHE_PATH) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(document, indent=1, sort_keys=True), encoding="utf-8")
    return destination


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - thin CLI wrapper
    import argparse

    parser = argparse.ArgumentParser(description="Measure road distances with OSRM and cache them.")
    parser.add_argument("--build", action="store_true", help="fetch the matrix and write the cache")
    parser.add_argument("--url", default="http://localhost:5000", help="OSRM base URL")
    parser.add_argument("--profile", default="driving", help="OSRM profile (default: driving)")
    parser.add_argument("--out", default=str(ROAD_CACHE_PATH), help="where to write the cache")
    parser.add_argument("--delay", type=float, default=None,
                        help="seconds between requests (default: 1.0 against the public demo server, 0 locally)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not args.build:
        parser.print_help()
        return 0

    dataset = get_dataset()
    delay = args.delay if args.delay is not None else (1.0 if "project-osrm.org" in args.url else 0.0)
    document = build_cache(dataset, OsrmClient(args.url, args.profile, delay=delay))
    path = write_cache(document, args.out)
    meta = document["meta"]
    print(f"\nWrote {meta['pairs']} measured pairs to {path}")
    if meta["unroutable_pairs"]:
        print(f"{meta['unroutable_pairs']} pair(s) could not be routed; they keep the straight-line estimate.")
    for warning in meta["snap_warnings"]:
        print(f"  warning: {warning}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
