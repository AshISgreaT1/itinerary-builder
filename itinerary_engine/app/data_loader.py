"""Loading and cleaning of the Excel dataset.

Only the sheets the planner needs are read: cities, city_distances, sightseeing and
sight_descriptions. Everything is normalised once at load time (keys, numbers, month
windows, aliases) so the rest of the engine never parses raw strings.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

from .geo import CityGraph
from .normalization import DeduplicationResult, deduplicate_sights, normalize_key

logger = logging.getLogger(__name__)

DEFAULT_DATASET = Path(os.environ.get("ITINERARY_DATASET", "data/sample_dataset_v4_clean.xlsx"))

REQUIRED_COLUMNS: dict[str, set[str]] = {
    "cities": {"city", "lat", "lon"},
    "city_distances": {"from_city", "to_city", "road_km", "drive_hours"},
    "sightseeing": {"city", "sight", "duration_hours"},
    "sight_descriptions": {"city", "sight", "description"},
}

MONTH_NAMES = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
ALL_MONTHS = frozenset(range(1, 13))


def parse_open_months(value: object) -> frozenset[int]:
    """Parse ``open_months`` / ``season_open`` into a set of month numbers.

    Handles ``all``, ``5,6,7,8``, ``Apr-Oct`` and ``Oct-Mar`` (wrapping over new year).
    Unparseable values fall back to "open all year" so a bad cell never hides a sight.
    """
    text = str(value or "").strip().lower()
    if not text or text in {"all", "all year", "year-round", "nan"}:
        return frozenset(ALL_MONTHS)
    if all(part.strip().isdigit() for part in text.split(",") if part.strip()):
        months = {int(part) for part in text.split(",") if part.strip()}
        return frozenset(m for m in months if 1 <= m <= 12) or frozenset(ALL_MONTHS)
    parts = [p.strip()[:3] for p in text.replace("to", "-").split("-")]
    if len(parts) == 2 and all(p in MONTH_NAMES for p in parts):
        start, end = MONTH_NAMES[parts[0]], MONTH_NAMES[parts[1]]
        if start <= end:
            return frozenset(range(start, end + 1))
        return frozenset(list(range(start, 13)) + list(range(1, end + 1)))  # wraps the new year
    logger.warning("Could not parse month window %r; treating as open all year", value)
    return frozenset(ALL_MONTHS)


def _split_aliases(value: object) -> set[str]:
    """``"Ajmer-e-Sharief Dargah;  Dargah"`` -> {"ajmer e sharief dargah", "dargah"}."""
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return set()
    return {normalize_key(part) for part in text.split(";") if normalize_key(part)}


@dataclass(frozen=True)
class Dataset:
    """Everything the planner needs, cleaned and indexed."""

    cities: pd.DataFrame
    distances: pd.DataFrame
    sights: pd.DataFrame
    descriptions: pd.DataFrame
    graph: CityGraph
    dedup: DeduplicationResult
    source: Path

    def city_row(self, city: str) -> pd.Series | None:
        """Look up a city by any capitalisation/spacing; None when unknown."""
        matches = self.cities[self.cities["city_key"] == normalize_key(city)]
        return None if matches.empty else matches.iloc[0]

    def city_sights(self, city_key: str) -> pd.DataFrame:
        """All canonical sights of a city, in dataset priority order."""
        subset = self.sights[self.sights["city_key"] == city_key]
        return subset.sort_values(
            ["must_visit_flag", "must_see_tier", "priority"], ascending=[False, True, True]
        ).reset_index(drop=True)

    def description(self, city: str, sight: str) -> str | None:
        key = (normalize_key(city), normalize_key(sight))
        return self._description_index.get(key)

    @property
    def _description_index(self) -> dict[tuple[str, str], str]:
        return self.descriptions.attrs["index"]

    @property
    def city_names(self) -> list[str]:
        return sorted(self.cities["city"].tolist())


def _read_sheet(path: Path, sheet: str) -> pd.DataFrame:
    frame = pd.read_excel(path, sheet_name=sheet)
    frame.columns = [str(c).strip() for c in frame.columns]
    missing = REQUIRED_COLUMNS[sheet] - set(frame.columns)
    if missing:
        raise ValueError(f"Sheet '{sheet}' is missing required column(s): {sorted(missing)}")
    for column in frame.columns:
        if frame[column].dtype == object:
            frame[column] = frame[column].map(lambda v: v.strip() if isinstance(v, str) else v)
    return frame


def _column(frame: pd.DataFrame, name: str, default: object) -> pd.Series:
    """Column as a Series, or a Series of ``default`` when the sheet omits it."""
    if name in frame.columns:
        return frame[name]
    return pd.Series([default] * len(frame), index=frame.index)


def _clean_cities(frame: pd.DataFrame) -> pd.DataFrame:
    cities = frame.copy()
    cities["city"] = cities["city"].astype(str)
    cities["city_key"] = cities["city"].map(normalize_key)
    for column in ("lat", "lon", "altitude_m"):
        if column in cities:
            cities[column] = pd.to_numeric(cities[column], errors="coerce")
    cities["altitude_m"] = pd.to_numeric(_column(cities, "altitude_m", float("nan")), errors="coerce")
    cities["gateway_flag"] = _column(cities, "gateway", "no").astype(str).str.lower().eq("yes")
    cities["season_months"] = _column(cities, "season_open", "all").map(parse_open_months)
    # Optional in some dataset versions; the planner falls back to 1 night per city.
    cities["min_days"] = pd.to_numeric(_column(cities, "min_days", 1), errors="coerce").fillna(1).astype(int)
    return cities.dropna(subset=["lat", "lon"]).drop_duplicates("city_key").reset_index(drop=True)


def _clean_distances(frame: pd.DataFrame) -> pd.DataFrame:
    distances = frame.copy()
    distances["from_city_key"] = distances["from_city"].map(normalize_key)
    distances["to_city_key"] = distances["to_city"].map(normalize_key)
    for column in ("road_km", "drive_hours"):
        distances[column] = pd.to_numeric(distances[column], errors="coerce")
    distances["mode"] = _column(distances, "mode", "road").fillna("road").astype(str)
    distances = distances.dropna(subset=["road_km", "drive_hours"])
    return distances[distances["from_city_key"] != distances["to_city_key"]].reset_index(drop=True)


def _clean_sights(frame: pd.DataFrame) -> pd.DataFrame:
    sights = frame.copy()
    sights["city"] = sights["city"].astype(str)
    sights["sight"] = sights["sight"].astype(str)
    sights["city_key"] = sights["city"].map(normalize_key)
    sights["sight_key"] = sights["sight"].map(normalize_key)
    sights["alias_keys"] = _column(sights, "aliases", "").map(_split_aliases)

    sights["duration_hours"] = pd.to_numeric(sights["duration_hours"], errors="coerce").fillna(1.5).clip(lower=0.25)
    sights["must_see_tier"] = pd.to_numeric(_column(sights, "must_see_tier", 3), errors="coerce").fillna(3).astype(int)
    sights["priority"] = pd.to_numeric(_column(sights, "priority", 99), errors="coerce").fillna(99).astype(int)
    sights["mentions"] = pd.to_numeric(_column(sights, "mentions", 0), errors="coerce").fillna(0).astype(int)
    sights["min_days"] = pd.to_numeric(_column(sights, "min_days", 1), errors="coerce").fillna(1).astype(int)
    for column in ("lat", "lon", "altitude_m", "entry_fee_inr"):
        if column in sights:
            sights[column] = pd.to_numeric(sights[column], errors="coerce")

    sights["must_visit_flag"] = _column(sights, "must_visit", "no").astype(str).str.lower().eq("yes")
    sights["best_time"] = _column(sights, "best_time", "any").fillna("any").astype(str).str.lower()
    sights["zone"] = _column(sights, "zone", "").fillna("").astype(str)
    sights["visit_type"] = _column(sights, "visit_type", "").fillna("").astype(str)
    sights["open_months_set"] = _column(sights, "open_months", "all").map(parse_open_months)

    sights = sights.dropna(subset=["lat", "lon"])
    return sights.reset_index(drop=True)


def _clean_descriptions(frame: pd.DataFrame) -> pd.DataFrame:
    descriptions = frame.copy()
    descriptions["city_key"] = descriptions["city"].map(normalize_key)
    descriptions["sight_key"] = descriptions["sight"].map(normalize_key)
    descriptions = descriptions.dropna(subset=["description"])
    descriptions.attrs["index"] = {
        (row["city_key"], row["sight_key"]): str(row["description"]).strip()
        for _, row in descriptions.iterrows()
    }
    return descriptions.reset_index(drop=True)


def load_dataset(path: str | Path | None = None) -> Dataset:
    """Read, clean and index the dataset (not cached; use :func:`get_dataset`)."""
    dataset_path = Path(path or DEFAULT_DATASET)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. Set ITINERARY_DATASET or pass an explicit path."
        )

    cities = _clean_cities(_read_sheet(dataset_path, "cities"))
    distances = _clean_distances(_read_sheet(dataset_path, "city_distances"))
    sights = _clean_sights(_read_sheet(dataset_path, "sightseeing"))
    descriptions = _clean_descriptions(_read_sheet(dataset_path, "sight_descriptions"))

    unknown_cities = sorted(set(sights["city_key"]) - set(cities["city_key"]))
    if unknown_cities:
        logger.warning("Sights reference %d city/cities missing from the cities sheet: %s",
                       len(unknown_cities), unknown_cities[:5])

    dedup = deduplicate_sights(sights)
    logger.info(
        "Loaded %s: %d cities, %d distance rows, %d sights (%d after merging duplicates)",
        dataset_path.name, len(cities), len(distances), len(sights), len(dedup.sights),
    )
    return Dataset(
        cities=cities,
        distances=distances,
        sights=dedup.sights,
        descriptions=descriptions,
        graph=CityGraph(distances, cities),
        dedup=dedup,
        source=dataset_path,
    )


@lru_cache(maxsize=4)
def get_dataset(path: str | None = None) -> Dataset:
    """Cached dataset load, so the API reads the workbook once per process."""
    return load_dataset(path)
