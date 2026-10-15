"""Spatial clustering of a city's sights into day-groups.

A day should stay in one part of town. The dataset's ``zone`` column already encodes that
when it is populated with enough distinct zones; otherwise KMeans on (lat, lon) is used.
Excursions (places far outside the city) are pulled out first, because they are day trips
rather than members of a city cluster.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .config import EXCURSION_KM, RANDOM_SEED
from .geo import road_km

logger = logging.getLogger(__name__)


@dataclass
class ClusterResult:
    """Sight groups for one city, most important group first."""

    clusters: list[pd.DataFrame]
    method: str
    excursions: pd.DataFrame

    @property
    def count(self) -> int:
        return len(self.clusters)


def mark_excursions(sights: pd.DataFrame, centre_lat: float, centre_lon: float) -> pd.DataFrame:
    """Add ``distance_from_centre_km`` and ``is_excursion`` columns."""
    marked = sights.copy()
    marked["distance_from_centre_km"] = [
        road_km(centre_lat, centre_lon, row["lat"], row["lon"]) for _, row in marked.iterrows()
    ]
    marked["is_excursion"] = (marked["distance_from_centre_km"] >= EXCURSION_KM) | (
        marked["visit_type"].str.lower() == "excursion"
    )
    return marked


def _sort_clusters(clusters: list[pd.DataFrame]) -> list[pd.DataFrame]:
    """Most important cluster first (must-visits, then tier, then priority)."""
    def rank(cluster: pd.DataFrame) -> tuple:
        return (
            -int(cluster["must_visit_flag"].sum()),
            float(cluster["must_see_tier"].mean()),
            float(cluster["priority"].min()),
        )

    return sorted([c for c in clusters if not c.empty], key=rank)


def cluster_city_sights(
    sights: pd.DataFrame,
    n_clusters: int,
    centre_lat: float,
    centre_lon: float,
) -> ClusterResult:
    """Group a city's sights into at most ``n_clusters`` spatial day-groups.

    Edge cases handled: fewer sights than days, a single sight, missing zones, and more
    zones than days (zones are then merged by KMeans over their centroids).
    """
    if sights.empty or n_clusters < 1:
        return ClusterResult(clusters=[], method="none", excursions=sights.iloc[0:0])

    marked = mark_excursions(sights, centre_lat, centre_lon)
    excursions = marked[marked["is_excursion"]].copy()
    core = marked[~marked["is_excursion"]].copy()
    if core.empty:  # everything is far out: treat them as ordinary groups
        core, excursions = marked.copy(), marked.iloc[0:0].copy()

    if len(core) <= n_clusters:  # one sight per day at most; no clustering needed
        clusters = [core.iloc[[i]] for i in range(len(core))]
        return ClusterResult(_sort_clusters(clusters), "one-per-day", excursions)

    zones = [z for z in core["zone"].unique() if str(z).strip()]
    if len(zones) >= n_clusters and len(zones) > 1:
        groups = [core[core["zone"] == zone] for zone in zones]
        if len(groups) > n_clusters:
            groups = _merge_groups_by_centroid(groups, n_clusters)
        return ClusterResult(_sort_clusters(groups), "zone", excursions)

    coordinates = core[["lat", "lon"]].to_numpy(dtype=float)
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10)
    labels = kmeans.fit_predict(coordinates)
    clusters = [core[labels == label] for label in range(n_clusters)]
    return ClusterResult(_sort_clusters(clusters), "kmeans", excursions)


def _merge_groups_by_centroid(groups: list[pd.DataFrame], n_clusters: int) -> list[pd.DataFrame]:
    """Merge zone groups down to ``n_clusters`` by clustering their centroids."""
    centroids = np.array([[g["lat"].mean(), g["lon"].mean()] for g in groups])
    labels = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10).fit_predict(centroids)
    merged = []
    for label in range(n_clusters):
        members = [g for g, l in zip(groups, labels) if l == label]
        if members:
            merged.append(pd.concat(members, ignore_index=True))
    logger.debug("Merged %d zones into %d day clusters", len(groups), len(merged))
    return merged
