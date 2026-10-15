"""Attraction-name normalisation and de-duplication.

Ported from the Qwen itinerary validation layer: the same attraction often appears under
several names ("Tiger Hill" / "Tiger Hills" / "Tiger Hill Sunrise Point"), which produces
itineraries that visit one place twice. Here the dataset's own ``aliases`` column does most
of the work, and the rest is caught with spelling, containment and coordinate checks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

import pandas as pd

from .config import (
    DESCRIPTOR_WORDS,
    GENERIC_SIGHT_WORDS,
    SAME_PLACE_KM,
    SAME_PLACE_NAME_KM,
)
from .geo import haversine_km

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize_key(value: object) -> str:
    """Lowercase, punctuation-free key used for all name comparisons."""
    text = str(value or "").lower().replace("&", " and ")
    return re.sub(r"\s+", " ", _NON_ALNUM.sub(" ", text)).strip()


def _compact(key: str) -> str:
    return re.sub(r"^the ", "", key).replace(" ", "")


def _extra_words(key_a: str, key_b: str) -> bool:
    """True if one name has a real extra word ("war" in "Jaisalmer War Museum")."""
    tokens_a, tokens_b = key_a.split(), key_b.split()
    unmatched = [w for w in tokens_a if len(w) >= 3 and not any(SequenceMatcher(None, w, v).ratio() >= 0.6 for v in tokens_b)]
    unmatched += [w for w in tokens_b if len(w) >= 3 and not any(SequenceMatcher(None, w, v).ratio() >= 0.6 for v in tokens_a)]
    return bool(unmatched) and "".join(tokens_a) != "".join(tokens_b)


def names_contain(key_a: str, key_b: str) -> bool:
    """True if one name contains the other as whole words."""
    return bool(key_a and key_b) and (f" {key_a} " in f" {key_b} " or f" {key_b} " in f" {key_a} ")


def same_place(row_a: pd.Series, row_b: pd.Series) -> str | None:
    """Reason why two sights of one city are the same place, or None.

    Checked in order: identical key, declared alias, spelling variant, containment with
    only descriptor words added, and finally coordinates (which alone are not enough --
    two different places are often geocoded to the same point).
    """
    key_a, key_b = row_a["sight_key"], row_b["sight_key"]
    if not key_a or not key_b:
        return None
    if key_a == key_b:
        return "identical name"
    if key_b in row_a["alias_keys"] or key_a in row_b["alias_keys"]:
        return "declared alias"
    if re.findall(r"\d+", key_a) != re.findall(r"\d+", key_b):
        return None  # "Beach 1" and "Beach 2" are different places

    compact_a, compact_b = _compact(key_a), _compact(key_b)
    if compact_a == compact_b:
        return "spacing/punctuation"
    if compact_a[:3] == compact_b[:3] and SequenceMatcher(None, compact_a, compact_b).ratio() >= 0.9 and not _extra_words(key_a, key_b):
        return "spelling"

    distance = haversine_km(row_a["lat"], row_a["lon"], row_b["lat"], row_b["lon"])
    shorter, longer = sorted([key_a, key_b], key=len)
    if names_contain(key_a, key_b) and shorter not in GENERIC_SIGHT_WORDS and len(shorter) >= 4:
        extra = set(longer.split()) - set(shorter.split())
        if extra <= DESCRIPTOR_WORDS or distance <= SAME_PLACE_NAME_KM:
            return "one name contains the other"
    if distance <= SAME_PLACE_KM and (names_contain(key_a, key_b) or _share_distinctive_word(key_a, key_b)):
        return "same coordinates"
    return None


def _share_distinctive_word(key_a: str, key_b: str) -> bool:
    words_a = {w for w in key_a.split() if len(w) >= 4 and w not in GENERIC_SIGHT_WORDS}
    words_b = {w for w in key_b.split() if len(w) >= 4 and w not in GENERIC_SIGHT_WORDS}
    return bool(words_a & words_b)


@dataclass
class DeduplicationResult:
    """Canonical sightseeing rows plus the merges that produced them."""

    sights: pd.DataFrame
    merges: list[dict] = field(default_factory=list)
    #: (city_key, any known name key) -> canonical sight name
    alias_index: dict[tuple[str, str], str] = field(default_factory=dict)

    def resolve(self, city: str, sight: str) -> str:
        """Canonical name for a possibly non-canonical sight name."""
        return self.alias_index.get((normalize_key(city), normalize_key(sight)), str(sight))


def deduplicate_sights(sights: pd.DataFrame) -> DeduplicationResult:
    """Collapse same-place rows per city, keeping the most important row of each group."""
    merges: list[dict] = []
    alias_index: dict[tuple[str, str], str] = {}
    kept_rows: list[pd.Series] = []

    for city_key, group in sights.groupby("city_key", sort=False):
        # Most important row first, so a group keeps the best-ranked name.
        ordered = group.sort_values(
            ["must_visit_flag", "must_see_tier", "priority", "mentions"],
            ascending=[False, True, True, False],
        )
        rows: list[pd.Series] = []
        for _, candidate in ordered.iterrows():
            twin_index = next(
                (i for i, kept in enumerate(rows) if same_place(kept, candidate)), None
            )
            if twin_index is None:
                rows.append(candidate)
                continue
            kept = rows[twin_index]
            reason = same_place(kept, candidate)
            merges.append(
                {
                    "city": kept["city"],
                    "kept": kept["sight"],
                    "merged": candidate["sight"],
                    "reason": reason,
                }
            )
            # The merged row's names stay reachable, and its aliases carry over.
            alias_index[(city_key, candidate["sight_key"])] = kept["sight"]
            for alias_key in candidate["alias_keys"]:
                alias_index[(city_key, alias_key)] = kept["sight"]
            rows[twin_index] = kept.copy()
            rows[twin_index]["alias_keys"] = kept["alias_keys"] | candidate["alias_keys"] | {candidate["sight_key"]}

        for kept in rows:
            alias_index[(city_key, kept["sight_key"])] = kept["sight"]
            for alias_key in kept["alias_keys"]:
                alias_index.setdefault((city_key, alias_key), kept["sight"])
            kept_rows.append(kept)

    deduplicated = pd.DataFrame(kept_rows).reset_index(drop=True) if kept_rows else sights.iloc[0:0].copy()
    return DeduplicationResult(sights=deduplicated, merges=merges, alias_index=alias_index)
