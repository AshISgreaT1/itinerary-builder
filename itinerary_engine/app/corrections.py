"""Reviewed corrections to the dataset, applied at load time.

The data-quality report finds things the workbook gets wrong: a sight filed 166 km from
the city it belongs to, the same monastery listed under two cities, a coordinate that is
nowhere near a road. Some of those have a determinable answer -- the dataset itself says
which city a point is nearest -- and some need a human with a map.

This module handles the first kind, and deliberately does not decide anything itself:

* ``python -m app.datacheck --propose`` writes the proposals to a CSV
* a person reads them, deletes what they disagree with, and commits the file
* the loader applies what is in the file, and reports how many were applied

So the workbook stays exactly as it was shipped, every change is one reviewable line with
a reason attached, and reverting is deleting a row. Nothing is corrected silently, which
matters here more than usual: the whole engine rests on not inventing data.

The file is ``data/corrections.csv``:

    action,city,sight,value,reason
    rehome,Bikaner,Kheechan village,Manvar,166 km from Bikaner, 34 km from Manvar
    drop,Leh,Alchi Monastery,,duplicate of the Alchi row at identical coordinates
"""

from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .normalization import normalize_key

logger = logging.getLogger(__name__)

CORRECTIONS_PATH = Path(os.environ.get("ITINERARY_CORRECTIONS", "data/corrections.csv"))

#: ``rehome`` moves a sight to another city; ``drop`` removes a duplicate row.
ACTIONS = ("rehome", "drop")
FIELDS = ("action", "city", "sight", "value", "reason")


@dataclass
class Correction:
    action: str
    city: str
    sight: str
    value: str = ""
    reason: str = ""

    @property
    def key(self) -> tuple[str, str]:
        return normalize_key(self.city), normalize_key(self.sight)

    def as_row(self) -> dict:
        return {"action": self.action, "city": self.city, "sight": self.sight,
                "value": self.value, "reason": self.reason}


def _path(path: Path | str | None) -> Path:
    """Where the corrections live, resolved now rather than at import time."""
    return Path(CORRECTIONS_PATH if path is None else path)


def load(path: Path | str | None = None) -> list[Correction]:
    """Read the reviewed corrections, or nothing when the file is absent."""
    file = _path(path)
    if not file.is_file():          # a missing file, or a directory, means no corrections
        return []
    found: list[Correction] = []
    try:
        with file.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                action = str(row.get("action", "")).strip().lower()
                if action not in ACTIONS:
                    continue
                found.append(Correction(
                    action=action,
                    city=str(row.get("city", "")).strip(),
                    sight=str(row.get("sight", "")).strip(),
                    value=str(row.get("value", "")).strip(),
                    reason=str(row.get("reason", "")).strip(),
                ))
    except OSError as error:  # pragma: no cover - unreadable file
        logger.warning("Corrections at %s could not be read (%s); none applied.", file, error)
        return []
    return found


def write(corrections: list[Correction], path: Path | str | None = None) -> Path:
    file = _path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for correction in corrections:
            writer.writerow(correction.as_row())
    return file


def apply(sights: pd.DataFrame, cities: pd.DataFrame,
          path: Path | str | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Apply the reviewed corrections to the sights frame.

    Returns the frame and a line per correction describing what happened -- including the
    ones that did not match anything, because a correction pointing at a row that no longer
    exists is a stale file rather than a silent no-op.
    """
    corrections = load(path)
    if corrections:
        logger.info("Applying %d reviewed correction(s) from %s.", len(corrections), _path(path))
    if not corrections or sights.empty:
        return sights, []

    frame = sights.copy()
    known = {normalize_key(str(row["city"])): str(row["city"]) for _, row in cities.iterrows()}
    applied: list[str] = []
    drop_index: list = []

    for correction in corrections:
        city_key, sight_key = correction.key
        match = frame[(frame["city"].map(normalize_key) == city_key)
                      & (frame["sight"].map(normalize_key) == sight_key)]
        if match.empty:
            applied.append(f"skipped: no '{correction.sight}' under '{correction.city}' "
                           f"(the workbook may have changed)")
            continue

        if correction.action == "drop":
            drop_index += list(match.index)
            applied.append(f"dropped '{correction.sight}' from {correction.city}"
                           + (f" -- {correction.reason}" if correction.reason else ""))
            continue

        target = known.get(normalize_key(correction.value))
        if not target:
            applied.append(f"skipped: '{correction.value}' is not a city in the dataset, so "
                           f"'{correction.sight}' stays under {correction.city}")
            continue
        frame.loc[match.index, "city"] = target
        applied.append(f"re-homed '{correction.sight}' from {correction.city} to {target}"
                       + (f" -- {correction.reason}" if correction.reason else ""))

    if drop_index:
        frame = frame.drop(index=drop_index)
    return frame.reset_index(drop=True), applied


def summary(path: Path | str | None = None) -> dict:
    corrections = load(path)
    counts: dict[str, int] = {}
    for correction in corrections:
        counts[correction.action] = counts.get(correction.action, 0) + 1
    return {"file": str(_path(path)), "count": len(corrections), "by_action": counts}
