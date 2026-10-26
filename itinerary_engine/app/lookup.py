"""Finding a sight from what someone actually typed.

People type "bagore haveli", "sajangarh", "city plce" or just "fort". The dataset has
"Bagore ki Haveli", "Sajjangarh fort", "City Palace" and eleven forts. This module matches
all of those without ever guessing silently: every result carries why it matched and how
confident the match is, so the caller can show a list rather than pick for the user.

Matching happens in four widening steps, best first:

1. **exact** -- the normalised name is the same
2. **prefix / contains** -- every word typed appears in the name ("bagore haveli")
3. **fuzzy** -- close enough by edit distance to survive a typo ("sajangarh")
4. **sounds like** -- same consonant skeleton, for a phonetic spelling ("jodpur")

Only the dataset is searched for sights; the generic activities (dinner, a spa, shopping)
are a separate list, because inventing a sight that is not in the data is exactly what the
rest of the engine refuses to do.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

import pandas as pd

from .data_loader import Dataset
from .normalization import normalize_key

#: Below this, a fuzzy match is noise rather than a typo.
FUZZY_FLOOR = 0.62
#: Things people add that are not sights. Offered when the typing looks like one of them,
#: and as starting points in the "suggested activities" list.
GENERIC_ACTIVITIES: list[dict] = [
    {"name": "Dinner", "duration_hours": 1.5, "at": "20:00", "note": "a table somewhere good"},
    {"name": "Lunch with a view", "duration_hours": 1.5, "at": "13:00", "note": ""},
    {"name": "Shopping", "duration_hours": 2.0, "at": None, "note": "local markets"},
    {"name": "Spa / massage", "duration_hours": 1.5, "at": None, "note": ""},
    {"name": "Rest at the hotel", "duration_hours": 2.0, "at": "15:00", "note": "a quiet afternoon"},
    {"name": "Cooking class", "duration_hours": 2.5, "at": "17:00", "note": ""},
    {"name": "Cultural show", "duration_hours": 1.5, "at": "19:00", "note": ""},
    {"name": "Morning walk", "duration_hours": 1.0, "at": "07:00", "note": ""},
    {"name": "Photography stop", "duration_hours": 0.5, "at": None, "note": ""},
    {"name": "Free time", "duration_hours": 2.0, "at": None, "note": "nothing planned"},
]


@dataclass
class Match:
    """One candidate for what the user meant."""

    name: str
    kind: str          # sight | activity
    how: str           # exact | contains | fuzzy | sounds-like | suggested
    score: float
    city: str | None = None
    duration_hours: float | None = None
    best_time: str | None = None
    must_visit: bool = False
    scheduled_on: int | None = None
    detail: str = ""
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "how": self.how,
            "score": round(self.score, 3),
            "city": self.city,
            "duration_hours": self.duration_hours,
            "best_time": self.best_time,
            "must_visit": self.must_visit,
            "scheduled_on": self.scheduled_on,
            "detail": self.detail,
            **self.extra,
        }


def _words(text: str) -> list[str]:
    return [word for word in re.split(r"[^a-z0-9]+", str(text or "").lower()) if word]


def _skeleton(text: str) -> str:
    """Consonants only, doubles collapsed: "Sajjangarh" and "sajangar" both -> "sjngrh"."""
    letters = re.sub(r"[^a-z]", "", str(text or "").lower())
    letters = re.sub(r"[aeiou]", "", letters)
    out: list[str] = []
    for letter in letters:
        if not out or out[-1] != letter:
            out.append(letter)
    return "".join(out)


def _ratio(typed: str, candidate: str) -> float:
    return difflib.SequenceMatcher(None, typed, candidate).ratio()


def _match_one(typed: str, name: str) -> tuple[str, float] | None:
    """How (and how well) this name matches what was typed, or None."""
    typed_key, name_key = normalize_key(typed), normalize_key(name)
    if not typed_key:
        return None
    if typed_key == name_key:
        return "exact", 1.0

    typed_words, name_words = _words(typed), _words(name)
    if name_key.startswith(typed_key):
        return "contains", 0.95
    if typed_words and all(
        any(word == other or other.startswith(word) for other in name_words) for word in typed_words
    ):
        # "bagore haveli" -> "Bagore ki Haveli": every word typed is in the name
        return "contains", 0.9 - 0.01 * max(0, len(name_words) - len(typed_words))
    if typed_key in name_key:
        return "contains", 0.85

    ratio = _ratio(typed_key, name_key)
    if ratio >= FUZZY_FLOOR:
        return "fuzzy", ratio
    # A single mistyped word inside a longer name ("sajangarh fort").
    best_word = max((_ratio(typed_key, word) for word in name_words), default=0.0)
    if best_word >= 0.78:
        return "fuzzy", best_word * 0.9
    # A skeleton of one or two consonants matches far too much ("lake" -> "Lok Kala"), so
    # phonetic matching only applies once there is enough of a shape to compare.
    typed_skeleton = _skeleton(typed)
    if len(typed_skeleton) < 3:
        return None
    if _skeleton(name) == typed_skeleton:
        return "sounds-like", 0.7
    skeleton = max((_ratio(typed_skeleton, _skeleton(word)) for word in name_words
                    if len(_skeleton(word)) >= 3), default=0.0)
    if skeleton >= 0.85:
        return "sounds-like", 0.65
    return None


def _sight_matches(dataset: Dataset, typed: str, cities: list[str],
                   scheduled: dict[str, int]) -> list[Match]:
    frames = [dataset.city_sights(normalize_key(city)) for city in cities] if cities else [dataset.sights]
    pool = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True) \
        if any(not frame.empty for frame in frames) else dataset.sights.iloc[0:0]

    found: list[Match] = []
    seen: set[str] = set()
    for _, row in pool.iterrows():
        name = str(row["sight"])
        key = normalize_key(name)
        if key in seen:
            continue
        hit = _match_one(typed, name)
        if hit is None:
            continue
        seen.add(key)
        how, score = hit
        found.append(Match(
            name=name,
            kind="sight",
            how=how,
            score=score,
            city=str(row.get("city", "")),
            duration_hours=float(row.get("duration_hours", 0) or 0),
            best_time=str(row.get("best_time", "") or ""),
            must_visit=bool(row.get("must_visit_flag", False)),
            scheduled_on=scheduled.get(key),
        ))
    return found


def _activity_matches(typed: str) -> list[Match]:
    found = []
    for activity in GENERIC_ACTIVITIES:
        hit = _match_one(typed, activity["name"])
        if hit is None:
            continue
        how, score = hit
        found.append(Match(
            name=activity["name"],
            kind="activity",
            how=how,
            score=score * 0.95,  # a real sight of equal score is the better answer
            duration_hours=activity["duration_hours"],
            detail=activity["note"],
            extra={"at": activity["at"]},
        ))
    return found


def _scheduled_days(payload: dict | None) -> dict[str, int]:
    if not payload:
        return {}
    return {normalize_key(sight["sight"]): day["day"]
            for day in payload.get("days", []) for sight in day.get("sights", [])}


def search(dataset: Dataset, typed: str, cities: list[str] | None = None,
           payload: dict | None = None, limit: int = 8) -> list[dict]:
    """What the user might mean, best match first.

    Sights already in the plan are kept but ranked below ones that are not: the point of
    the box is usually to add something new, and the entry says which day it is already on
    rather than hiding it.
    """
    scheduled = _scheduled_days(payload)
    found = _sight_matches(dataset, typed, cities or [], scheduled) + _activity_matches(typed)
    found.sort(key=lambda match: (
        match.scheduled_on is not None,       # unscheduled first
        -match.score,
        not match.must_visit,
        match.name,
    ))
    return [match.as_dict() for match in found[:limit]]


def suggestions(dataset: Dataset, cities: list[str] | None = None,
                payload: dict | None = None, day: int | None = None, limit: int = 12) -> dict:
    """The "suggested activities" list: what is worth adding to this trip, unprompted.

    Unscheduled sights in the cities being visited come first, most important first, because
    those are the things the traveller came for and the plan could not fit. The generic
    activities follow as starting points.
    """
    scheduled = _scheduled_days(payload)
    unscheduled: list[Match] = []
    frames = [dataset.city_sights(normalize_key(city)) for city in (cities or [])]
    pool = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True) \
        if any(not frame.empty for frame in frames) else dataset.sights.iloc[0:0]

    held_back = {normalize_key(label.split(" / ")[-1]): reason
                 for label, reason in ((payload or {}).get("unplaceable_reasons") or {}).items()}

    for _, row in pool.iterrows():
        key = normalize_key(str(row["sight"]))
        if key in scheduled:
            continue
        unscheduled.append(Match(
            name=str(row["sight"]),
            kind="sight",
            how="suggested",
            score=1.0 if bool(row.get("must_visit_flag")) else 0.5,
            city=str(row.get("city", "")),
            duration_hours=float(row.get("duration_hours", 0) or 0),
            best_time=str(row.get("best_time", "") or ""),
            must_visit=bool(row.get("must_visit_flag", False)),
            detail=held_back.get(key, ""),
        ))
    unscheduled.sort(key=lambda match: (not match.must_visit, -match.score, match.name))

    return {
        "sights": [match.as_dict() for match in unscheduled[:limit]],
        "activities": [
            {"name": item["name"], "kind": "activity", "how": "suggested",
             "duration_hours": item["duration_hours"], "at": item["at"], "detail": item["note"]}
            for item in GENERIC_ACTIVITIES
        ],
        "day": day,
    }
