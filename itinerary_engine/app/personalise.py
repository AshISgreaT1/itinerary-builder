"""Matching sights to who is travelling and what they came for.

Two inputs, both grounded in the dataset rather than in assumption.

**Interests** map onto the sightseeing sheet's own ``category`` column -- twelve clean
values (heritage, religious, nature, adventure, wildlife, market, culture, leisure,
waterbody, viewpoint, village, transit). The sheet's ``visit_type`` has thirty-six values
with a long tail, so ``category`` is what the matching uses and ``visit_type`` is kept for
display.

**Group type** earns its place. Measured across 1,689 sight visits in the workbook's own
agent-written itineraries, it predicts which categories get visited (chi-square p < 0.0001):

| group | heritage | religious | nature | adventure | waterbody |
| --- | --- | --- | --- | --- | --- |
| family | 41.7% | 15.9% | 2.6% | 2.6% | 14.6% |
| couple | 28.6% | 20.1% | 12.2% | 4.6% | 14.5% |
| friends | 21.9% | 28.7% | 13.4% | 8.4% | 10.9% |
| solo | 35.1% | 21.6% | 4.1% | 1.4% | 9.5% |

Note what this does *not* say. The same field was measured earlier against how full a day
is and showed nothing (2.63 stops a day for couples, 2.63 for families, 2.65 for groups,
ANOVA p=0.87), which is why group type steers *what* is recommended and never the pace.

The effect is deliberately a nudge. A must-visit stays a must-visit and the tiers are
untouched; preference only chooses between sights the planner already considers equal.
With no preferences given, every score is zero and plans are byte-identical to before.
"""

from __future__ import annotations

from .preferences import CustomerPreferences

#: What a customer calls it -> what the dataset calls it.
INTEREST_CATEGORIES: dict[str, tuple[str, ...]] = {
    "nature": ("nature", "waterbody", "wildlife"),
    "adventure": ("adventure",),
    "religious": ("religious",),
    "cultural": ("heritage", "culture", "village"),
    "shopping": ("market",),
    "wildlife": ("wildlife",),
    "scenic": ("viewpoint", "nature", "waterbody"),
    "leisure": ("leisure",),
    "water": ("waterbody",),
}

#: Share of each group's visits by category, from the agent itineraries (see the module
#: docstring). Only categories a group visits more than the average are listed: these are
#: leanings, not rules, and a category missing here simply gets no nudge.
GROUP_LEANINGS: dict[str, dict[str, float]] = {
    "family": {"heritage": 1.0, "waterbody": 0.5, "leisure": 0.5, "viewpoint": 0.3},
    "couple": {"nature": 0.8, "waterbody": 0.6, "heritage": 0.4, "leisure": 0.3},
    "friends": {"adventure": 1.0, "nature": 0.7, "religious": 0.5, "leisure": 0.3},
    "solo": {"culture": 1.0, "market": 0.8, "heritage": 0.5, "viewpoint": 0.4},
}
#: ``group`` in the dataset is a party of friends; the request vocabulary says "friends".
GROUP_ALIASES = {"group": "friends", "friends": "friends", "solo": "solo",
                 "couple": "couple", "family": "family"}

#: How much a matched interest and a group leaning are worth when ranking. An interest the
#: customer named outweighs a tendency inferred from people like them, which is the right
#: way round: one is a statement, the other is a statistic.
INTEREST_WEIGHT = 2.0
GROUP_WEIGHT = 1.0


def categories_for(interests: list[str] | tuple[str, ...]) -> set[str]:
    """The dataset categories a customer's stated interests cover."""
    found: set[str] = set()
    for interest in interests or []:
        found.update(INTEREST_CATEGORIES.get(str(interest).strip().lower(), ()))
    return found


def score_category(category: str, interests: list[str] | tuple[str, ...],
                   group_type: str = "") -> float:
    """How well one sight's category suits this customer. Zero when nothing is known."""
    value = str(category or "").strip().lower()
    if not value:
        return 0.0
    score = 0.0
    if value in categories_for(interests):
        score += INTEREST_WEIGHT
    leanings = GROUP_LEANINGS.get(GROUP_ALIASES.get(str(group_type).strip().lower(), ""), {})
    score += GROUP_WEIGHT * leanings.get(value, 0.0)
    return round(score, 3)


def score_row(row, interests: list[str] | tuple[str, ...], group_type: str = "") -> float:
    """The same, for a dataset row (falls back to ``visit_type`` when a category is blank)."""
    category = row.get("category") if hasattr(row, "get") else None
    if not str(category or "").strip():
        category = row.get("visit_type") if hasattr(row, "get") else ""
    return score_category(str(category or ""), interests, group_type)


def why_it_suits(category: str, interests: list[str] | tuple[str, ...],
                 group_type: str = "") -> str:
    """One short phrase for the customer, or empty when it is just a good sight."""
    value = str(category or "").strip().lower()
    reasons = []
    matched = [name for name in (interests or [])
               if value in INTEREST_CATEGORIES.get(str(name).strip().lower(), ())]
    if matched:
        reasons.append(f"matches your interest in {matched[0]}")
    group = GROUP_ALIASES.get(str(group_type).strip().lower(), "")
    if group and value in GROUP_LEANINGS.get(group, {}):
        reasons.append(f"popular with {group} travellers in past trips")
    return "; ".join(reasons)


def from_preferences(preferences: CustomerPreferences | None) -> tuple[list[str], str]:
    """``(interests, group_type)`` from a stored record, or empty when there is none."""
    if preferences is None:
        return [], ""
    return list(preferences.interests), preferences.group_type
