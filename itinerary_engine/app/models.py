"""Pydantic request/response models for the API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class PinnedSightIn(BaseModel):
    """A sight the user wants in the plan, optionally on a chosen day."""

    sight: str
    city: str | None = None
    day: int | None = Field(None, ge=1, le=30)


class SightEditIn(BaseModel):
    """One edit: add a sight, swap one for another, or drop one."""

    action: Literal["add", "replace", "remove"]
    sight: str
    city: str | None = None
    day: int | None = Field(None, ge=1, le=30, description="Preferred day; the engine suggests another if it cannot fit.")
    replaces: str | None = Field(None, description="Sight being swapped out, for action='replace'.")


class CustomActivityIn(BaseModel):
    """Something of the traveller's own: a dinner, a show, an afternoon off."""

    name: str = Field(..., min_length=1)
    day: int = Field(..., ge=1)
    duration_hours: float = Field(1.0, gt=0, le=24,
                                  description="Any length, up to a full day.")
    at: str | None = Field(None, description='Clock time, e.g. "20:00" or "06:30". Any hour: the '
                                             'day starts earlier or runs later to fit it. Omit to '
                                             'let it fall where there is room.')
    note: str = ""


class ItineraryRequest(BaseModel):
    """Body of ``POST /generate-itinerary``."""

    cities: list[str] = Field(..., min_length=1, description="Base cities in travel order.")
    duration_days: int = Field(..., ge=1, le=30, description="Total days including arrival and departure.")
    pace: Literal["relaxed", "moderate", "active"] = "moderate"
    origin_city: str | None = Field(
        None, description="Where the traveller starts (e.g. Delhi); used to charge day 1's drive."
    )
    start_month: int | None = Field(None, ge=1, le=12, description="Travel month, for seasonal closures.")
    include_descriptions: bool = True
    pinned_sights: list[PinnedSightIn] = []
    excluded_sights: list[str] = []
    #: The user's own split of days across cities, e.g. {"Jaipur": 4, "Udaipur": 2}.
    city_days: dict[str, int] = Field(default_factory=dict)
    #: Extra hours per day number, for a day the traveller wants to stretch.
    extra_day_hours: dict[int, float] = Field(default_factory=dict)
    #: The user's own order for a day's stops, by day number.
    day_order: dict[int, list[str]] = Field(default_factory=dict)
    #: Times asked for, by day number then sight: {2: {"Lake Pichola": "16:30"}}.
    sight_times: dict[int, dict[str, str]] = Field(default_factory=dict)
    #: Start times chosen per day number, e.g. {2: "07:30"}.
    day_starts: dict[int, str] = Field(default_factory=dict)
    custom_activities: list[CustomActivityIn] = []
    #: Who is travelling. Steers which sights are chosen between equals and what is offered
    #: for free time; it never changes the pace, which the data says it does not predict.
    group_type: Literal["solo", "couple", "friends", "family"] | None = None
    #: What they came for: nature, adventure, religious, cultural, shopping, wildlife,
    #: scenic, leisure, water. Mapped onto the dataset's own category column.
    interests: list[str] = []
    #: Load this customer's stored preferences, and fill in anything not given here.
    customer_id: str | None = Field(None, max_length=64)

    @field_validator("cities")
    @classmethod
    def _strip_cities(cls, value: list[str]) -> list[str]:
        cleaned = [city.strip() for city in value if city and city.strip()]
        if not cleaned:
            raise ValueError("cities must contain at least one non-empty name")
        return cleaned

    model_config = {
        "json_schema_extra": {
            "example": {
                "cities": ["Manali", "Kasol"],
                "duration_days": 5,
                "pace": "moderate",
                "origin_city": "Delhi",
                "start_month": 6,
            }
        }
    }


class AdjustRequest(BaseModel):
    """A plan request plus the edits to apply to it."""

    base: ItineraryRequest
    edits: list[SightEditIn] = Field(..., min_length=1)


class PlacementQuery(BaseModel):
    """Ask where a sight could go in a plan built from ``base``."""

    base: ItineraryRequest
    sight: str
    city: str | None = None


class EditOutcomeOut(BaseModel):
    action: str
    sight: str
    replaces: str | None = None
    requested_day: int | None = None
    applied: bool
    placed_on_day: int | None = None
    message: str
    suggestions: list[dict] = []


class SightOut(BaseModel):
    sight: str
    visit_type: str = ""
    duration_hours: float
    buffer_hours: float
    must_see_tier: int
    must_visit: bool
    priority: int
    best_time: str = "any"
    zone: str = ""
    lat: float
    lon: float
    is_excursion: bool = False
    distance_from_centre_km: float = 0.0
    description: str | None = None
    #: Set on stops made during an intercity drive rather than from the city's hotel.
    en_route: bool = False
    detour_km: float | None = None
    detour_hours: float | None = None
    along_route_percent: int | None = None
    home_city: str | None = None


class TransferOut(BaseModel):
    from_city: str
    to_city: str
    road_km: float | None = None
    straight_line_km: float | None = None
    #: None when the dataset has no journey time; the engine never invents one.
    drive_hours: float | None = None
    mode: str
    source: str
    data_gap: bool = False
    travel_days_needed: int = 1
    is_overnight: bool = False
    is_long_travel_day: bool = False
    notes: list[str] = []


class TimeBudgetOut(BaseModel):
    pace_allowance: float
    intercity_travel_hours: float | None = None
    day_overhead_hours: float = 0.0
    departure_cost_hours: float = 0.0
    #: Hours the user added to this day on top of the pace.
    extra_hours: float = 0.0


class ScheduleBlockOut(BaseModel):
    """One stretch of a day on the clock."""

    start: str
    end: str
    hours: float
    kind: str   # travel | sight | transfer | break | lunch | check-in | checkout | free | wait
    label: str
    detail: str = ""
    sight: str | None = None
    #: Of a travel block, the share that is parking and tickets rather than driving.
    buffer_hours: float = 0.0


class ScheduleTotalsOut(BaseModel):
    """Every hour of the day, split the way the itinerary header reports it."""

    sightseeing: float = 0.0
    local_travel: float = 0.0
    stop_buffers: float = 0.0
    intercity_travel: float = 0.0
    meals_and_breaks: float = 0.0
    waiting: float = 0.0
    check_in_out: float = 0.0
    your_own_plans: float = 0.0
    free: float = 0.0


class ScheduleAlternativeOut(BaseModel):
    """What to do about a preferred time the day could not honour."""

    kind: str = "slot"
    sight: str
    day: int | None = None
    city: str | None = None
    window_start: float | None = None
    problem: str = ""
    suggestion: str = ""


class DayScheduleOut(BaseModel):
    starts: str | None = None
    ends: str | None = None
    blocks: list[ScheduleBlockOut] = []
    totals: ScheduleTotalsOut | None = None
    #: The clock span of the day, the part of it that counts against the pace, and the sum
    #: of the blocks -- which must equal the span.
    span_hours: float = 0.0
    activity_hours: float = 0.0
    accounted_hours: float = 0.0
    #: Preferred times that could not be honoured, or missing timing data.
    flags: list[str] = []
    notes: list[str] = []
    alternatives: list[ScheduleAlternativeOut] = []
    #: Hard-slot sights this day could not fit inside their window. Always empty in a plan
    #: that passed validation: the builder re-homes or drops them first.
    deferred: list[dict] = []


class DayTotalsOut(BaseModel):
    """The day's hours as the clock has them, for the itinerary header."""

    activity_hours: float = 0.0
    meal_and_break_hours: float = 0.0
    waiting_hours: float = 0.0
    intercity_travel_hours: float = 0.0
    check_in_out_hours: float = 0.0
    span_hours: float = 0.0
    starts: str | None = None
    ends: str | None = None


class DayOut(BaseModel):
    day: int
    city: str
    transfer: TransferOut | None = None
    is_arrival_day: bool = False
    is_departure_day: bool = False
    is_travel_only: bool = False
    time_budget: TimeBudgetOut | None = None
    available_hours: float
    used_hours: float
    spare_hours: float
    sightseeing_hours: float
    buffer_hours: float
    local_travel_hours: float
    en_route_hours: float = 0.0
    route_km: float
    max_leg_km: float
    sights: list[SightOut]
    schedule: DayScheduleOut | None = None
    #: The same hours as ``schedule.totals``, flattened for the header.
    day_totals: DayTotalsOut | None = None
    #: Stretches of the day with nothing booked, after sightseeing, travel, meals and rest.
    free_windows: list[dict] = []
    #: Optional nearby activities for those windows. Never part of the itinerary: not
    #: scheduled, not counted in any total, not priced unless the customer takes one.
    explore_on_your_own: list[dict] = []
    notes: list[str] = []


class ValidationIssueOut(BaseModel):
    severity: str
    category: str
    day: int | None = None
    message: str
    fix: str = ""


class ValidationOut(BaseModel):
    #: True only when no critical check failed.
    passed: bool
    critical_count: int = 0
    warning_count: int = 0
    info_count: int = 0
    #: Blocking problems, grouped by category.
    issues: dict[str, list[str]] = {}
    entries: list[ValidationIssueOut] = []
    warnings: list[str] = []
    notes: list[str] = []


class ItineraryResponse(BaseModel):
    request: dict
    summary: dict
    route: list[TransferOut]
    days: list[DayOut]
    data_gaps: list[str] = []
    unscheduled_sights: list[str] = []
    unscheduled_must_visits: list[str] = []
    unscheduled_reasons: dict[str, str] = {}
    #: Sights the engine left out because no day could give them their time of day, with
    #: the reason for each. Distinct from a user's own exclusions.
    unplaceable_reasons: dict[str, str] = {}
    closed_this_month: list[str] = []
    validation: ValidationOut
    edits: list[EditOutcomeOut] = []


class OperationIn(BaseModel):
    """One post-generation customisation (see ``app/customize.py``)."""

    operation: Literal["retime", "move", "swap", "swap_days", "reorder", "to_morning",
                       "extend_day", "set_day_start", "set_pace", "reallocate", "pin", "drop",
                       "add_activity"]
    sight: str | None = None
    city: str | None = None
    day: int | None = Field(None, ge=1)
    to_day: int | None = Field(None, ge=1)
    from_day: int | None = Field(None, ge=1)
    at: str | None = None
    hours: float | None = Field(None, ge=0, le=12)
    order: list[str] = []
    city_days: dict[str, int] = Field(default_factory=dict)
    name: str | None = None
    duration_hours: float | None = Field(None, gt=0, le=24)
    note: str = ""
    with_sight: str | None = None
    with_day: int | None = Field(None, ge=1)
    pace: Literal["relaxed", "moderate", "active"] | None = None


class CustomizeRequest(BaseModel):
    """Body of ``POST /itinerary/customize``: a plan, and what to change about it."""

    base: ItineraryRequest
    operations: list[OperationIn] = Field(..., min_length=1)

    model_config = {
        "json_schema_extra": {
            "example": {
                "base": {"cities": ["Udaipur"], "duration_days": 3, "pace": "moderate"},
                "operations": [
                    {"operation": "move", "sight": "Lake Pichola", "to_day": 2},
                    {"operation": "extend_day", "day": 2, "hours": 1.5},
                    {"operation": "add_activity", "name": "Dinner at Ambrai", "day": 2,
                     "duration_hours": 1.5, "at": "20:00"},
                ],
            }
        }
    }


class SuggestRequest(BaseModel):
    """Body of ``POST /itinerary/suggestions``."""

    base: ItineraryRequest


class ActivitySearchQuery(BaseModel):
    """Body of ``POST /itinerary/activity-search``: what has been typed so far."""

    text: str = Field(..., min_length=1, max_length=120)
    #: The trip, so results can be ranked against what is already planned. Optional: without
    #: it the whole dataset is searched.
    base: ItineraryRequest | None = None
    #: Cities to search, when there is no trip to take them from.
    cities: list[str] = []
    limit: int = Field(8, ge=1, le=25)


class ActivitySuggestQuery(BaseModel):
    """Body of ``POST /itinerary/activity-suggestions``."""

    base: ItineraryRequest
    day: int | None = Field(None, ge=1)
    limit: int = Field(12, ge=1, le=40)


class PreferencesIn(BaseModel):
    """Body of ``PUT /customers/{customer_id}/preferences``."""

    name: str = ""
    group_type: Literal["solo", "couple", "friends", "family"] | None = None
    interests: list[str] = []
    pace: Literal["relaxed", "moderate", "active"] | None = None
    notes: str = ""

    model_config = {
        "json_schema_extra": {
            "example": {"name": "R. Mehta", "group_type": "family",
                        "interests": ["cultural", "nature"], "pace": "relaxed"}
        }
    }
