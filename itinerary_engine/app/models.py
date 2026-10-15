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
    closed_this_month: list[str] = []
    validation: ValidationOut
    edits: list[EditOutcomeOut] = []
