"""A small, stable REST surface for other systems to call.

The planning engine already has an HTTP interface, but it is the one the browser UI uses:
it speaks in the engine's own vocabulary, returns everything the UI needs, and changes
whenever the UI does. That is the wrong contract for another team to build against.

This module is a *facade*, not a second engine. It calls :func:`app.builder.build_itinerary`
exactly as the UI does and then projects the result onto a deliberately small, documented
shape, so the engine can keep changing without breaking a caller. Three things are
different from the internal endpoints, and all three are for the caller's benefit:

* **one envelope** -- every response, success or failure, is ``{"success": ..., ...}``
* **an API key** -- ``X-API-Key``, because this is reachable from outside
* **a frozen vocabulary** -- ``destination`` and ``days`` rather than ``cities`` and
  ``duration_days``, mapped here so renaming an internal field is not a breaking change.

**On the fields this engine cannot use.** The request accepts ``travellers``, ``budget``
and ``traveller_type`` because callers have them and leaving them out would mean losing
them at the boundary. None of them change the plan, and the response says so in
``meta.unused_inputs`` rather than quietly implying otherwise:

* *budget* -- the engine does no pricing. Only 2 of the 196 past itineraries in the dataset
  carry a total price, which is far too few to estimate from.
* *traveller_type* -- measured against those same itineraries it does not predict how full
  a day is: couples 2.63 stops a day, families 2.63, groups 2.65 (ANOVA p=0.87). A mapping
  from traveller type to pace would be invention, so ``pace`` is exposed directly instead.
* *travellers* -- the engine models time and distance, not vehicles or group logistics.
"""

from __future__ import annotations

import os
import time
from functools import lru_cache
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, field_validator

from .builder import PlanningError, PlanRequest, build_itinerary
from .data_loader import get_dataset

API_VERSION = "1.0.0"
PREFIX = "/api/v1"
#: Comma-separated keys, e.g. ITINERARY_API_KEYS="live-key-1,live-key-2".
API_KEYS_ENV = "ITINERARY_API_KEYS"
#: Set to "1" to run without a key. Local development only -- it is never the default,
#: because an API that silently accepts anyone when its configuration is missing is worse
#: than one that refuses to start.
ALLOW_ANONYMOUS_ENV = "ITINERARY_API_ALLOW_ANONYMOUS"

STARTED_AT = time.time()
#: How many distinct plans to keep. The engine is deterministic -- the same inputs always
#: produce the same plan -- so a repeat call can be served from memory instead of spending
#: several seconds re-planning. Small, because each entry holds a full itinerary.
PLAN_CACHE_SIZE = 64

api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
    description="Your API key. Set ITINERARY_API_KEYS on the server to issue keys.",
)

router = APIRouter(prefix=PREFIX, tags=["public API v1"])


# --- authentication ---------------------------------------------------------------------------
def configured_keys() -> set[str]:
    raw = os.environ.get(API_KEYS_ENV, "")
    return {key.strip() for key in raw.split(",") if key.strip()}


def anonymous_allowed() -> bool:
    return os.environ.get(ALLOW_ANONYMOUS_ENV, "").strip() in {"1", "true", "yes"}


def require_api_key(key: str | None = Depends(api_key_header)) -> str:
    """Reject anything without a valid key, and fail closed when none are configured."""
    keys = configured_keys()
    if not keys:
        if anonymous_allowed():
            return "anonymous"
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"This API has no keys configured, so it cannot authenticate anyone. Set "
                f"{API_KEYS_ENV} to one or more comma-separated keys and restart, or set "
                f"{ALLOW_ANONYMOUS_ENV}=1 for local development."
            ),
        )
    if not key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Missing X-API-Key header.")
    if key not in keys:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key.")
    return key


# --- the envelope -------------------------------------------------------------------------------
def ok(data: Any, **meta: Any) -> dict:
    return {"success": True, "data": data, "meta": {"api_version": API_VERSION, **meta}}


def failed(code: str, message: str, details: Any = None) -> dict:
    body: dict[str, Any] = {"code": code, "message": message}
    if details is not None:
        body["details"] = details
    return {"success": False, "error": body, "meta": {"api_version": API_VERSION}}


# --- request --------------------------------------------------------------------------------------
class ItineraryRequestV1(BaseModel):
    """What another system sends to plan a trip."""

    destination: str | list[str] = Field(
        ...,
        description='One city or several, in travel order. "Jaipur", "Jaipur, Udaipur" or '
                    '["Jaipur", "Udaipur"] are all accepted.',
        examples=["Jaipur, Udaipur"],
    )
    days: int = Field(..., ge=1, le=30, description="Total days, including arrival and departure.")
    travellers: int | None = Field(
        None, ge=1, le=60,
        description="Party size. Recorded and echoed back; it does not change the plan.",
    )
    budget: float | None = Field(
        None, ge=0,
        description="Total budget. Recorded and echoed back; this engine does no pricing.",
    )
    traveller_type: Literal["solo", "couple", "family", "group", "other"] | None = Field(
        None,
        description="Recorded and echoed back. Measured against the dataset's own past "
                    "itineraries it does not predict trip intensity, so it does not change "
                    "the plan. Use `pace` to control that.",
    )
    pace: Literal["relaxed", "moderate", "active"] = Field(
        "moderate", description="Sightseeing hours a day: 5 / 7 / 9.")
    origin_city: str | None = Field(None, description="Where the trip starts, if day 1 is a journey.")
    start_month: int | None = Field(None, ge=1, le=12, description="Travel month, for seasonal closures.")
    currency: str | None = Field(None, max_length=8, description="Echoed with `budget`.")

    @field_validator("destination")
    @classmethod
    def _clean_destination(cls, value: str | list[str]) -> list[str]:
        names = value if isinstance(value, list) else str(value).split(",")
        cleaned = [name.strip() for name in names if str(name).strip()]
        if not cleaned:
            raise ValueError("destination must name at least one city")
        if len(cleaned) > 10:
            raise ValueError("destination may name at most 10 cities")
        return cleaned

    model_config = {
        "json_schema_extra": {
            "example": {
                "destination": "Jaipur, Udaipur",
                "days": 6,
                "travellers": 2,
                "budget": 60000,
                "currency": "INR",
                "traveller_type": "couple",
                "pace": "moderate",
                "start_month": 11,
            }
        }
    }


# --- response -------------------------------------------------------------------------------------
class ItemV1(BaseModel):
    """One entry on a day: a visit, a drive, a meal, something of the traveller's own."""

    start: str
    end: str
    kind: str = Field(description="sight | travel | transfer | lunch | break | wait | check-in | checkout | custom")
    title: str
    description: str = ""
    duration_hours: float = 0.0
    best_time: str | None = None
    must_visit: bool = False


class DayV1(BaseModel):
    day: int
    city: str
    is_arrival_day: bool = False
    is_departure_day: bool = False
    transfer: dict | None = Field(None, description="from_city, to_city, road_km, drive_hours, mode")
    starts: str | None = None
    ends: str | None = None
    sightseeing_hours: float = 0.0
    travel_hours: float = 0.0
    items: list[ItemV1] = []


class ItineraryV1(BaseModel):
    destination: list[str]
    days: int
    pace: str
    nights_per_city: dict[str, int] = {}
    total_sights: int = 0
    total_sightseeing_hours: float = 0.0
    total_travel_hours: float = 0.0
    feasible: bool = True
    warnings: list[str] = []
    not_included: list[str] = Field(
        default_factory=list,
        description="Sights the plan could not fit, with the reason for each.",
    )
    itinerary: list[DayV1] = []


# --- projection -------------------------------------------------------------------------------------
def _items_for(day: dict) -> list[ItemV1]:
    """One day's blocks, with each sight's own facts attached."""
    by_name = {sight["sight"]: sight for sight in day.get("sights", [])}
    items: list[ItemV1] = []
    for block in (day.get("schedule") or {}).get("blocks", []):
        sight = by_name.get(block.get("sight") or "")
        items.append(ItemV1(
            start=block["start"],
            end=block["end"],
            kind=block["kind"],
            title=(block.get("sight") or block.get("label") or "") if block["kind"] == "sight"
                  else (block.get("label") or ""),
            # Either source can be absent; the field is a string, not a maybe-string.
            description=((sight or {}).get("description") if block["kind"] == "sight"
                         else block.get("detail")) or "",
            duration_hours=block.get("hours", 0.0),
            best_time=(sight or {}).get("best_time"),
            must_visit=bool((sight or {}).get("must_visit", False)),
        ))
    return items


def project(payload: dict, request: ItineraryRequestV1) -> ItineraryV1:
    """Turn the engine's full plan into the shape this API promises.

    Deliberately lossy. The engine's payload carries budgets, validation entries, learned
    ordering patterns and the controls the UI needs; a caller wants the trip. Anything left
    out here can be added later without breaking anyone, which is the point of projecting
    rather than forwarding.
    """
    summary = payload.get("summary", {})
    validation = payload.get("validation", {})
    days = [
        DayV1(
            day=day["day"],
            city=day["city"],
            is_arrival_day=bool(day.get("is_arrival_day")),
            is_departure_day=bool(day.get("is_departure_day")),
            transfer=({
                "from_city": day["transfer"]["from_city"],
                "to_city": day["transfer"]["to_city"],
                "road_km": day["transfer"].get("road_km"),
                "drive_hours": day["transfer"].get("drive_hours"),
                "mode": day["transfer"].get("mode"),
            } if day.get("transfer") else None),
            starts=(day.get("schedule") or {}).get("starts"),
            ends=(day.get("schedule") or {}).get("ends"),
            sightseeing_hours=day.get("sightseeing_hours", 0.0),
            travel_hours=round(day.get("local_travel_hours", 0.0)
                               + ((day.get("transfer") or {}).get("drive_hours") or 0.0), 2),
            items=_items_for(day),
        )
        for day in payload.get("days", [])
    ]
    return ItineraryV1(
        destination=list(request.destination),  # type: ignore[arg-type]
        days=summary.get("days", request.days),
        pace=request.pace,
        nights_per_city=summary.get("nights_per_city", {}),
        total_sights=summary.get("total_sights", 0),
        total_sightseeing_hours=summary.get("total_sightseeing_hours", 0.0),
        total_travel_hours=round(summary.get("total_local_travel_hours", 0.0)
                                 + summary.get("total_intercity_drive_hours", 0.0), 2),
        feasible=bool(validation.get("passed", False)),
        warnings=list(validation.get("warnings", []))[:20],
        not_included=[f"{label}: {reason}"
                      for label, reason in (payload.get("unplaceable_reasons") or {}).items()],
        itinerary=days,
    )


def unused_inputs(request: ItineraryRequestV1) -> dict:
    """What the caller sent that the plan does not depend on, and why.

    Returned on every response so an integrator finds out here, rather than from a customer
    asking why their budget made no difference.
    """
    notes: dict[str, str] = {}
    if request.budget is not None:
        notes["budget"] = ("This engine does no pricing. Only 2 of the 196 past itineraries in "
                           "the dataset carry a total price, too few to estimate from.")
    if request.travellers is not None:
        notes["travellers"] = "The engine plans time and distance, not vehicles or group logistics."
    if request.traveller_type is not None:
        notes["traveller_type"] = ("Measured on the dataset's past itineraries, traveller type does "
                                   "not predict how full a day is (couples 2.63 stops/day, families "
                                   "2.63, groups 2.65; ANOVA p=0.87). Use `pace` instead.")
    return notes


# --- planning -------------------------------------------------------------------------------------
@lru_cache(maxsize=PLAN_CACHE_SIZE)
def _plan_cached(cities: tuple[str, ...], days: int, pace: str,
                 origin: str | None, month: int | None) -> dict:
    """Plan a trip, remembering the result.

    Safe only because planning is deterministic: the seed is fixed, distances come from a
    file, and nothing consults a clock or a random source. If that ever stops being true
    this cache has to go with it.
    """
    return build_itinerary(PlanRequest(
        cities=list(cities),
        duration_days=days,
        pace=pace,
        origin_city=origin,
        start_month=month,
    ))


def plan_cache_info() -> dict:
    stats = _plan_cached.cache_info()
    return {"hits": stats.hits, "misses": stats.misses, "size": stats.currsize,
            "capacity": stats.maxsize}


# --- endpoints ------------------------------------------------------------------------------------
@router.get("/info", summary="What this API is and what it can plan")
def info(_: str = Depends(require_api_key)) -> dict:
    """Version, limits, the cities that can be planned, and what is deliberately not modelled."""
    dataset = get_dataset()
    return ok(
        {
            "name": "Deterministic Itinerary Engine",
            "api_version": API_VERSION,
            "description": "Builds logistically feasible, time-bounded, spatially clustered day "
                           "plans from a structured travel dataset.",
            "capabilities": {
                "plans": ["day-by-day sightseeing", "intercity transfers", "clock schedules",
                          "feasibility validation"],
                "does_not_model": ["pricing and budgets", "flights and ticketing",
                                   "hotel selection", "live availability"],
            },
            "limits": {"max_days": 30, "max_cities_per_trip": 10,
                       "pace_options": ["relaxed", "moderate", "active"]},
            "dataset": {
                "cities_plannable": int(dataset.sights["city_key"].nunique()),
                "cities_known": int(len(dataset.cities)),
                "sights": int(len(dataset.sights)),
                "source": dataset.source.name,
            },
            "endpoints": {
                "health": "/health",
                "info": f"{PREFIX}/info",
                "itinerary": f"{PREFIX}/itinerary",
                "docs": "/docs",
            },
        },
        uptime_seconds=round(time.time() - STARTED_AT, 1),
        plan_cache=plan_cache_info(),
    )


@router.post("/itinerary", summary="Plan a trip")
def create_itinerary(body: ItineraryRequestV1, request: Request,
                     _: str = Depends(require_api_key)) -> dict:
    """Plan a trip and return it as JSON.

    `destination` and `days` drive the plan; `pace` controls how full each day is. The
    remaining fields are recorded and echoed back under `meta.unused_inputs` -- see the
    module docstring for why each one does not change the result.
    """
    started = time.monotonic()
    before = _plan_cached.cache_info().hits
    try:
        plan = _plan_cached(tuple(body.destination), body.days, body.pace,  # type: ignore[arg-type]
                            body.origin_city, body.start_month)
    except PlanningError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except FileNotFoundError as error:  # pragma: no cover - deployment misconfiguration
        raise HTTPException(status_code=503, detail=f"Dataset unavailable: {error}") from error

    return ok(
        project(plan, body).model_dump(),
        unused_inputs=unused_inputs(body),
        echo={"travellers": body.travellers, "budget": body.budget,
              "currency": body.currency, "traveller_type": body.traveller_type},
        took_ms=round((time.monotonic() - started) * 1000),
        cached=_plan_cached.cache_info().hits > before,
        request_id=request.headers.get("x-request-id"),
    )
