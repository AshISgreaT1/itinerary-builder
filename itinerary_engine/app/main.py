"""FastAPI application exposing the itinerary engine.

Run with:  uvicorn app.main:app --reload
Docs at:   http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import logging
import os
import threading
import time

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .api_v1 import PREFIX as API_V1_PREFIX, API_VERSION, failed, router as api_v1_router
from .builder import CustomActivity, PinnedSight, PlanningError, PlanRequest, build_itinerary
from .customize import Operation, apply_operations
from .edits import SightEdit, apply_edits, placement_options
from .data_loader import get_dataset
from .datacheck import run_checks
from .models import (ActivitySearchQuery, ActivitySuggestQuery, AdjustRequest, CustomizeRequest,
                     ItineraryRequest, ItineraryResponse, PlacementQuery, PreferencesIn,
                     SuggestRequest)
from .lookup import search, suggestions
from . import preferences
from .normalization import normalize_key
from .personalise import INTEREST_CATEGORIES
from .suggest import suggestions_for

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

@asynccontextmanager
async def _lifespan(_: FastAPI):
    """Load the dataset and plan one small trip, in the background, at startup.

    Nothing here changes what the engine does -- it only moves work off the first
    traveller's request. A cold instance otherwise pays the dataset load and the
    scikit-learn import (about 2.5s on a laptop, more on a small cloud instance) while
    somebody waits, which is exactly when it is most noticeable. Failures are logged and
    ignored: a warm-up that cannot run must never stop the service from starting.
    """
    def warm() -> None:
        try:
            started = time.monotonic()
            dataset = get_dataset()
            city = str(dataset.sights.iloc[0]["city"])
            build_itinerary(PlanRequest(cities=[city], duration_days=2, pace="moderate"), dataset)
            logger.info("Warm-up finished in %.1fs (dataset and planner ready).",
                        time.monotonic() - started)
        except Exception:  # pragma: no cover - never let warm-up break startup
            logger.exception("Warm-up failed; the first request will be slower.")

    if os.environ.get("ITINERARY_SKIP_WARMUP", "").strip() not in {"1", "true", "yes"}:
        threading.Thread(target=warm, name="warm-up", daemon=True).start()
    yield


app = FastAPI(
    lifespan=_lifespan,
    title="Deterministic Itinerary Engine",
    version="1.0.0",
    description=(
        "Builds logistically feasible, time-bounded, spatially clustered day plans from a "
        "structured travel dataset. No pricing, flights or hotel selection.\n\n"
        "**Integrating another system?** Use the `/api/v1` endpoints: a small, versioned "
        "surface with one response envelope and an API key. The other endpoints belong to "
        "this project's own UI and change with it."
    ),
    openapi_tags=[
        {"name": "public API v1",
         "description": "The stable surface for other systems. Send `X-API-Key`; press "
                        "**Authorize** above to try these here."},
        {"name": "planning", "description": "Internal endpoints used by the planner UI."},
        {"name": "catalogue", "description": "What the dataset contains."},
        {"name": "meta", "description": "Health and data quality."},
        {"name": "customers", "description": "Stored preferences, reused on the next trip."},
    ],
)
app.include_router(api_v1_router)


# One envelope for the public API, and only for it: the UI endpoints keep the error shape
# they already have, so adding this API changes nothing for the existing front end.
def _is_public_api(request: Request) -> bool:
    return request.url.path.startswith(API_V1_PREFIX)


@app.exception_handler(RequestValidationError)
async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    if not _is_public_api(request):
        return JSONResponse(status_code=422, content={"detail": exc.errors()})
    details = [
        {"field": ".".join(str(part) for part in error.get("loc", []) if part != "body"),
         "problem": error.get("msg", "")}
        for error in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content=failed("invalid_request", "The request body did not validate.", details),
    )


@app.exception_handler(HTTPException)
async def _http_error(request: Request, exc: HTTPException) -> JSONResponse:
    if not _is_public_api(request):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail},
                            headers=getattr(exc, "headers", None))
    codes = {401: "unauthorized", 403: "forbidden", 404: "not_found",
             422: "unprocessable", 429: "rate_limited", 503: "unavailable"}
    return JSONResponse(
        status_code=exc.status_code,
        content=failed(codes.get(exc.status_code, "error"), str(exc.detail)),
        headers=getattr(exc, "headers", None),
    )


@app.exception_handler(Exception)
async def _unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    # Never leak a traceback to an integrator; the log keeps the detail.
    logger.exception("Unhandled error on %s", request.url.path)
    if not _is_public_api(request):
        raise exc
    return JSONResponse(
        status_code=500,
        content=failed("internal_error", "The engine failed to handle this request."),
    )


STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def ui() -> FileResponse:
    """The planning UI."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", tags=["meta"])
def health() -> dict:
    """Liveness probe that also reports what the dataset contains."""
    try:
        dataset = get_dataset()
    except Exception as exc:  # dataset missing or malformed
        raise HTTPException(status_code=503, detail=f"Dataset unavailable: {exc}") from exc
    import time

    from .api_v1 import STARTED_AT, configured_keys

    return {
        "status": "ok",
        "dataset": dataset.source.name,
        "cities": len(dataset.cities),
        "sights": len(dataset.sights),
        "duplicate_names_merged": len(dataset.dedup.merges),
        # Added for the public API; the keys above are unchanged for existing callers.
        "api_version": API_VERSION,
        "uptime_seconds": round(time.time() - STARTED_AT, 1),
        "api_keys_configured": len(configured_keys()),
    }


@app.get("/cities", tags=["catalogue"])
def list_cities(with_sights_only: bool = Query(True, description="Only cities that have sightseeing rows")) -> dict:
    """Cities the engine can plan for."""
    dataset = get_dataset()
    keys_with_sights = set(dataset.sights["city_key"])
    rows = [
        {
            "city": row["city"],
            "state": row.get("state"),
            "city_type": row.get("city_type"),
            "gateway": bool(row["gateway_flag"]),
            "sights": int((dataset.sights["city_key"] == row["city_key"]).sum()),
        }
        for _, row in dataset.cities.iterrows()
        if not with_sights_only or row["city_key"] in keys_with_sights
    ]
    return {"count": len(rows), "cities": sorted(rows, key=lambda r: r["city"])}


@app.get("/cities/{city}/sights", tags=["catalogue"])
def city_sights(city: str) -> dict:
    """Canonical sights of one city, in priority order."""
    dataset = get_dataset()
    row = dataset.city_row(city)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Unknown city '{city}'.")
    sights = dataset.city_sights(normalize_key(city))
    return {
        "city": row["city"],
        "count": len(sights),
        "sights": [
            {
                "sight": item["sight"],
                "duration_hours": float(item["duration_hours"]),
                "must_visit": bool(item["must_visit_flag"]),
                "must_see_tier": int(item["must_see_tier"]),
                "priority": int(item["priority"]),
                "zone": item["zone"],
                "best_time": item["best_time"],
            }
            for _, item in sights.iterrows()
        ],
    }


def _plan_request(request: ItineraryRequest) -> PlanRequest:
    return PlanRequest(
        cities=request.cities,
        duration_days=request.duration_days,
        pace=request.pace,
        origin_city=request.origin_city,
        start_month=request.start_month,
        include_descriptions=request.include_descriptions,
        pinned_sights=[PinnedSight(sight=p.sight, city=p.city, day=p.day) for p in request.pinned_sights],
        excluded_sights=list(request.excluded_sights),
        city_days=dict(request.city_days),
        extra_day_hours={int(day): float(hours) for day, hours in request.extra_day_hours.items()},
        day_order={int(day): list(order) for day, order in request.day_order.items()},
        sight_times={int(day): dict(times) for day, times in request.sight_times.items()},
        day_starts={int(day): str(at) for day, at in request.day_starts.items()},
        custom_activities=[
            CustomActivity(name=a.name, day=a.day, duration_hours=a.duration_hours, at=a.at, note=a.note)
            for a in request.custom_activities
        ],
        **_customer_bits(request),
    )


def _customer_bits(request: ItineraryRequest) -> dict:
    """Group type and interests, falling back to what the customer told us last time.

    A request that names a customer but says nothing else gets their stored preferences;
    anything stated on the request wins, because this trip may not be like the last one.
    """
    stored = preferences.get(request.customer_id) if request.customer_id else None
    group = request.group_type or (stored.group_type if stored else "")
    interests = list(request.interests) or (list(stored.interests) if stored else [])
    return {"group_type": group or "", "interests": interests,
            "customer_id": request.customer_id or ""}


@app.get("/data-quality", tags=["meta"])
def data_quality(kind: str | None = Query(None, description="Only this finding kind")) -> dict:
    """Dataset problems that distort itineraries, with the evidence and a suggested fix."""
    findings = [finding.__dict__ for finding in run_checks(get_dataset())]
    if kind:
        findings = [finding for finding in findings if finding["kind"] == kind]
    counts: dict[str, int] = {}
    for finding in findings:
        counts[finding["kind"]] = counts.get(finding["kind"], 0) + 1
    return {"count": len(findings), "by_kind": counts, "findings": findings}


@app.post("/generate-itinerary", response_model=ItineraryResponse, tags=["planning"])
def generate_itinerary(request: ItineraryRequest) -> dict:
    """Generate a day-by-day sight sequence for the given cities, days and pace."""
    try:
        payload = build_itinerary(_plan_request(request))
        if request.customer_id:
            preferences.record_trip(request.customer_id)
        return payload
    except PlanningError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/itinerary/adjust", response_model=ItineraryResponse, tags=["planning"])
def adjust_itinerary(request: AdjustRequest) -> dict:
    """Add, replace or remove sights and rebuild the itinerary around the change.

    The whole plan is recalculated, so the route, travel time and daily budgets stay
    consistent. Each edit comes back in ``edits`` saying what happened; an edit that does
    not fit is reported with alternative days instead of being dropped silently.
    """
    try:
        payload, _ = apply_edits(
            get_dataset(),
            _plan_request(request.base),
            [SightEdit(action=e.action, sight=e.sight, city=e.city, day=e.day, replaces=e.replaces) for e in request.edits],
        )
        return payload
    except PlanningError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/customers/{customer_id}/preferences", tags=["customers"])
def read_preferences(customer_id: str) -> dict:
    """What we know about this customer. 404 when they are new."""
    stored = preferences.get(customer_id)
    if stored is None:
        raise HTTPException(status_code=404, detail=f"No preferences stored for '{customer_id}'.")
    return stored.as_dict()


@app.put("/customers/{customer_id}/preferences", tags=["customers"])
def write_preferences(customer_id: str, body: PreferencesIn) -> dict:
    """Create or update a customer's preferences.

    Unknown interests and group types are dropped rather than stored: a preference the
    planner cannot act on is worse than none, because it looks like it is doing something.
    """
    saved = preferences.update(
        customer_id,
        name=body.name or None,
        group_type=body.group_type,
        interests=body.interests or None,
        pace=body.pace,
        notes=body.notes or None,
    )
    return saved.as_dict()


@app.get("/customers", tags=["customers"])
def list_customers() -> dict:
    """Everyone whose preferences we hold."""
    return {"customers": preferences.everyone(), "store": str(preferences.STORE_PATH)}


@app.get("/preferences/options", tags=["customers"])
def preference_options() -> dict:
    """The vocabulary the planner understands, and what each interest covers."""
    return {
        "group_types": list(preferences.GROUP_TYPES),
        "interests": [
            {"interest": name, "dataset_categories": list(categories)}
            for name, categories in INTEREST_CATEGORIES.items()
        ],
        "note": "Interests map onto the sightseeing sheet's own `category` column; nothing "
                "here is invented vocabulary.",
    }


@app.post("/itinerary/customize", tags=["planning"])
def customize_itinerary(request: CustomizeRequest) -> dict:
    """Change a generated plan: move, swap, reorder, retime, lengthen a day, or add your own.

    Each operation is applied to the *request* and the plan is then rebuilt, so the route,
    the clock, the day budgets and the validation report all stay consistent. The response
    is the rebuilt itinerary with ``operations`` saying what each one did, and fresh
    ``suggestions`` for whatever is still worth improving.
    """
    try:
        plan_request = _plan_request(request.base)
        payload, outcomes = apply_operations(
            get_dataset(),
            plan_request,
            [Operation(**op.model_dump()) for op in request.operations],
        )
        payload["operations"] = [outcome.as_dict() for outcome in outcomes]
        payload["suggestions"] = suggestions_for(payload, get_dataset(), plan_request.pace)
        return payload
    except PlanningError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/itinerary/suggestions", tags=["planning"])
def itinerary_suggestions(request: SuggestRequest) -> dict:
    """What could be improved about this plan, each with the call that would do it.

    Nothing is changed. Every suggestion carries an ``action`` that can be sent straight
    back to ``/itinerary/customize``.
    """
    try:
        plan_request = _plan_request(request.base)
        payload = build_itinerary(plan_request)
        return suggestions_for(payload, get_dataset(), plan_request.pace)
    except PlanningError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/itinerary/activity-search", tags=["planning"])
def activity_search(query: ActivitySearchQuery) -> dict:
    """What the user might mean by what they have typed so far.

    Tolerates partial names, typos and phonetic spellings ("bagore haveli", "sajangarh",
    "city plce"), and says how each result matched so the caller can show a list rather
    than guess. Sights come from the dataset only; generic activities (a dinner, a spa) are
    offered alongside them and marked as such.
    """
    try:
        payload = build_itinerary(_plan_request(query.base)) if query.base else None
        cities = query.cities or (list(query.base.cities) if query.base else [])
        return {
            "query": query.text,
            "matches": search(get_dataset(), query.text, cities, payload, limit=query.limit),
        }
    except PlanningError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/itinerary/activity-suggestions", tags=["planning"])
def activity_suggestions(query: ActivitySuggestQuery) -> dict:
    """Things worth adding to this trip, unprompted.

    The sights the plan could not fit come first (most important first), because those are
    what the traveller came for; generic activities follow as starting points.
    """
    try:
        payload = build_itinerary(_plan_request(query.base))
        return suggestions(get_dataset(), list(query.base.cities), payload, query.day, query.limit)
    except PlanningError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/itinerary/placement-options", tags=["planning"])
def sight_placement_options(query: PlacementQuery) -> dict:
    """Which days could hold this sight, and what would have to move to make room."""
    try:
        plan_request = _plan_request(query.base)
        payload = build_itinerary(plan_request)
        return placement_options(payload, get_dataset(), query.sight, query.city, plan_request.pace)
    except PlanningError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
