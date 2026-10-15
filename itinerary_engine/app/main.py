"""FastAPI application exposing the itinerary engine.

Run with:  uvicorn app.main:app --reload
Docs at:   http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import logging

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .builder import PinnedSight, PlanningError, PlanRequest, build_itinerary
from .edits import SightEdit, apply_edits, placement_options
from .data_loader import get_dataset
from .datacheck import run_checks
from .models import AdjustRequest, ItineraryRequest, ItineraryResponse, PlacementQuery
from .normalization import normalize_key

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

app = FastAPI(
    title="Deterministic Itinerary Engine",
    version="1.0.0",
    description=(
        "Builds logistically feasible, time-bounded, spatially clustered day plans from a "
        "structured travel dataset. No pricing, flights or hotel selection."
    ),
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
    return {
        "status": "ok",
        "dataset": dataset.source.name,
        "cities": len(dataset.cities),
        "sights": len(dataset.sights),
        "duplicate_names_merged": len(dataset.dedup.merges),
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
    )


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
        return build_itinerary(_plan_request(request))
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


@app.post("/itinerary/placement-options", tags=["planning"])
def sight_placement_options(query: PlacementQuery) -> dict:
    """Which days could hold this sight, and what would have to move to make room."""
    try:
        plan_request = _plan_request(query.base)
        payload = build_itinerary(plan_request)
        return placement_options(payload, get_dataset(), query.sight, query.city, plan_request.pace)
    except PlanningError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
