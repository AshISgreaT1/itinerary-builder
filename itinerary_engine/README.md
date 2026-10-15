# Deterministic Itinerary Engine

Generates logistically feasible, time-bounded, spatially clustered day plans from
`sample_dataset_v4_clean.xlsx`. No pricing, no flights, no hotel selection — only which
sights happen on which day, in which order.

Given the same dataset and request, the output is always identical: no model calls and a
fixed seed for clustering.

## Install and run

```bash
pip install -r requirements.txt
```

```bash
uvicorn app.main:app --reload
```

The planning **UI is at <http://127.0.0.1:8000/>** and the interactive API docs at
<http://127.0.0.1:8000/docs>. The dataset is read from
`data/sample_dataset_v4_clean.xlsx`; override with `ITINERARY_DATASET=/path/to/file.xlsx`.

Terminal version, handy while tuning the rules:

```bash
python -m app.cli --cities Manali Kasol --days 5 --pace moderate --from Delhi --month 6
```

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Planning UI (pick cities, days, pace, origin, month; edit the result; prints cleanly) |
| POST | `/generate-itinerary` | Build the day-by-day plan (accepts `pinned_sights` / `excluded_sights`) |
| POST | `/itinerary/adjust` | Add, replace or remove sights and rebuild the plan |
| POST | `/itinerary/placement-options` | Which days could hold a sight, and what would have to move |
| GET | `/cities` | Cities that can be planned, with sight counts |
| GET | `/cities/{city}/sights` | A city's canonical sights in priority order |
| GET | `/health` | Dataset loaded, row counts, duplicates merged |

```bash
curl -X POST http://127.0.0.1:8000/generate-itinerary \
  -H "Content-Type: application/json" \
  -d '{"cities":["Manali","Kasol"],"duration_days":5,"pace":"moderate","origin_city":"Delhi","start_month":6}'
```

`origin_city` and `start_month` are optional. The origin makes day 1 a real travel day
(Delhi → Manali is 12 h, so nothing is scheduled), and the month removes seasonal closures
such as Rohtang Pass in January.

## How a plan is built

1. **`data_loader.py`** reads the `cities`, `city_distances`, `sightseeing` and
   `sight_descriptions` sheets, coerces types, and parses `open_months` / `season_open`
   (`all`, `5,6,7`, `Apr-Oct`, and `Oct-Mar` wrapping the new year).
2. **`normalization.py`** merges attractions that appear under several names, using the
   dataset's `aliases` column plus spelling, containment and coordinate checks.
3. **`builder.py`** splits the trip's days between cities by *usable* hours rather than day
   counts, so a city whose only day is eaten by a drive gets a second day before a
   well-covered city gets a fourth.
4. **`clustering.py`** groups each city's sights into one group per day, using `zone` when
   the city has enough distinct zones and KMeans on (lat, lon) otherwise. Day trips far
   outside the city are pulled out first.
5. **`solver.py`** packs each day: a day reserved for a big-ticket sight is built around it,
   then sights are added in `must_visit` → `must_see_tier` → `priority` order while the day
   still fits. Stops are ordered by road distance (nearest-neighbour + 2-opt) and nudged so
   morning sights come first and sunset ones last.
6. **`validation.py`** re-checks the finished plan and returns a report with every response.

### The day budget

```
available = pace allowance − intercity drive − transfer/arrival/departure overheads
used      = Σ duration_hours + 0.3 h per sight + local driving between stops
```

Local driving comes from the real coordinates (straight line × 1.3, at 28 km/h, or 22 km/h
above 1200 m), so a scattered day runs out of hours before a compact one does.

## Travel realism

A day is only as long as what is left of it:

```
available = pace allowance
          − intercity travel (capped at one travel day)
          − transfer / arrival overheads
          − departure day: checkout + transfer to the airport or station + reporting buffer
```

* A journey longer than `MAX_SINGLE_DAY_TRAVEL_HOURS` (10 h) leaves the day with **no**
  sightseeing hours; longer than `MAX_OVERNIGHT_TRAVEL_HOURS` (14 h) fails validation,
  because it cannot be done in one day at all.
* **Journey times are never invented.** If `city_distances` has no row and no connecting
  path (Delhi → Goa, ~1 949 km apart), the leg comes back with `drive_hours: null`, the
  day is held empty, and validation fails with the row that needs adding.
* A dataset row whose implied speed is impossible (say 300 km in 2 h) is recomputed at
  45 km/h and the correction is reported.
* The departure day pays for checkout plus the transfer to the airport or railhead, using
  the distance in the dataset's own `nearest_airport` / `nearest_railhead` cell. When that
  cell has no distance, a clearly labelled assumption is used and listed in `data_gaps`.

## Rules the engine enforces

| Rule | Where | Default |
|---|---|---|
| Daily active hours | `PACE_ACTIVE_HOURS` | 5 / 7 / 9 h |
| Per-sight buffer | `SIGHT_BUFFER_HOURS` | 0.3 h |
| Stops per day | `MAX_SIGHTS_PER_DAY` | 4 / 5 / 6 |
| Distance between consecutive stops | `MAX_LEG_KM` | 30 km |
| Sight-to-sight route per day | `MAX_DAY_ROUTE_KM` | 70 km |
| Day trip (far from the city) | `EXCURSION_KM`, `MAX_STOPS_WITH_EXCURSION` | 35 km, 2 other stops |
| Wrong-city detection | `WRONG_CITY_KM` | 120 km from the city centre |
| Every city gets usable time | `MIN_CITY_USABLE_FRACTION` | half a day |

All of them live in `app/config.py`.

## Validation report

`passed` is true **only when no critical check fails**. Warnings and notes never flip the
gate; every critical finding carries a `fix`. Nothing in the payload is taken on trust:
distances, driving time and hour totals are recomputed from the coordinates and compared
with what the planner claimed.

| Category | Catches |
|---|---|
| `travel_feasibility` | journeys with no time in the dataset, legs too long for a day, sightseeing on a travel day |
| `route_data` | reported metrics that disagree with the coordinates, corrected rows, suboptimal visiting order (warning) |
| `time_and_activity_load` | more hours or stops than the day holds |
| `departure_feasibility` | checkout + airport transfer not reserved, last-day overruns, a day trip on the departure day |
| `hotel_nights` | nights that do not match the trip length or the per-city days |
| `duplicates_and_normalisation` | the same place twice, including under another name |
| `geographic_distance` | stops too far apart, or a day that criss-crosses the region |
| `destination_and_logic` | a sight that belongs to another city, or a day trip crammed in |
| `seasonality` | an **attraction** closed in the travel month |

Seasonality is attraction-level. A city's `season_open` window is its *peak* season, not a
shutdown: Goa in June is wet and quieter but open, so that is a note, never a failure.

When a must-visit sight does not make it in, `unscheduled_reasons` says why, in hours:

```
"Manali / Rohtang Pass": "needs about 7.2h on its own (visit 5.0h + 1.9h driving),
                          but the longest Manali day has 5.0h at relaxed pace."
```

These checks are carried over from the Qwen itinerary validation layer; the difference is
that the distances are now computed from real coordinates instead of a lookup table.

## Editing a plan

A finished itinerary is never patched in place. An edit becomes part of the request -- a
pin ("this sight, optionally on this day") or an exclusion -- and the planner rebuilds
everything, so the route, travel time and day budgets stay consistent after every change.

In the UI, each scheduled sight has **Replace** and **Remove**, and everything left out
appears under **"Not in the plan -- add any of these"**, must-visits first with the reason
they did not fit. **Add to Itinerary** asks the engine where the sight could go and shows a
button per day:

| Button | Meaning |
|---|---|
| *Day 3 · 1.4h spare* | fits as it is |
| *Day 2 · swap out Mall Road* | fits if one stop goes |
| *Day 1 · needs 3 stops dropped* | fits only after clearing most of the day |
| *Day 4 · full* (disabled) | nothing can make room; hover for the reason |
| *Let the planner choose* | the engine picks the best day |

Nothing is ever dropped silently. If a pinned sight will not fit the chosen day it is moved
and the change log says so ("did not fit day 4, so it was scheduled on day 3 instead"); if
no day can hold it, the edit is reported as rejected with the alternatives, for example
*"No day of this trip can hold Rohtang Pass (5.0h plus travel). Try a faster pace or an
extra day in Manali."*

```bash
curl -X POST http://127.0.0.1:8000/itinerary/adjust   -H "Content-Type: application/json"   -d '{"base":{"cities":["Manali"],"duration_days":4,"pace":"active"},
       "edits":[{"action":"replace","sight":"Jogini Waterfall","city":"Manali","day":4,"replaces":"Naggar Castle"}]}'
```

Each edit comes back in `edits[]` with `applied`, `placed_on_day`, a plain-English
`message` and, when it failed, `suggestions`.

## Tests

```bash
python -m pytest tests -q
```

60 tests: dataset loading and month parsing, clustering determinism, pace and distance caps
on real plans, seasonal filtering, day-allocation fairness, the API endpoints, travel
realism (missing route data, overnight legs, travel-only days), departure-day reservations,
hotel-night consistency, the Goa December and June/September cases, fault injection
(a deliberately corrupted plan must fail validation), and the editing API (add, replace,
remove, pinning to a day, and the suggestions when a sight will not fit).

## Layout

```
app/
  config.py         tunable rules and thresholds
  data_loader.py    Excel -> cleaned DataFrames (cached)
  normalization.py  canonical attraction names, duplicate merging
  geo.py            haversine, local travel time, intercity graph (networkx), route order
  clustering.py     zone / KMeans day groups, excursion detection
  solver.py         day budgets and time-packing
  builder.py        orchestration, descriptions join, JSON payload
  validation.py     feasibility report
  edits.py          add / replace / remove, placement options, rebuild
  models.py         pydantic request/response schemas
  main.py           FastAPI app (also serves the UI)
  static/           the planning UI (single page, no build step)
  cli.py            terminal runner
data/               the Excel dataset
tests/              pytest suite
```

## Known limits

- `city_distances` covers 45 of 52 cities and has no long-haul legs (for example Delhi to
  Goa). Missing pairs are routed over connecting roads where possible; otherwise the leg is
  reported as a data gap and the trip fails validation rather than being planned on a
  guessed journey time.
- Trip days are assumed to be full days bounded by arrival and departure overheads; there
  is no clock, so opening hours are not checked (the dataset's `opening_time` columns are
  empty). `best_time` only orders a day, it does not pin a sight to a time.
- Hotels are not selected: every day's local driving is measured from the city centre.
