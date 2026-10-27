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

## Public API (v1)

Another system integrates through `/api/v1`. It is a facade over the same engine the UI
calls -- not a second implementation -- with three differences that exist for the caller:
one response envelope, an API key, and a frozen vocabulary (`destination`, `days`) so
renaming an internal field is not a breaking change.

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | `/health` | none | Liveness, dataset counts, uptime |
| GET | `/api/v1/info` | key | Version, limits, what is and is not modelled |
| POST | `/api/v1/itinerary` | key | Plan a trip |

Open `/docs`, press **Authorize**, paste a key, and both endpoints are testable in the
browser.

### Authentication

Keys come from the `ITINERARY_API_KEYS` environment variable, comma-separated, and are sent
as `X-API-Key`. **With no keys configured the API returns 503 rather than running open** --
a missing setting must never mean "let everyone in". For local work,
`ITINERARY_API_ALLOW_ANONYMOUS=1` turns the check off explicitly.

### Request

```bash
curl -X POST https://your-service.onrender.com/api/v1/itinerary \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{
    "destination": "Jaipur, Udaipur",
    "days": 6,
    "travellers": 2,
    "budget": 60000,
    "currency": "INR",
    "traveller_type": "couple",
    "pace": "moderate",
    "start_month": 11
  }'
```

`destination` takes one city, a comma-separated list, or a JSON array. `days` is 1-30 and
`pace` is `relaxed` / `moderate` / `active` (5 / 7 / 9 sightseeing hours a day).
`origin_city` charges day 1 with the journey in; `start_month` applies seasonal closures.

### Response

```json
{
  "success": true,
  "data": {
    "destination": ["Jaipur", "Udaipur"],
    "days": 6,
    "pace": "moderate",
    "nights_per_city": {"Jaipur": 4, "Udaipur": 1},
    "total_sights": 16,
    "total_sightseeing_hours": 18.25,
    "total_travel_hours": 13.69,
    "feasible": true,
    "warnings": [],
    "not_included": [],
    "itinerary": [
      {
        "day": 1, "city": "Jaipur", "is_arrival_day": true, "transfer": null,
        "starts": "09:00", "ends": "15:18",
        "sightseeing_hours": 4.0, "travel_hours": 1.3,
        "items": [
          {"start": "09:00", "end": "09:30", "kind": "check-in",
           "title": "Arrival and hotel check-in, Jaipur", "description": "", "duration_hours": 0.5},
          {"start": "09:54", "end": "10:09", "kind": "sight", "title": "Lassiwala",
           "description": "A long-running lassi shop on MI Road\u2026",
           "duration_hours": 0.25, "best_time": "any", "must_visit": false}
        ]
      }
    ]
  },
  "meta": {
    "api_version": "1.0.0",
    "unused_inputs": {
      "budget": "This engine does no pricing\u2026",
      "travellers": "The engine plans time and distance, not vehicles or group logistics.",
      "traveller_type": "\u2026does not predict how full a day is (ANOVA p=0.87). Use `pace` instead."
    },
    "echo": {"travellers": 2, "budget": 60000.0, "currency": "INR", "traveller_type": "couple"},
    "took_ms": 4109,
    "cached": false
  }
}
```

Errors use the same envelope, so a client parses one shape:

```json
{
  "success": false,
  "error": {
    "code": "invalid_request",
    "message": "The request body did not validate.",
    "details": [{"field": "days", "problem": "Input should be greater than or equal to 1"}]
  },
  "meta": {"api_version": "1.0.0"}
}
```

`invalid_request` (422), `unauthorized` (401), `unprocessable` (422, e.g. an unknown city --
the message lists the known ones), `unavailable` (503), `internal_error` (500). **The
envelope applies to `/api/v1` only**; the UI endpoints keep the shapes they already had, and
there is a test holding that line.

### What the engine does not use

`budget`, `travellers` and `traveller_type` are accepted, echoed back, and reported in
`meta.unused_inputs` with the reason. They are not quietly ignored, because an integrator who
believes `budget` did something is a support ticket waiting to happen:

| Field | Why it does not change the plan |
|---|---|
| `budget` | The engine does no pricing. Only **2 of 196** past itineraries in the dataset carry a total price -- far too few to estimate from. |
| `traveller_type` | Measured against those same itineraries it does not predict trip intensity: couples **2.63** stops a day, families **2.63**, groups **2.65** (ANOVA p=0.87). A mapping to pace would be invention, so `pace` is exposed directly. |
| `travellers` | The engine models time and distance, not vehicles or group logistics. |

A test asserts that four traveller types produce byte-identical plans, so if that ever stops
being true the documentation has to change with the code.

### Performance

A cold plan takes roughly 2-5 seconds; the engine clusters, routes, schedules, validates and
re-plans around time-of-day conflicts. Because planning is deterministic, identical requests
are served from an in-process cache (`meta.cached`) in about a millisecond. The cache keys on
everything that changes a plan -- cities, days, pace, origin, month.

## Deploying on Render

`render.yaml` is a blueprint: **New -> Blueprint** in Render, point it at this repository, and
it creates the service. By hand instead:

| Setting | Value |
|---|---|
| Root directory | `itinerary_engine` |
| Build command | `pip install -r requirements.txt` |
| Start command | `uvicorn app.main:app --host 0.0.0.0 --port $PORT` |
| Health check path | `/health` |
| Environment | `PYTHON_VERSION=3.12.3`, `ITINERARY_API_KEYS=<your keys>`, `WEB_CONCURRENCY=1` |

The dataset and the measured-distance cache are committed (about 770 KB together), so the
service needs no disk, database or network at runtime. One worker per instance: planning is
CPU-bound and each worker holds its own copy of the dataset, so scale with instances.

Smoke-test a deployment:

```bash
curl https://your-service.onrender.com/health
curl -H "X-API-Key: your-key" https://your-service.onrender.com/api/v1/info
```

On Render's free plan the service sleeps when idle and the first request afterwards pays the
dataset load as well as the plan, so allow ten seconds or so for a cold start.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Planning UI (pick cities, days, pace, origin, month; edit the result; prints cleanly) |
| POST | `/generate-itinerary` | Build the day-by-day plan (accepts `pinned_sights` / `excluded_sights`) |
| POST | `/itinerary/adjust` | Add, replace or remove sights and rebuild the plan |
| POST | `/itinerary/customize` | Move, swap, reorder, retime, lengthen a day, re-split the trip, add your own activity |
| POST | `/itinerary/suggestions` | What could be improved, each with the call that would do it |
| POST | `/itinerary/activity-search` | Find a sight from a partial or misspelt name |
| POST | `/itinerary/activity-suggestions` | What is worth adding to this trip, unprompted |
| POST | `/itinerary/placement-options` | Which days could hold a sight, and what would have to move |
| GET | `/cities` | Cities that can be planned, with sight counts |
| GET | `/cities/{city}/sights` | A city's canonical sights in priority order |
| GET | `/health` | Dataset loaded, row counts, duplicates merged |
| GET | `/data-quality` | Dataset problems that distort itineraries, with suggested fixes |

```bash
curl -X POST http://127.0.0.1:8000/generate-itinerary \
  -H "Content-Type: application/json" \
  -d '{"cities":["Manali","Kasol"],"duration_days":5,"pace":"moderate","origin_city":"Delhi","start_month":6}'
```

```bash
curl -X POST http://127.0.0.1:8000/itinerary/customize   -H "Content-Type: application/json"   -d '{"base":{"cities":["Jaipur","Udaipur"],"duration_days":6},
       "operations":[{"operation":"reallocate","city_days":{"Jaipur":3,"Udaipur":3}},
                     {"operation":"extend_day","day":2,"hours":1.5},
                     {"operation":"add_activity","name":"Dinner at Ambrai","day":2,"at":"20:00"}]}'
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

## Road distances: measured, not estimated

By default the engine approximates road distance as a straight line inflated by
`ROAD_DISTANCE_FACTOR` (1.3). Measured against OSRM over this dataset's 6 106 comparable
pairs, that factor is **systematically low**:

| | |
| --- | --- |
| median measured / estimated | **1.14** (the real detour factor is 1.48, not 1.30) |
| estimate too low | 56% of pairs |
| estimate too high | 4% of pairs |
| worst case | 3.9 km estimated, **63.2 km** by road |

So the engine prefers a measured distance wherever it has one. `app/osrm.py` fetches the
whole matrix once from [OSRM](https://project-osrm.org/) into `data/road_distances.json`,
and `road_km()` reads it from there.

**Building the cache.** OSRM is open source and self-hosted, so there is no key, no quota
and no network dependency at planning time:

```bash
# one-off: build the routing graph (needs ~8 GB RAM for an India extract)
wget https://download.geofabrik.de/asia/india-latest.osm.pbf
docker run -t -v "${PWD}:/data" ghcr.io/project-osrm/osrm-backend     osrm-extract -p /opt/car.lua /data/india-latest.osm.pbf
docker run -t -v "${PWD}:/data" ghcr.io/project-osrm/osrm-backend     osrm-partition /data/india-latest.osrm
docker run -t -v "${PWD}:/data" ghcr.io/project-osrm/osrm-backend     osrm-customize /data/india-latest.osrm
docker run -p 5000:5000 -v "${PWD}:/data" ghcr.io/project-osrm/osrm-backend     osrm-routed --algorithm mld --max-table-size 200 /data/india-latest.osrm
```

```bash
python -m app.osrm --build --url http://localhost:5000
```

That is 33 table requests for this dataset -- one per city (its sights plus the centre) and
one for the 52 city centres -- and writes about 6 300 measured pairs. OSRM's public demo
server (`--url https://router.project-osrm.org`) works for a one-off build and is what the
shipped cache was built with, but it is rate-limited and not meant for production, so the
builder paces itself at one request a second against it.

**What is cached and what is not.** Distances are cached and used. Durations are cached but
**not** used: OSRM puts Manali to Rohtang Pass at 1.0 h for 61 km of mountain road, where
the engine's altitude-aware model says 2.8 h, which is far closer to reality. Speed stays
with the engine; distance comes from OSRM.

**Nothing is faked.** A pair the router could not route is simply left out of the cache and
keeps the straight-line estimate, and every plan reports the split in
`summary.road_distances` (`measured_legs`, `estimated_legs`, and which legs are still
estimates). Delete the cache file and the engine behaves exactly as it did before.

**Measurement finds bad coordinates.** How far OSRM has to move a point to reach a road is
itself a data check: 16 coordinates in this dataset sit more than 2 km from any road, and
the worst -- Leh / Nakeela at 13.3 km -- is rounded to two decimal places in a region where
that matters. They are reported by `python -m app.datacheck --kind coordinate_far_from_road`.

## Grouping nearby stops

Clustering decides which sights share a day before any day has a clock. The time-of-day
groups inside a day then stretch the route -- a day runs morning stops, then afternoon, then
evening, which is right for the traveller and wrong for the odometer. Measured across 88
days, 17% drove more than two kilometres further than their own stops allowed.

`refine_grouping()` sweeps each city's days again afterwards. For every pair of days it tries
each single move and each pairwise swap, keeping a change only when the driving gets shorter
*and* both days still pass every rule -- budget, stop cap, leg limits, evening capacity,
arrival-day distance. Feasibility is checked with `evaluate_day()`, the same function the
packer and the edit API use, so the refinement cannot invent a day the rest of the engine
would reject.

| | before | after |
| --- | --- | --- |
| driving hours per sight | 0.391 | **0.377** |
| days driving well over their best order | 17% | 15% |
| sights scheduled | 315 | 315 |

Two things are held back from it. A **pinned** sight never moves -- the refinement optimises,
it does not overrule the user. A **must-visit on the arrival day** never moves either: day 1
carries the city's headline sights on purpose, and a kilometre of driving is not worth
trading that away.

The search is pruned geometrically: a stop is only tested against another day when it is
nearer that day's stops (within `REFINE_MARGIN_KM`). Without the prune the same result costs
six times the runtime; with it the whole pass is about 0.2s across three plans.

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

## The timed day

Every day comes back with a clock schedule: travel, visits, lunch, a short break and
waiting, laid out from `09:00` (`08:00` on driving days) and finishing by `20:00`.

**A day starts when its sights ask it to.** Udaipur's lakes are evening sights, and laying
such a day out from 09:00 put a sunset boat ride at 12:37 while the evening stood empty. So
the day is laid out once and, if a hard slot was missed only because the day began too
early, laid out again from later in the morning -- the smallest shift that rescues a stop
without running past `20:00`. The shift is stated on the day ("the day starts at 13:25 ...").

```
09:00 – 09:29  Set off for City Palace      0.5h including parking and tickets
09:29 – 11:29  City Palace                  morning visit
11:29 – 11:47  Drive to Jantar Mantar
11:47 – 12:47  Jantar Mantar                morning visit
12:47 – 13:47  Lunch
...
15:34 – 16:30  Free time before Birla Temple   it is best seen from 16:30
16:30 – 17:30  Birla Temple                 evening visit
```

Times are derived, never invented: visit lengths come from `duration_hours`, driving from
the coordinates, intercity legs from `city_distances`. Day start, lunch and break lengths
are the engine's own assumption, stated in `summary.schedule_assumptions` on every plan.
Opening hours are read from the workbook's `opening_time` / `closing_time` columns and, where
present, narrow a sight's window; **in the shipped dataset all 254 rows are empty**, so the
plan says so rather than assuming a door is open.

**The totals are the schedule.** A day's reported hours are read back off its own blocks, so
the header cannot drift from the timeline:

| field | what it holds |
| --- | --- |
| `schedule.totals.sightseeing` | time inside the sights |
| `schedule.totals.local_travel` | driving between them, hotel to hotel |
| `schedule.totals.stop_buffers` | parking and tickets, `0.3h` a stop |
| `schedule.totals.intercity_travel` | the day's transfer, from `city_distances` |
| `schedule.totals.meals_and_breaks` | lunch and the afternoon break |
| `schedule.totals.waiting` | time held for a slot that has not opened yet |
| `schedule.activity_hours` | sightseeing + local travel + buffers: what the pace budgets |
| `schedule.span_hours` | first block to last: how long the day really is |

`accounted_hours` is the sum of the totals and must equal `span_hours`; `day.used_hours`
must equal `activity_hours`. Both are **critical** validation checks, so a day whose header
disagrees with its own clock fails the plan rather than being shown.

**Hard slots: evening and sunset.** These are constraints, not preferences. A sight marked
`evening` is visited between **16:30 and 20:00**, `sunset` between **16:30 and 19:30**, or it
is not visited at all. There is no "close enough" and no grace: a plan that starts one before
16:30 fails validation. Four things enforce it, in order:

1. **The packer refuses an impossible booking.** `hard_slot_blocker()` checks that the day's
   evening sights fit the window at all -- three of them do not fit 3.5 hours, and a departure
   day that owes four hours to the onward journey has no evening to give.
2. **The day starts later.** Rather than see a sunset lake at noon, the day is laid out again
   from later in the morning -- the smallest shift that rescues a stop without running past
   20:00. Udaipur day 1 begins at 12:44 so Lake Pichola lands at 17:30.
3. **A flexible stop fills the wait.** `_pull_flexible_forward()` moves an `any` sight into the
   gap before an evening one, so the wait becomes a visit. Only genuinely flexible sights move:
   dragging a morning sight into a late gap would just trade one misplacement for another.
4. **Otherwise the sight leaves the day.** The clock never places it at the wrong hour; it goes
   into `schedule.deferred`, and the builder re-plans to put it on another day whose evening is
   free. If no day can take it, it is dropped from the trip with its reason in
   `unplaceable_reasons`, still offered in `unscheduled_sights`, and shown in the UI under
   "Kept for another trip". A sight left deferred in a finished plan is a critical failure.

Across 36 plans in 12 cities this produces **zero** hard-slot violations, with six sights held
back and explained. `morning` and `afternoon` are preferences --
nearly half the dataset says "morning", so no day could honour them all -- and are only
reported when a visit lands in the wrong half of the day. A sight with no `best_time` is
placed in day order and flagged, never given an invented time.

## What the past itineraries teach

The workbook's `itineraries` sheet holds 1 241 agent-written days. Two things decide how
far it is trusted:

* **It carries no clock times at all**, so no time of day is ever taken from it. Every plan
  says so in its notes.
* **Its order does not follow `best_time`** -- averaged over the sheet, "morning" sights sit
  at position 0.52 of a day and "evening" ones at 0.54.

What it does give is pairwise ordering, and that is strong: Shanti Stupa before Leh Palace
27 times against 6, Echo Point before Mount Carmel 23 against 0. The planner uses those
habits **only to settle ties inside one time slot, and only when the swap costs no extra
driving**. Where the plan still differs it says why, for example *"past itineraries usually
visit 'Echo point' before 'Kundala Lake' (11 times against 0); this plan keeps the shorter
driving order instead"*.

**Day 1 carries the main sights.** In the agent itineraries **85.3%** of day-1 stops are
must-visits, against roughly 47% on every later day (mean priority 3.7 against 5-7.5). The
first day is when the famous sights are seen, not what is left over -- so the arrival day now
claims the cluster richest in must-visits, instead of choosing last because it has the fewest
hours. That moved the engine's own day-1 must-visit share from **36.8% to 70.7%**, and Udaipur
day 1 now opens with City Palace and Lake Pichola.

**Day 1 stays in town.** The arrival day is short and the traveller has just spent it
travelling, so no distant destination is scheduled on it. The limit is measured from the
same sheet rather than chosen:

| | day 1 | later days |
|---|---|---|
| median distance from the city centre | 4.0 km | 8.5 – 11.7 km |
| 90th percentile | 23.2 km | 27.9 km and up |
| stops 35 km or more out | **1.0%** (3 of 286) | 18.3% |

`ARRIVAL_DAY_MAX_KM` is therefore 25 km, which keeps 93% of what the agents actually did on
day 1. A sight over the limit is **deferred, not dropped**: the packer leaves it for a later
day (Jaisalmer's Kuldhara, 30 km out, moves to day 2), and if no day can take it, it stays
in `unscheduled_sights` where the UI offers it. An explicit pin outranks the rule -- if you
ask for Kuldhara on day 1 you get it, with a warning saying it is past the usual limit.

## Stops on the way, and days that are only travel

**En-route stops.** A sight you simply drive past does not belong to either city: Chang La
sits 1.4 km off the Leh to Pangong Lake road, a third of the way along, but 77 km from Leh
and 95 km from the lake, so neither city could ever host it. Such a sight is attached to the
transfer day and charged only the detour off the road plus the stop itself. It is taken en
route only when **no** city on the trip could reasonably hold it, so Thiksey (19 km from
Leh) stays a Leh day trip. Controlled by `EN_ROUTE_MAX_DETOUR_KM` (15 km) and
`EN_ROUTE_MAX_STOPS` (3); each stop reports `detour_km` and `along_route_percent`.

**Travel-only days stretch.** A day given over entirely to driving has no sightseeing to
pace, so it may start early and run to `TRAVEL_DAY_MAX_DRIVE_HOURS` (13 h) instead of the
pace allowance. Delhi to Manali (12 h) is therefore one long travel day rather than an
impossible one. Past 13 h it needs an overnight bus or train, and past
`MAX_OVERNIGHT_TRAVEL_HOURS` (14 h) validation fails.

## Rules the engine enforces

| Rule | Where | Default |
|---|---|---|
| Road distance source | `ROAD_CACHE_PATH`, `ROAD_DISTANCE_FACTOR` | measured if cached, else straight line x 1.3 |
| Daily active hours | `PACE_ACTIVE_HOURS` | 5 / 7 / 9 h |
| Per-sight buffer | `SIGHT_BUFFER_HOURS` | 0.3 h |
| Stops per day | `MAX_SIGHTS_PER_DAY` | 4 / 5 / 6 |
| Distance between consecutive stops | `MAX_LEG_KM` | 30 km |
| Sight-to-sight route per day | `MAX_DAY_ROUTE_KM` | 70 km |
| Day trip (far from the city) | `EXCURSION_KM`, `MAX_STOPS_WITH_EXCURSION` | 35 km, 2 other stops |
| Arrival day stays in town | `ARRIVAL_DAY_MAX_KM` | 25 km from the centre |
| Hard time slots | `HARD_TIME_SLOTS`, `TIME_WINDOWS` | evening 16:30-20:00, sunset 16:30-19:30 |
| Re-homing a missed slot | `MAX_SLOT_ROUNDS`, `MAX_SLOT_REPAIRS` | 6 rounds, 2 target days each |
| Regrouping days to cut driving | `REFINE_PASSES`, `REFINE_MARGIN_KM` | 2 sweeps, 2 km |
| Stop made during a drive | `EN_ROUTE_MAX_DETOUR_KM`, `EN_ROUTE_MAX_STOPS` | 15 km off the road, 3 stops |
| Longest pure travel day | `TRAVEL_DAY_MAX_DRIVE_HOURS` | 13 h driving |
| Wrong-city detection | `WRONG_CITY_KM` | 120 km from the city centre |
| Every city gets usable time | `MIN_CITY_USABLE_FRACTION` | half a day |

All of them live in `app/config.py`.

## Changing a plan after it is generated

A generated plan is a starting point. `POST /itinerary/customize` takes the original request
plus a list of operations and rebuilds from them -- every change goes back through the whole
pipeline, because a payload patched in place is exactly how a header starts disagreeing with
its own timeline.

| Operation | What it does |
| --- | --- |
| `move` | put a stop on another day |
| `swap` | exchange two stops between days |
| `reorder` | set a day's order outright (what a drag ends with) |
| `retime` | ask for a stop at a particular time |
| `to_morning` | push a flexible stop to the front of its day |
| `extend_day` | lengthen one day beyond the pace (`hours: 0` undoes it) |
| `reallocate` | change the split of days between cities |
| `pin` / `drop` | add or remove a sight |
| `add_activity` | add something of your own -- a dinner, a show |
| `swap_days` | exchange two whole days |
| `set_day_start` | start a day earlier or later than 09:00 |
| `set_pace` | change the pace of the whole trip without starting again |

**All of it is reachable after generating.** The plan comes back with a *Customise this trip*
panel: one click moves a day between cities (the total is fixed, so every click is a valid
split), another changes the pace. Each day carries its own controls -- lengthen it, set when it
starts, add your own activity -- and each stop can be dragged to another day, swapped with a
stop on another day, replaced or removed. Nothing has to be re-entered.

**Every operation is checked against its own intent.** A stop dragged onto a full day does
not quietly disappear: the change is rolled back and the response says what would make room
-- *"'Jagdish temple' could not be placed on day 1 ... Day 1 has 0.1h spare of 5.0h. Add
hours to it (extend_day), or swap with one of 'City Palace', 'Lake Pichola'."* Run
`extend_day` first and the same move succeeds.

**A manual move moves one thing.** `move` and `swap` pin every other stop where it already
is, so re-planning around a drag re-costs the route and the clock without rearranging four
days the user never touched.

**Longer days, in both directions.** `extra_day_hours` raises a day's budget and, with it,
the stop cap -- one more stop per `EXTRA_HOURS_PER_STOP` (2h). Validation allows the longer
day and notes it rather than failing; the 20:00 clock limit still applies, so the extra hours
buy sights, not miracles.

**Rearranging never shrinks the trip.** Beyond each operation's own intent, one invariant
holds across all of them: a change that would drop a must-visit, or leave the trip with fewer
stops than it had, is undone and explained. Trading two stops for three is a better trip and
is allowed; losing one quietly is not. Changing the *shape* of the trip -- the pace, the city
split, a day's length -- is exempt, because changing what fits is the point.

**Your own plans are time on the clock.** A custom activity has no coordinates, so it costs
no travel -- it is simply time the day owes to something, counted in
`schedule.totals.your_own_plans` and never against the sightseeing pace.

## Suggestions, not just complaints

`POST /itinerary/suggestions` returns what could be improved, each with the call that would
do it. The same list comes back on every `customize` response, so acting on one suggestion
shows you what is left.

| Suggestion | Raised when |
| --- | --- |
| `retime` | a hard-slot sight is at the wrong hour **and that day still has evening room** |
| `move_day` | it does not, but another day's evening does |
| `free_evening` | a flexible stop is occupying evening space while an evening sight waits |
| `extend_day` | a day is full while must-visit sights did not fit, and it ends before 20:00 |
| `reallocate` | one city is starved of must-visits while another has hours to spare |
| `add` | a sight the engine held back, and what pinning it would mean |

A `retime` is only offered when it can actually be taken: suggesting 16:30 to two sights at
once is advice nobody can act on, so the engine checks the evening's remaining room first and
falls through to `move_day`.

## Adding something: search that forgives typing

"Add your own activity" searches the dataset rather than taking a name on trust. `app/lookup.py`
matches in four widening steps and says which one hit, so the caller shows a list instead of
guessing:

| How | Example |
| --- | --- |
| `exact` | `city palace` -> City Palace |
| `contains` | `bagore haveli` -> Bagore ki Haveli |
| `fuzzy` | `sajangarh` -> Sajjangarh fort, `city plce` -> City Palace |
| `sounds-like` | same consonant skeleton, for a phonetic spelling |

Generosity has a floor. Nonsense matches nothing rather than returning the least-bad row, and
phonetic matching needs at least three consonants -- `lake` once matched *Bhartiya Lok Kala
Mandal* on a two-consonant skeleton.

A **Suggested** button lists what is worth adding without typing: the must-visit sights the
plan could not fit come first, each carrying the reason it was held back, then the generic
activities (a dinner, a spa, a morning walk). Picking a dataset sight adds it as a real stop,
routed and timed; only something that matches nothing becomes a free-text note on the day.

## Two kinds of validation

The hard-slot rule is enforced against **the engine** and advisory for **the user**:

* A plan the engine produced on its own never puts an evening sight outside 16:30-20:00. If
  it cannot honour the slot it re-homes the sight or leaves it out with a reason, and a
  violation is a **critical** failure.
* A clash the user chose -- pinning three evening sights to one day, or reordering a day so a
  sunset lands at noon -- is kept, reported as a **warning**, and paired with the better time
  in `schedule.conflicts`. Being locked out of your own itinerary helps nobody.

The difference is `user_pinned` on the stop, set by a pin, a hand-set order or a retime.

## Validation report

`passed` is true **only when no critical check fails**. Warnings and notes never flip the
gate; every critical finding carries a `fix`. Nothing in the payload is taken on trust:
distances, driving time and hour totals are recomputed from the coordinates and compared
with what the planner claimed.

| Category | Catches |
|---|---|
| `schedule` | overlapping blocks, a sight scheduled twice, **an evening or sunset sight outside 16:30-20:00**, a sight left off the clock, days running past 20:00 |
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

## The printed itinerary

Printing produces a separate customer-facing document, built from the plan -- not the app with
parts hidden by CSS. That distinction is the point: hiding the screen UI rule by rule means
every panel added later leaks onto the customer's printout. `@media print` hides
`.wrap > *:not(#print-doc)` wholesale and shows the document instead, so new screen features
are invisible to the printer by default.

The document carries the trip title and length, then each day: the day number, city, the
transfer when there is one, and a time / activity table with a one-sentence description from
the dataset. It ends with what the itinerary covers and what it does not. Budgets, validation
reports, suggestions, drag handles and every other working detail stay on screen.

**Only what the engine can stand behind is listed as included.** Sightseeing as planned, local
travel, the intercity transfers with their mode -- all facts from the plan. The engine models
no hotels, meals or prices, so those are listed as *not* included rather than promised; a
printed promise the engine cannot keep is worse than a blank line. There is a test for this.

Times round to the nearest five minutes. Rounding starts down and ends up was the first
attempt and made consecutive rows overlap on paper (`12:50` then `12:45`), which reads as a
mistake even when the plan underneath is sound; rounding every instant the same way keeps one
row's end exactly the next row's start. Page setup is A4 with 14mm margins and no day split
across a page break.

## Dataset quality report

The planner is only as sensible as the rows it reads, so the same rules run against the
dataset itself:

```bash
python -m app.datacheck                    # readable report
python -m app.datacheck --csv findings.csv # one row per finding
curl http://127.0.0.1:8000/data-quality    # same thing as JSON
```

| Finding | Meaning |
|---|---|
| `far_from_city` | the sight sits far from the city it is filed under, with the nearest city suggested |
| `cannot_fit_any_day` | visit + driving exceeds the longest possible day, so no trip can ever include it |
| `no_route_data` | two cities the `itineraries` sheet sells together have no journey time |
| `road_only_long_haul` | the only route is a drive longer than a travel day |
| `sight_named_after_city` | cosmetic: "Pangong Lake / Pangong Lake" |

Nothing is changed and nothing is guessed: every suggestion comes from coordinates and rows
already in the workbook. On `sample_dataset_v4_clean.xlsx` it reports 69 findings, including
"Leh / Pangong Lake: 141 km from the Leh centre -> re-home to Pangong Lake (0 km away)".

## Tests

```bash
python -m pytest tests -q
```

285 tests: dataset loading and month parsing, clustering determinism, pace and distance caps
on real plans, seasonal filtering, day-allocation fairness, the API endpoints, travel
realism (missing route data, overnight legs, travel-only days), departure-day reservations,
hotel-night consistency, the Goa December and June/September cases, fault injection
(a deliberately corrupted plan must fail validation), and the editing API (add, replace,
remove, pinning to a day, and the suggestions when a sight will not fit), and the dataset
quality report, en-route stops, travel-only days, clock schedules, preferred visiting
times, the patterns mined from past itineraries, and the Udaipur regressions (day totals
reconciling with the clock on four city/pace combinations, meals counted outside the
sightseeing budget, en-route detours charged to the day rather than to the drive, evening
sights landing in the evening after a late start, every hard-slot clash carrying an
alternative, an unknown pinned sight being refused instead of dropped, and the API not
stripping the totals out of the response), and the arrival-day rule (six city/pace
combinations keep day 1 within the limit, a displaced sight moves to a later day rather than
vanishing, an explicit pin still wins, and the limit is re-derived from the workbook so it
stays honest if the data changes), and the OSRM layer (the client against a stub server,
unroutable pairs keeping the estimate, distant snaps being reported, the cache preferring
the measured direction, a missing cache file being harmless, planning making no network
call, and two runs of one request producing identical plans), the hard slots (eight
city/pace combinations where no evening or sunset sight starts before 16:30, nothing is left
deferred, only flexible sights are pulled into a wait, the packer refusing a departure-day
evening, and the validator rejecting both a hand-broken morning placement and a deferred
sight), and day 1 (must-visit share, staying in town, and the 85%/47% split re-derived from
the workbook), and customisation (day splits that do and do not add up, lengthened days and
their stop cap, a move that cannot fit being rolled back rather than losing the sight, a
manual move leaving other days alone, swaps, reorders, custom activities landing on the clock
and in the totals, a user's slot clash not blocking the plan, the advised time matching where
the sight actually is, suggestions carrying actions that apply cleanly, and both endpoints), and the trip-wide controls (pace changed after the fact, a day started
early, two days swapped wholesale, a cross-city day swap refused with the reason, and the
no-shrinking invariant held across move, swap and add), grouping (refined days still obey
every limit, no hard slot leaves its window, pinned sights and day-1 must-visits are left
alone, and refining is measured against the same plan unrefined rather than an absolute bar),
typo-tolerant search (nine spellings of real sights, nonsense matching nothing, short words
not matching by sound, and both endpoints), and the printed document (A4 setup, page breaks,
the app chrome hidden wholesale, consistent time rounding, and nothing promised that the
engine does not model), and the public API (the envelope, a key being required, failing
closed when none is configured, every input validated, the three shapes `destination`
accepts, unused inputs reported rather than ignored, traveller type provably not changing
the plan while pace does, the cache keying on everything that matters, `/docs` advertising
the key, and the UI endpoints keeping their own response shapes).

## Layout

```
app/
  config.py         tunable rules and thresholds
  api_v1.py         the public REST facade: envelope, API key, frozen vocabulary
  customize.py      post-generation operations (move, swap, extend, reallocate, ...)
  lookup.py         typo-tolerant search over the dataset, and what to suggest
  suggest.py        what to do about a plan, as actions rather than complaints
  osrm.py           builds the measured-distance cache (run once, needs OSRM)
  roadcache.py      reads it; the planner never calls the network
  data_loader.py    Excel -> cleaned DataFrames (cached)
  normalization.py  canonical attraction names, duplicate merging
  geo.py            haversine, local travel time, intercity graph (networkx), route order
  clustering.py     zone / KMeans day groups, excursion detection
  solver.py         day budgets and time-packing
  builder.py        orchestration, descriptions join, JSON payload
  validation.py     feasibility report
  edits.py          add / replace / remove, placement options, rebuild
  datacheck.py      dataset quality report (CLI + /data-quality)
  enroute.py        stops made on the way between two cities
  schedule.py       the clock: blocks, preferred times, lunch and breaks
  patterns.py       ordering habits mined from past itineraries
  models.py         pydantic request/response schemas
  main.py           FastAPI app (also serves the UI)
  static/           the planning UI (single page, no build step)
  cli.py            terminal runner
data/               the Excel dataset
tests/              pytest suite
```

## Known limits

- OSM road coverage is thinner in Ladakh and rural Rajasthan than in the cities, and OSRM
  knows nothing about seasonal closures (Khardung La, Rohtang). A measured distance is still
  a fair-weather one.
- `city_distances` covers 45 of 52 cities and has no long-haul legs (for example Delhi to
  Goa). Missing pairs are routed over connecting roads where possible; otherwise the leg is
  reported as a data gap and the trip fails validation rather than being planned on a
  guessed journey time.
- Opening hours are honoured when the dataset carries them, but the shipped workbook's
  `opening_time` / `closing_time` columns are empty for all 254 sights, so in practice only
  `best_time` constrains the clock. No door is ever assumed to be open or shut.
- The repair pass re-plans at most `MAX_SLOT_REPAIRS` (2) times per plan, so a day with
  three unreachable evening sights may still report one by alternative rather than moving it.
- Hotels are not selected: every day's local driving is measured from the city centre.
