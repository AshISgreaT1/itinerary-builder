# How the engine works

Four views of the same system: where a request goes, how a plan is built, what the data
layer feeds it, and what happens after a plan exists. The diagrams are Mermaid, so they
render on GitHub and in most editors, and they can be edited as text.

Module names are given as `app/<file>.py` so each box can be found in the code.

## 1. A request, end to end

```mermaid
flowchart TD
    UI["Browser UI<br/><i>app/static/index.html</i>"] --> REQ
    API["REST API v1<br/><i>app/api_v1.py</i>"] --> REQ
    CLI["Command line<br/><i>app/cli.py</i>"] --> REQ

    REQ["PlanRequest<br/><i>cities, days, pace, preferences</i>"] --> PREF
    PREF{"customer_id given?"} -->|yes| LOAD["Load stored preferences<br/><i>app/preferences.py</i>"]
    PREF -->|no| CACHE
    LOAD --> CACHE

    CACHE{"Seen this exact request?"} -->|yes| OUT
    CACHE -->|no| PLAN["Build the plan<br/><i>see diagram 2</i>"]

    PLAN --> CUST["Apply customisations<br/><i>order, times, own activities</i>"]
    CUST --> CLOCK["Put the day on a clock<br/><i>app/schedule.py</i>"]
    CLOCK --> SLOTS["Rescue missed time slots<br/><i>re-home, or drop with a reason</i>"]
    SLOTS --> FREE["Free windows and optional extras<br/><i>app/freetime.py</i>"]
    FREE --> VAL["Validate<br/><i>app/validation.py</i>"]
    VAL --> STORE["Remember the plan<br/><i>app/plancache.py</i>"]
    STORE --> OUT["Response<br/><i>itinerary, warnings, options</i>"]
```

The cache is what makes editing usable: every customisation rebuilds the plan, and the UI
replays the whole list of changes each time, so the repeated prefixes are answered from
memory rather than planned again.

## 2. Building a plan

```mermaid
flowchart TD
    ALLOC["Allocate days per city<br/><i>fairness, then usable hours</i>"] --> SLOTS
    SLOTS["Lay out day slots<br/><i>transfers, arrival, departure costs</i>"] --> ENROUTE
    ENROUTE["Stops along the drive<br/><i>app/enroute.py</i>"] --> LOOP

    LOOP["For each city"] --> ELIG
    ELIG["Eligible sights<br/><i>season, length of stay, exclusions</i>"] --> SCORE
    SCORE["Score against interests<br/><i>app/personalise.py</i>"] --> CLUSTER
    CLUSTER["Cluster by geography<br/><i>zones, else k-means</i>"] --> PACK
    PACK["Pack each day<br/><i>app/solver.py</i>"] --> REFINE
    REFINE["Regroup to cut driving<br/><i>swap stops between days</i>"] --> ASSEMBLE
    ASSEMBLE["Assemble the payload<br/><i>days, routes, summary</i>"]
```

Every rule a day must obey — the hour budget, the stop cap, leg and route distances, the
evening window, the arrival-day distance limit — is checked by one function,
`evaluate_day()`. The packer, the regrouper and the edit API all call it, so none of them
can invent a day the others would reject.

## 3. Where the facts come from

```mermaid
flowchart LR
    XL[("sample_dataset_v4_clean.xlsx")] --> LOADER["app/data_loader.py"]
    LOADER --> DS["Dataset<br/><i>cities, sights, distances, descriptions</i>"]

    XL --> PAT["app/patterns.py<br/><i>1,241 agent-written days</i>"]
    PAT --> ORDER["Visiting order<br/><i>ties inside one time slot</i>"]

    OSRM[("road_distances.json<br/>6,285 measured pairs")] --> RC["app/roadcache.py"]
    RC --> GEO["app/geo.py<br/><i>measured, else straight line x 1.3</i>"]

    CUST[("customers.json")] --> PREFS["app/preferences.py"]

    DS --> PLAN["The planner"]
    ORDER --> PLAN
    GEO --> PLAN
    PREFS --> PLAN
```

Nothing is invented. A missing journey time is reported as a data gap rather than guessed,
a distance the router could not measure falls back to the estimate and is counted as one,
and an unknown sight in a pin is refused rather than dropped.

## 4. After the plan exists

```mermaid
flowchart TD
    PLAN["A finished plan"] --> ACT{"What does the agent do?"}

    ACT -->|read| VIEW["Itinerary tab<br/><i>what the customer sees</i>"]
    ACT -->|change| OPS["Customise tab<br/><i>app/customize.py</i>"]
    ACT -->|improve| SUG["Suggestions<br/><i>app/suggest.py</i>"]
    ACT -->|send| PRINT["Printed itinerary<br/><i>A4, inclusions, exclusions</i>"]

    OPS --> APPLY["Apply one operation<br/><i>move, swap, extend, reallocate, add</i>"]
    APPLY --> REBUILD["Rebuild the whole plan"]
    REBUILD --> CHECK{"Did it do what it said?"}
    CHECK -->|yes| KEEP["Keep it, say what it cost"]
    CHECK -->|no| UNDO["Undo it, say what would help"]
    KEEP --> PLAN
    UNDO --> PLAN
```

Changes are applied to the *request* and the plan rebuilt, never patched into the finished
payload — that is the only way the route, the clock, the day budgets and the validator stay
in agreement. The check afterwards is what stops a stop quietly disappearing: a sight the
customer asked for either lands where they asked, or the change is undone and the response
says what would make room.

## The validation gate

A plan is returned either way, but `passed` is true only when **no critical check fails**.

| Severity | Meaning | Examples |
| --- | --- | --- |
| critical | the plan is wrong | hours that disagree with the clock, an evening sight before 16:30 that nobody asked for, a sight scheduled twice, a journey with no time in the dataset |
| warning | worth knowing | a morning sight seen in the afternoon, a slot clash the customer chose, a day that runs late because of something they added |
| note | context | what was learned from past itineraries, which legs are estimated rather than measured |

The distinction that matters: the engine holds *itself* to the hard rules, and treats the
customer's own choices as choices. Pinning three evening sights to one day is a warning
with the better times offered; the engine producing that plan on its own is a failure.
