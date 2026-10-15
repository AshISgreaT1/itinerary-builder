"""Command-line runner, for checking plans without starting the API.

    python -m app.cli --cities Manali Kasol --days 5 --pace moderate --from Delhi --month 6
    python -m app.cli --cities Jaipur --days 4 --json > jaipur.json
"""

from __future__ import annotations

import argparse
import json
import sys

from .builder import PlanningError, PlanRequest, build_itinerary
from .data_loader import get_dataset


def _print_human(plan: dict) -> None:
    summary = plan["summary"]
    print(f"{' + '.join(plan['request']['cities'])} - {summary['days']} days, {plan['request']['pace']} pace")
    print(f"Nights: {summary['nights_per_city']} | sights: {summary['total_sights']} | "
          f"sightseeing {summary['total_sightseeing_hours']}h, local travel {summary['total_local_travel_hours']}h, "
          f"intercity {summary['total_intercity_drive_hours']}h")
    for day in plan["days"]:
        header = f"\nDay {day['day']} - {day['city']}"
        if day["transfer"]:
            header += f"  (drive from {day['transfer']['from_city']}, {day['transfer']['drive_hours']}h)"
        print(f"{header}   [{day['used_hours']}h used of {day['available_hours']}h, route {day['route_km']} km]")
        for sight in day["sights"]:
            tag = " [day trip]" if sight["is_excursion"] else ""
            print(f"   - {sight['sight']} ({sight['duration_hours']}h, {sight['best_time']}){tag}")
        for note in day["notes"]:
            print(f"     * {note}")
    validation = plan["validation"]
    print("\nFeasibility:", "PASS" if validation["passed"] else "ISSUES FOUND")
    for category, issues in validation["issues"].items():
        for issue in issues:
            print(f"   [{category}] {issue}")
    for label, reason in plan.get("unscheduled_reasons", {}).items():
        print(f"   [not scheduled] {label}: {reason}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a deterministic day-wise itinerary.")
    parser.add_argument("--cities", nargs="+", required=True, help="Base cities, in travel order")
    parser.add_argument("--days", type=int, required=True, help="Total trip days")
    parser.add_argument("--pace", default="moderate", choices=["relaxed", "moderate", "active"])
    parser.add_argument("--from", dest="origin", default=None, help="Origin city for day 1's drive")
    parser.add_argument("--month", type=int, default=None, help="Travel month (1-12) for seasonal closures")
    parser.add_argument("--dataset", default=None, help="Path to the Excel dataset")
    parser.add_argument("--json", action="store_true", help="Print the raw JSON payload")
    args = parser.parse_args(argv)

    try:
        dataset = get_dataset(args.dataset)
        plan = build_itinerary(
            PlanRequest(
                cities=args.cities, duration_days=args.days, pace=args.pace,
                origin_city=args.origin, start_month=args.month,
            ),
            dataset,
        )
    except (PlanningError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(plan, indent=2, ensure_ascii=False)) if args.json else _print_human(plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
