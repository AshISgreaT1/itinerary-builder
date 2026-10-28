"""What a customer likes, remembered between trips.

There was no customer store in this project -- the engine took a request and answered it,
with nothing kept. This is the smallest thing that meets the brief: a JSON file holding one
record per customer, written atomically, read on demand.

**It is a file, so it lives where the file lives.** On Render's default filesystem that
means preferences are lost on redeploy and are not shared between instances. Attach a disk
and point ``ITINERARY_CUSTOMERS`` at it, or move this behind a real database later -- the
three functions below are the whole surface, so swapping the storage is a contained job.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

STORE_PATH = Path(os.environ.get("ITINERARY_CUSTOMERS", "data/customers.json"))

#: Who is travelling. Measured against the dataset's own past itineraries, this predicts
#: *what kind* of sight gets visited (see :mod:`app.personalise`) but not how full a day
#: is -- so it steers recommendations and never the pace.
GROUP_TYPES = ("solo", "couple", "friends", "family")

#: What a customer says they are interested in. Each maps to the dataset's own ``category``
#: column; nothing here is invented vocabulary.
INTERESTS = ("nature", "adventure", "religious", "cultural", "shopping",
             "wildlife", "scenic", "leisure", "water")


@dataclass
class CustomerPreferences:
    """One customer's standing preferences."""

    customer_id: str
    name: str = ""
    group_type: str = ""
    interests: list[str] = field(default_factory=list)
    #: Their usual pace, when they have one. Left empty means "whatever the trip needs".
    pace: str = ""
    notes: str = ""
    updated_at: str = ""
    trips: int = 0

    def clean(self) -> "CustomerPreferences":
        """Normalise and drop anything the engine would not understand.

        Unknown values are discarded rather than stored: a preference the planner cannot
        act on is worse than none, because it looks like it is doing something.
        """
        self.customer_id = str(self.customer_id).strip()
        self.group_type = str(self.group_type or "").strip().lower()
        if self.group_type not in GROUP_TYPES:
            self.group_type = ""
        seen: list[str] = []
        for interest in self.interests or []:
            value = str(interest).strip().lower()
            if value in INTERESTS and value not in seen:
                seen.append(value)
        self.interests = seen
        self.pace = str(self.pace or "").strip().lower()
        if self.pace not in {"relaxed", "moderate", "active"}:
            self.pace = ""
        return self

    def as_dict(self) -> dict:
        return asdict(self)


def _load(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        logger.warning("Customer store at %s could not be read (%s); starting empty.", path, error)
        return {}
    return data if isinstance(data, dict) else {}


def _write(path: Path, data: dict[str, dict]) -> None:
    """Write through a temporary file so a crash cannot leave a half-written store."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         delete=False, suffix=".tmp")
    try:
        json.dump(data, handle, indent=1, sort_keys=True)
        handle.close()
        os.replace(handle.name, path)
    except Exception:  # pragma: no cover - disk failure
        handle.close()
        Path(handle.name).unlink(missing_ok=True)
        raise


def get(customer_id: str, path: Path | str = STORE_PATH) -> CustomerPreferences | None:
    """One customer's preferences, or None when they are new."""
    record = _load(Path(path)).get(str(customer_id).strip())
    if not record:
        return None
    known = {key: record[key] for key in CustomerPreferences.__dataclass_fields__ if key in record}
    return CustomerPreferences(**known).clean()


def save(preferences: CustomerPreferences, path: Path | str = STORE_PATH) -> CustomerPreferences:
    """Create or replace a customer's preferences."""
    preferences = preferences.clean()
    if not preferences.customer_id:
        raise ValueError("a customer needs an id")
    store = Path(path)
    data = _load(store)
    existing = data.get(preferences.customer_id, {})
    preferences.trips = int(existing.get("trips", 0))
    preferences.updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    data[preferences.customer_id] = preferences.as_dict()
    _write(store, data)
    return preferences


def update(customer_id: str, path: Path | str = STORE_PATH, **changes) -> CustomerPreferences:
    """Change some of a customer's preferences, leaving the rest alone."""
    current = get(customer_id, path) or CustomerPreferences(customer_id=customer_id)
    for key, value in changes.items():
        if value is None or key not in CustomerPreferences.__dataclass_fields__:
            continue
        setattr(current, key, value)
    return save(current, path)


def record_trip(customer_id: str, path: Path | str = STORE_PATH) -> None:
    """Count a trip against a customer, so "reuse for future trips" is visible."""
    store = Path(path)
    data = _load(store)
    record = data.get(str(customer_id).strip())
    if not record:
        return
    record["trips"] = int(record.get("trips", 0)) + 1
    record["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _write(store, data)


def everyone(path: Path | str = STORE_PATH) -> list[dict]:
    return sorted(_load(Path(path)).values(), key=lambda record: record.get("customer_id", ""))
