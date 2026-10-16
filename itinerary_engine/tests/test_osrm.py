"""The OSRM client, the distance cache, and the promise that planning stays offline."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from app import roadcache
from app.config import ROAD_DISTANCE_FACTOR
from app.geo import haversine_km, road_km, road_km_is_measured
from app.osrm import OsrmClient, OsrmError, build_cache, pair_key, write_cache

# Two points in Udaipur, and what the real OSRM demo server answers for them.
CITY_PALACE = (24.5760, 73.6833)
FATEHSAGAR = (24.5925, 73.6800)


class _StubOsrm(BaseHTTPRequestHandler):
    """Answers like OSRM's table service: metres, seconds, and directional distances.

    ``build`` is handed the number of coordinates in the request, because a table answer
    has to be the size it was asked for -- which is most of what the builder relies on.
    """

    build = staticmethod(lambda count: {"code": "Ok", "distances": [], "durations": [], "sources": []})

    def do_GET(self):  # noqa: N802 - name fixed by BaseHTTPRequestHandler
        coordinates = self.path.split("/driving/")[-1].split("?")[0]
        count = len([part for part in coordinates.split(";") if part])
        body = json.dumps(type(self).build(count)).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # keep the test output clean
        return


@pytest.fixture
def stub_server():
    """Start a stub OSRM. Pass a fixed payload, or a function of the request size."""
    servers = []

    def factory(payload):
        _StubOsrm.build = staticmethod(payload if callable(payload) else (lambda count: payload))
        server = HTTPServer(("127.0.0.1", 0), _StubOsrm)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return f"http://127.0.0.1:{server.server_port}"

    yield factory
    for server in servers:
        server.shutdown()


def square(count: int, metres, seconds, snap_m: float = 20.0) -> dict:
    """An OSRM answer of the right shape for ``count`` coordinates."""
    return {
        "code": "Ok",
        "distances": [[0 if i == j else metres for j in range(count)] for i in range(count)],
        "durations": [[0 if i == j else seconds for j in range(count)] for i in range(count)],
        "sources": [{"distance": snap_m} for _ in range(count)],
        "destinations": [{"distance": snap_m} for _ in range(count)],
    }


OK_TABLE = {
    "code": "Ok",
    # metres: the two directions differ, as they really do (one-way streets)
    "distances": [[0, 3247.3], [3721.1, 0]],
    "durations": [[0, 435], [352.6, 0]],
    "sources": [{"distance": 43.15}, {"distance": 30.81}],
    "destinations": [{"distance": 43.15}, {"distance": 30.81}],
}


# --- the client ---------------------------------------------------------------------------
def test_table_converts_to_km_and_hours(stub_server):
    client = OsrmClient(stub_server(OK_TABLE))
    result = client.table([CITY_PALACE, FATEHSAGAR])
    assert result.distances_km[0][1] == pytest.approx(3.2473)
    assert result.distances_km[1][0] == pytest.approx(3.7211), "direction must be preserved"
    assert result.durations_hours[0][1] == pytest.approx(435 / 3600, abs=1e-5)
    assert result.snap_km == [pytest.approx(0.0432), pytest.approx(0.0308)]


def test_an_osrm_error_is_raised_not_guessed(stub_server):
    url = stub_server({"code": "NoRoute", "message": "no route found"})
    with pytest.raises(OsrmError) as error:
        OsrmClient(url).table([CITY_PALACE, FATEHSAGAR])
    assert "NoRoute" in str(error.value)


def test_an_oversized_table_is_refused_before_the_request():
    with pytest.raises(OsrmError) as error:
        OsrmClient("http://127.0.0.1:1").table([CITY_PALACE] * 500)
    assert "max-table-size" in str(error.value)


def test_unroutable_pairs_are_left_out_rather_than_filled_in(stub_server, dataset):
    """A null distance keeps the estimate; it never becomes a measured-looking number."""
    url = stub_server(lambda count: square(count, metres=None, seconds=None))
    document = build_cache(dataset, OsrmClient(url))
    assert document["distances_km"] == {}
    assert document["meta"]["unroutable_pairs"] > 0


def test_a_distant_snap_is_reported(stub_server, dataset):
    """A coordinate 6 km from any road makes its distances suspect: say so."""
    url = stub_server(lambda count: square(count, metres=1000, seconds=100, snap_m=6100))
    document = build_cache(dataset, OsrmClient(url))
    assert any("6.1 km from the nearest road" in warning for warning in document["meta"]["snap_warnings"])


# --- the cache ----------------------------------------------------------------------------
def test_road_km_prefers_the_measured_value(tmp_path):
    path = tmp_path / "roads.json"
    write_cache({
        "meta": {"source": "OSRM", "pairs": 1},
        "distances_km": {pair_key(*CITY_PALACE, *FATEHSAGAR): 3.2473},
        "durations_hours": {},
    }, path)
    roadcache.reload(path)
    try:
        assert roadcache.measured_km(*CITY_PALACE, *FATEHSAGAR, path=path) == pytest.approx(3.2473)
        # the reverse direction falls back to the measured value, not to the estimate
        assert roadcache.measured_km(*FATEHSAGAR, *CITY_PALACE, path=path) == pytest.approx(3.2473)
        assert roadcache.info(path)["measured_pairs"] == 1
    finally:
        roadcache.reload()


def test_a_pair_outside_the_cache_keeps_the_estimate(tmp_path):
    path = tmp_path / "empty.json"
    write_cache({"meta": {}, "distances_km": {}, "durations_hours": {}}, path)
    roadcache.reload(path)
    try:
        assert roadcache.measured_km(*CITY_PALACE, *FATEHSAGAR, path=path) is None
    finally:
        roadcache.reload()


def test_a_missing_cache_file_is_not_an_error(tmp_path):
    assert roadcache.measured_km(*CITY_PALACE, *FATEHSAGAR, path=tmp_path / "nope.json") is None
    assert roadcache.info(tmp_path / "nope.json")["source"] == "estimated"


def test_the_estimate_is_still_the_old_formula(tmp_path):
    """Without a cache the engine behaves exactly as it did before OSRM existed."""
    roadcache.reload(tmp_path / "nope.json")
    try:
        from app import geo

        expected = haversine_km(*CITY_PALACE, *FATEHSAGAR) * ROAD_DISTANCE_FACTOR
        assert geo.road_km(*CITY_PALACE, *FATEHSAGAR) == pytest.approx(expected)
    finally:
        roadcache.reload()


def test_measured_distances_are_flagged_as_measured():
    """Whatever the shipped cache holds, the two sources are never confused."""
    measured = road_km_is_measured(*CITY_PALACE, *FATEHSAGAR)
    estimate = haversine_km(*CITY_PALACE, *FATEHSAGAR) * ROAD_DISTANCE_FACTOR
    if measured:
        assert road_km(*CITY_PALACE, *FATEHSAGAR) != pytest.approx(estimate)
    else:
        assert road_km(*CITY_PALACE, *FATEHSAGAR) == pytest.approx(estimate)


def test_planning_never_touches_the_network(monkeypatch, dataset):
    """The planner reads the cache file; it must not call OSRM while building a plan."""
    import urllib.request

    from app.builder import PlanRequest, build_itinerary

    def forbidden(*args, **kwargs):  # pragma: no cover - only runs if the promise breaks
        raise AssertionError("planning made a network call")

    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    plan = build_itinerary(PlanRequest(cities=["Udaipur"], duration_days=3, pace="moderate"), dataset)
    assert plan["days"]


def test_plans_are_identical_across_runs(dataset):
    """Cached distances keep the engine deterministic."""
    from app.builder import PlanRequest, build_itinerary

    request = PlanRequest(cities=["Jaisalmer"], duration_days=4, pace="moderate")
    first = build_itinerary(request, dataset)
    second = build_itinerary(request, dataset)
    assert [[s["sight"] for s in day["sights"]] for day in first["days"]] == \
           [[s["sight"] for s in day["sights"]] for day in second["days"]]
    assert [day["route_km"] for day in first["days"]] == [day["route_km"] for day in second["days"]]
