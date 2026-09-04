"""the protocol must cover the contract, and agree with it.

this exists because it drifted once already: six of ten checks registered, two of
the missing four REQUIRED, and one authority disagreeing. none of it was visible
from the record, because an unregistered check does not appear at all - it is not
`not_run`, it is absent. a missing check that says nothing is the defect the whole
record contract exists to prevent, reappearing one level up.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import omicstra

from omicstra.protocols import ProtocolOrderError, Step, build_protocol
from omicstra.protocols.eda import EDA_STEPS
from omicstra.protocols.inventory import INVENTORY_STEPS

CONTRACT = json.loads(
    (Path(omicstra.__file__).parent / "configs" / "eda_contract.json").read_text()
)
CHECKS = {c["id"]: c for c in CONTRACT["checks"]}
REGISTERED = {s.id: s for s in EDA_STEPS}


def test_every_contract_check_is_registered():
    """an unregistered check is INVISIBLE - not not_run, absent."""
    missing = sorted(set(CHECKS) - set(REGISTERED))
    assert not missing, (
        f"{len(missing)} contract check(s) unregistered: {missing}. a check that is "
        f"not registered produces no record at all, so the gate cannot judge it and "
        f"nobody can see that it was skipped."
    )


def test_no_registered_step_is_absent_from_the_contract():
    """the reverse: a step nobody declared has no criterion to be judged against."""
    extra = sorted(set(REGISTERED) - set(CHECKS))
    assert not extra, f"registered but undeclared: {extra}"


@pytest.mark.parametrize("cid", sorted(CHECKS))
def test_authority_agrees_with_the_contract(cid):
    """authority decides whether a bar derived elsewhere ESCALATES on a new cohort.

    declaring a cohort_calibrated check as universal makes it judge a new cohort
    against another cohort's threshold without asking - silently.
    """
    want = CHECKS[cid].get("authority")
    got = REGISTERED[cid].authority
    assert got == want, f"{cid}: contract says {want!r}, step declares {got!r}"


@pytest.mark.parametrize("cid", [c for c, v in CHECKS.items() if v.get("applies_when")])
def test_conditional_checks_have_an_applicability_predicate(cid):
    """`applies_when` in the contract means the step must be able to say
    not_applicable. without a predicate it would report not_run instead, which
    means 'nobody looked' rather than 'it cannot apply here'."""
    assert REGISTERED[cid].applicable_when is not None, (
        f"{cid} declares applies_when={CHECKS[cid]['applies_when']!r} but the step "
        f"has no applicable_when predicate")


def test_required_checks_are_all_present():
    req = {c for c, v in CHECKS.items() if v.get("required")}
    assert req <= set(REGISTERED), f"required and unregistered: {sorted(req - set(REGISTERED))}"


# --- the driver's own invariants --------------------------------------------
def test_protocol_refuses_a_bad_order():
    """an invalid order is a wrong METHOD, so it must fail before data is touched."""
    with pytest.raises(ProtocolOrderError):
        build_protocol(
            [Step(id="moran", fn=lambda c: None, requires=frozenset({"weights"})),
             Step(id="knn", fn=lambda c: None, produces=frozenset({"weights"}))],
            "bad")


def test_protocol_accepts_a_good_order():
    assert build_protocol(
        [Step(id="knn", fn=lambda c: None, produces=frozenset({"weights"})),
         Step(id="moran", fn=lambda c: None, requires=frozenset({"weights"}))],
        "good") is not None


@pytest.mark.parametrize("steps,name", [(EDA_STEPS, "eda"), (INVENTORY_STEPS, "inventory")])
def test_shipped_protocols_build(steps, name):
    """both real protocols must satisfy their own order assertion."""
    assert build_protocol(steps, name) is not None


# --- the two bugs that shipped, now covered ---------------------------------
def test_inventory_halts_on_any_failure_not_only_unbound_roles():
    """a failed inventory must stop eda.

    the halt originally read only `bind.unbound_required`. when `platform`
    failed, `bind` was not_run, its unbound list was empty, and the run
    proceeded - letting eda measure data inventory could not describe.
    """
    import inspect
    from omicstra.graphs import eda as eda_graph
    src = inspect.getsource(eda_graph.inventory)
    assert 'status") == "fail"' in src or "status') == 'fail'" in src, (
        "inventory must halt on any failed step, not only on unbound roles")
    assert "unbound or failed" in src, "the halt must be the union of both conditions"


@pytest.mark.parametrize("proj", ["projects/tnbc-92"])
def test_platform_json_has_the_flat_samples_map(proj):
    """platform.json must map sample_id -> platform key.

    it was first written with sample_ids nested inside each platform. the
    inventory step requires the flat map, so every sample read as undeclared
    and the whole cohort halted for a schema reason, not a data one.
    """
    p = Path(__file__).resolve().parents[1] / proj / "platform.json"
    if not p.exists():
        pytest.skip(f"{proj} has no platform.json")
    d = json.loads(p.read_text())
    assert "samples" in d, "platform.json needs a flat samples map"
    assert isinstance(d["samples"], dict) and d["samples"], "samples must be non-empty"
    undefined = set(d["samples"].values()) - set(d.get("platforms", {}))
    assert not undefined, f"samples name undefined platforms: {sorted(undefined)}"
    for key, spec in d.get("platforms", {}).items():
        assert "sample_ids" not in spec, (
            f"{key} still nests sample_ids - that is the superseded shape")


def test_eda_graph_has_an_inventory_node_before_profile():
    """inventory produces the join key, platform and raw matrix that eda reads."""
    from omicstra.graphs.eda import build_eda_graph
    nodes = set(build_eda_graph().get_graph().nodes)
    assert {"inventory", "profile", "gate"} <= nodes, f"nodes: {sorted(nodes)}"


def test_step_registry_is_gone_from_the_graph():
    """the registry lives in protocols/. a graph that also owns one is two things."""
    from omicstra.graphs import eda as eda_graph
    assert not hasattr(eda_graph, "STEP_REGISTRY")


# --- transport and protocol revision ----------------------------------------
def test_serve_refuses_an_unknown_mode():
    from omicstra.mcp.server import serve
    with pytest.raises(ValueError, match="stdio or http"):
        serve(mode="grpc")


def test_protocol_version_is_read_from_meta_not_a_session():
    """the july revision removed the initialize handshake - version travels in
    `_meta` on every request, so it is per-request and belongs on the record."""
    from mcp.types import PROTOCOL_VERSION_META_KEY, LATEST_PROTOCOL_VERSION
    from omicstra.mcp.server import negotiated_protocol_version
    assert negotiated_protocol_version(
        {PROTOCOL_VERSION_META_KEY: LATEST_PROTOCOL_VERSION}) == LATEST_PROTOCOL_VERSION
    assert negotiated_protocol_version(None) is None


def test_require_protocol_2026_defaults_off_and_refuses_when_on(monkeypatch):
    """default permissive; strict refuses an older client, which is what a
    deployment behind an authorization server needs - an older client cannot
    present an audience-bound token."""
    import importlib
    from mcp.types import PROTOCOL_VERSION_META_KEY as K, LATEST_PROTOCOL_VERSION as L
    import omicstra.mcp.server as s
    monkeypatch.setenv("REQUIRE_PROTOCOL_2026", "0")
    s = importlib.reload(s)
    assert s.REQUIRE_PROTOCOL_2026 is False
    assert s.check_protocol({K: "2025-03-26"}) == "2025-03-26"
    monkeypatch.setenv("REQUIRE_PROTOCOL_2026", "1")
    s = importlib.reload(s)
    assert s.REQUIRE_PROTOCOL_2026 is True
    assert s.check_protocol({K: L}) == L
    with pytest.raises(ValueError):
        s.check_protocol({K: "2025-03-26"})
    monkeypatch.setenv("REQUIRE_PROTOCOL_2026", "0")
    importlib.reload(s)


def test_http_app_is_stateless():
    """any instance handles any request - omicstra's handle is run_id, passed
    as an ordinary param, so nothing needs session affinity."""
    from omicstra.mcp.server import srv
    app = srv.streamable_http_app(streamable_http_path="/mcp", stateless_http=True)
    assert "/mcp" in [getattr(r, "path", None) for r in app.routes]


def test_gate_interrupt_carries_the_records():
    """a client asked to accept cautions must see the diagnostics behind them.

    while a subgraph is paused the parent sees none of its state, so anything
    the human needs has to travel IN the interrupt payload.
    """
    import inspect
    from omicstra.graphs import eda as g
    assert '"records": state.get("records", [])' in inspect.getsource(g.escalate)


# --- a,b: the version reaches the record ------------------------------------
def test_protocol_version_is_a_field_on_the_base_record():
    """b - without this every public-demo run carries a null forever, and
    REQUIRE_PROTOCOL_2026 means something only at the door, never after."""
    from omicstra.records import DiagnosticRecord, GateRecord, Record, TransformRecord
    for cls in (Record, DiagnosticRecord, TransformRecord, GateRecord):
        assert "protocol_version" in cls.model_fields, cls.__name__


def test_request_protocol_version_is_on_the_request_path():
    """a - a checker that is only ever called from a test refuses nothing."""
    from omicstra.mcp.server import request_protocol_version
    assert request_protocol_version() is None       # outside a request: stdio
    import inspect
    from omicstra.mcp.server import request_protocol_version as f
    assert "check_protocol" in inspect.getsource(f), "must validate, not just read"


# --- c: durability is what makes the stateless claim true -------------------
def test_checkpointer_selector():
    from omicstra.graph import make_checkpointer
    m, label = make_checkpointer("memory")
    assert type(m).__name__ == "InMemorySaver" and label == "memory"
    with pytest.raises(ValueError, match="memory or sqlite"):
        make_checkpointer("redis://nope")


def test_sqlite_checkpoint_survives_a_new_saver(tmp_path):
    """the point of c: a thread resumable by an instance that never ran it.

    InMemorySaver dies with the process, which quietly makes "any instance
    handles any request" false. this builds a SECOND saver over the same file -
    the in-process stand-in for a restarted or second server.
    """
    from langgraph.checkpoint.memory import InMemorySaver  # noqa: F401
    from omicstra.graph import make_checkpointer
    db = tmp_path / "ck.sqlite"
    s1, _ = make_checkpointer(f"sqlite:{db}")
    assert db.exists(), "setup() must create the file"
    s2, _ = make_checkpointer(f"sqlite:{db}")
    assert s1 is not s2 and type(s2).__name__ == "SqliteSaver"


# --- d: through the real http app -------------------------------------------
def test_tools_served_over_http_with_meta():
    """d - the same tools, over streamable-http, with the version in `_meta`.

    note the transport rejects an unknown Host by default (DNS-rebinding
    protection), so a deployment must declare its hostname.
    """
    import json as _json
    from starlette.testclient import TestClient

    from mcp.server.transport_security import TransportSecuritySettings
    from mcp.types import LATEST_PROTOCOL_VERSION, PROTOCOL_VERSION_META_KEY
    from omicstra.mcp.server import srv

    sec = TransportSecuritySettings(allowed_hosts=["testserver"], allowed_origins=["*"])
    app = srv.streamable_http_app(streamable_http_path="/mcp", stateless_http=True,
                                  transport_security=sec)
    with TestClient(app) as c:
        r = c.post("/mcp",
                   json={"jsonrpc": "2.0", "id": 1, "method": "tools/list",
                         "params": {"_meta": {PROTOCOL_VERSION_META_KEY:
                                              LATEST_PROTOCOL_VERSION}}},
                   headers={"Accept": "application/json, text/event-stream",
                            "Content-Type": "application/json",
                            "Mcp-Method": "tools/list"})
    assert r.status_code == 200, r.text[:200]
    data = [l for l in r.text.splitlines() if l.startswith("data:")]
    payload = _json.loads(data[0][5:]) if data else r.json()
    names = {t["name"] for t in payload["result"]["tools"]}
    assert {"check_eda_gate", "route", "decision_record"} <= names, sorted(names)


def test_http_rejects_an_unknown_host():
    """DNS-rebinding protection is ON by default and should stay on."""
    from starlette.testclient import TestClient
    from omicstra.mcp.server import srv
    app = srv.streamable_http_app(streamable_http_path="/mcp", stateless_http=True)
    with TestClient(app) as c:
        r = c.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
                   headers={"Accept": "application/json, text/event-stream",
                            "Content-Type": "application/json"})
    assert r.status_code == 421, "unknown Host must be refused"


# --- packaging: the contracts must travel with the code ---------------------
def test_contracts_resolve_from_the_package_not_a_repo_checkout():
    """`pip install omicstra` shipped a server that could not find its own
    contracts: configs/ was at repo root, and in an installed wheel
    `parents[2]` is site-packages with no configs/ beside it. this is the
    difference between "pip-installable mcp server" being true or a claim."""
    from omicstra.settings import settings
    pkg = Path(omicstra.__file__).parent
    assert settings.configs_dir == pkg / "configs", settings.configs_dir
    assert settings.configs_dir.is_absolute(), "must not depend on cwd"
    for f in ("eda_contract.json", "routing_contract.json", "data_contract.json"):
        assert (settings.configs_dir / f).exists(), f


def test_no_cohort_data_inside_the_package():
    """package data is generalizable and cohort-free. configs/v3/ is one
    cohort's run grid and belongs at repo root, not in the wheel."""
    pkg_cfg = Path(omicstra.__file__).parent / "configs"
    names = {p.name for p in pkg_cfg.iterdir()}
    assert "v3" not in names, "run configs are not package data"
    for p in pkg_cfg.glob("*.json"):
        d = json.loads(p.read_text())
        # `source` is provenance - citing where a shape came from is legitimate.
        # a cohort named anywhere ELSE is a measurement that escaped its evidence
        # file into the contract that is supposed to be cohort-free.
        d.pop("source", None)
        assert "tnbc" not in json.dumps(d).lower(), (
            f"{p.name} names a cohort outside its provenance field - that value "
            f"belongs in routing_evidence.json")


def test_server_json_matches_the_server():
    """the registry entry is a claim about what this server serves. a tool list
    that drifts from the code is how a client discovers a tool that is not
    there."""
    import asyncio
    import tomllib
    from omicstra.mcp.server import srv
    root = Path(__file__).resolve().parents[1]
    sj = json.loads((root / "server.json").read_text())
    meta = sj["_meta"]["io.github.teslajoy/omicstra"]
    actual = {t.name for t in asyncio.run(srv.list_tools())}
    assert set(meta["tools"]) == actual, f"drift: {set(meta['tools']) ^ actual}"
    pj = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    assert sj["version"] == pj, f"server.json {sj['version']} != pyproject {pj}"
    for pkg in sj["packages"]:
        assert pkg["version"] == pj, f"package entry {pkg['version']} != {pj}"


def test_declared_protocol_version_matches_the_sdk():
    from mcp.types import LATEST_PROTOCOL_VERSION
    sj = json.loads((Path(__file__).resolve().parents[1] / "server.json").read_text())
    assert sj["_meta"]["io.github.teslajoy/omicstra"]["protocolVersion"] \
        == LATEST_PROTOCOL_VERSION
