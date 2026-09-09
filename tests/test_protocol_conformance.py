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


# --- the compute path's interrupt points ------------------------------------
COMPUTE = json.loads(
    (Path(omicstra.__file__).parent / "configs" / "compute_contract.json").read_text())
GATES = {g["id"]: g for g in COMPUTE["gates"]}


def test_five_preflight_one_mid_run():
    """the split is the argument. five of six can be asked before a unit is
    embedded, so durable execution is justified by ONE mid-run gate plus
    waiting-without-an-allocation - not by 'six human pauses overnight'."""
    pre = [g for g in GATES.values() if g["when"] == "preflight"]
    mid = [g for g in GATES.values() if g["when"] == "mid_run"]
    assert len(pre) == COMPUTE["summary"]["preflight"] == 5
    assert len(mid) == COMPUTE["summary"]["mid_run"] == 1
    assert mid[0]["id"] == "shard_failure_threshold"
    assert "cannot be asked before submission" in mid[0]["why_it_cannot_be_preflight"]


@pytest.mark.parametrize("gid", sorted(GATES))
def test_every_gate_has_a_closed_option_set_and_no_default(gid):
    """the model or the person picks from an enumerated set; nothing is free
    text but the rationale. and no gate has a default - the system does not
    pick, which is what makes an override a decision someone took."""
    g = GATES[gid]
    assert isinstance(g["options"], list) and len(g["options"]) >= 2, gid
    assert g["default"] is None, f"{gid} declares a default"
    assert g["emits"] in {"SelectionRecord", "GateRecord", "DiagnosticRecord"}, gid
    assert g["forecloses"], f"{gid} must say what accepting gives up"


def test_preflight_gates_are_pre_answerable_and_the_mid_run_one_is_not():
    """a gate with a declared answer does not fire. that is what makes an
    unattended run possible without lowering the bar - the answer is still
    recorded, with the declaration as its source."""
    for g in GATES.values():
        if g["when"] == "preflight":
            assert g["pre_answerable_by"], f"{g['id']} has no cohort declaration"
            assert g["pre_answerable_by"].startswith("cohort.json#"), g["id"]
        else:
            assert g["pre_answerable_by"] is None, (
                f"{g['id']} is mid_run - it depends on failures that have not "
                f"happened, so it cannot be declared in advance")


def test_one_schema_two_hosts():
    """the contract must not know which host asks. a preflight gate is a
    langgraph interrupt; the mid-run one is a signal on a durable workflow;
    both emit the same record."""
    h = COMPUTE["gate_schema"]["hosting"]
    assert "interrupt()" in h["preflight"] and "signal" in h["mid_run"]
    assert {g["emits"] for g in GATES.values()} <= {"SelectionRecord", "GateRecord"}


# --- the layer boundary: measures compute, protocols order ------------------
def test_measures_are_outside_protocols():
    """a file's location says what it is allowed to do. a measure that lives
    inside protocols/ makes that boundary stop meaning anything."""
    import omicstra.measures
    pkg = Path(omicstra.__file__).parent
    assert (pkg / "measures" / "__init__.py").exists()
    assert not (pkg / "protocols" / "measures.py").exists()
    assert not (pkg / "protocols" / "eda_steps.py").exists()


def test_measures_import_nothing_from_protocols():
    """the dependency runs one way. protocols call measures; a measure that
    imported a protocol would have acquired an order it is not allowed to have."""
    src = (Path(omicstra.__file__).parent / "measures" / "__init__.py").read_text()
    assert "omicstra.protocols" not in src and "from omicstra import protocols" not in src


def test_measures_take_no_ctx():
    """no ctx, no state, no conditions - give it data and parameters, get a
    record. that is what makes it callable from a notebook."""
    import inspect
    from omicstra import measures
    for name in ("count_statistics", "marker_expression",
                 "spatial_autocorrelation", "batch_structure"):
        params = list(inspect.signature(getattr(measures, name)).parameters)
        assert "ctx" not in params, f"{name} takes ctx"
        assert params[0] in ("adata", "load_sample_fn"), f"{name} first arg: {params[0]}"


# --- the alignment / evaluation port ----------------------------------------
def _cfg():
    from omicstra.contracts.project import ProjectConfig
    from omicstra.settings import settings
    settings.project_dir = Path(__file__).resolve().parents[1] / "projects" / "tnbc-92"
    return ProjectConfig.load("tnbc-92")


def test_all_four_stages_record_whether_they_computed():
    """`resolves_only` is how a ledger reader tells a stage that computed from
    one that read. without it the two are indistinguishable after the fact."""
    from omicstra.protocols import align
    cfg = _cfg()
    for fn in (align.run_embed_he, align.run_embed_st):
        _, rec = fn(cfg, "tnbc-92")
        assert rec.params["resolves_only"] is True, f"{rec.step_id} must never compute"


def test_encode_stages_never_encode():
    """the interface is satisfied; the execution is not claimed. running a
    foundation model is a backend question and deliberately outside this layer."""
    import inspect
    from omicstra.protocols import align
    for fn in (align.run_embed_he, align.run_embed_st):
        src = inspect.getsource(fn) + inspect.getsource(align._embed)
        assert "_run_script" not in src, f"{fn.__name__} invokes a script"


def test_baseline_dispatch_does_not_guess():
    """B4 is random-init with its own script and no --baseline. routing it
    through align_classical would produce a CCA number wearing B4's name."""
    import inspect
    from omicstra.protocols import align
    src = inspect.getsource(align.run_align)
    assert "align_b4.py" in src, "B4 must use its own script"
    assert 'base == "B4"' in src
    assert "raise ValueError" in src, "an unknown baseline must raise, not default"
    assert set(align._BASELINE_OF) == {"B1", "B2", "B3"}


def test_eval_scripts_are_mapped_per_hypothesis():
    """H1 scores the whole grid in one call; H2/H3 take one run at a time."""
    from omicstra.protocols import align
    assert align._EVAL_SCRIPT["H1"] == "eval.py"
    assert align._EVAL_SCRIPT["H2"] == align._EVAL_SCRIPT["H3"] == "eval_alignment_biology.py"


def test_run_eval_resolves_the_pack_and_never_generates_it():
    """no script writes routing_evidence.json. deciding which metric answers
    which task family, what the floors are, and which caveat attaches where is
    editorial - generating it would invent the judgements it records."""
    import inspect
    from omicstra.protocols import align
    src = inspect.getsource(align.run_eval)
    assert "curated" in src.lower()
    assert 'ev.write_text' not in src and 'json.dump' not in src


def test_project_relative_paths_resolve_against_the_project():
    """project.json writes niches_dir/runs_dir relative to the PROJECT dir.
    resolving against repo_root leaves `../..` in the path - a wrong answer
    that looks like a missing artifact."""
    from omicstra.protocols import align
    cfg = _cfg()
    p = align._project_rel(cfg.niches_dir, "tnbc-92")
    assert ".." not in p.parts, p
    assert p.is_absolute()


def test_declared_runs_exist_on_disk():
    """project.json must name the runs that exist. it listed R1..B4 while the
    directories are R1_v3..B4_v3, so the grid resolved to nothing."""
    from omicstra.protocols import align
    cfg = _cfg()
    root = align._project_rel(cfg.runs_dir, "tnbc-92")
    if not root.is_dir():
        pytest.skip("run grid not present on this machine")
    for rid in list(cfg.contrastive_runs) + list(cfg.classical_runs):
        assert (root / rid).is_dir(), f"{rid} declared but absent under {root}"


# --- the acceptance test, and WHY its tolerance is what it is ---------------
#
# B1 is CCA: deterministic linear algebra on a fixed split. a rerun is
# BIT-IDENTICAL, and anything else means the port changed something. verified
# 2026-09-08 - 13/13 metrics at delta 0.00e+00, and H1 over the full grid at
# 90/90 numeric fields identical.
#
# do NOT copy this assertion to a contrastive run. R1-R6 have a training loop
# on MPS, seeded inside a subprocess, so they need a stated tolerance and a
# stated reason - not exact equality that happens to be true for CCA.
EXACT_RUNS = {"B1_v3", "B2_v3", "B3_v3"}      # classical: deterministic
TOLERANCE_RUNS = {"R1_v3", "R2_v3", "R3_v3", "R4_v3", "R5_v3", "R6_v3", "B4_v3"}


def test_the_acceptance_split_is_declared():
    """which runs are exact and which need a tolerance is a property of the
    METHOD, not of the machine. classical baselines are closed-form; anything
    with a training loop or a random init is not."""
    from omicstra.protocols import align
    cfg = _cfg()
    declared = set(cfg.contrastive_runs) | set(cfg.classical_runs)
    assert EXACT_RUNS | TOLERANCE_RUNS == declared, (
        f"unclassified: {declared ^ (EXACT_RUNS | TOLERANCE_RUNS)}")
    assert "B4_v3" in TOLERANCE_RUNS, "B4 is random-init - not reproducible exactly"


# --- the acceptance test, pinned and guarded --------------------------------
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_pinned_h1_summary_is_intact():
    """the verified summary, pinned so 90/90 runs on a fresh clone.

    runs/ is gitignored, so an acceptance test that reads it passes on one
    machine and skips everywhere else - which is not a test.
    """
    import hashlib
    f = FIXTURES / "h1_summary_v3.json"
    meta = json.loads((FIXTURES / "h1_summary_v3.meta.json").read_text())
    assert hashlib.sha256(f.read_bytes()).hexdigest() == meta["sha256"]
    d = json.loads(f.read_text())
    assert sorted(d["per_run"]) == meta["runs"] and len(meta["runs"]) == 10
    assert not any(v is None for r in d["per_run"].values() for v in r.values())


def test_pinned_summary_agrees_with_the_evidence_pack():
    """the pack cites numbers; the summary produced them. if they disagree,
    one of the two is stale and the router is quoting a number nothing made."""
    from omicstra.settings import settings
    root = Path(__file__).resolve().parents[1] / "projects" / "tnbc-92"
    ev = json.loads((root / "routing_evidence.json").read_text())
    per_run = json.loads((FIXTURES / "h1_summary_v3.json").read_text())["per_run"]
    for cand in ev["tasks"]["cross_modal_retrieval"]["candidates"]:
        rid = cand["id"]
        if rid in per_run:
            assert abs(per_run[rid]["AUC"] - cand["value"]) < 5e-4, (
                f"{rid}: summary {per_run[rid]['AUC']:.6f} vs pack {cand['value']}")


def test_compute_cannot_write_into_the_declared_runs_dir():
    """the GUARD, not the behaviour.

    passing runs_root to a scratch path is what the acceptance test does;
    this asserts the stage cannot write to the cohort's declared grid when
    given one. a test that only demonstrates the safe call proves nothing
    about the unsafe one.
    """
    import inspect
    from omicstra.protocols import align
    src = inspect.getsource(align.run_align)
    # every write goes through `root`, which is runs_root when supplied
    assert "root = Path(runs_root) if runs_root else" in src
    assert '"--runs-root", str(root)' in src, "the script must be told where to write"
    # cfg.runs_dir may appear exactly ONCE - as the fallback when no runs_root
    # is supplied. a second occurrence means some path bypasses the override.
    assert src.count("cfg.runs_dir") == 1, (
        f"cfg.runs_dir referenced {src.count('cfg.runs_dir')} times; every write "
        "must go through `root` so runs_root can redirect it")
    # every mention of the flag passes `root` - two invocations share a
    # `common` arg list, so counting call sites would be wrong here.
    assert src.count('"--runs-root"') > 0
    assert src.count('"--runs-root", str(root)') == src.count('"--runs-root"'), (
        "a --runs-root is passed something other than the resolved root")


def test_declared_runs_dir_untouched_by_a_scratch_run(tmp_path):
    """mtime guard: a compute run into a scratch root leaves the cohort's own
    grid byte-for-byte unmodified. this is the assertion that would have caught
    an eval writing to a module-level RUNS_ROOT instead of --runs-dir."""
    from omicstra.protocols import align
    cfg = _cfg()
    declared = align._project_rel(cfg.runs_dir, "tnbc-92")
    if not declared.is_dir():
        pytest.skip("run grid not present on this machine")
    before = {p: p.stat().st_mtime_ns for p in declared.rglob("*") if p.is_file()}
    nj, _ = align.run_niche_join(cfg, "tnbc-92")
    with pytest.raises(align.ComputeUnavailable):
        # scratch root is empty, compute not requested -> must refuse, and must
        # not have reached into the declared grid to satisfy itself
        align.run_align(cfg, nj, project_id="tnbc-92", runs_root=tmp_path)
    after = {p: p.stat().st_mtime_ns for p in declared.rglob("*") if p.is_file()}
    assert before == after, "a scratch run modified the declared grid"


# --- the contracts layer: reads a declaration, decides nothing --------------
def test_contracts_only_read():
    """a contract module answers "what did someone declare", never "what
    follows from it". a decision in here would be a rule nobody could find."""
    import inspect
    from omicstra.contracts import eda as ceda, project as cproj, routing as crout
    for mod in (ceda, cproj, crout):
        src = inspect.getsource(mod)
        assert "interrupt(" not in src, f"{mod.__name__} asks a human"
        assert "subprocess" not in src, f"{mod.__name__} runs something"
        for verb in ("def resolve(", "def evaluate(", "def run_gate("):
            assert verb not in src, f"{mod.__name__} decides: {verb}"


def test_settings_is_the_bottom_layer():
    """settings imports nothing from omicstra. everything else may import it,
    which is what keeps the package/project split from becoming a cycle -
    and is why ProjectConfig belongs in contracts/, not merged into settings."""
    src = (Path(omicstra.__file__).parent / "settings.py").read_text()
    assert "from omicstra" not in src and "import omicstra" not in src


def test_project_config_reads_a_cohort_declaration():
    """ProjectConfig is a COHORT's declaration, not package settings. merging
    it into settings.py would put a cohort-shaped object in the module that is
    meant to know about no cohort at all."""
    from omicstra.contracts.project import ProjectConfig
    assert "project" in ProjectConfig.__module__
    src = (Path(omicstra.__file__).parent / "contracts" / "project.py").read_text()
    assert "project.json" in src


def test_init_public_scaffolds_without_crashing(tmp_path):
    """`init --public` read payload["provenance"], but provenance is written into
    cohort.json - so the flag that matters most raised KeyError at the last line.
    the assertion is exit 0; the rest guards the split that caused it."""
    import json

    from click.testing import CliRunner

    from omicstra.cli import main
    r = CliRunner().invoke(main, ["init", "--project-dir", str(tmp_path / "c"), "--public"])
    assert r.exit_code == 0, r.output + str(r.exception)

    cohort = json.loads((tmp_path / "c" / "cohort.json").read_text())
    project = json.loads((tmp_path / "c" / "project.json").read_text())
    assert cohort["data_classification"] == "public"
    assert cohort["provenance"] == {"source": "", "licence": "", "gated": ""}
    assert "cohort.json" in r.output, "the prompt must name the file provenance is in"
    # one home per declaration: classification is cohort's, never project's
    assert "data_classification" not in project
    # a scaffold must not claim a model backend the server does not have
    assert cohort["client_model_backend"] is None


def test_init_defaults_to_restricted(tmp_path):
    """forgetting the flag fails closed, and no provenance block is invented."""
    import json

    from click.testing import CliRunner

    from omicstra.cli import main
    r = CliRunner().invoke(main, ["init", "--project-dir", str(tmp_path / "d")])
    assert r.exit_code == 0, r.output + str(r.exception)
    cohort = json.loads((tmp_path / "d" / "cohort.json").read_text())
    assert cohort["data_classification"] == "restricted"
    assert "provenance" not in cohort
    assert "data/inputs/" in (tmp_path / "d" / ".gitignore").read_text()
