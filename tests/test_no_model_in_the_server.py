"""`modelInsideServer: false` is a claim in the registry entry, so it is a test.

the claim is what makes a client-side model swap - Claude on Bedrock, say -
a configuration change rather than a port: every consequence this server
produces resolves from lookup tables and declarations, so no model is on the
path and swapping one cannot change a route.

a claim like that is false the moment somebody imports an SDK "just to try it",
which is why it is checked by source rather than trusted. the two judges are the
places where the temptation lives: judge #1 maps a question onto a task family,
judge #2 states a verdict per hypothesis, and both are deterministic.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "src" / "omicstra"

# an SDK that TALKS to a model. langgraph and langchain-core are state machines
# and are not on this list; `langchain_aws`, `langchain_anthropic` and the raw
# clients are.
MODEL_SDKS = ("anthropic", "openai", "boto3", "botocore", "langchain_aws",
              "langchain_anthropic", "langchain_openai", "litellm", "ollama",
              "google.generativeai", "vertexai", "mistralai", "cohere")


def _imports(path: Path):
    """every module name imported anywhere in a file, including inside functions."""
    tree = ast.parse(path.read_text())
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                yield a.name
        elif isinstance(n, ast.ImportFrom) and n.module:
            yield n.module


def test_no_module_in_the_package_imports_a_model_sdk():
    bad = []
    for f in sorted(PKG.rglob("*.py")):
        for mod in _imports(f):
            root = mod.split(".")[0]
            if root in MODEL_SDKS or mod in MODEL_SDKS:
                bad.append(f"{f.relative_to(ROOT)}: {mod}")
    assert not bad, (
        "the registry entry claims modelInsideServer: false and these import a "
        f"model SDK: {bad}")


def test_no_model_sdk_is_a_dependency_declared_or_optional():
    """an extra counts. `pip install omicstra[everything]` must not be able to
    put a model client in the server's own environment."""
    import tomllib

    pj = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    deps = list(pj.get("dependencies") or [])
    for extra in (pj.get("optional-dependencies") or {}).values():
        deps += list(extra)
    named = [d for d in deps
             if any(d.lower().startswith(s.replace("_", "-")) or
                    d.lower().startswith(s) for s in MODEL_SDKS)]
    assert not named, f"a model SDK is a dependency: {named}"


def test_the_registry_entry_still_makes_the_claim_this_file_checks():
    """if the claim is ever dropped, this test should be deleted deliberately
    rather than left passing against nothing."""
    meta = json.loads((ROOT / "server.json").read_text())["_meta"][
        "io.github.teslajoy/omicstra"]
    assert meta["modelInsideServer"] is False
    assert "client-side" in meta["note"]
    assert "lookup tables" in meta["note"]


def test_both_judges_are_deterministic():
    """judge #1 routes, judge #2 states a verdict per hypothesis. a model on
    either path would make the claim above false in the one place it matters.
    """
    import inspect

    from omicstra.graphs import evaluate as ev
    from omicstra.graphs import route as rt
    from omicstra.protocols import evaluate as pev

    for mod in (rt, ev, pev):
        src = inspect.getsource(mod)
        for sdk in MODEL_SDKS:
            assert sdk not in src, f"{mod.__name__} names {sdk}"
    # and the verdict is a function of its inputs: four states, no sampling
    assert pev.claims({"evaluable": True, "guards_ran": ["a"],
                       "guards_failed": []}) == "biological"
    assert pev.claims({"evaluable": True, "guards_ran": [],
                       "guards_failed": []}) == "unguarded"


# --- the server a client actually gets ------------------------------------
def test_both_documented_start_paths_serve_the_same_server():
    """`python -m omicstra.mcp.server` served 9 of the 18 tools.

    the `if __name__ == "__main__": main()` block sat in the MIDDLE of the
    module, above the resources and the nine compute tools, so running it as a
    module called main() before those decorators had run - a client discovering
    half a server with nothing saying so. the README's command happened to use
    `python -c "... serve()"`, which imports the whole module first and was
    therefore correct by accident.

    found by the Bedrock probe, which lists the tools over the transport rather
    than importing them. that is the difference between the two: every other
    test in this suite imports `srv` and sees all 18.
    """
    import asyncio
    import os
    import sys

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def names(args):
        p = StdioServerParameters(command=sys.executable, args=args,
                                  env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
        async with stdio_client(p) as (r, w), ClientSession(r, w) as s:
            await s.initialize()
            return sorted(t.name for t in (await s.list_tools()).tools)

    as_module = asyncio.run(names(["-m", "omicstra.mcp.server"]))
    as_import = asyncio.run(names(["-c", "from omicstra.mcp.server import serve; serve()"]))
    assert as_module == as_import, (
        f"the two start paths differ: only in one of them are "
        f"{sorted(set(as_module) ^ set(as_import))} served")

    from omicstra.mcp.server import srv
    assert as_module == sorted(t.name for t in asyncio.run(srv.list_tools())), \
        "the transport serves fewer tools than the module defines"


def test_the_main_guard_is_the_last_thing_in_the_module():
    """the rule the bug above came from, as a line rather than a memory: a tool
    appended after the guard is a tool `python -m` does not serve."""
    src = (PKG / "mcp" / "server.py").read_text().rstrip()
    assert src.endswith('if __name__ == "__main__":\n    main()')
