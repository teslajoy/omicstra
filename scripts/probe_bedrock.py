"""drive this server from a Bedrock-backed client, and prove the model did not decide.

`server.json` claims `modelInsideServer: false`: a model has three narrow jobs,
all client-side - classify a question into a closed task set, choose among
enumerated options at a gate, write the rationale - and every consequence
resolves from lookup tables. the claim that follows is that swapping the model
cannot change a route.

`tests/test_no_model_in_the_server.py` proves the server imports no model SDK.
This proves the other half: the same question through a Bedrock-backed client
returns the SAME server-side answer as the deterministic path. The model picks
which tool to call; the server picks what the answer is.

boto3 is deliberately NOT a dependency of this package, in any extra - a model
client belongs to the client. install it where you run this:

    pip install boto3
    aws configure                 # or export AWS_PROFILE / AWS_REGION
    python scripts/probe_bedrock.py --project-id tnbc-92

and with no credentials, to check the plumbing and print what WOULD be sent:

    python scripts/probe_bedrock.py --project-id tnbc-92 --dry-run

what it does, in one turn each:

  1. starts the server over stdio and lists its tools
  2. hands those tool schemas to Bedrock's Converse API as a toolConfig
  3. asks one question and executes whatever tool the model asks for
  4. calls the same tool with the same arguments IN PROCESS, with no model
     anywhere, and diffs the two answers

step 4 is the test. a difference means something on the server's path is not a
lookup.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import click

REPO = Path(__file__).resolve().parents[1]

# a READ-ONLY question on purpose. a probe that could dispatch an encode would
# spend GPU hours to test a transport.
QUESTION = (
    "For the cohort {pid}: which method should I use for cross-modal retrieval, "
    "and what do the three hypotheses H1, H2 and H3 currently claim? "
    "Use the tools. Do not guess a number."
)

# the model is the client's choice; this is a default, not a requirement.
DEFAULT_MODEL = "us.anthropic.claude-sonnet-5-20260514-v1:0"


async def _tools(session):
    """the server's own tool list, as a Bedrock toolConfig.

    `input_schema` with a camelCase fallback: mcp 2.0 renamed the field, and a
    probe that only works against one SDK minor version tests the SDK.
    """
    res = await session.list_tools()
    return [{"toolSpec": {"name": t.name,
                          "description": (t.description or "")[:1000],
                          "inputSchema": {"json": getattr(t, "input_schema", None)
                                          or t.inputSchema}}}
            for t in res.tools]


async def _call(session, name: str, args: dict):
    out = await session.call_tool(name, args)
    text = "".join(getattr(c, "text", "") for c in out.content)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"_raw": text}


def _bedrock_turn(client, model: str, tool_config: dict, messages: list) -> dict:
    return client.converse(modelId=model, messages=messages,
                           toolConfig=tool_config,
                           inferenceConfig={"maxTokens": 2000, "temperature": 0})


def _in_process(name: str, args: dict):
    """the same tool, with no model and no transport anywhere on the path."""
    import omicstra.mcp.server as S

    return getattr(S, name)(**args)


async def _run(project_id: str, model: str, dry_run: bool, turns: int) -> int:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command=sys.executable, args=["-m", "omicstra.mcp.server"],
        env={**os.environ, "PYTHONPATH": str(REPO / "src")})

    async with stdio_client(params) as (r, w), ClientSession(r, w) as session:
        await session.initialize()
        specs = await _tools(session)
        click.echo(f"server serves {len(specs)} tool(s) over stdio")
        tool_config = {"tools": specs}
        messages = [{"role": "user",
                     "content": [{"text": QUESTION.format(pid=project_id)}]}]

        if dry_run:
            click.echo("\n--dry-run: nothing is sent. the toolConfig would be:\n")
            click.echo(json.dumps(
                {"modelId": model,
                 "toolConfig": {"tools": [t["toolSpec"]["name"] for t in specs]},
                 "messages": messages}, indent=1))
            # the server half is still exercised, because that is the half this
            # repo owns: one read-only tool, called over the transport.
            got = await _call(session, "list_task_families", {"project_id": project_id})
            n = len(got.get("families") or [])
            click.echo(f"\nover the transport: list_task_families -> {n} families")
            same = _in_process("list_task_families", {"project_id": project_id})
            ok = got == same
            click.echo(f"identical in process: {ok}")
            return 0 if ok else 1

        try:
            import boto3
        except ModuleNotFoundError:
            click.echo("boto3 is not installed. it is the CLIENT's dependency and "
                       "deliberately not this package's: pip install boto3")
            return 2

        client = boto3.client("bedrock-runtime")
        mismatches, calls = [], 0
        for _ in range(turns):
            resp = _bedrock_turn(client, model, tool_config, messages)
            out = resp["output"]["message"]
            messages.append(out)
            uses = [c["toolUse"] for c in out["content"] if "toolUse" in c]
            for c in out["content"]:
                if "text" in c:
                    click.echo(f"\nmodel: {c['text'].strip()[:600]}")
            if not uses:
                break

            results = []
            for u in uses:
                calls += 1
                name, args = u["name"], u["input"]
                click.echo(f"\n-> {name}({json.dumps(args)})")
                served = await _call(session, name, args)
                # THE TEST. same tool, same arguments, no model on the path.
                direct = _in_process(name, args)
                if _comparable(served) != _comparable(direct):
                    mismatches.append({"tool": name, "args": args})
                    click.echo(click.style("   DIFFERS from the in-process call",
                                           fg="red"))
                else:
                    click.echo("   identical in process")
                results.append({"toolResult": {
                    "toolUseId": u["toolUseId"],
                    "content": [{"text": json.dumps(served)[:20000]}]}})
            messages.append({"role": "user", "content": results})

        click.echo(f"\n{calls} tool call(s), {len(mismatches)} mismatch(es)")
        if mismatches:
            click.echo(json.dumps(mismatches, indent=1))
            click.echo("a mismatch means something on the server's path is not a "
                       "lookup - modelInsideServer: false is the claim it breaks.")
            return 1
        return 0


_VOLATILE = ("run_id", "checkpointer", "ledger", "submit_seconds", "seconds",
             "started_at", "recorded", "cohort_decisions")


def _comparable(d):
    """drop what cannot be equal across two calls, and nothing else.

    a run handle is minted per call and a duration is measured per call. every
    DECISION in the response has to match, which is what the diff is for.
    """
    if isinstance(d, dict):
        return {k: _comparable(v) for k, v in sorted(d.items())
                if k not in _VOLATILE}
    if isinstance(d, list):
        return [_comparable(x) for x in d]
    return d


@click.command()
@click.option("--project-id", default=None, help="the cohort to ask about.")
@click.option("--model", default=DEFAULT_MODEL, show_default=True,
              help="a Bedrock model id, inference profile or ARN.")
@click.option("--dry-run", is_flag=True,
              help="send nothing. print the toolConfig and check the transport.")
@click.option("--turns", default=6, show_default=True,
              help="maximum model turns before stopping.")
def main(project_id, model, dry_run, turns):
    """probe this server from a Bedrock-backed client."""
    sys.path.insert(0, str(REPO / "src"))
    raise SystemExit(asyncio.run(_run(project_id, model, dry_run, turns)))


if __name__ == "__main__":
    main()
