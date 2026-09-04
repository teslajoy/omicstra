"""LEVEL 1 - one compiled StateGraph per file, each holding exactly one gate.

the rule: a thing gets its own graph only if it can ask a human. everything
else is a protocol, and lives in `protocols/`.

each graph here compiles with `checkpointer=None` and is added to `graph.py` as
ONE node, inheriting the parent's checkpointer. an `interrupt()` raised inside
pauses the PARENT at that node, and the caller resumes on the parent's
thread_id - so the CLI and the MCP server only ever talk to one graph.
"""
