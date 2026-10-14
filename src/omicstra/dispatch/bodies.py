"""the queue's name table: which activity name holds which shard body.

the submitter names a body and the worker holds it. `graphs/submit.py` passes
`fn_name=` into the scheduler, and the process that executes the shard is a
different one - possibly on a different machine, started before the submitter
existed. a function is not serialisable, so the only thing keeping the two sides
in agreement is this table.

it is a function and not a dict literal because the bodies live in
`protocols.encode`, which imports a tensor library on no path but is still not
free, and a worker is the only caller that needs them resolved.

adding an arm means adding its name HERE as well as in its graph.
`tests/test_rule_lints.py` asserts every `fn_name=` in `graphs/` is a key, so a
new arm that forgets this line fails a test rather than orphaning its runs on a
queue no worker can serve.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omicstra.dispatch import ShardFn


def shard_bodies() -> dict[str, ShardFn]:
    """every activity name a worker can serve, with the body it names."""
    from omicstra.protocols.encode import he_shard_body, st_shard_body

    return {"encode_he": he_shard_body, "encode_st": st_shard_body}
