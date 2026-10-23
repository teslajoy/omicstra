"""the label vocabularies that cross a graph boundary. no logic, no cohort.

every one of these was a bare string, and four separate defects came from the
same cause: a vocabulary defined at one layer, consumed at another, with nothing
asserting the two agree. a plain string makes a typo (`refuse` written, `refused`
read) and a semantic clash (`arm` meaning two different enums) look identical to
the interpreter, and both only surface as a symptom several layers away.

    arm                 route label vs handle label. CONTAINED - `_runner_for`
                        raises `NotAnArmHandle` rather than defaulting to the
                        H&E graph, so a wrong value cannot get through
    refuse / refused    the router writes `refuse`; the chain's stop list holds
                        `refused`. SILENT - the intersection of the two
                        vocabularies was EMPTY, so every routing outcome,
                        including a tie, continued into evaluate

`StrEnum` and not `Enum`: these values are written into langgraph checkpoints,
appended to the ledger and returned in MCP tool output. a StrEnum member IS a
str, so nothing downstream changes shape - json.dumps emits the same bytes an
old run wrote, and an existing checkpoint still compares equal.

WHO DECLARES WHAT
-----------------
the vocabulary lives here, with the producer. what a consumer DOES with each
label is declared by the consumer, beside the code that branches on it -
`graph.py`'s stop/continue partitions. that split is deliberate: a partition
declared in this file would be this file checking itself, and the check that
matters is whether two modules agree.
"""
from __future__ import annotations

from enum import StrEnum
from typing import TypeVar


class Route(StrEnum):
    """LEVEL 0: which arm of the top-level graph answers THIS question.

    a property of the question CROSSED with the cohort, not of the cohort alone.
    `COMPUTE` and `REFUSE` are the load-bearing distinction: a declared task this
    cohort has not run is work outstanding, and an undeclared one is a question
    the system has no method for.
    """

    ASK = "ask"
    COMPUTE = "compute"
    REFUSE = "refuse"


class HandleArm(StrEnum):
    """the MCP run handle: which compiled graph a thread belongs to.

    the arms share `EncodeState`, so any compiled graph can READ any checkpoint
    and only the gates and the compute node differ - which is exactly why the
    arm has to be recorded. resuming an ST run into the H&E node would re-enter
    a node that resolves a tile geometry for a cohort never asked for one.
    """

    HE = "he"
    ST = "st"
    ALIGN = "align"
    EVALUATE = "evaluate"


class Resolution(StrEnum):
    """what the routing subgraph concluded about one task.

    `TIE` and `ESCALATE` are the abstain cases, and they exist because a
    confidence-bearing decision needs an outcome that means "I will not pick".
    a consumer that handles only the positive outcomes lets an abstention
    through as if it were an answer, which is what happened here.
    """

    RECOMMEND = "recommend"
    TIE = "tie"
    ESCALATE = "escalate"
    REFUSE = "refuse"
    OVERRIDE_ACK = "override_ack"


class StageStatus(StrEnum):
    """what one pipeline stage reported, as its `report.status`.

    a SEPARATE vocabulary from `Resolution`, and conflating the two is the whole
    defect: `_keeper` collapsed a stage's status and the router's resolution into
    one field, and `_continue` compared whatever arrived against a list built
    only from these values.
    """

    READY = "ready"
    RESOLVED = "resolved"
    COMPUTED = "computed"
    EVALUATED = "evaluated"
    MEASURED = "measured"
    NOTHING_TO_RUN = "nothing_to_run"
    SUBMITTED = "submitted"
    REFUSED = "refused"
    HALTED = "halted"
    UNAVAILABLE = "unavailable"
    NOT_RUN = "not_run"


E = TypeVar("E", bound=StrEnum)


def parse(vocab: type[E], value: object, where: str) -> E:
    """`value` as a member of `vocab`, or a refusal that names both.

    the point of the whole module. `_runner_for` already did this by hand for
    the handle arm - read, check membership, raise with the two meanings spelled
    out - and it is the reason that collision is the one that never silently
    served the wrong graph. every other consumer fell through to a default.

    the error names the vocabulary, the value and the reading site, because the
    useful question on an unknown label is which of two producers wrote it.
    """
    try:
        return vocab(value)
    except ValueError:
        raise UnknownLabel(
            f"{where} read {value!r}, which is not a {vocab.__name__}. "
            f"known: {sorted(m.value for m in vocab)}. an unknown label here is "
            "a producer and a consumer disagreeing about a vocabulary, not a "
            "value to default") from None


def is_yes(answer: object) -> bool:
    """did a person actually say yes.

    `bool(answer)` is the defect this replaces: `bool("no")` is True, so at
    `graphs/promote.py` a human DECLINING the evidence pack approved it - at the
    one gate whose whole job is to refuse a pack nobody vouched for.

    an explicit allow-list, not a truthiness test, and it lives here because
    `graphs/route.py` already had the correct version inline. two copies of one
    vocabulary is the pattern this module exists to end, and a second copy is
    how the two drift.

    anything unrecognised is NO. a gate that asks a person a question answers it
    in the safe direction when it cannot read the reply.
    """
    if isinstance(answer, bool):
        return answer
    if answer is None:
        return False
    return str(answer).strip().lower() in YES


# the words a person may use. `override` and `proceed` are here because the
# routing gate asks "proceed anyway?" rather than "yes?".
YES = frozenset({"yes", "y", "proceed", "true", "override", "approve", "approved"})


class UnknownLabel(ValueError):
    """a label no vocabulary at this boundary declares.

    a subclass so a consumer can let it through deliberately. nothing catches it
    today: every site that reads a label is a site where a wrong one means two
    layers disagree, and continuing on a default is what this module exists to
    stop.
    """


def exhaustive(vocab: type[E], *handled: object) -> set[str]:
    """the members of `vocab` that no consumer group covers.

    used by the build-time lint. returning the gap rather than asserting keeps
    the maths here and the failure message in the test, where a reader looking
    at a red run is already standing.
    """
    covered = {str(v) for group in handled for v in group}
    return {m.value for m in vocab} - covered


__all__ = ["YES", "HandleArm", "Resolution", "Route", "StageStatus",
           "UnknownLabel", "exhaustive", "is_yes", "parse"]
