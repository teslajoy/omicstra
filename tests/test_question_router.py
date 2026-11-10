"""TF-IDF router eval: accuracy on held-out questions and OOS refusal rate.

three reported numbers (printed even on pass):
  top-1 accuracy   fraction of held-out + near-miss questions routed correctly
  OOS refusal rate fraction of out-of-scope questions refused (task_id=None)
  confusion pairs  which family pairs are mixed up (printed if any)

pre-registered bar (2026-11-10):
  top-1 accuracy >= 0.90  (no silent misroutes above threshold)
  OOS refusal rate >= 0.75

swap rule: if bar not met, add sentence-transformers as omicstra[router-embed]
and plug EmbeddingRouter behind the Router protocol.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parents[0] / "fixtures"


@pytest.fixture(scope="module")
def router():
    from omicstra.question_router import TFIDFRouter, DEFAULT_THRESHOLD
    return TFIDFRouter(threshold=DEFAULT_THRESHOLD)


@pytest.fixture(scope="module")
def questions():
    return json.loads((FIXTURES / "router_questions.json").read_text())


def test_router_implements_protocol():
    """TFIDFRouter satisfies the Router protocol at runtime."""
    from omicstra.question_router import Router, TFIDFRouter
    r = TFIDFRouter()
    assert isinstance(r, Router)
    assert hasattr(r, "threshold")
    assert hasattr(r, "version")


def test_to_record_carries_metadata(router):
    """to_record() returns router=tfidf with version and threshold."""
    rec = router.to_record()
    assert rec["router"] == "tfidf"
    assert rec["version"] == "1"
    assert "threshold" in rec
    assert isinstance(rec["threshold"], float)


def test_route_returns_required_keys(router):
    """route() always returns task_id, confidence, options."""
    result = router.route("does the alignment work?")
    assert "task_id" in result
    assert "confidence" in result
    assert "options" in result
    assert isinstance(result["options"], list)
    assert len(result["options"]) == 7  # one per family


def test_top1_accuracy_on_held_out(router, questions, capsys):
    """top-1 accuracy on held-out + near-miss questions meets the bar.

    prints accuracy, OOS refusal rate, and confusion pairs regardless of pass/fail.
    """
    bar = questions["bar"]
    min_acc = bar["top1_accuracy_min"]

    eval_set = questions["held_out"] + questions["near_miss"]
    correct = []
    wrong = []
    confusion: dict[tuple[str, str], list[str]] = {}

    for item in eval_set:
        q = item["question"]
        expected = item["task_id"]
        result = router.route(q)
        predicted = result["task_id"]  # may be None if below threshold

        if predicted == expected:
            correct.append(q)
        else:
            wrong.append({
                "question": q,
                "expected": expected,
                "predicted": predicted,
                "confidence": result["confidence"],
                "note": item.get("note", ""),
            })
            key = (expected, str(predicted))
            confusion.setdefault(key, []).append(q)

    accuracy = len(correct) / len(eval_set)

    # out-of-scope refusal rate (computed here for the joint print)
    oos = questions["out_of_scope"]
    refused = sum(1 for item in oos if router.route(item["question"])["task_id"] is None)
    oos_rate = refused / len(oos) if oos else 1.0

    with capsys.disabled():
        print(f"\n--- router eval (threshold={router.threshold}) ---")
        print(f"top-1 accuracy : {accuracy:.2%}  ({len(correct)}/{len(eval_set)})  bar={min_acc:.0%}")
        print(f"OOS refusal    : {oos_rate:.2%}  ({refused}/{len(oos)})  bar={bar['out_of_scope_refusal_min']:.0%}")
        if confusion:
            print("confusion pairs:")
            for (exp, pred), qs in sorted(confusion.items()):
                for q in qs:
                    print(f"  {exp} -> {pred}: {q[:80]}")
        else:
            print("confusion pairs: none")

    assert accuracy >= min_acc, (
        f"top-1 accuracy {accuracy:.2%} < bar {min_acc:.0%}. "
        f"wrong: {[w['question'][:60] for w in wrong]}"
    )


def test_out_of_scope_refusal_rate(router, questions, capsys):
    """out-of-scope questions are refused (task_id=None) at or above the bar."""
    bar = questions["bar"]
    oos = questions["out_of_scope"]
    refused = []
    misrouted = []

    for item in oos:
        result = router.route(item["question"])
        if result["task_id"] is None:
            refused.append(item["question"])
        else:
            misrouted.append({
                "question": item["question"],
                "predicted": result["task_id"],
                "confidence": result["confidence"],
            })

    rate = len(refused) / len(oos)
    min_rate = bar["out_of_scope_refusal_min"]

    assert rate >= min_rate, (
        f"OOS refusal rate {rate:.2%} < bar {min_rate:.0%}. "
        f"misrouted: {misrouted}"
    )


def test_refusal_below_threshold():
    """a question that scores below threshold gets task_id=None."""
    from omicstra.question_router import TFIDFRouter
    # a very high threshold forces refusals
    r = TFIDFRouter(threshold=0.99)
    result = r.route("what is the retrieval accuracy?")
    assert result["task_id"] is None
    assert result["confidence"] < 0.99


def test_above_threshold_returns_task_id(router):
    """a clearly phrased retrieval question returns a task_id above threshold."""
    result = router.route("What is the Recall@K for cross-modal retrieval in the aligned space?")
    assert result["task_id"] is not None
    assert result["task_id"] == "cross_modal_retrieval"
