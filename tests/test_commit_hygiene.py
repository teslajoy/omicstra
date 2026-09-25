"""the commit-msg hook, tested rather than trusted.

a hook lives in `.git/hooks` by default, which is not versioned - so it exists on
one machine and nowhere else. this one is committed under `.githooks/` and
activated with `git config core.hooksPath .githooks`, and the test is what keeps
it honest after the activation is forgotten.

why it exists: tooling appends provenance trailers by default, and four commits
went in carrying them before anyone read the log. a message is about the code.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HOOK = ROOT / ".githooks" / "commit-msg"

REFUSED = [
    "fix: a thing\n\nCo-Authored-By: Claude Opus 5 <noreply@anthropic.com>\n",
    "fix: a thing\n\nClaude-Session: https://claude.ai/code/session_abc\n",
    "docs: a thing\n\n\N{ROBOT FACE} Generated with [Claude Code](https://claude.com/claude-code)\n",
    "fix: a thing\n\nco-authored-by: claude <noreply@anthropic.com>\n",   # case
]
ACCEPTED = [
    "encode: the executor is chosen at dispatch\n\nboth go in the ledger.\n",
    # the point of the rule is provenance, not the word. a message that is about
    # a model id is about the code.
    "stack: the router moves to the current model\n\nsee MODELS.md.\n",
]


def _run(msg: str, tmp_path: Path):
    p = tmp_path / "MSG"
    p.write_text(msg)
    return subprocess.run([str(HOOK), str(p)], capture_output=True, text=True, check=False)


def test_the_hook_is_committed_and_executable():
    assert HOOK.is_file(), "the hook is not in the repo, so it exists on one machine only"
    assert os.access(HOOK, os.X_OK), "not executable; git would skip it silently"


@pytest.mark.parametrize("msg", REFUSED)
def test_a_message_carrying_session_or_tooling_context_is_refused(msg, tmp_path):
    r = _run(msg, tmp_path)
    assert r.returncode != 0, f"accepted: {msg!r}"
    assert "refused" in r.stderr


@pytest.mark.parametrize("msg", ACCEPTED)
def test_an_ordinary_message_passes(msg, tmp_path):
    r = _run(msg, tmp_path)
    assert r.returncode == 0, r.stderr


def test_no_commit_on_this_branch_carries_a_trailer():
    """the history itself, not just the gate on new commits.

    scoped to what this branch adds over main, because that is what a review sees
    and what a rewrite can still reach.
    """
    r = subprocess.run(["git", "log", "--format=%B", "main..HEAD"],
                       cwd=ROOT, capture_output=True, text=True, check=False)
    if r.returncode != 0:
        pytest.skip("no main to compare against")
    bad = [ln for ln in r.stdout.splitlines()
           if ln.lower().startswith(("co-authored-by: claude", "claude-session:"))]
    assert not bad, f"trailers in the history: {bad[:5]}"
