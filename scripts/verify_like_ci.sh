#!/usr/bin/env bash
# run the test suite the way CI runs it: a CLONE, and only the declared extras.
#
# this exists because a green local run is not evidence. the machine that writes
# the code has every extra installed AND 151 GB of cohort data sitting where the
# tests look, so two whole classes of failure are invisible here and obvious in
# CI:
#
#   1. a test that needs an optional dependency the server deliberately omits
#   2. a test gated on a COMMITTED record while the data it names is git-ignored,
#      which skips on this machine and fails in every clone
#
# both shipped to CI on 2026-09-11 and both were caught there. a venv with the
# right packages was not enough - it still had the data. the clone is the part
# that matters.
#
# usage:  bash scripts/verify_like_ci.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

BRANCH="$(git -C "$REPO" rev-parse --abbrev-ref HEAD)"
echo "==> cloning $BRANCH (working tree is NOT used - commit first)"
git clone -q --depth 1 --single-branch --branch "$BRANCH" "file://$REPO" "$WORK/clone"

echo "==> what a clone actually carries"
printf '    data/inputs/          %s\n' "$([ -d "$WORK/clone/data/inputs" ] && echo present || echo absent)"
printf '    canonical parquet     %s\n' \
  "$(ls "$WORK"/clone/projects/*/data/canonical/*_spots.parquet 2>/dev/null | wc -l | tr -d ' ') file(s)"
printf '    embeddings cache      %s\n' \
  "$([ -d "$WORK/clone/data/embeddings" ] && echo present || echo absent)"

echo "==> installing with measure+dev only, as the workflow does"
python3 -m venv "$WORK/venv"
"$WORK/venv/bin/pip" install -q --upgrade pip
"$WORK/venv/bin/pip" install -q -e "$WORK/clone[measure,dev]"

echo "==> bare-import check: the server must serve without a tensor library"
for m in torch timm novae PIL; do
  if "$WORK/venv/bin/python" -c "import $m" 2>/dev/null; then
    echo "    WARNING: $m is importable; this env is not CI-like"; fi
done

echo "==> tests"
cd "$WORK/clone" && "$WORK/venv/bin/python" -m pytest tests -q
