#!/usr/bin/env bash
#
# Regenerate the three lockfiles from the hand-edited requirements/*.in files.
#
#   scripts/lock.sh                            # after editing any .in file
#   scripts/lock.sh --upgrade-package pandas   # bump one unpinned package
#   scripts/lock.sh --upgrade                  # bump everything not pinned in a .in file
#
# Order matters: dev.txt is resolved first, then api.txt and mlflow-server.txt
# are constrained to it (-c), so every environment shares the same versions.
# Extra options go to the dev compile only; the other two follow through the
# constraint. Never edit the generated .txt files by hand.
#
# Needs uv on PATH (pinned in dev.in): activate the venv first.

set -euo pipefail

# Run from the repo root, wherever the script was called from.
cd "$(dirname "$0")/.."

# Shared by all three compiles. Keep the Python version in step with the
# Docker base images and CI. --universal writes one lock for every platform
# (Linux in CI and Docker, macOS for local dev): platform-only packages get
# environment markers instead of being pinned unconditionally.
COMMON=(--universal --python-version 3.12 --custom-compile-command "scripts/lock.sh" --quiet)

uv pip compile requirements/dev.in "${COMMON[@]}" -o requirements/dev.txt "$@"

uv pip compile requirements/api.in -c requirements/dev.txt "${COMMON[@]}" \
    -o requirements/api.txt

uv pip compile requirements/mlflow-server.in -c requirements/dev.txt "${COMMON[@]}" \
    -o requirements/mlflow-server.txt

echo "Locks updated: requirements/dev.txt, api.txt, mlflow-server.txt"
