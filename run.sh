#!/usr/bin/env bash
# Entry point of the container (and a convenience wrapper for `uv run`): forwards every argument to the
# pipeline, e.g. `./run.sh all`, `./run.sh oof_infer --limit 10`, `./run.sh score`.
set -euo pipefail
cd "$(dirname "$0")"
exec uv run --no-sync python -m biohub046.pipeline "$@"
