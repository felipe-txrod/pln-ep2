#!/usr/bin/env bash
# Bootstrap the standard-library-only installer with explicit optional profiles.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"
setup_python="${EP2_SETUP_PYTHON:-python3.12}"
export PYTHONPATH="$project_root/src"
export PYTHONDONTWRITEBYTECODE=1
exec "$setup_python" -u -m ep2_delivery.setup "$@"
