#!/usr/bin/env bash
# Run native CPU inference by default; GPU is an explicit option.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"

if [[ -n "${EP2_DELIVERY_PYTHON:-}" ]]; then
    delivery_python="$EP2_DELIVERY_PYTHON"
elif [[ -x "$project_root/.venv-delivery/bin/python" ]]; then
    delivery_python="$project_root/.venv-delivery/bin/python"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
    delivery_python="$VIRTUAL_ENV/bin/python"
elif [[ -x "$project_root/.venv/bin/python" ]]; then
    delivery_python="$project_root/.venv/bin/python"
elif command -v python3.12 >/dev/null 2>&1; then
    delivery_python="$(command -v python3.12)"
else
    delivery_python="$(command -v python3)"
fi

# Do not mix binary caches compiled for different Python versions.
export PYTHONPATH="$project_root/src"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-10.3.0}"
exec "$delivery_python" -u -m ep2_delivery "$@"
