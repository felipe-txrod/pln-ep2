#!/usr/bin/env bash
# Score an explicitly selected SVM artifact offline, without exposing GPU devices.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
exec docker run --rm --pull=never --network=none --cpus=2 \
    --user "$(id -u):$(id -g)" --security-opt label=disable \
    -v "$project_root:/workspace" -w /workspace \
    -e PYTHONPATH=/workspace/src -e PYTHONDONTWRITEBYTECODE=1 \
    --entrypoint python3 "${SVM_IMAGE:-pln-rocm:6.4.1}" \
    -u scripts/svm/evaluate-test1.py "$@"
