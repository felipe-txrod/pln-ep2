#!/usr/bin/env bash
# Finalize the selected offline SVM; final-set evaluation is a separate command.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"
export PYTHONPATH="${SVM_RESEARCH_RUNTIME:-.cache/cascade-runtime:.cache/rag-runtime}:src"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 PYTHONDONTWRITEBYTECODE=1
exec "${SVM_RESEARCH_PYTHON:-python3.12}" -u -m svm_finalization "$@"
