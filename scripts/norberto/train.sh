#!/usr/bin/env bash
# Delegate to the stable training launcher retained for running jobs and manifests.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "$project_root/scripts/train-norberto-full.sh" "$@"
