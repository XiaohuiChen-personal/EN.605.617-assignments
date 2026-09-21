#!/usr/bin/env bash
# Run assignment.exe. Arguments are forwarded (e.g. ./run.sh 512 256).
set -euo pipefail
export PATH="/usr/local/cuda/bin:${PATH}"
cd "$(dirname "$0")"
if [[ ! -x ./assignment.exe ]]; then
    echo "assignment.exe not found; run ./build.sh first" >&2
    exit 1
fi
exec ./assignment.exe "$@"
