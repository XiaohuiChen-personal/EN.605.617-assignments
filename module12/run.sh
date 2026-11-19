#!/usr/bin/env bash
# Run assignment.exe. Arguments are forwarded.
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -x ./assignment.exe ]]; then
    echo "assignment.exe not found; run ./build.sh first" >&2
    exit 1
fi
exec ./assignment.exe "$@"
