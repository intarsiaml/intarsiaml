#!/usr/bin/env bash
# Slot with vLLM UP: build w4-dev (this tree, with the CLI and test binaries) and w4-base (an export of perf/cuda-moe-nvfp4-mma
# 2aff2199, FP4 kernel off by default) on spark1 before a window.
#   fleetq run --host spark1 --mem 16 --label "w4a16 prep" -- scripts/cuda-moe-w4a16/prep.sh <base-export-dir>
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${1:?base export dir}" INTARSIA_STAMP=w4a16-base "$S/../cuda-decode/build.sh" w4-base
"$S/../cuda-decode/build.sh" w4-dev
