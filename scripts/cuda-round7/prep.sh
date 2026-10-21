#!/usr/bin/env bash
# Round 7, small slot (vLLM up): rebuild r7-dev from this tree (syncs scripts/), then r7-base from BASE_SRC
# (scripts/cuda-round7/build-base.sh, checks the base binary has no round-7 symbol).
#   BASE_SRC=<dir> fleetq run --host spark1 --mem 8 --label "..." -- scripts/cuda-round7/prep.sh
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$S/../.." && pwd)"
"$REPO/scripts/cuda-decode/build.sh" r7-dev 2>&1 | grep -E "^error|Finished|intarsia$|stamp"
"$S/build-base.sh"
