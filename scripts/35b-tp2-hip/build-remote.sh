#!/usr/bin/env bash
# Lane G2b: run build.sh on tatooine from the Mac (under a fleetq CPU slot). TREES / TEST pass through.
#   fleetq run --host tatooine --mem 0 --label "g2b build" -- env TREES="dev=head=..." scripts/35b-tp2-hip/build-remote.sh
set -uo pipefail
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ssh -o BatchMode=yes tatooine "FLEETQ_JOB=${FLEETQ_JOB:-} TREES='${TREES:?}' TEST=${TEST:-0} bash -s" < "$S/build.sh"
