#!/usr/bin/env bash
# Lane asp (from the Mac, inside a fleetq cpu job): sync HEAD to tatooine ~/intarsiaml-asp and build it (CLI + HIP tests).
#   fleetq run --host tatooine --load cpu --expect 30 --label "asp build" -- scripts/asp/syncbuild.sh
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
stamp=$(scripts/asp/sync.sh dev HEAD) || exit 1
echo "$stamp"
ssh -o BatchMode=yes tatooine "TREES='dev=$stamp' HIPTEST=1 MTESTS='${MTESTS:-batch batch_recurrent mtp hybrid fusion_p2 graphs hybrid_mxfp4 spec tp_hybrid tp_mtp tp_batch tp}' bash -s" < scripts/asp/build.sh
