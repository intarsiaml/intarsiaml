#!/usr/bin/env bash
# Lane asa (from the Mac, inside a fleetq cpu job): sync HEAD to tatooine ~/intarsiaml-asa and build it (CLI + HIP tests).
#   fleetq run --host tatooine --load cpu --expect 30 --label "asa build" -- scripts/asa/syncbuild.sh
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
stamp=$(scripts/asa/sync.sh dev HEAD) || exit 1
echo "$stamp"
ssh -o BatchMode=yes tatooine "TREES='dev=$stamp' HIPTEST=${HIPTEST:-0} MTESTS='${MTESTS:-batch batch_recurrent mtp hybrid fusion_p2 graphs hybrid_mxfp4 spec tp_hybrid tp_mtp tp_batch tp}' bash -s" < scripts/asa/build.sh
