#!/usr/bin/env bash
# Lane ck, CUDA: run chunk-job.sh ON spark1 inside this fleetq slot, then bring the outputs back.
#   fleetq run --host spark1 --exclusive --mem 45 --expect 45 --label "ck: ..." -- scripts/ck/cuda-run.sh "acc speed"
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run --host spark1}"
STEPS=${CK_STEPS:-acc gen speed accdec}   # speed before accdec: if the 11:30 stop comes, the speed table is complete
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RAW=$REPO/docs/results/2026-10-22-ck-prefill-chunk/raw/${RAWSUB:-cuda}; mkdir -p "$RAW"
H=${HOST:-spark1}
rsync -a "$REPO/scripts/ck/" $H:ck/src/scripts/ck/
O=ck/out/run-$(date +%H%M%S)
ssh -n -o BatchMode=yes $H "export FLEETQ_JOB=$FLEETQ_JOB; rm -rf ~/$O; mkdir -p ~/$O; CHUNKS='${CHUNKS:-2048 4096 8192}' SPEED_CHUNKS='${SPEED_CHUNKS:-}' MODELS_RUN='${MODELS_RUN:-35b 8b}' TAILS='${TAILS:-}' bash \$HOME/ck/src/scripts/ck/chunk-job.sh cuda \$HOME/$O '$STEPS'" 2>&1 < /dev/null
rsync -a $H:$O/ "$RAW/"
