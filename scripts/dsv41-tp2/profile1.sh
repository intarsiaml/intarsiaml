#!/usr/bin/env bash
# Lane dsv41/tp2: one-GPU profile of the eager runtime on the few-layer cache (spark2, vLLM stays up, ~25 GB):
#   fleetq run --host spark2 --quiet gpu,cpu --mem 30 --expect 10 --label "dsv41 tp2 eager profile" -- scripts/dsv41-tp2/profile1.sh
# 1) host-bound check: INTARSIA_DSV41_PROFILE (build / enqueue / whole decode pass); 2) per-kernel device time
# (INTARSIA_DSV41_OPPROF, sync per step) for decode and a 2048-token prefill. Engram off (tables not on spark2's cache path
# for l3; the two Engram ops are timed anyway).
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
ssh -n -o BatchMode=yes spark2 "export FLEETQ_JOB=$FLEETQ_JOB; cat ~/intarsia-dsv41-graphs/bin/STAMP; B=~/intarsia-dsv41-graphs/bin/intarsia; C=~/intarsia-dsv41-tp2/cache/l3-r0of1; \
  P=\$(python3 -c 'print((\"The history of the lighthouse is long and full of storms. \")*180)'); \
  export INTARSIA_DSV41_ENGRAM=off INTARSIA_DSV41_CAP=8192 INTARSIA_DSV41_CHUNK=512; \
  echo '== eager timing'; INTARSIA_DSV41_PROFILE=32 timeout 300 \$B generate -m \$C --backend cuda -p \"\$P\" --max-tokens 97 --ignore-eos 2>&1 | grep -E 'profile|finish=' ; \
  for ch in 256 512 1024 2048; do echo \"== prefill chunk \$ch\"; INTARSIA_DSV41_CHUNK=\$ch timeout 300 \$B generate -m \$C --backend cuda -p \"\$P\" --max-tokens 2 --ignore-eos 2>&1 | grep -E 'finish='; done; \
  echo '== per-kernel'; INTARSIA_DSV41_OPPROF=1 INTARSIA_DSV41_PROFILE=32 timeout 300 \$B generate -m \$C --backend cuda -p \"\$P\" --max-tokens 33 --ignore-eos 2>&1 | grep -E 'opprof|^  |profile|finish='"
