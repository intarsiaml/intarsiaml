#!/usr/bin/env bash
# Lane sp2: moe_w4a16 bitwise tests (w4a16 vs moe_experts_mma_bf16) under each schedule setting "cfg0,cfg1".
#   fleetq run --host spark1 --mem 12 --label "..." -- docs/results/2026-10-22-sp2-cuda-prefill/scripts/bitwise-job.sh "0,0 1,1 ..."
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
ssh -n -o BatchMode=yes "${HOST:-spark1}" "export FLEETQ_JOB=$FLEETQ_JOB PATH=\$HOME/.cargo/bin:/usr/local/cuda/bin:\$PATH; cd ~/intarsia-decfuse/${TREE:-sp2-dev}
BIN=\$(ls -t target/test/release/deps/moe_w4a16-* | grep -v '\\.d\$' | head -1)
for st in $1; do IFS=, read c0 c1 <<< \"\$st\"
  echo \"== cfg0=\$c0 cfg1=\$c1: \$(env INTARSIA_W4_CFG0=\$c0 INTARSIA_W4_CFG1=\$c1 INTARSIA_REQUIRE_GPU=1 \$BIN --test-threads=1 2>&1 | grep -aE 'test result|FAILED|panicked|differ' | head -4 | tr '\n' '|')\"
done"
