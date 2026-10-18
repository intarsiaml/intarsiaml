#!/usr/bin/env bash
# Measure vLLM itself (no downtime; vLLM stays up, this slot is exclusive so nothing else loads the GPU): 3 rounds of
#   intarsia bench speed --url http://100.120.88.104:8000 --model guidance --sizes 1024,4096,16384 --gen 128 --reps 5
# (scripts/prefill-cuda/ab-vllm.sh). Run:
#   fleetq run --host spark1 --exclusive --mem 4 --label "w4a16 vLLM bench" -- scripts/cuda-moe-w4a16/vllm-bench.sh
# Needs ~/intarsia-decfuse/w4-dev built (scripts/cuda-moe-w4a16/prep.sh). Output ~/intarsia-decfuse/out/w4vllm
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
HOST=${HOST:-spark1}
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
rsh() { ssh -n -o BatchMode=yes "$HOST" "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
scp -q "$REPO/scripts/prefill-cuda/ab-vllm.sh" "$HOST:/tmp/ab-vllm-$FLEETQ_JOB.sh"
for r in 1 2 3; do
  rsh "BENCH_BIN=\$HOME/intarsia-decfuse/w4-dev/target/acc/release/intarsia AB_DIR=\$HOME/intarsia-decfuse/out/w4vllm SIZES=1024,4096,16384 bash /tmp/ab-vllm-$FLEETQ_JOB.sh $r"
done
rsh "rm -f /tmp/ab-vllm-$FLEETQ_JOB.sh"
