#!/usr/bin/env bash
# Lane lt, spark side: the f64 CPU oracle (fx5c / fx1 recipe: `accuracy --backend cpu --mode prefill --prefill-chunk 256`,
# top-64 self-golden) over one-prompt inputs made by scripts/fx5c/make_oracle_input.py. Inside a fleetq slot (--load cpu).
# Usage: oracle.sh <outname> <tree> <name> ...   reads/writes out/<outname>/<name>/{input,oracle.dump}.safetensors
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}"
P=$HOME/intarsia-decfuse; OUT=$P/out/$1; T=$2; shift 2
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
B=$P/$T/target/acc/release/intarsia
for n in "$@"; do
  D=$OUT/$n
  echo "== oracle $n $(date +%T)"
  /usr/bin/time -v $B accuracy --model $M35 --backend cpu --golden $D/input.safetensors --mode prefill --prefill-chunk 256 --mem-budget 40 --max-seq 8192 \
    --allow-dirty --write-self-golden $D/oracle.dump.safetensors < /dev/null > $D/oracle.log 2>&1
  echo "exit $?"; grep -aE "Maximum resident|Elapsed" $D/oracle.log
done
echo "== done $(date +%T)"
