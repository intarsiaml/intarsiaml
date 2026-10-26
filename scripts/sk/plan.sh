#!/usr/bin/env bash
# Lane sk: `plan show` of trees/arms on spark2 (the no-op check with the knob off, and the memory plan with it on).
#   fleetq run --host spark2 --mem 30 --expect 8 --label "sk: plan show" -- scripts/sk/plan.sh <outname> <arm> ...    arm = name:tree[:ENV=V,ENV=V]
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
OUTN=$1; shift; HOST=${HOST:-spark2}
RAW=/opt/code/intarsiaml/.worktrees/sk3/docs/results/2026-10-26-sk3-spark-decode/raw/$OUTN; mkdir -p "$RAW"
for a in "$@"; do
  IFS=: read -r name tree envs <<< "$a"; envs=${envs//,/ }
  ssh -n -o BatchMode=yes "$HOST" "export FLEETQ_JOB=$FLEETQ_JOB; M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
    env $envs timeout 600 \$HOME/intarsiaml-sk-$tree/target/acc/release/intarsia plan show --backend cuda --model \$M35 --max-seq 20480 --mem-budget 45 < /dev/null" > "$RAW/$name.txt" 2> "$RAW/$name.err"
  ssh -n -o BatchMode=yes "$HOST" "export FLEETQ_JOB=$FLEETQ_JOB; M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
    env $envs timeout 600 \$HOME/intarsiaml-sk-$tree/target/acc/release/intarsia plan show --backend cuda --model \$M35 --max-seq 20480 --mem-budget 45 --json < /dev/null" > "$RAW/$name.json" 2>> "$RAW/$name.err"
  echo "$name: $(wc -c < "$RAW/$name.txt") B text, $(wc -c < "$RAW/$name.json") B json, md5 $(md5 -q "$RAW/$name.json")"
done
