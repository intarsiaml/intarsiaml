#!/usr/bin/env bash
# Lane mp, spark side: `intarsia qualify` accuracy stages (no bench/ab/tp) for the 35B on CUDA, one run per arm. Inside a fleetq slot.
# Usage: qualify.sh <outname> <arm> ...   arm = name:tree[:ENV=V,ENV=V]
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}"
P=$HOME/intarsia-decfuse; OUT=$P/out/$1; shift
export PATH=/usr/local/cuda/bin:$PATH
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
mkdir -p $OUT
for spec in "$@"; do
  IFS=: read -r name tree envs <<< "$spec"; envs=${envs//,/ }
  d=$OUT/$name; rm -rf $d; mkdir -p $d
  echo "== qualify $name tree=$tree env='$envs' $(date +%T)"
  env $envs timeout ${QTIMEOUT:-5400} $P/$tree/target/acc/release/intarsia qualify --model $M35 --backend cuda --golden-dir $P/pf-golden \
    --skip bench --skip bench-long --skip ab --skip tp --skip doctor --skip kernels --no-unim --mem-budget 45 --out $d/report --stage-timeout-min 40 \
    < /dev/null > $d/qualify.log 2>&1
  echo "qualify $name exit $? $(date +%T)"
  grep -aE "PASS|FAIL|SKIP|ERROR|MISSING" $d/qualify.log | tail -40 | cut -c1-220
done
echo "== done $(date +%T)"
