#!/usr/bin/env bash
# Lane dhm: pre-merge CUDA no-op of the lane dh head against origin/main, ON spark1 inside one fleetq slot, with vLLM left up:
# `plan show --json` (35B NVFP4 and 8B) base vs head must be byte-identical, then `kernels test --quick` on the head. No model is loaded.
#   scripts/dh/cuda-noop.sh sync <base-rev> <head-rev>      (Mac; git archive of both revs to spark1 ~/intarsia-decfuse/dhm-src-{base,dev})
#   fleetq run --host spark1 --mem 20 --expect 40 --label "dhm cuda no-op" -- ssh spark1 'bash ~/intarsia-decfuse/dhm-src-dev/scripts/dh/cuda-noop.sh run'
# Build load is capped for the 121 GB unified memory: CARGO_BUILD_JOBS=4, MAX_JOBS=4, FLASHINFER_NVCC_THREADS=1.
set -uo pipefail
P=$HOME/intarsia-decfuse
if [ "${1:?sync|run}" = sync ]; then
  cd "$(git rev-parse --show-toplevel)"
  for spec in base=$2 dev=$3; do
    n=${spec%%=*}; r=${spec#*=}
    git archive --format=tar "$r" | ssh -o BatchMode=yes spark1 "rm -rf ~/intarsia-decfuse/dhm-src-$n && mkdir -p ~/intarsia-decfuse/dhm-src-$n && tar -x -C ~/intarsia-decfuse/dhm-src-$n"
    echo "$n=$(git rev-parse --short "$r")"
  done
  exit 0
fi
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH CARGO_BUILD_JOBS=4 MAX_JOBS=4 FLASHINFER_NVCC_THREADS=1
O=$P/out/dhm; mkdir -p $O
H=/home/nvidia/.cache/huggingface/hub
M35=$(ls -d $H/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/* | head -1)
M8=$(ls -d $H/models--Qwen--Qwen3-8B/snapshots/* | head -1)
echo "free GB: $(free -g | awk '/Mem:/ {print $7}')"
for t in base dev; do
  echo "== build $t $(date +%T)"
  (cd $P/dhm-src-$t && CARGO_TARGET_DIR=$P/dhm-src-$t/target/acc cargo build --release --features cuda -p intarsia-cli 2>&1 | grep -E '^error|Finished' | tail -3)
  [ -x $P/dhm-src-$t/target/acc/release/intarsia ] || { echo "BUILD FAILED $t"; exit 1; }
  for m in 35b:$M35 8b:$M8; do
    $P/dhm-src-$t/target/acc/release/intarsia plan show --backend cuda --model ${m#*:} --max-seq 8192 --mem-budget 30 --json < /dev/null > $O/plan-${m%%:*}-$t.json 2> $O/plan-${m%%:*}-$t.err
  done
done
for m in 35b 8b; do
  cmp -s $O/plan-$m-base.json $O/plan-$m-dev.json && [ -s $O/plan-$m-dev.json ] && echo "plan $m json: byte-identical ($(wc -c < $O/plan-$m-dev.json) B)" || echo "plan $m json: DIFFER or empty"
done
echo "== kernels test --quick (dev) $(date +%T)"
$P/dhm-src-dev/target/acc/release/intarsia kernels test --quick --backend cuda < /dev/null > $O/kernels-quick.log 2>&1
echo "exit $? FAIL lines: $(grep -c FAIL $O/kernels-quick.log)"; tail -3 $O/kernels-quick.log
echo "== done $(date +%T)"
