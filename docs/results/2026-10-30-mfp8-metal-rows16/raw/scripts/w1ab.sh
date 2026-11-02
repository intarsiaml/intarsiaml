#!/bin/bash
# MFP8 harness A/B: batch_metal_real::real_35b_batch_step_real_traffic, base (default: the two 8-row launches
# of main) vs r16 (INTARSIA_METAL_FP8_ROWS16=1: one 16-row launch), same binary, arms interleaved per round, B order flipped per round.
# usage: w1ab.sh OUTDIR [ROUNDS]   (inside a fleetq slot)
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
W=/opt/code/intarsiaml/.worktrees/mfp8; O=$1; R=${2:-3}; mkdir -p $O
BR=$(ls -t $W/.lane/target/release/deps/batch_metal_real-* | grep -v '\.d$' | head -1)
export INTARSIA_TEST_REAL_MODEL=1 INTARSIA_REQUIRE_GPU=1 INTARSIA_QWEN36_DIR=$HOME/models/Qwen3.6-35B-A3B-NVFP4 MM_STEPS=32
ARMS=${ARMS:-"base r16"}
armenv() { case $1 in r16) echo "INTARSIA_METAL_FP8_ROWS16=1";; *) echo "X=1";; esac; }  # base = default (two 8-row launches); NOTE the runs in raw/w1ab* used a build where the 16-row form was default-on and base was ROWS16=0
run() { local tag=$1 arm=$2 bs=$3; shift 3
  echo "## $tag $(date +%T)"
  env $(armenv $arm) MM_BATCHES=$bs "$@" $BR --ignored real_35b_batch_step_real_traffic --nocapture --test-threads=1 > $O/$tag.log 2>&1
  echo "exit $?"; grep REALTRAFFIC $O/$tag.log; }
for r in $(seq 1 $R); do
  # arms rotate each round (and reverse on even rounds), B order flips on even rounds
  set -- $ARMS; n=$#; list=""; for i in $(seq 0 $((n-1))); do idx=$(( (i + r - 1) % n + 1 )); eval "a=\${$idx}"; list="$list $a"; done
  if [ $((r % 2)) = 1 ]; then PB=1,4,8,9,12,16; CB=8,9,12,16; else PB=16,12,9,8,4,1; CB=16,12,9,8; list=$(echo $list | tr ' ' '\n' | tail -r | tr '\n' ' '); fi
  for a in $list; do run plain-$a-r$r $a $PB; done
  for a in $(echo $list | tr ' ' '\n' | tail -r | tr '\n' ' '); do run ce1-$a-r$r $a $CB INTARSIA_METAL_COMMIT_EVERY=1 INTARSIA_METAL_PROFILE=cb:$O/ce1-$a-r$r.jsonl; done
done
echo "## done $(date +%T)"
