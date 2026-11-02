#!/bin/bash
# MNARROW harness A/B: batch_metal_real::real_35b_batch_step_real_traffic, arms interleaved per round (rotated, reversed on even
# rounds), B order flipped per round; plain (wall) and ce1 (per-dispatch GPU records) runs.
# usage: ARMS="name,bintag,KEY=VAL ..." w1ab.sh OUTDIR [ROUNDS]   (inside a fleetq slot). bintag = .lane/bins/<tag>/batch_metal_real
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
W=/opt/code/intarsiaml/.worktrees/mnarrow; O=$1; R=${2:-3}; mkdir -p $O
export INTARSIA_TEST_REAL_MODEL=1 INTARSIA_REQUIRE_GPU=1 INTARSIA_QWEN36_DIR=$HOME/models/Qwen3.6-35B-A3B-NVFP4 MM_STEPS=32
ARMS=${ARMS:?}
CE1=${CE1:-1}
armfield() { echo "$1" | cut -d, -f$2; }
run() { local tag=$1 arm=$2 bs=$3; shift 3
  local bin=$W/.lane/bins/$(armfield $arm 2)/batch_metal_real kv=$(armfield $arm 3)
  echo "## $tag $(date +%T)"
  env $kv MM_BATCHES=$bs "$@" $bin --ignored real_35b_batch_step_real_traffic --nocapture --test-threads=1 > $O/$tag.log 2>&1
  echo "exit $?"; grep REALTRAFFIC $O/$tag.log; }
for r in $(seq 1 $R); do
  set -- $ARMS; n=$#; list=""; for i in $(seq 0 $((n-1))); do idx=$(( (i + r - 1) % n + 1 )); eval "a=\${$idx}"; list="$list $a"; done
  if [ $((r % 2)) = 1 ]; then PB=1,4,8,9,12,16; CB=8,9,12,16; else PB=16,12,9,8,4,1; CB=16,12,9,8; list=$(echo $list | tr ' ' '\n' | tail -r | tr '\n' ' '); fi
  for a in $list; do run plain-$(armfield $a 1)-r$r $a $PB; done
  if [ "$CE1" = 1 ]; then
    for a in $(echo $list | tr ' ' '\n' | tail -r | tr '\n' ' '); do run ce1-$(armfield $a 1)-r$r $a $CB INTARSIA_METAL_COMMIT_EVERY=1 INTARSIA_METAL_PROFILE=cb:$O/ce1-$(armfield $a 1)-r$r.jsonl; done
  fi
done
gzip -f $O/*.jsonl 2>/dev/null
echo "## done $(date +%T)"
