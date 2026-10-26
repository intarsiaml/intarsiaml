#!/usr/bin/env bash
# Lane lb, ON tatooine inside a fleetq slot that takes radiance down (card 0 only): the correctness gates of the LDS-barrier change.
# Needs build.sh (dev tests built "tests", base binary). Logs: ~/intarsiaml-lb/logs/gates. A trap restores radiance.
#   gates.sh [sections: "HIPT QUICK IDS27 IDS GC"]
#   HIPT  : every test of the HIP crate (rows_exact, moe_pairs, moe_block, quantmoe, fp8_*, ...), one thread
#   QUICK : intarsia kernels test --backend hip --quick (dev binary)
#   IDS   : 256 greedy ids, base vs dev, 8B and 35B, short and 4k prompt (scripts/hg/noop.sh, + plan show)
#   IDS27 : 256 greedy ids, base vs dev, the 27B MXFP4 (short and 4k)
#   GC    : G-C, the 35B batched == solo (batch_cuda_real real_35b..., batch_recurrent) on the HIP backend
set -uo pipefail
SECTIONS=${*:-HIPT QUICK IDS27 IDS GC}
LB=$HOME/intarsiaml-lb
O=$LB/logs/gates; mkdir -p "$O"
D=$LB/dev/deploy/tatooine/dev.sh
RD=$LB/dev; RB=$LB/base
VD=intarsia-cargo-lb-dev; VB=intarsia-cargo-lb-base
M8=/models/Qwen3-8B; M35=/models/Qwen3.6-35B-A3B-NVFP4
HUB=/data/models/radiance/hf-cache/hub
M27=/models/models--amd--Qwen3.8-27B-Quark-AWQ-MXFP4/snapshots/5233554c5fa56afda40150556b95573c2d7d29c0
export INTARSIA_REGISTRY_VOL=intarsia-cargo-lb-registry
want() { [[ " $SECTIONS " == *" $1 "* ]]; }
trap '$LB/dev/scripts/lb/win.sh restore' EXIT
$LB/dev/scripts/lb/win.sh stop
dev() { # dev <repo> <vol> <cmd...>
  r=$1; v=$2; shift 2
  INTARSIA_REPO=$r INTARSIA_TARGET_VOL=$v HIP_VISIBLE_DEVICES=0 $D "$@" < /dev/null
}
F='^test |^test result|FAILED|panicked|^error|pairs|bitwise|launches|negative|SKIPPED|PASS|FAIL'
if want HIPT; then
  echo "[gates] HIPT $(date +%T)"
  dev $RD $VD env INTARSIA_REQUIRE_GPU=1 cargo test --release --no-fail-fast -p intarsia-backend-hip -- --test-threads=1 2>&1 | tee "$O/hipt.full" | grep -aE "$F" | tee "$O/hipt.log" | tail -150
fi
if want QUICK; then
  echo "[gates] QUICK $(date +%T)"
  dev $RD $VD /target/release/intarsia kernels test --backend hip --device 0 --quick 2>&1 | tee "$O/quick.log" | tail -25
  echo "quick exit ${PIPESTATUS[0]}"
fi
if want IDS; then
  echo "[gates] IDS $(date +%T)"
  for t in base dev; do
    if [ $t = base ]; then r=$RB; v=$VB; else r=$RD; v=$VD; fi
    RUN="env INTARSIA_REPO=$r INTARSIA_TARGET_VOL=$v HIP_VISIBLE_DEVICES=0 $D /target/release/intarsia" \
      bash $RD/scripts/hg/noop.sh run "$O" $t hip $M8 $M35 $RD/scripts/hg/prompts < /dev/null 2>&1 | tee -a "$O/ids.log"
  done
  bash $RD/scripts/hg/noop.sh cmp "$O" base dev | tee -a "$O/ids.log"
fi
if want IDS27; then
  echo "[gates] IDS27 $(date +%T)"
  P4=$RD/scripts/hip-decode/prompt4k.txt
  for t in base dev; do
    if [ $t = base ]; then r=$RB; v=$VB; else r=$RD; v=$VD; fi
    for p in short 4k; do
      if [ $p = short ]; then prompt='Write a long story about a robot who learns to paint.'; else prompt=$(cat $P4); fi
      INTARSIA_REPO=$r INTARSIA_TARGET_VOL=$v INTARSIA_MODELS=$HUB HIP_VISIBLE_DEVICES=0 $D /target/release/intarsia generate -m $M27 --backend hip \
        --mem-budget 29 --max-seq 8192 --profile-every 0 --ignore-eos --show-ids -p "$prompt" --max-tokens 256 > "$O/ids27-$p-$t.out" 2> "$O/ids27-$p-$t.err" < /dev/null
      grep -a '^ids:' "$O/ids27-$p-$t.err" > "$O/ids27-$p-$t.ids"
      echo "[ids27] $t $p: $(wc -c < "$O/ids27-$p-$t.ids") bytes md5 $(md5sum < "$O/ids27-$p-$t.ids" | cut -c1-8)"
    done
  done
  for p in short 4k; do
    if cmp -s "$O/ids27-$p-base.ids" "$O/ids27-$p-dev.ids" && [ -s "$O/ids27-$p-base.ids" ]; then echo "[ids27] $p IDENTICAL"; else echo "[ids27] $p DIFFER"; fi
  done | tee -a "$O/ids.log"
fi
if want GC; then
  echo "[gates] GC $(date +%T)"
  dev $RD $VD env INTARSIA_BATCH_TEST_BACKEND=hip INTARSIA_REQUIRE_GPU=1 INTARSIA_QWEN36_DIR=$M35 cargo test --release --no-fail-fast -p intarsia-models --test batch_cuda_real --test batch_recurrent -- --test-threads=1 --nocapture 2>&1 | grep -aE "$F" | tee "$O/g-c.log"
fi
echo "[gates] done $(date +%T)"
