#!/usr/bin/env bash
# Window 4 (spark1, vLLM down, inside scripts/cuda-decode/window.sh): `intarsia qualify --quick-only` of the dev build on the 35B,
# then the 8B A/B (base vs dev, 2 interleaved rounds). Steps are skipped when the deadline is near (W4_BUDGET_MIN, default 14).
set -u
P=$HOME/intarsia-decfuse
D=$(dirname "$0")
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
GD=$HOME/intarsiaml-golden
O=$P/out/w4; mkdir -p $O
DEV=$P/dev/target/acc/release/intarsia
DEADLINE=$(( $(date +%s) + ${W4_BUDGET_MIN:-14} * 60 ))
left() { echo $(( (DEADLINE - $(date +%s)) / 60 )); }
need() { if (( $(left) < $1 )); then echo "SKIP $2: $(left) min left, needs $1"; return 1; fi; echo "== $2 ($(left) min left) $(date +%T)"; return 0; }
gb=$(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) / 1024 / 1024 )); echo "MemAvailable ${gb} GB"; (( gb >= 60 )) || { echo "ABORT: memory"; exit 1; }
if need 5 "qualify --quick-only, 35B"; then
  rm -rf $O/qualify-35b
  $DEV qualify --model $M35 --backend cuda --quick-only --gen 128 --reps 5 --no-unim --out $O/qualify-35b --golden-dir $GD --mem-budget 45 > $O/qualify-35b.log 2>&1
  echo "qualify exit $?"; tail -40 $O/qualify-35b.log | cut -c1-200
fi
need 3 "8B A/B round 1" && bash $D/w-ab.sh 8b 1 base:base dev:dev
need 3 "8B A/B round 2" && bash $D/w-ab.sh 8b 2 dev:dev base:base
echo "== w4 done $(date +%T), $(left) min left"
