#!/bin/bash
# on endor inside a spark2 slot: run the matmul_mxfp4 sweep (quick + full), the whole quick sweep, and plan show for head and origin/main
set -uo pipefail
O=/opt/code/intarsiaml/.worktrees/m27-gemv/docs/results/2026-11-02-m27-gemv/raw/cuda; mkdir -p $O
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
M8=/home/nvidia/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
for n in head main; do
  B='$HOME/intarsia-deploy/m27-gemv-'$n'/target/release/intarsia'
  ssh -n -o BatchMode=yes spark2 "cat \$HOME/intarsia-deploy/m27-gemv-$n/DEPLOY_STAMP.json 2>/dev/null | head -c 600; echo" > $O/$n-stamp.txt 2>&1
  ssh -n -o BatchMode=yes spark2 "$B kernels test --backend cuda --op matmul_mxfp4 --quick -v" > $O/$n-mxfp4-quick.log 2>&1; echo "$n mxfp4 quick rc=$?"
  ssh -n -o BatchMode=yes spark2 "$B kernels test --backend cuda --op matmul_mxfp4 -v" > $O/$n-mxfp4-full.log 2>&1; echo "$n mxfp4 full rc=$?"
  ssh -n -o BatchMode=yes spark2 "$B kernels test --backend cuda --quick" > $O/$n-all-quick.log 2>&1; echo "$n all quick rc=$?"
  ssh -n -o BatchMode=yes spark2 "$B plan show -m $M35 --backend cuda 2>&1" > $O/$n-plan-35b.txt; echo "$n plan 35b rc=$?"
  ssh -n -o BatchMode=yes spark2 "$B plan show -m $M8 --backend cuda 2>&1" > $O/$n-plan-8b.txt; echo "$n plan 8b rc=$?"
done
for m in 35b 8b; do
  if diff -q <(grep -v -iE 'built|stamp|took|loaded in|ms\b' $O/main-plan-$m.txt) <(grep -v -iE 'built|stamp|took|loaded in|ms\b' $O/head-plan-$m.txt) >/dev/null; then echo "plan show $m: identical ($(wc -l < $O/head-plan-$m.txt) lines)"; else echo "plan show $m: DIFFERS"; fi
done
