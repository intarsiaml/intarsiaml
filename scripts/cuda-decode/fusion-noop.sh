#!/usr/bin/env bash
# CUDA no-op window for core-op fusion phase 1 (docs/results/2026-10-19-core-op-fusion-p1.md). Runs ON spark1 inside
# scripts/cuda-decode/window.sh (vLLM down):
#   scripts/cuda-decode/window.sh 16 "core-op-fusion P1 CUDA no-op" '~/intarsia-decfuse/fusion-dev/scripts/cuda-decode/fusion-noop.sh'
# A = ~/intarsia-decfuse/fusion-base (origin/main code), B = fusion-dev (the branch). CUDA registers none of the new ops, so B
# must be A: (1) w-noop.sh lanes without the speed rounds (35b-short golden on both + self-golden cmp, 256 greedy ids of the
# 35B and the 8B, graphs and eager, the two 8B fp32 goldens on B); (2) `plan show --json` of the 35B and the 8B on both,
# byte-compared; (3) `kernels test --backend cuda --quick` kernel / case / pass counts on both, for the 35B and the 8B.
set -u
P=$HOME/intarsia-decfuse
D=$(dirname "$0")
A=fusion-base; B=fusion-dev
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
M8=/home/nvidia/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
O=$P/out/noop-$B
BA=$P/$A/target/acc/release/intarsia; BB=$P/$B/target/acc/release/intarsia
echo "== fusion-noop start $(date +%T)"
W_BUDGET_MIN=${W_BUDGET_MIN:-9} bash $D/w-noop.sh $A $B nospeed
mkdir -p $O/plans $O/adm
echo "== plans $(date +%T)"
for t in A B; do
  bin=$BA; [ $t = B ] && bin=$BB
  $bin plan show --backend cuda --model $M35 --mem-budget 29 --max-seq 4096 --json > $O/plans/$t-35b.json 2> $O/plans/$t-35b.err; echo "plan $t 35b exit $?"
  $bin plan show --backend cuda --model $M8 --mem-budget 25 --max-seq 4096 --json > $O/plans/$t-8b.json 2> $O/plans/$t-8b.err; echo "plan $t 8b exit $?"
done
for m in 35b 8b; do
  if cmp -s $O/plans/A-$m.json $O/plans/B-$m.json; then echo "PLAN $m JSON IDENTICAL ($(md5sum < $O/plans/A-$m.json | cut -c1-12))"; else echo "PLAN $m JSON DIFFERS"; diff $O/plans/A-$m.json $O/plans/B-$m.json | head -20; fi
done
python3 $D/../hip-decode/plan_diff.py $O/plans B A 2>&1 | tail -12
echo "== admission $(date +%T)"
for t in A B; do
  bin=$BA; [ $t = B ] && bin=$BB
  for m in 35b 8b; do
    mp=$M35; [ $m = 8b ] && mp=$M8
    timeout 150 $bin kernels test --backend cuda --model $mp --quick > $O/adm/$t-$m.log 2>&1; echo "kernels test $t $m exit $?: $(grep -aE 'kernels,|cases' $O/adm/$t-$m.log | tail -1 | cut -c1-160)"
  done
done
for m in 35b 8b; do
  if diff <(grep -avE "wall|took|elapsed|[0-9]+(\.[0-9]+)? ?(ms|s)\b" $O/adm/A-$m.log) <(grep -avE "wall|took|elapsed|[0-9]+(\.[0-9]+)? ?(ms|s)\b" $O/adm/B-$m.log) > /dev/null; then
    echo "ADMISSION $m IDENTICAL (timing lines aside)"
  else
    echo "ADMISSION $m DIFFERS"; diff <(grep -avE "wall|took|elapsed" $O/adm/A-$m.log) <(grep -avE "wall|took|elapsed" $O/adm/B-$m.log) | head -12
  fi
done
echo "== fusion-noop done $(date +%T)"
