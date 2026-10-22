#!/usr/bin/env bash
# Lane T2 window 2 (ON tatooine, radiance down): no-op and tp=2 identity gates, kernels, HIP crate GPU tests.
#   base = 0d2d6eda (P1 #67 + H1 #59, no T2 change), head = this branch.
#   1. tp=1 no-op, 256 greedy ids base vs head: 27B short / 4k (and P1's md5s 2086b6a4 / ef88d237), 35B short / 4k, 8B short.
#   2. tp=2 no-op on the models that had tp=2 before: 8B short (bf16 dense FFN arm), 35B short (NVFP4 / MoE / GDN).
#   3. 27B tp=2 4k: eager vs graphs vs graphs + fusion (must be identical), first divergence from tp=1.
#   4. kernels test --backend hip --model <27B>; the HIP crate's GPU tests (both cards, comm tests included).
#   5. rocprofv3 decode traces after a 1k prompt: tp=1, tp=2 graphs, tp=2 graphs + fusion (per-rank step table, prof2.py).
set -uo pipefail
export O=${1:?out dir}
source "$HOME/t2/lib.sh"
for t in base head; do
  ids $t 0 27-tp1-short-$t $M27 short
  ids $t 0 27-tp1-4k-$t $M27 4k
done
same ids-27-tp1-short-base.ids ids-27-tp1-short-head.ids "27B tp=1 short base vs head"
same ids-27-tp1-4k-base.ids ids-27-tp1-4k-head.ids "27B tp=1 4k base vs head"
for t in base head; do
  ids $t 0 35-tp1-short-$t $M35 short
  ids $t 0 35-tp1-4k-$t $M35 4k
  ids $t 0 8-tp1-short-$t $M8 short
  ids $t 0,1 8-tp2-short-$t $M8 short $TP2
  ids $t 0,1 35-tp2-short-$t $M35 short $TP2
done
for x in 35-tp1-short 35-tp1-4k 8-tp1-short 8-tp2-short 35-tp2-short; do same ids-$x-base.ids ids-$x-head.ids "$x base vs head"; done
ids head 0,1 27-tp2ng-4k $M27 4k $TP2 --no-graphs
ids head 0,1 27-tp2g-4k $M27 4k $TP2
E="INTARSIA_TP_FUSION=1" ids head 0,1 27-tp2gf-4k $M27 4k $TP2
same ids-27-tp2ng-4k.ids ids-27-tp2g-4k.ids "27B tp=2 4k graphs vs eager"
same ids-27-tp2g-4k.ids ids-27-tp2gf-4k.ids "27B tp=2 4k graphs+fusion vs graphs"
firstdiff ids-27-tp1-4k-head.ids ids-27-tp2g-4k.ids "27B tp=2 vs tp=1 4k"
log "kernels test 27B"
run head 0 900 kernels-27b "$HUB" kernels test --backend hip --model $M27
grep -aE "FAIL|REJECT|kernels,|mxfp4" "$O/kernels-27b.out" "$O/kernels-27b.err" | tail -12 | tee -a "$O/steps.log"
log "HIP crate GPU tests"
INTARSIA_REPO=$(T head) INTARSIA_TARGET_VOL=intarsia-cargo-target-t2-head-hiptest HIP_VISIBLE_DEVICES=0,1 timeout 900 \
  "$(T head)/deploy/tatooine/dev.sh" cargo test --release -p intarsia-backend-hip > "$O/hiptest.log" 2>&1 < /dev/null
log "hiptest exit $?: $(grep -E '^test result' "$O/hiptest.log" | awk '{p+=$4; f+=$6} END {print p" passed, "f" failed"}') $(grep -cE 'FAILED|panicked' "$O/hiptest.log") FAILED/panicked lines"
prof tp1 0
prof tp2g 0,1 $TP2
E="INTARSIA_TP_FUSION=1" prof tp2gf 0,1 $TP2
log done
