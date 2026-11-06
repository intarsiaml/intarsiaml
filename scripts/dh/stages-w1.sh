#!/usr/bin/env bash
# Lane dh window w1 (ON tatooine, radiance down; PRE=w1 of cat-window.sh): items 0-3, everything but the category ABAB.
#   plan show, base (main 982ec8b27, dh-intarsia-base) vs dev (the branch, dh-intarsia), 27B tp2: --spec off, --spec mtp k2 (must be identical),
#   --spec dflash k7 (item 0: the window kernel planned); dev --spec dflash k7 with INTARSIA_DFLASH_TP=1 (recorded).
#   generate, short-0 and 4k-1, INTARSIA_SPEC_TRACE=1, ids against off:
#     A (knobs unset) / B (INTARSIA_DFLASH_TP=1) / C (B + INTARSIA_DFLASH_DEVICE_TOPK=1) in the order A B C A B C (item 1: A B A B, item 2: B C B C);
#     K = B with the device top-k in check mode;
#     item 3 on top of C: D (INTARSIA_HIP_AR_LL_MAX=262144), E (INTARSIA_HIP_A8_ILP=2), F (INTARSIA_HIP_A8_ILP=4), runs C D E F C D E F.
#   item 3 data: MTP k1..k6 traces at short-0 (verify + rollback against rows); profiled op tables of off, dfl7 A, dfl7 C.
#   item 3b kernel gates: tests/mxfp4_a8.rs + tests/rows_exact.rs at INTARSIA_HIP_A8_ILP unset / 2 / 4; VRC_BENCH=1 verify_gemv_bench at 1 / 2 / 4.
set -uo pipefail
export O=${1:?out dir}
source "$HOME/intarsiaml-dh/.dh/lib.sh"
D=dev
clear "w1" || exit 2
TRK="$(T "$D")/scripts/dfh/ids-prompts"
DFL="--spec dflash --spec-draft $DRAFT --spec-k 7"
B="INTARSIA_DFLASH_TP=1"; C="INTARSIA_DFLASH_TP=1 INTARSIA_DFLASH_DEVICE_TOPK=1"
declare -A ENVS=([A]="" [B]="$B" [C]="$C" [K]="INTARSIA_DFLASH_TP=1 INTARSIA_DFLASH_DEVICE_TOPK=check"
  [D]="$C INTARSIA_HIP_AR_LL_MAX=262144" [E]="$C INTARSIA_HIP_A8_ILP=2" [F]="$C INTARSIA_HIP_A8_ILP=4")
# 3b kernel gates and bench first (one GPU each, no model): a failing bitwise gate stops 3b's arms below
for il in 1 2 4; do
  E=$([[ $il == 1 ]] && echo "" || echo "INTARSIA_HIP_A8_ILP=$il") htest "$D" "a8-ilp$il" 0 600 --test mxfp4_a8 --test rows_exact
  grep -aE 'hash|fac47447935a8383' "$O/htest-a8-ilp$il.log" | head -4 > "$O/hash-ilp$il.txt"
  E="VRC_BENCH=1 $([[ $il == 1 ]] || echo "INTARSIA_HIP_A8_ILP=$il")" htest "$D" "bench-ilp$il" 0 600 --test verify_gemv_bench -- --nocapture
done
for b in base dev; do
  BIN=dh-intarsia; [[ $b == base ]] && BIN=dh-intarsia-base
  BIN=$BIN plan "$D" 0,1 "$b-off" "$M27" $TP2 --spec off
  BIN=$BIN plan "$D" 0,1 "$b-mtp2" "$M27" $TP2 --spec mtp --spec-k 2
  # shellcheck disable=SC2086
  BIN=$BIN plan "$D" 0,1 "$b-dfl7" "$M27" $TP2 $DFL
done
same plan-base-off.out plan-dev-off.out "plan json 27B tp2 off"
same plan-base-mtp2.out plan-dev-mtp2.out "plan json 27B tp2 mtp k2"
same plantext-base-off.out plantext-dev-off.out "plan text 27B tp2 off"
same plantext-base-mtp2.out plantext-dev-mtp2.out "plan text 27B tp2 mtp k2"
log "item 0: window kernel in the dfl7 plan: base $(grep -c attn_decode_bf16_window "$O/plantext-base-dfl7.out") dev $(grep -c attn_decode_bf16_window "$O/plantext-dev-dfl7.out")"
# shellcheck disable=SC2086
E="INTARSIA_DFLASH_TP=1" plan "$D" 0,1 "dev-dfl7-tp" "$M27" $TP2 $DFL
trace() { # trace <gen tag>
  log "trace $1: $(grep -a 'verify + rollback' "$O/gen-$1.err" | tail -n +5 | awk '{s+=$(NF-1); d+=$4; c++} END{if(c)printf "mean draft %.2f verify+rollback %.2f ms n=%d", d/c, s/c, c}') | $(grep -a 'dflash2 draft flush' "$O/gen-$1.err" | tail -n +5 | awk '{f+=$5; b+=$7; h+=$9; sp+=$11; dl+=$13; se+=$15; t+=$17; c++} END{if(c)printf "dflash2 flush %.2f block %.2f head %.2f selproj %.2f dl %.2f select %.2f total %.2f n=%d", f/c,b/c,h/c,sp/c,dl/c,se/c,t/c,c}') | $(grep -aE 'decode .*tok/s' "$O/gen-$1.err" | tail -1 | cut -c1-160)"
}
for n in short-0 4k-1; do
  P=$(cat "$TRK/$n.txt")
  g() { # g <tag> <env> [args]
    local tag=$1 e=$2; shift 2
    E="INTARSIA_SPEC_TRACE=1 $e" run "$D" 0,1 420 "gen-$tag" generate -m "$M27" --backend hip --mem-budget 29 --max-seq 8192 --chat --no-think \
      --profile-every 0 --ignore-eos --show-ids -p "$P" --max-tokens 256 $TP2 "$@"
    grep -a '^ids:' "$O/gen-$tag.err" > "$O/gen-$tag.ids"
  }
  g "$n-off" "" --spec off
  for a in A1 B1 C1 A2 B2 C2 K D1 E1 F1 C3 D2 E2 F2 C4; do
    # shellcheck disable=SC2086
    g "$n-$a" "${ENVS[${a:0:1}]}" $DFL
    if cmp -s "$O/gen-$n-off.ids" "$O/gen-$n-$a.ids" && [[ -s "$O/gen-$n-off.ids" ]]; then log "PASS ids $a tp2 $n"; else log "FAIL ids $a tp2 $n"; fi
    trace "$n-$a"
  done
  log "topk check $n: $(grep -a 'dflash2 topk check: rank .* rows' "$O/gen-$n-K.err" | awk '{r+=$7; d+=$9; c++} END{printf "%d calls, %d rows, %d differing", c, r, d}')"
done
P=$(cat "$TRK/short-0.txt")
for kk in 1 2 3 4 5 6; do
  E="INTARSIA_SPEC_TRACE=1" run "$D" 0,1 420 "gen-mtp$kk" generate -m "$M27" --backend hip --mem-budget 29 --max-seq 8192 --chat --no-think \
    --profile-every 0 --ignore-eos --show-ids -p "$P" --max-tokens 256 $TP2 --spec mtp --spec-k $kk
  log "vcurve mtp k$kk ($((kk + 1)) rows): $(grep -a 'verify + rollback' "$O/gen-mtp$kk.err" | tail -n +5 | awk '{s+=$(NF-1); d+=$4; c++} END{if(c)printf "mean draft %.2f verify+rollback %.2f ms n=%d", d/c, s/c, c}') | $(grep -aE 'decode .*tok/s' "$O/gen-mtp$kk.err" | tail -1 | cut -c1-120)"
done
E="INTARSIA_SPEC_TRACE=1 INTARSIA_HIP_AR_LL_MAX=262144" run "$D" 0,1 420 "gen-mtp2-D" generate -m "$M27" --backend hip --mem-budget 29 --max-seq 8192 \
  --chat --no-think --profile-every 0 --ignore-eos --show-ids -p "$P" --max-tokens 256 $TP2 --spec mtp --spec-k 2
log "mtp k2 with INTARSIA_HIP_AR_LL_MAX=262144: $(grep -a 'verify + rollback' "$O/gen-mtp2-D.err" | tail -n +5 | awk '{s+=$(NF-1); d+=$4; c++} END{if(c)printf "mean draft %.2f verify+rollback %.2f ms n=%d", d/c, s/c, c}')"
for a in "off::--spec off" "dflA::$DFL" "dflC:$C:$DFL"; do
  tag=${a%%:*}; rest=${a#*:}; e=${rest%%:*}; args=${rest#*:}
  # shellcheck disable=SC2086
  E="$e" run "$D" 0,1 420 "optable-$tag" generate -m "$M27" --backend hip --mem-budget 29 --max-seq 8192 --chat --no-think \
    --profile-every 1 --op-table --ignore-eos -p "$P" --max-tokens 128 $TP2 $args
done
log "w1 stages done"
