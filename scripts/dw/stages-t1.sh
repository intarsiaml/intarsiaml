#!/usr/bin/env bash
# Lane dw window t1 (ON tatooine, radiance down; scripts/dw/window.sh with W=t1): the win-split gates and traces.
#   1. card 0: `kernels test --backend hip --quick` in the background; card 1: tests/win_splitkv.rs, then WS_BENCH=1 (microseconds a call, old vs new).
#   2. plan show, base (dw-intarsia-base = main 3f46da4f7) vs dev (dw-intarsia), knob unset: 27B tp2 off / mtp2 / dfl7, 35B tp2 off / mtp2, 8B tp1 off;
#      dev with INTARSIA_DFLASH_WIN_SPLIT=1 at serve's --max-seq 20480: the new kernel must be in the dfl7 plan.
#   3. generate, short-0 and 4k-1, INTARSIA_SPEC_TRACE=1, ids against --spec off, arms A B C D A B C D (per prompt):
#      A = knobs unset (whole draft), B = WIN_SPLIT, C = INTARSIA_DFLASH_TP, D = C + WIN_SPLIT. DEVICE_TOPK is on by default.
set -uo pipefail
export O=${1:?out dir}
source "$HOME/intarsiaml-dh/.dw/lib.sh"
D=dev
clear "t1" || exit 2
TRK="$(T "$D")/scripts/dfh/ids-prompts"
DFL="--spec dflash --spec-draft $DRAFT --spec-k 7"
# 1. kernel gates
if [[ ${DRY:-0} == 1 ]]; then log "DRY kernels test --backend hip --quick gpu 0"; else
  ( INTARSIA_REPO=$(T "$D") HIP_VISIBLE_DEVICES=0 timeout 1500 "$(T "$D")/deploy/tatooine/dev.sh" /work/dw-intarsia kernels test --backend hip --quick \
      > "$O/kernels-test.log" 2>&1 < /dev/null; echo "exit $?" >> "$O/kernels-test.log" ) &
  KT=$!
fi
E="" htest "$D" "win-splitkv" 1 900 --test win_splitkv -- --nocapture
E="WS_BENCH=1" htest "$D" "win-bench" 1 900 --test win_splitkv window_splitkv_bench -- --nocapture
log "WSBENCH: $(grep -ah '^WSBENCH' "$O/htest-win-bench.log" | tr '\n' '|' | cut -c1-1500)"
if [[ -n ${KT:-} ]]; then wait "$KT"; log "kernels test: $(tail -4 "$O/kernels-test.log" | tr '\n' '|' | cut -c1-300)"; log "kernels test win kernel: $(grep -a 'attn_decode_bf16_window' "$O/kernels-test.log" | tr '\n' '|' | cut -c1-400)"; fi
# 2. plans
for b in base dev; do
  BIN=dw-intarsia; [[ $b == base ]] && BIN=dw-intarsia-base
  BIN=$BIN plan "$D" 0,1 "$b-off" "$M27" $TP2 --spec off
  BIN=$BIN plan "$D" 0,1 "$b-mtp2" "$M27" $TP2 --spec mtp --spec-k 2
  # shellcheck disable=SC2086
  BIN=$BIN plan "$D" 0,1 "$b-dfl7" "$M27" $TP2 $DFL
  BIN=$BIN plan "$D" 0,1 "$b-35off" "$M35" $TP2 --spec off
  BIN=$BIN plan "$D" 0,1 "$b-35mtp2" "$M35" $TP2 --spec mtp --spec-k 2
  BIN=$BIN plan "$D" 0 "$b-8off" "$M8" --spec off
done
for t in off mtp2 dfl7 35off 35mtp2 8off; do
  same "plan-base-$t.out" "plan-dev-$t.out" "plan json $t"
  same "plantext-base-$t.out" "plantext-dev-$t.out" "plan text $t"
done
# shellcheck disable=SC2086
E="INTARSIA_DFLASH_WIN_SPLIT=1" run "$D" 0,1 300 "plantext-dev-dfl7-ws-20k" plan show --backend hip --model "$M27" --mem-budget 29 --max-seq 20480 $TP2 $DFL
# shellcheck disable=SC2086
E="INTARSIA_DFLASH_WIN_SPLIT=1 INTARSIA_DFLASH_TP=1" run "$D" 0,1 300 "plantext-dev-dfl7-ws-tp-20k" plan show --backend hip --model "$M27" --mem-budget 29 --max-seq 20480 $TP2 $DFL
log "plan lists the new kernel: ws $(grep -ac attn_decode_bf16_window_splitkv "$O/plantext-dev-dfl7-ws-20k.out") ws+tp $(grep -ac attn_decode_bf16_window_splitkv "$O/plantext-dev-dfl7-ws-tp-20k.out") unset $(grep -ac attn_decode_bf16_window_splitkv "$O/plantext-dev-dfl7.out")"
# 3. traces
trace() {
  log "trace $1: $(grep -a 'verify + rollback' "$O/gen-$1.err" | tail -n +5 | awk '{s+=$(NF-1); d+=$4; c++} END{if(c)printf "mean draft %.2f verify+rollback %.2f pass %.2f ms n=%d", d/c, s/c, (d+s)/c, c}') | $(grep -a 'dflash2 draft flush' "$O/gen-$1.err" | tail -n +5 | awk '{f+=$5; b+=$7; h+=$9; sp+=$11; dl+=$13; se+=$15; t+=$17; c++} END{if(c)printf "dflash2 flush %.2f block %.2f head %.2f selproj %.2f dl %.2f select %.2f total %.2f n=%d", f/c,b/c,h/c,sp/c,dl/c,se/c,t/c,c}') | $(grep -aE 'decode .*tok/s' "$O/gen-$1.err" | tail -1 | cut -c1-160) | $(grep -aE 'tokens per pass' "$O/gen-$1.err" | tail -1 | cut -c1-120)"
}
declare -A ENVS=([A]="" [B]="INTARSIA_DFLASH_WIN_SPLIT=1" [C]="INTARSIA_DFLASH_TP=1" [D]="INTARSIA_DFLASH_TP=1 INTARSIA_DFLASH_WIN_SPLIT=1")
for n in short-0 4k-1; do
  P=$(cat "$TRK/$n.txt")
  g() { local tag=$1 e=$2; shift 2
    E="INTARSIA_SPEC_TRACE=1 $e" run "$D" 0,1 420 "gen-$tag" generate -m "$M27" --backend hip --mem-budget 29 --max-seq 8192 --chat --no-think \
      --profile-every 0 --ignore-eos --show-ids -p "$P" --max-tokens 256 $TP2 "$@"
    grep -a '^ids:' "$O/gen-$tag.err" > "$O/gen-$tag.ids" 2>/dev/null; }
  g "$n-off" "" --spec off
  for a in A1 B1 C1 D1 A2 B2 C2 D2; do
    # shellcheck disable=SC2086
    g "$n-$a" "${ENVS[${a:0:1}]}" $DFL
    if cmp -s "$O/gen-$n-off.ids" "$O/gen-$n-$a.ids" && [[ -s "$O/gen-$n-off.ids" ]]; then log "PASS ids $a tp2 $n"; else log "FAIL ids $a tp2 $n"; fi
    trace "$n-$a"
  done
done
log "t1 stages done"
