#!/usr/bin/env bash
# Lane dcp (ON tatooine, radiance down): attribution of one DFlash2 pass at tp=2 (trace phases + op tables), draft fp8 on / off,
# ids against --spec off. Run by window.sh (W=<name> STAGES=attr).
set -uo pipefail
export O=${1:?out dir}
source "$HOME/intarsiaml-dcp/.dcp/lib.sh"
D=dev
clear "attr" || exit 2
TRK="$(T "$D")/scripts/dfh/ids-prompts"
for n in short-0 4k-1; do
  P=$(cat "$TRK/$n.txt")
  g() { # g <tag> <env> [args]
    local tag=$1 e=$2; shift 2
    E="$e" run "$D" 0,1 420 "gen-$tag" generate -m "$M27" --backend hip --mem-budget 29 --max-seq 8192 --chat --no-think \
      --profile-every 0 --ignore-eos --show-ids -p "$P" --max-tokens 256 $TP2 "$@"
    grep -a '^ids:' "$O/gen-$tag.err" > "$O/gen-$tag.ids"
  }
  g "$n-off" "" --spec off
  g "$n-dfl7" "INTARSIA_SPEC_TRACE=1" --spec dflash --spec-draft "$DRAFT" --spec-k 7
  g "$n-dfl7f" "INTARSIA_SPEC_TRACE=1 INTARSIA_DFLASH2_FP8=1" --spec dflash --spec-draft "$DRAFT" --spec-k 7
  for a in dfl7 dfl7f; do
    if cmp -s "$O/gen-$n-off.ids" "$O/gen-$n-$a.ids" && [[ -s "$O/gen-$n-off.ids" ]]; then log "PASS ids $a tp2 $n"; else log "FAIL ids $a tp2 $n"; fi
  done
done
# op tables (eager, synchronised per op: shares, not wall ms)
P=$(cat "$TRK/short-0.txt")
for f in "" 1; do
  E="${f:+INTARSIA_DFLASH2_FP8=1}" run "$D" 0,1 420 "optable-dfl7${f:+f}" generate -m "$M27" --backend hip --mem-budget 29 --max-seq 8192 --chat --no-think \
    --profile-every 1 --op-table --ignore-eos -p "$P" --max-tokens 96 $TP2 --spec dflash --spec-draft "$DRAFT" --spec-k 7
done
for f in dfl7 dfl7f; do
  log "trace $f short-0: $(grep -a 'dflash2 draft' "$O/gen-short-0-$f.err" | tail -n +5 | awk '{for(i=5;i<=NF;i+=2){s[i]+=$i;l[i]=$(i-1)} n++} END{for(i in s) printf "%s %.2f ", l[i], s[i]/n; print "(n=" n ")"}')"
  log "verify $f short-0: $(grep -a 'verify + rollback' "$O/gen-short-0-$f.err" | tail -n +5 | awk '{s+=$(NF-1); n++} END{printf "mean verify+rollback %.2f ms n=%d", s/n, n}')"
done
log "attr done"
