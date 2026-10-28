#!/usr/bin/env bash
# Lane vat (ON tatooine, radiance down; PRE=phase of cat-window.sh): per-phase wall time of speculative passes at tp=2
# (INTARSIA_SPEC_TRACE=1), mtp k2 and dflash k7, the lane's new paths on and off (OLDENV, default the verify attention rows knob), short and 4k prompts.
set -uo pipefail
export O=${1:?out dir}
source "$HOME/intarsiaml-vat/.vat/lib.sh"
D=dev
clear "phase" || exit 2
TRK="$(T "$D")/scripts/dfh/ids-prompts"
OLD="${OLDENVV:-INTARSIA_HIP_VERIFY_ATTN_ROWS=0}"
for n in short-0 4k-1; do
  P=$(cat "$TRK/$n.txt")
  g() { # g <tag> <env> [args]
    local tag=$1 e=$2; shift 2
    E="INTARSIA_SPEC_TRACE=1 $e" run "$D" 0,1 420 "gen-$tag" generate -m "$M27" --backend hip --mem-budget 29 --max-seq 8192 --chat --no-think \
      --profile-every 0 --ignore-eos --show-ids -p "$P" --max-tokens 256 $TP2 "$@"
    grep -a '^ids:' "$O/gen-$tag.err" > "$O/gen-$tag.ids"
  }
  g "$n-off" "" --spec off
  g "$n-mtp2" "" --spec mtp --spec-k 2
  g "$n-mtp2old" "$OLD" --spec mtp --spec-k 2
  g "$n-dfl7" "" --spec dflash --spec-draft "$DRAFT" --spec-k 7
  g "$n-dfl7old" "$OLD" --spec dflash --spec-draft "$DRAFT" --spec-k 7
  for a in mtp2 mtp2old dfl7 dfl7old; do
    if cmp -s "$O/gen-$n-off.ids" "$O/gen-$n-$a.ids" && [[ -s "$O/gen-$n-off.ids" ]]; then log "PASS ids $a tp2 $n"; else log "FAIL ids $a tp2 $n"; fi
    log "note $a $n: $(grep -ah 'verify attention rows' "$O/gen-$n-$a.err" | sort -u | tr '\n' '|')"
    log "verify $a $n: $(grep -a 'verify + rollback' "$O/gen-$n-$a.err" | tail -n +5 | awk '{s+=$(NF-1); c++} END{if(c)printf "mean verify+rollback %.2f ms n=%d", s/c, c}') | $(grep -aE 'decode .*tok/s' "$O/gen-$n-$a.err" | tail -1 | cut -c1-120)"
  done
done
log "phase done"
