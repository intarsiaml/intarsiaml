#!/usr/bin/env bash
# Lane vrc (ON tatooine, radiance down): DFlash2 draft kernel variants at tp=2 (INTARSIA_DFLASH_ROWS_KERNELS),
# phase trace + ids against --spec off on two prompts. Arms in ARMS as tag=ENV (ENV with commas), run by window.sh (STAGES=rows).
set -uo pipefail
export O=${1:?out dir}
source "$HOME/intarsiaml-vrc/.vrc/lib.sh"
D=dev
clear "rows" || exit 2
TRK="$(T "$D")/scripts/dfh/ids-prompts"
ARMS=${ARMS:-"base=X=0 rows=INTARSIA_DFLASH_ROWS_KERNELS=1"}
for n in short-0 4k-1; do
  P=$(cat "$TRK/$n.txt")
  g() { local tag=$1 e=$2; shift 2
    E="$e" run "$D" 0,1 420 "gen-$tag" generate -m "$M27" --backend hip --mem-budget 29 --max-seq 8192 --chat --no-think \
      --profile-every 0 --ignore-eos --show-ids -p "$P" --max-tokens 256 $TP2 "$@"
    grep -a '^ids:' "$O/gen-$tag.err" > "$O/gen-$tag.ids"; }
  g "$n-off" "" --spec off
  for a in $ARMS; do
    t=${a%%=*}; e=${a#*=}; e=${e//,/ }
    g "$n-$t" "INTARSIA_SPEC_TRACE=1 $e" --spec dflash --spec-draft "$DRAFT" --spec-k 7
    if cmp -s "$O/gen-$n-off.ids" "$O/gen-$n-$t.ids" && [[ -s "$O/gen-$n-off.ids" ]]; then log "PASS ids $t tp2 $n"; else log "FAIL ids $t tp2 $n"; fi
    log "trace $t $n: $(grep -a 'dflash2 draft' "$O/gen-$n-$t.err" | tail -n +5 | awk '{for(i=5;i<=NF;i+=2){s[i]+=$i;l[i]=$(i-1)} n++} END{if(n)for(i in s) printf "%s %.2f ", l[i], s[i]/n; print "(n=" n ")"}') | $(grep -a 'verify + rollback' "$O/gen-$n-$t.err" | tail -n +5 | awk '{s+=$(NF-1); n++} END{if(n)printf "verify+rollback %.2f ms n=%d", s/n, n}') | $(grep -a -o 'verify passes=[^|]*' "$O/gen-$n-$t.err" | tail -1 | cut -c1-120)"
  done
done
log "rows done"
