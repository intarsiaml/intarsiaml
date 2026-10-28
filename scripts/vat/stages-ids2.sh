#!/usr/bin/env bash
# Lane vat (ON tatooine, radiance down): is the plain decode (--spec off) of ids prompt short-5 reproducible run to run at tp=2? Four
# off runs and one run of each spec arm (new path, knob off), md5 of the id lists; then the whole ids gate again (stages-ids.sh,
# arms mtp2 dfl, tp2).
set -uo pipefail
export O=${1:?out dir}
source "$HOME/intarsiaml-vat/.vat/lib.sh"
D=dev
clear "ids2" || exit 2
P="$(T "$D")/scripts/dfh/ids-prompts"
g() { # g <tag> <env> [args]
  local tag=$1 e=$2; shift 2
  E="$e" run "$D" 0,1 420 "gen-$tag" generate -m "$M27" --backend hip --mem-budget 29 --max-seq 20480 --chat --no-think \
    --profile-every 0 --ignore-eos --show-ids -p "$(cat "$P/short-5.txt")" --max-tokens 256 $TP2 "$@"
  grep -a '^ids:' "$O/gen-$tag.err" > "$O/gen-$tag.ids"
}
for i in 1 2 3 4; do g "s5-off$i" "" --spec off; done
g "s5-mtp2" "" --spec mtp --spec-k 2
g "s5-mtp2-old" "INTARSIA_HIP_VERIFY_ATTN_ROWS=0" --spec mtp --spec-k 2
g "s5-dfl" "" --spec dflash --spec-draft "$DRAFT" --spec-k 7
g "s5-dfl-old" "INTARSIA_HIP_VERIFY_ATTN_ROWS=0" --spec dflash --spec-draft "$DRAFT" --spec-k 7
for f in "$O"/gen-s5-*.ids; do log "md5 $(basename "$f" .ids) $(md5sum < "$f" | cut -c1-12)"; done
bash "$HOME/intarsiaml-vat/.vat/stages-ids.sh" "$O"
log "ids2 done"
