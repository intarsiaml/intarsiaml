#!/bin/zsh
# usage: run_id.sh <outdir> <new binary name in bin/> <variants...>  (inside an exclusive slot)
# variants: base (8254311 binary, spec off), bon4 (8254311 --spec ngram), off (new, spec off), on4 (new --spec ngram)
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a slot" >&2; exit 2; }
O=$1; NEWNAME=$2; shift 2
S=/private/tmp/claude-501/-opt-code-intarsiaml/0e8026c5-6dd0-4b83-b375-5ff0a66894fc/scratchpad
P=/opt/code/intarsiaml-cb-m1/docs/results/raw/2026-10-15-metal-cb-m0/prompts
BASE=$S/bin/intarsia-cbm0-8254311
NEW=$S/bin/$NEWNAME
M=$HOME/models/Qwen3-8B
mkdir -p $O
while IFS='|' read -r name maxtok flags pfile samp; do
  [ -z "$name" ] && continue
  [ -z "$samp" ] && samp="--temperature 0"
  for v in "$@"; do
    case $v in
      base) BIN=$BASE; SPEC=(--max-batch 1);;
      bon4) BIN=$BASE; SPEC=(--max-batch 1 --spec ngram);;
      off) BIN=$NEW; SPEC=(--max-batch 1 --spec off);;
      on4) BIN=$NEW; SPEC=(--max-batch 1 --spec ngram);;
    esac
    echo "$(date +%T) $name $v"
    INTARSIA_SPEC_TRACE=1 $BIN generate --model $M --backend metal --max-seq 8192 --profile-every 0 ${=samp} --show-ids --max-tokens $maxtok ${=flags} $SPEC --prompt "$(cat $P/$pfile.txt)" < /dev/null > $O/$v-$name.out 2> $O/$v-$name.err
    echo "  exit $? $(grep '^finish=' $O/$v-$name.err)"
  done
done < $S/cases-sub.txt
