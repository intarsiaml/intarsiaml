#!/bin/zsh
# usage: run_identity.sh <outdir> <cases file> <variants...>   (inside an exclusive endor slot: FLEETQ_JOB set)
# variants: base (spike/metal 56ec2d5 binary), bon4 (same, --spec ngram), off (cb-m0, --max-batch 1 --spec off), on4 (--spec ngram), on6 (--spec ngram --spec-k 6)
# cases: name|max_tokens|flags|prompt file|[sampling flags, default --temperature 0]
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a slot" >&2; exit 2; }
[ -d "$DEVELOPER_DIR" ] || { echo "DEVELOPER_DIR missing" >&2; exit 3; }
O=$1; CASES=$2; shift 2
HERE=${0:A:h}
P=$HERE/prompts
BIN_DIR=/private/tmp/claude-501/-opt-code-intarsiaml/0e8026c5-6dd0-4b83-b375-5ff0a66894fc/scratchpad/bin
BASE=$BIN_DIR/intarsia-base-56ec2d5
NEW=$BIN_DIR/intarsia-new-${NEW_REV:?NEW_REV}
M=$HOME/models/Qwen3-8B
mkdir -p $O
while IFS='|' read -r name maxtok flags pfile samp; do
  [ -z "$name" ] && continue
  [ -z "$samp" ] && samp="--temperature 0"
  for v in "$@"; do
    case $v in
      base) BIN=$BASE; SPEC=();;
      bon4) BIN=$BASE; SPEC=(--spec ngram);;
      off) BIN=$NEW; SPEC=(--max-batch 1 --spec off);;
      on4) BIN=$NEW; SPEC=(--max-batch 1 --spec ngram);;
      on6) BIN=$NEW; SPEC=(--max-batch 1 --spec ngram --spec-k 6);;
    esac
    echo "$(date +%T) $name $v"
    INTARSIA_SPEC_TRACE=1 $BIN generate --model $M --backend metal --max-seq 8192 --profile-every 0 ${=samp} --show-ids --max-tokens $maxtok ${=flags} $SPEC --prompt "$(cat $P/$pfile.txt)" < /dev/null > $O/$v-$name.out 2> $O/$v-$name.err
    echo "  exit $? $(grep '^finish=' $O/$v-$name.err)"
  done
done < $CASES
