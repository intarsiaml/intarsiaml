#!/bin/zsh
# usage: run_identity.sh <outdir> <cases file> <variants...>   (inside a slot: FLEETQ_JOB set, see slot.sh)
# variants: base (spike/metal c556008), off (new, --spec off), on4 (new, --spec ngram), on6 (new, --spec ngram --spec-k 6)
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a slot" >&2; exit 2; }
[ -d "$DEVELOPER_DIR" ] || { echo "DEVELOPER_DIR missing" >&2; exit 3; }
O=$1; CASES=$2; shift 2
P=/opt/code/intarsiaml-spec/docs/results/raw/2026-10-15-metal-spec-g1/prompts
BASE=/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/bin/intarsia-base-c556008
NEW=/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/bin/intarsia-new-5a5162d
M=$HOME/models/Qwen3-8B
mkdir -p $O
while IFS='|' read -r name maxtok flags pfile; do
  [ -z "$name" ] && continue
  for v in "$@"; do
    case $v in
      base) BIN=$BASE; SPEC=();;
      off) BIN=$NEW; SPEC=(--spec off);;
      on4) BIN=$NEW; SPEC=(--spec ngram);;
      on6) BIN=$NEW; SPEC=(--spec ngram --spec-k 6);;
    esac
    echo "$(date +%T) $name $v"
    INTARSIA_SPEC_TRACE=1 $BIN generate --model $M --backend metal --max-seq 8192 --profile-every 0 --temperature 0 --show-ids --max-tokens $maxtok ${=flags} $SPEC --prompt "$(cat $P/$pfile.txt)" < /dev/null > $O/$v-$name.out 2> $O/$v-$name.err
    echo "  exit $? $(grep '^finish=' $O/$v-$name.err)"
  done
done < $CASES
