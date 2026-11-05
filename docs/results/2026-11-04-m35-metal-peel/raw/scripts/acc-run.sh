#!/bin/zsh
# usage (inside a fleetq slot): acc-run.sh OUTDIR MODEL "MODEXTRA" GOLDEN TAG:BIN[:ENV=VAL] ...   -- `intarsia accuracy --backend metal`, one run per TAG, per-position dump kept.
set -u
[ -n "${FLEETQ_JOB:-}" ] || exit 2
cd /opt/code/intarsiaml/.worktrees/m35-peel
O=${1:?out}; M=${2:?model}; MODEX=(${=3}); G=${4:?golden}; shift 4; mkdir -p $O
for spec in "$@"; do
  tag=${spec%%:*}; rest=${spec#*:}; bin=${rest%%:*}; envs=""; [[ "$rest" == *:* ]] && envs=${rest#*:}
  rm -f $O/$tag.pos.tsv
  env ${=envs} INTARSIA_ACC_POS_DUMP=$O/$tag.pos.tsv $bin accuracy --backend metal --model $M $MODEX --golden $G --allow-unstamped --allow-dirty > $O/$tag.out 2> $O/$tag.err
  echo "accuracy $tag exit $?"
done
