#!/bin/zsh
# DSpine draft (Metal): greedy ids vs --spec off, tokens a pass and tok/s, on endor, inside a fleetq slot.
#   fleetq run --host endor --quiet gpu,cpu --mem 40 --expect 40 --label "dspine ids" -- zsh scripts/dspine/ids.sh <binary> <out dir> [arms...]
# Arms: off, dsd (DSpine with no --spec-k: the DSpine default of 3), dsK (DSpine draft, chain kK, e.g. ds3 ds7), dfK (DFlash2, for the comparison), mtp2 (pipelined default MTP).
# PROMPTS (default "short-0 4k-1 9k-0"), TOKENS (default 256), ENVS (extra env for every run), DRAFT (the DSpine checkpoint dir).
# Timings are only meaningful in a quiet slot: check `fleetq status` (nothing standing on endor, e.g. endor-tensorfold, may run) and
# compare against the mtp2 arm; the review's quiet-box runs had spec off 31 tok/s, DSpine k=3 40, k=7 24, mtp2 50 on short-0.
set -u
BIN=${1:?binary}; O=${2:?out dir}; shift 2; mkdir -p $O
[[ -n "${FLEETQ_JOB:-}" ]] || { echo "not inside a fleetq slot" >&2; exit 2; }
R=${0:A:h:h:h}
M=${M:-/Users/josef/models/Qwen3.8-27B-Quark-AWQ-MXFP4}
DRAFT=${DRAFT:-/opt/code/intarsiaml/.worktrees/dspine/model}
DFL2=${DFL2:-/Users/josef/.cache/huggingface/hub/models--z-lab--Qwen3.8-27B-DFlash2/snapshots/50307d4c4cde6860d4eee73e2547cd786fe8e8a4}
ARMS=(${@:-off ds3 ds7}); TOKENS=${TOKENS:-256}
say() { echo "$(date +%T) $*" | tee -a $O/steps.log; }
for n in ${=PROMPTS:-short-0 4k-1 9k-0}; do
  f=$R/scripts/dfh/ids-prompts/$n.txt; [[ -f $f ]] || f=$R/scripts/dspine/prompts/$n.txt
  P="$(cat $f)"
  for tag in $ARMS; do
    envs=(INTARSIA_SPEC_TRACE=1 ${=ENVS:-})
    case $tag in
      off) args=(--spec off) ;;
      mtp2) args=(--spec mtp --spec-k 2) ;;
      dsd) args=(--spec dflash --spec-draft $DRAFT) ;;
      ds*) args=(--spec dflash --spec-k ${tag#ds} --spec-draft $DRAFT) ;;
      df*) args=(--spec dflash --spec-k ${tag#df} --spec-draft $DFL2) ;;
      *) echo "unknown arm $tag" >&2; exit 2 ;;
    esac
    env $envs $BIN generate -m $M --backend metal --max-seq 20480 --chat --no-think --profile-every 0 --ignore-eos --show-ids -p "$P" --max-tokens $TOKENS $args \
      > $O/gen-$n-$tag.out 2> $O/gen-$n-$tag.err < /dev/null
    rc=$?; grep -a '^ids:' $O/gen-$n-$tag.err > $O/gen-$n-$tag.ids
    same=FAIL; [[ -s $O/gen-$n-off.ids ]] && cmp -s $O/gen-$n-off.ids $O/gen-$n-$tag.ids && same=PASS
    say "$n $tag exit $rc ids-vs-off $same | $(python3 $R/scripts/dm/passes.py $O/gen-$n-$tag.err) | $(grep -aE 'tok/s' $O/gen-$n-$tag.err | tail -1 | cut -c1-160)"
  done
done
