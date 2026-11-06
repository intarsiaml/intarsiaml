#!/bin/zsh
# Lane dm (Metal DFlash2): generate traces + greedy ids vs --spec off on endor, inside a fleetq slot.
#   fleetq run --host endor --exclusive --mem 45 --expect 30 --label "dm gen" -- zsh scripts/dm/gen.sh <binary> <out dir> [arms...]
# Arms: off, mtp2e (eager MTP k2, prints draft / verify split), mtp2 (pipelined default), dflK (DFlash2 chain kK, e.g. dfl7).
# PROMPTS (default "short-0 4k-1"), TOKENS (default 256), ENVS (extra env for every run, e.g. INTARSIA_METAL_MXFP4_MR=1).
set -u
BIN=${1:?binary}; O=${2:?out dir}; shift 2; mkdir -p $O
[[ -n "${FLEETQ_JOB:-}" ]] || { echo "not inside a fleetq slot" >&2; exit 2; }
R=${0:A:h:h:h}
M=${M:-/Users/josef/models/Qwen3.8-27B-Quark-AWQ-MXFP4}
DRAFT=${DRAFT:-/Users/josef/.cache/huggingface/hub/models--z-lab--Qwen3.8-27B-DFlash2/snapshots/50307d4c4cde6860d4eee73e2547cd786fe8e8a4}
ARMS=(${@:-off mtp2e dfl7}); TOKENS=${TOKENS:-256}
say() { echo "$(date +%T) $*" | tee -a $O/steps.log; }
for n in ${=PROMPTS:-short-0 4k-1}; do
  P="$(cat $R/scripts/dfh/ids-prompts/$n.txt)"
  for tag in $ARMS; do
    envs=(INTARSIA_SPEC_TRACE=1 ${=ENVS:-})
    case $tag in
      off) args=(--spec off) ;;
      mtp2e) args=(--spec mtp --spec-k 2); envs+=(INTARSIA_METAL_MTP_GPU_ACCEPT=0) ;;
      mtp2) args=(--spec mtp --spec-k 2) ;;
      dfl*) args=(--spec dflash --spec-k ${tag#dfl} --spec-draft $DRAFT) ;;
      *) echo "unknown arm $tag" >&2; exit 2 ;;
    esac
    env $envs $BIN generate -m $M --backend metal --max-seq 20480 --chat --no-think --profile-every 0 --ignore-eos --show-ids -p "$P" --max-tokens $TOKENS $args \
      > $O/gen-$n-$tag.out 2> $O/gen-$n-$tag.err < /dev/null
    rc=$?; grep -a '^ids:' $O/gen-$n-$tag.err > $O/gen-$n-$tag.ids
    same=FAIL; [[ -s $O/gen-$n-off.ids ]] && cmp -s $O/gen-$n-off.ids $O/gen-$n-$tag.ids && same=PASS
    say "$n $tag exit $rc ids-vs-off $same | $(python3 $R/scripts/dm/passes.py $O/gen-$n-$tag.err) | $(grep -aE 'tok/s' $O/gen-$n-$tag.err | tail -1 | cut -c1-160)"
  done
done
