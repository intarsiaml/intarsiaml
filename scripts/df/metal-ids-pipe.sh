#!/bin/zsh
# Lane df: greedy ids of the DEFAULT (pipelined) MTP step with INTARSIA_METAL_MXFP4_MR=1 vs --spec off, 27B Metal, short-0 and 4k-1, MTP k1..k3.
set -u
BIN=${1:?binary}; O=${2:?out dir}; mkdir -p $O; R=${0:A:h:h:h}
[[ -n "${FLEETQ_JOB:-}" ]] || { echo "not inside a fleetq slot" >&2; exit 2; }
M=/Users/josef/models/Qwen3.8-27B-Quark-AWQ-MXFP4
say() { echo "$(date +%T) $*" | tee -a $O/steps.log; }
for n in short-0 4k-1; do
  P="$(cat $R/scripts/dfh/ids-prompts/$n.txt)"
  for a in off k1 k2 k3; do
    args=(--spec mtp --spec-k ${a#k}); envs=(INTARSIA_METAL_MXFP4_MR=1); [[ $a == off ]] && { args=(--spec off); envs=(); }
    env $envs $BIN generate -m $M --backend metal --max-seq 8192 --chat --no-think --profile-every 0 --ignore-eos --show-ids -p "$P" --max-tokens 256 $args > $O/gen-$n-$a.out 2> $O/gen-$n-$a.err < /dev/null
    rc=$?; grep -a '^ids:' $O/gen-$n-$a.err > $O/gen-$n-$a.ids
    same=FAIL; [[ -s $O/gen-$n-off.ids ]] && cmp -s $O/gen-$n-off.ids $O/gen-$n-$a.ids && same=PASS
    say "$n $a (pipelined, knob ${envs[*]:-unset}) exit $rc ids-vs-off $same | $(grep -aoE 'decode [0-9]+ ms \([0-9.]+ tok/s\)' $O/gen-$n-$a.err | tail -1)"
  done
done
