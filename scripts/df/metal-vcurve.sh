#!/bin/zsh
# Lane df: Metal verify-cost curve on endor, ONE exclusive slot: eager MTP k1..k6 (verify passes of 2..7 rows,
# INTARSIA_METAL_MTP_GPU_ACCEPT=0, INTARSIA_SPEC_TRACE=1) and off, at short-0 and 4k-1; mean draft and verify + rollback per pass.
#   fleetq run --host endor --exclusive --mem 30 --expect 20 --label "df metal vcurve" -- zsh scripts/df/metal-vcurve.sh <binary> <out dir>
set -u
BIN=${1:?binary}; O=${2:?out dir}; mkdir -p $O
[[ -n "${FLEETQ_JOB:-}" ]] || { echo "not inside a fleetq slot" >&2; exit 2; }
R=${0:A:h:h:h}; M=${M:-/Users/josef/models/Qwen3.8-27B-Quark-AWQ-MXFP4}
say() { echo "$(date +%T) $*" | tee -a $O/steps.log; }
for n in ${=PROMPTS:-short-0 4k-1}; do
  P="$(cat $R/scripts/dfh/ids-prompts/$n.txt)"
  for k in ${=KS:-0 1 2 3 4 5 6}; do
    args=(--spec mtp --spec-k $k); [[ $k == 0 ]] && args=(--spec off)
    env INTARSIA_SPEC_TRACE=1 INTARSIA_METAL_MTP_GPU_ACCEPT=0 ${=XENV:-} $BIN generate -m $M --backend metal --max-seq 8192 --chat --no-think --profile-every 0 --ignore-eos --show-ids -p "$P" --max-tokens 256 $args \
      > $O/gen-$n-k$k.out 2> $O/gen-$n-k$k.err < /dev/null
    rc=$?; grep -a '^ids:' $O/gen-$n-k$k.err > $O/gen-$n-k$k.ids
    same=FAIL; [[ -s $O/gen-$n-k0.ids ]] && cmp -s $O/gen-$n-k0.ids $O/gen-$n-k$k.ids && same=PASS
    say "$n k$k exit $rc ids $same | $(grep -a 'verify + rollback' $O/gen-$n-k$k.err | tail -n +5 | awk '{s+=$(NF-1); d+=$4; c++} END{if(c)printf "draft %.2f verify+rollback %.2f ms n=%d", d/c, s/c, c}') | $(grep -aE 'decode [0-9]+ ms' $O/gen-$n-k$k.err | tail -1 | grep -oE 'decode [0-9]+ ms \([0-9.]+ tok/s\)')"
  done
done

# op tables (synchronised per op: shares, not wall time) for off and the 7-row eager verify
if [[ ${OPTABLE:-1} == 1 ]]; then
  P="$(cat $R/scripts/dfh/ids-prompts/short-0.txt)"
  for k in 0 6; do
    args=(--spec mtp --spec-k $k); [[ $k == 0 ]] && args=(--spec off)
    env INTARSIA_METAL_MTP_GPU_ACCEPT=0 $BIN generate -m $M --backend metal --max-seq 8192 --chat --no-think --profile-every 1 --op-table --ignore-eos -p "$P" --max-tokens 96 $args \
      > $O/optable-k$k.out 2> $O/optable-k$k.err < /dev/null
    say "optable k$k exit $?"
  done
fi
say "all done"
