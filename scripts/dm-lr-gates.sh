#!/bin/zsh
# Lane dm, logits-rows PR gates on endor in ONE slot (fleetq --quiet gpu,cpu): no-op gates (plan show 8B/27B/35B, kernels list/test) lane
# vs base; greedy ids (8B, 35B lane vs base; 27B MTP k2 and knob=0 vs --spec off); ABAB of INTARSIA_METAL_LOGITS_ROWS (A=0, B=unset).
#   zsh scripts/dm-lr-gates.sh <lane bin> <base bin> <out>
set -u
B=${1:?lane}; A=${2:?base}; O=${3:?out}; mkdir -p $O
[[ -n "${FLEETQ_JOB:-}" ]] || { echo "not inside a fleetq slot" >&2; exit 2; }
R=${0:A:h:h}
say() { echo "$(date +%T) $*" | tee -a $O/steps.log; }
typeset -A MOD; MOD=(8b /Users/josef/models/Qwen3-8B 27b /Users/josef/models/Qwen3.8-27B-Quark-AWQ-MXFP4 35b /Users/josef/models/Qwen3.6-35B-A3B-NVFP4)
if [[ ${PARTS:-NIA} == *N* ]]; then
for t in 8b 27b 35b; do
  for side in base lane; do bin=$A; [[ $side == lane ]] && bin=$B
    $bin plan show --backend metal --model $MOD[$t] --max-seq 4096 --json > $O/plan-$t-$side.json 2> $O/plan-$t-$side.err < /dev/null; done
  cmp -s $O/plan-$t-base.json $O/plan-$t-lane.json && say "SAME plan $t" || say "DIFF plan $t"
done
for side in base lane; do bin=$A; [[ $side == lane ]] && bin=$B
  $bin kernels list --backend metal > $O/klist-$side.txt 2>&1 < /dev/null
  $bin kernels test --backend metal > $O/ktest-$side.txt 2>&1 < /dev/null; say "kernels test $side exit $? $(tail -1 $O/ktest-$side.txt | cut -c1-160)"; done
cmp -s $O/klist-base.txt $O/klist-lane.txt && say "SAME kernels list" || say "DIFF kernels list"
diff <(grep -vE '[0-9]+\.[0-9]+ ?(ms|us|s)\b' $O/ktest-base.txt) <(grep -vE '[0-9]+\.[0-9]+ ?(ms|us|s)\b' $O/ktest-lane.txt) > $O/ktest.diff && say "SAME kernels test (timing lines dropped)" || say "DIFF kernels test $(wc -l < $O/ktest.diff) lines"
fi
gen() { # gen <tag> <bin> <model> <prompt> [env...] -- [args]
  local tag=$1 bin=$2 m=$3 pn=$4; shift 4; local envs=(); while [[ $# -gt 0 && $1 != -- ]]; do envs+=($1); shift; done; shift
  env ${envs[@]} $bin generate -m $m --backend metal --max-seq 8192 --chat --no-think --profile-every 0 --ignore-eos --show-ids \
    -p "$(cat $R/scripts/dfh/ids-prompts/$pn.txt)" --max-tokens 256 "$@" > $O/gen-$tag.out 2> $O/gen-$tag.err < /dev/null
  grep -a '^ids:' $O/gen-$tag.err > $O/gen-$tag.ids
  grep -aoE 'decode [0-9]+ ms \([0-9.]+ tok/s\)' $O/gen-$tag.err | grep -oE '[0-9.]+ tok/s' | cut -d' ' -f1
}
same() { cmp -s $O/gen-$1.ids $O/gen-$2.ids && [[ -s $O/gen-$1.ids ]] && echo SAME || echo DIFF; }
if [[ ${PARTS:-NIA} == *I* ]]; then
for t in 8b 35b; do for pn in short-0 4k-1; do
  gen $t-$pn-base $A $MOD[$t] $pn -- >/dev/null; gen $t-$pn-lane $B $MOD[$t] $pn -- >/dev/null
  say "ids $t $pn lane vs base (default spec): $(same $t-$pn-base $t-$pn-lane)"; done; done
for pn in short-0 4k-1; do
  gen 27b-$pn-off $B $MOD[27b] $pn -- --spec off >/dev/null
  gen 27b-$pn-mtp2 $B $MOD[27b] $pn -- --spec mtp --spec-k 2 >/dev/null
  gen 27b-$pn-mtp2-k0 $B $MOD[27b] $pn INTARSIA_METAL_LOGITS_ROWS=0 -- --spec mtp --spec-k 2 >/dev/null
  gen 27b-$pn-mtp2-eager $B $MOD[27b] $pn INTARSIA_METAL_MTP_GPU_ACCEPT=0 -- --spec mtp --spec-k 2 >/dev/null
  gen 27b-$pn-mtp4 $B $MOD[27b] $pn -- --spec mtp --spec-k 4 >/dev/null
  say "ids 27b $pn vs --spec off: mtp2 $(same 27b-$pn-off 27b-$pn-mtp2) mtp2-knob0 $(same 27b-$pn-off 27b-$pn-mtp2-k0) mtp2-eager $(same 27b-$pn-off 27b-$pn-mtp2-eager) mtp4 $(same 27b-$pn-off 27b-$pn-mtp4)"
done
fi
if [[ ${PARTS:-NIA} == *A* ]]; then
for arm in mtp off; do for v in A1 B1 A2 B2; do
  envs=(); [[ $v == A* ]] && envs=(INTARSIA_METAL_LOGITS_ROWS=0); args=(); [[ $arm == off ]] && args=(--spec off)
  for pn in short-0 4k-1; do r=(); for i in 1 2 3; do r+=($(gen ab-$arm-$v-$pn-$i $B $MOD[27b] $pn ${envs[@]} -- ${args[@]})); done
    say "ABAB $arm $v $pn ${r[*]}"; done; done; done
fi
say done
