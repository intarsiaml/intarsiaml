#!/bin/zsh
# Lane df, default-on gates for INTARSIA_METAL_MXFP4_MR (user decision 2026-11-06), endor, one exclusive slot:
#   plan show (8B, 27B, 35B) lane vs each base; kernels test lane (0 FAIL); greedy ids (generate, recipe default spec, short-0 and 4k-1):
#   27B lane default == 27B lane --spec off; 8B / 35B lane == base; 27B lane with =0 == base.
#   zsh scripts/df/metal-default-gates.sh <lane bin> <out dir> <base name>=<base bin> [...]
set -u
B=${1:?lane binary}; O=${2:?out dir}; shift 2; mkdir -p $O; R=${0:A:h:h:h}
[[ -n "${FLEETQ_JOB:-}" ]] || { echo "not inside a fleetq slot" >&2; exit 2; }
say() { echo "$(date +%T) $*" | tee -a $O/steps.log; }
typeset -A MOD; MOD=(8b /Users/josef/models/Qwen3-8B 27b /Users/josef/models/Qwen3.8-27B-Quark-AWQ-MXFP4 35b /Users/josef/models/Qwen3.6-35B-A3B-NVFP4)
for t in 8b 27b 35b; do
  $B plan show --backend metal --model $MOD[$t] --max-seq 4096 --json > $O/plan-$t-lane.json 2> $O/plan-$t-lane.err < /dev/null
  for bb in "$@"; do bn=${bb%%=*}; bin=${bb#*=}
    $bin plan show --backend metal --model $MOD[$t] --max-seq 4096 --json > $O/plan-$t-$bn.json 2> $O/plan-$t-$bn.err < /dev/null
    cmp -s $O/plan-$t-$bn.json $O/plan-$t-lane.json && say "SAME plan $t lane vs $bn" || say "DIFF plan $t lane vs $bn ($(diff $O/plan-$t-$bn.json $O/plan-$t-lane.json | grep -c '^[<>]') lines)"
  done
done
$B kernels test --backend metal > $O/ktest-lane.txt 2>&1 < /dev/null
say "kernels test lane exit $?: FAIL lines $(grep -c 'FAIL' $O/ktest-lane.txt), $(tail -1 $O/ktest-lane.txt)"
gen() { # gen <tag> <bin> <model> [env...] -- [args]
  local tag=$1 bin=$2 m=$3; shift 3; local envs=(); while [[ $# -gt 0 && $1 != -- ]]; do envs+=($1); shift; done; shift
  env $envs $bin generate -m $m --backend metal --max-seq 8192 --chat --no-think --profile-every 0 --ignore-eos --show-ids -p "$P" --max-tokens 256 "$@" \
    > $O/gen-$tag.out 2> $O/gen-$tag.err < /dev/null
  echo "$tag exit $? $(grep -aoE 'decode [0-9]+ ms \([0-9.]+ tok/s\)' $O/gen-$tag.err | tail -1) $(grep -a 'verify passes' $O/gen-$tag.err | tail -1 | cut -c1-90)" >> $O/gen.log
  grep -a '^ids:' $O/gen-$tag.err > $O/gen-$tag.ids
}
same() { [[ -s $O/gen-$1.ids ]] && cmp -s $O/gen-$1.ids $O/gen-$2.ids && say "PASS ids $1 == $2" || say "FAIL ids $1 vs $2"; }
BASE1=${1#*=}; BN1=${1%%=*}
for n in short-0 4k-1; do
  P="$(cat $R/scripts/dfh/ids-prompts/$n.txt)"
  gen 27b-$n-lane-default $B $MOD[27b] --
  gen 27b-$n-lane-off $B $MOD[27b] -- --spec off
  same 27b-$n-lane-off 27b-$n-lane-default
  gen 27b-$n-lane-mr0 $B $MOD[27b] INTARSIA_METAL_MXFP4_MR=0 --
  gen 27b-$n-$BN1-default $BASE1 $MOD[27b] --
  same 27b-$n-$BN1-default 27b-$n-lane-mr0
  for t in 8b 35b; do
    gen $t-$n-lane-default $B $MOD[$t] --
    gen $t-$n-$BN1-default $BASE1 $MOD[$t] --
    same $t-$n-$BN1-default $t-$n-lane-default
  done
done
say done
