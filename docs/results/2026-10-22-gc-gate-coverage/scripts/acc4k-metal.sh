#!/usr/bin/env bash
# gc lane: qualify acc-4k for the 35B on Metal (endor). Run under fleetq from the worktree:
#   fleetq run --host endor --mem 45 --expect 25 --label "gc: Metal acc-4k" -- docs/results/2026-10-22-gc-gate-coverage/scripts/acc4k-metal.sh OUTDIR
# q-0 MISSING-BASELINE (records), install in golden/baselines/, q-1 / q-1b gated against it.
set -u
: "${FLEETQ_JOB:?run under fleetq}"
OUT=${1:?outdir}
cd "$(git rev-parse --show-toplevel)"
B=${BIN:-/opt/code/wt-gc-target/release/intarsia}
M=/Users/josef/Models/Qwen3.6-35B-A3B-NVFP4
G=$PWD/golden/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors     # canonical short golden (copied in, git-ignored)
OR=$PWD/golden/oracle/qwen36-35b-a3b-nvfp4-cpu-oracle.safetensors
mkdir -p "$OUT"; O=$(cd "$OUT" && pwd)
export INTARSIA_REPO=$PWD
SK="--skip kernels --skip gate --skip acc-format --skip acc-short --skip acc-oracle --skip bench --skip acc-long --skip envelope --skip bench-long --skip ab --skip tp"
Q="qualify --backend metal --golden $G --oracle-golden $OR --model-name Qwen3.6-35B-A3B-NVFP4 --mem-budget 45 --stage-timeout-min 25 --model $M"
qual() {
  local n=$1; shift
  local t0=$(date +%s)
  $B $Q --out $O/q-$n $SK "$@" < /dev/null > $O/q-$n.log 2>&1
  echo "qualify $n exit $? $(( $(date +%s) - t0 )) s"
  grep -E "\] [a-z0-9-]+: (PASS|FAIL|SKIP|ERROR|WARN|MISSING-BASELINE)" $O/q-$n.log | cut -c1-400
}
qual 0-nobaseline
cp $O/q-0-nobaseline/baselines/acc-4k.baseline.json golden/baselines/qwen36-35b-a3b-nvfp4-vllm-4k-top20.metal.baseline.json && echo "installed baseline"
qual 1-clean
qual 1b-clean
