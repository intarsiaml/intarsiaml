#!/usr/bin/env bash
# Lane hgv: the GPU gates and kernel micro-timings of both trees on the R9700s, driven FROM THE MAC under one fleetq job that holds both cards and
# takes radiance down (same guard as window.sh):
#   STEPS="gates timing" fleetq run --host tatooine-r9700,tatooine-r9700-2 --takes-down radiance --quiet gpu,cpu --expect 20 --mem 30 \
#     --label "HGV gates + timing" -- scripts/hgv/gate.sh
# gates (hgv tree): rows_exact (bf16 + all others), moe_block grouped tests, comm, ar_norm, ar_push (default env: the new LL limit)
# timing (hgvbase then hgv): rows_exact timing_gemv_rows, timing_mxfp4_rows (the lm_head at M = 5..8 must not move), moe_rows_timing
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
STEPS=${STEPS:-"gates timing"}; CAP=${CAP:-950}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"
OUT=${OUT:-$REPO/docs/results/2026-10-30-hgv-hip-small-gemv-grid/raw/gate-$(date +%H%M)}; mkdir -p "$OUT"
T0=$(date +%s); left() { echo $(( CAP - ($(date +%s) - T0) )); }
say() { echo "$(date +%T) $*" | tee -a "$OUT/steps.log"; }
SSH="ssh -o BatchMode=yes tatooine"
FIFO="$OUT/guard.fifo"; mkfifo "$FIFO"
$SSH 'mkdir -p ~/hgv' < /dev/null; scp -q "$HERE/guard.sh" tatooine:hgv/guard.sh
$SSH 'bash ~/hgv/guard.sh' > "$OUT/guard.log" 2>&1 < "$FIFO" &
GUARD=$!
exec 9> "$FIFO"
for i in $(seq 1 30); do grep -q "radiance stopped" "$OUT/guard.log" 2>/dev/null && break; sleep 1; done
say "guard: $(grep -h 'radiance stopped' "$OUT/guard.log")"
cleanup() {
  exec 9>&-; wait $GUARD; rm -f "$FIFO"
  say "guard: $(grep -h 'stopping radiance\|radiance answers\|DID NOT' "$OUT/guard.log" | tr '\n' '|')"
  echo "radiance down: $(grep -h 'stopping radiance' "$OUT/guard.log" | sed 's/.*radiance //') -> $(grep -h 'radiance answers\|DID NOT' "$OUT/guard.log" | sed 's/.*\] //')  (fleetq job $FLEETQ_JOB)" >> "$OUT/downtime.log"
}
trap cleanup EXIT

# t <tag> <tree> <secs> <test target> <env ...> -- <test filter / args...>
t() {
  local tag=$1 tree=$2 secs=$3 target=$4; shift 4
  local envs=() args=()
  while [[ $# -gt 0 && $1 != -- ]]; do envs+=("$1"); shift; done; shift
  args=("$@")
  if (( $(left) < secs )); then say "SKIPPED $tag (time left $(left)s < $secs)"; return; fi
  say "$tag: start ($tree, $target ${args[*]:-})"
  $SSH "INTARSIA_REPO=\$HOME/intarsiaml-tp2-$tree INTARSIA_TARGET_VOL=intarsia-cargo-target-tp2-$tree-hiptest HIP_VISIBLE_DEVICES=0,1 \
    INTARSIA_CONTAINER_NAME=hgv-test timeout -k 10 $secs \$HOME/intarsiaml-tp2-$tree/deploy/tatooine/dev.sh env INTARSIA_REQUIRE_GPU=1 ${envs[*]:-} \
    cargo test --release -p intarsia-backend-hip --test $target -- --test-threads=1 --nocapture ${args[*]:-}" < /dev/null > "$OUT/$tag.log" 2>&1
  local rc=$?
  $SSH 'docker rm -f hgv-test' > /dev/null 2>&1 < /dev/null
  say "$tag: exit $rc: $(grep -aE '^test result|FAILED|panicked' "$OUT/$tag.log" | head -4 | tr '\n' '|')"
}
for step in $STEPS; do
  case $step in
    gates)
      t gate-rows-exact hgv 400 rows_exact --
      t gate-moe-grouped hgv 300 moe_block -- moe_block_grouped
      t gate-ar-norm hgv 200 ar_norm --
      t gate-ar-push hgv 200 ar_push --
      t gate-comm hgv 300 comm -- ;;
    timing)
      for tree in hgvbase hgv; do
        t time-gemv-$tree $tree 240 rows_exact -- --ignored timing_gemv_rows
        t time-mxfp4-$tree $tree 240 rows_exact -- --ignored timing_mxfp4_rows
        t time-moe-$tree $tree 240 moe_block INTARSIA_HIP_MOE_ROWS_BENCH=1 INTARSIA_HIP_MOE_ROWS_BENCH_T=5,8,13,16 -- moe_rows_timing
      done ;;
  esac
done
say "done, time left $(left)s"
