#!/usr/bin/env bash
# Lane hload (from scripts/k4fp8/tests.sh): the correctness gates of the load-wait fixes (fp8 one-row family, MoE), driven FROM THE MAC under one fleetq job that holds both R9700 cards and
# takes radiance down (same guard as mbwin.sh / window.sh). The tree k4fp8 is built with scripts/tp2-hip/build.sh HIPTEST=1 HIPTEST_MODELS=1
# (the tests are precompiled in the volume intarsia-cargo-target-tp2-k4fp8-hiptest).
#   STEPS="hip tp1" OUT=<dir> fleetq run --host tatooine-r9700,tatooine-r9700-2 --takes-down radiance --quiet gpu,cpu --expect 20 --mem 30 \
#     --label "HLOAD gates" -- scripts/hload/tests.sh
# STEPS (space separated, in order):
#   hip   the HIP crate's fp8 tests on card 0 (plus the fold, pair and decode tests that run the one-row body): rows_exact (every M = 1..16 against the one-row kernel, reversed order, 6 scale modes,
#         shapes incl. the tp2 shards), fp8_f8, rows_registry, frag_layout, quick_coverage, fp8_pair_shapes, fusion_p1, fused_decode,
#         decode_v2, wmma_folds, ar_norm, packed_rows
#   moe   the HIP crate's MoE tests on card 0: moe_block (fused 1-2 row block, grouped block, G2 configs), moe_pairs, moe_v3, quantmoe
#   tp1   intarsia-models batch_cuda_real real_35b_batched_decode_is_bitwise_the_solo_step on the real 35B, card 0: N = 2..16 rows batched
#         are bit for bit (f32 logits and argmax) their solo steps
#   tp2   the same with INTARSIA_BATCH_TEST_TP=2 on both cards
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
STEPS=${STEPS:-"hip moe tp1 tp2"}; TREE=${TREE:-hload}; CAP=${CAP:-1150}
M35=/models/Qwen3.6-35B-A3B-NVFP4
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"
OUT=${OUT:?OUT dir}; mkdir -p "$OUT"
T0=$(date +%s); left() { echo $(( CAP - ($(date +%s) - T0) )); }
say() { echo "$(date +%T) $*" | tee -a "$OUT/steps.log"; }
SSH="ssh -o BatchMode=yes tatooine"
FIFO="$OUT/guard.fifo"; mkfifo "$FIFO"
$SSH 'mkdir -p ~/hprof' < /dev/null; scp -q "$REPO/scripts/hprof/guard.sh" tatooine:hprof/guard.sh
$SSH 'bash ~/hprof/guard.sh' > "$OUT/guard.log" 2>&1 < "$FIFO" &
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

TD='$HOME/intarsiaml-tp2-'$TREE
VOL=intarsia-cargo-target-tp2-$TREE-hiptest
# hiptest <tag> <gpus> <timeout s> <cargo test args...>
hiptest() {
  local tag=$1 gpus=$2 to=$3; shift 3
  $SSH "INTARSIA_REPO=$TD INTARSIA_TARGET_VOL=$VOL HIP_VISIBLE_DEVICES=$gpus timeout -k 10 $to $TD/deploy/tatooine/dev.sh env INTARSIA_REQUIRE_GPU=1 cargo test --release -p intarsia-backend-hip $* -- --test-threads=1 --nocapture" \
    > "$OUT/hiptest-$tag.log" 2>&1 < /dev/null
  say "hiptest $tag exit $?: $(grep -aE '^test result|FAILED|panicked|SKIPPED' "$OUT/hiptest-$tag.log" | head -6 | cut -c1-160 | tr '\n' '|')"
}
# mtest <tag> <gpus> <timeout s> <extra env> <test name>
mtest() {
  local tag=$1 gpus=$2 to=$3 e=$4 name=$5
  $SSH "INTARSIA_REPO=$TD INTARSIA_TARGET_VOL=$VOL HIP_VISIBLE_DEVICES=$gpus timeout -k 10 $to $TD/deploy/tatooine/dev.sh env INTARSIA_BATCH_TEST_BACKEND=hip INTARSIA_REQUIRE_GPU=1 INTARSIA_QWEN36_DIR=$M35 $e cargo test --release -p intarsia-models --test batch_cuda_real -- --test-threads=1 --nocapture --exact $name" \
    > "$OUT/mtest-$tag.log" 2>&1 < /dev/null
  say "mtest $tag exit $?: $(grep -aE '^test result|FAILED|panicked|SKIPPED|bitwise' "$OUT/mtest-$tag.log" | head -8 | cut -c1-200 | tr '\n' '|')"
}
for st in $STEPS; do
  if (( $(left) < 200 )); then say "SKIPPED $st (time left $(left)s)"; continue; fi
  case $st in
    hip) for t in rows_exact fp8_f8 rows_registry frag_layout quick_coverage fp8_pair_shapes fusion_p1 fused_decode decode_v2 wmma_folds ar_norm packed_rows; do hiptest "$t" 0 400 --test "$t"; done ;;
    moe) for t in moe_block moe_pairs moe_v3 quantmoe; do hiptest "$t" 0 400 --test "$t"; done ;;
    tp1) mtest tp1 0 900 "" real_35b_batched_decode_is_bitwise_the_solo_step ;;
    tp2) mtest tp2 0,1 900 "INTARSIA_BATCH_TEST_TP=2" real_35b_batched_decode_is_bitwise_the_solo_step ;;
  esac
done
say "done, time left $(left)s"
