#!/usr/bin/env bash
# Lane tp2 window W=w13 (ON tatooine, radiance down, called by window.sh), on the tree merged with main 5241377b (#145 row-exact MXFP4
# rows, #143): what w12 did not settle and what the merge needs.
#  - w12's mutant step ran with INTARSIA_MODELS exported for the 27B (hf-cache mount), so the mutant's test did not find the 35B and
#    died at its SKIPPED guard in 0.01 s: not a kill. Here the mutant runs BEFORE the 27B export, and a SKIPPED log is not a verdict.
#  - the dump compares (tp=2 run to run, 27B tp=2 against tp=1, floor budgets, --pre-change PC) on the merged build.
#  - 27B batched == solo at tp=2 (batch_cuda_real real_35b_batched_decode_is_bitwise_the_solo_step with INTARSIA_QWEN36_DIR = the 27B
#    snapshot, INTARSIA_BATCH_TEST_TP=2), and the 35B control of the same test on the merged tree.
set -uo pipefail
export O=${1:?out dir}
source "$HOME/tp2/lib.sh"
D=${D:-batch}; PC=${PC:?PC=<head sha of the build>}
CAP=${CAP:-1900}; T0=$(date +%s)
left() { echo $(( CAP - ($(date +%s) - T0) )); }
need() { if (( $(left) > $1 )); then return 0; fi; log "SKIPPED (time left $(left)s < $1): $2"; return 1; }
FLOOR="--self-max-kl 1e-9 --self-max-pos-kl 1e-6 --self-max-flip-pct 0 --self-max-tail 5e-8 --pre-change $PC"
M27=/models/models--amd--Qwen3.8-27B-Quark-AWQ-MXFP4/snapshots/5233554c5fa56afda40150556b95573c2d7d29c0
G27=/work/p1-golden/qwen3.8-27b-mxfp4-hfquark-w4a16-top20.safetensors
HUB=/data/models/radiance/hf-cache/hub
log "window start, CAP ${CAP}s, pre-change $PC"
# 1. 35B tp=2 run to run on the merged build (floor budgets, bitwise class)
acc $D 0,1 w13-a $M35 $SG $TP2
acc $D 0,1 w13-b $M35 $SG $TP2
self $D 0,1 w13-b-vs-a acc-w13-a.safetensors $TP2 $FLOOR
# 2. the mutant batch-peer-no-table: the batched-vs-solo test of the mutant's binary must FAIL on an identity assertion (not SKIPPED)
if need 400 "mutant batch"; then
  MT=mut-batch-peer-no-table
  INTARSIA_REPO=$(T $MT) INTARSIA_TARGET_VOL=intarsia-cargo-target-tp2-mutb-hiptest HIP_VISIBLE_DEVICES=0,1 timeout 300 \
    "$(T $MT)/deploy/tatooine/dev.sh" env INTARSIA_BATCH_TEST_BACKEND=hip INTARSIA_REQUIRE_GPU=1 INTARSIA_QWEN36_DIR=$M35 INTARSIA_BATCH_TEST_TP=2 \
    cargo test --release -p intarsia-models --test batch_cuda_real -- --test-threads=1 --nocapture --exact real_35b_batched_decode_is_bitwise_the_solo_step \
    > "$O/mtest-mut-batch-peer-no-table.log" 2>&1 < /dev/null
  rc=$?
  if grep -aq 'SKIPPED' "$O/mtest-mut-batch-peer-no-table.log"; then v="NO VERDICT (the test skipped)"; elif (( rc != 0 )); then v="KILLED"; else v="SURVIVED"; fi
  log "MUTANT batch-peer-no-table: $v (exit $rc): $(grep -aE 'test result|panicked|differ|mismatch|HSA_STATUS' "$O/mtest-mut-batch-peer-no-table.log" | head -4 | cut -c1-220 | tr '\n' '|')"
fi
# 3. 27B at W4A16: tp=1 dump, tp=2 against it (bitwise class)
export INTARSIA_MODELS=$HUB
if need 300 "27B dump compare"; then
  run $D 0 600 acc-27b-tp1 accuracy --backend hip --model $M27 --mxfp4-act none --mem-budget 29 --golden $G27 --mode both --write-self-golden /work/tp2-out/acc-27b-tp1.safetensors
  mv "$(T $D)/tp2-out/acc-27b-tp1.safetensors" "$O/" 2>/dev/null; cp "$O/acc-27b-tp1.safetensors" "$(T $D)/tp2-out/"
  run $D 0,1 600 self-27b-tp2-vs-tp1 accuracy --backend hip --model $M27 --mxfp4-act none --mem-budget 29 --mode both --self-golden /work/tp2-out/acc-27b-tp1.safetensors \
    --allow-same-code --allow-config-mismatch $TP2 $FLOOR
fi
# 4. 27B batched == solo at tp=2 (N = 2..16 sequences, 130 steps; ~10 min), and at tp=1 is main's own gate (mr, #145)
mtest27() {
  local tag=$1 to=$2; shift 2
  # shellcheck disable=SC2086
  INTARSIA_REPO=$(T $D) INTARSIA_TARGET_VOL=intarsia-cargo-target-tp2-$D-hiptest HIP_VISIBLE_DEVICES=0,1 timeout "$to" \
    "$(T $D)/deploy/tatooine/dev.sh" env INTARSIA_BATCH_TEST_BACKEND=hip INTARSIA_REQUIRE_GPU=1 INTARSIA_QWEN36_DIR="$1" INTARSIA_BATCH_TEST_TP=2 \
    cargo test --release -p intarsia-models --test batch_cuda_real -- --test-threads=1 --nocapture --exact real_35b_batched_decode_is_bitwise_the_solo_step \
    > "$O/mtest-$tag.log" 2>&1 < /dev/null
  local rc=$?
  log "mtest $tag exit $rc: $(grep -aE '^test result|FAILED|panicked|SKIPPED|pairs bitwise|tensor-parallel rank 1: batched' "$O/mtest-$tag.log" | head -8 | cut -c1-200 | tr '\n' '|')"
}
if need 900 "27B batched == solo at tp2"; then mtest27 tp2-27b-batched 840 "$M27"; fi
# 5. the 35B control of the same test on the merged tree (this is the unchanged-by-the-merge check)
unset INTARSIA_MODELS
if need 520 "35B batched == solo at tp2"; then mtest27 tp2-35b-batched 480 "$M35"; fi
log "done, time left $(left)s"
