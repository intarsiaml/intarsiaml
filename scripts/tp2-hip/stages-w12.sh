#!/usr/bin/env bash
# Lane tp2 window W=w12 (ON tatooine, radiance down, called by window.sh): what w8 / w9 could not do.
#  - w8's `self` compares (tp=2 run to run, INTARSIA_TP_VOCAB=0, INTARSIA_TP_ARNORM=1) all stopped at "unverifiable: pass --pre-change":
#    the dumps are recorded by this very build, so --pre-change is this build's head (PC=<sha>, passed in XENV).
#  - w9's 27B bitwise dump compare had the same flaw, and the 27B accuracy ran against the W4A4 golden, which fails at tp=1 too;
#    T2's gate for this model is the W4A16 golden (p1-golden/...-w4a16-top20).
#  - the mutant `batch-peer-no-table` is only exercised by a batched step: its models test binary (mutants-batch-build.sh) must FAIL
#    the batched-vs-solo identity test at tp=2.
set -uo pipefail
export O=${1:?out dir}
source "$HOME/tp2/lib.sh"
D=${D:-batch}; PC=${PC:?PC=<head sha of the build>}
CAP=${CAP:-1500}; T0=$(date +%s)
left() { echo $(( CAP - ($(date +%s) - T0) )); }
need() { if (( $(left) > $1 )); then return 0; fi; log "SKIPPED (time left $(left)s < $1): $2"; return 1; }
FLOOR="--self-max-kl 1e-9 --self-max-pos-kl 1e-6 --self-max-flip-pct 0 --self-max-tail 5e-8 --pre-change $PC"
log "window start, CAP ${CAP}s, pre-change $PC"
# 1. tp=2 35B: run to run, vocab head off, fused all-reduce+rmsnorm: fresh run against a first dump, bitwise class
acc $D 0,1 w12-a $M35 $SG $TP2
acc $D 0,1 w12-b $M35 $SG $TP2
self $D 0,1 w12-b-vs-a acc-w12-a.safetensors $TP2 $FLOOR
E="INTARSIA_TP_VOCAB=0" self $D 0,1 w12-vocab0-vs-a acc-w12-a.safetensors $TP2 $FLOOR
E="INTARSIA_TP_ARNORM=1" self $D 0,1 w12-arnorm-vs-a acc-w12-a.safetensors $TP2 $FLOOR
log "evidence arnorm: $(grep -ah 'fused all-reduce+rmsnorm' "$O/self-w12-arnorm-vs-a.err" | tr '\n' '|')"
E="INTARSIA_TP_FUSION=0" self $D 0,1 w12-fusion0-vs-a acc-w12-a.safetensors $TP2 $FLOOR
# 2. the 27B at W4A16: tp=1 dump, tp=2 against it (T2: bitwise class)
M27=/models/models--amd--Qwen3.8-27B-Quark-AWQ-MXFP4/snapshots/5233554c5fa56afda40150556b95573c2d7d29c0
G27=/work/p1-golden/qwen3.8-27b-mxfp4-hfquark-w4a16-top20.safetensors
N27=/work/golden/ref_noise/qwen3.8-27b-mxfp4-hfquark-top20.safetensors.json
export INTARSIA_MODELS=/data/models/radiance/hf-cache/hub
if need 400 "27B"; then
  run $D 0 600 acc-27b-tp1 accuracy --backend hip --model $M27 --mem-budget 29 --golden $G27 --mode both --write-self-golden /work/tp2-out/acc-27b-tp1.safetensors
  mv "$(T $D)/tp2-out/acc-27b-tp1.safetensors" "$O/" 2>/dev/null; cp "$O/acc-27b-tp1.safetensors" "$(T $D)/tp2-out/"
  run $D 0,1 600 self-27b-tp2-vs-tp1 accuracy --backend hip --model $M27 --mem-budget 29 --mode both --self-golden /work/tp2-out/acc-27b-tp1.safetensors \
    --allow-same-code --allow-config-mismatch $TP2 $FLOOR
fi
# 3. batch-peer-no-table: the mutant's models test binary must fail the batched-vs-solo test at tp=2
if need 300 "mutant batch"; then
  MT=mut-batch-peer-no-table
  INTARSIA_REPO=$(T $MT) INTARSIA_TARGET_VOL=intarsia-cargo-target-tp2-mutb-hiptest HIP_VISIBLE_DEVICES=0,1 timeout 280 \
    "$(T $MT)/deploy/tatooine/dev.sh" env INTARSIA_BATCH_TEST_BACKEND=hip INTARSIA_REQUIRE_GPU=1 INTARSIA_QWEN36_DIR=$M35 INTARSIA_BATCH_TEST_TP=2 \
    cargo test --release -p intarsia-models --test batch_cuda_real -- --test-threads=1 --nocapture --exact real_35b_batched_decode_is_bitwise_the_solo_step \
    > "$O/mtest-mut-batch-peer-no-table.log" 2>&1 < /dev/null
  rc=$?
  log "MUTANT batch-peer-no-table (batched test, exit $rc; want non-zero): $(grep -aE 'test result|panicked|logits differ|HSA_STATUS' "$O/mtest-mut-batch-peer-no-table.log" | head -4 | cut -c1-200 | tr '\n' '|')"
fi
log "done, time left $(left)s"
