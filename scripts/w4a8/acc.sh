#!/usr/bin/env bash
# Lane W4A8 gate 2 (KL vs the W4A16 golden), ON tatooine INSIDE a window that holds both cards (radiance down): the engine's
# `intarsia accuracy` against the 27B W4A16 golden with --mxfp4-act none (the reference run) on card 0 and fp8 (the candidate) on
# card 1, tp1, prefill + decode positions; then the batched == solo gate of the 27B at tp2 in fp8 mode (INTARSIA_TEST_MXFP4_ACT=fp8).
set -uo pipefail
R=$HOME/intarsiaml-w4a8; V=intarsia-cargo-w4a8; D=$R/deploy/tatooine/dev.sh
O=${O:-$HOME/intarsiaml-w4a8-logs/acc}; mkdir -p "$O" "$R/golden"
cp -n /data/models/intarsia/golden/qwen3.8-27b-mxfp4-hfquark-*top20.safetensors "$R/golden/" 2>/dev/null
H=/data/models/radiance/hf-cache/hub
M27=/models/models--amd--Qwen3.8-27B-Quark-AWQ-MXFP4/snapshots/5233554c5fa56afda40150556b95573c2d7d29c0
G=golden/qwen3.8-27b-mxfp4-hfquark-w4a16-top20.safetensors
acc() { # acc <gpu> <act> <tag>
  INTARSIA_REPO=$R INTARSIA_TARGET_VOL=$V HIP_VISIBLE_DEVICES=$1 INTARSIA_MODELS=$H $D /target/release/intarsia accuracy --backend hip --model $M27 \
    --mem-budget 29 --max-seq 4096 --mxfp4-act $2 --golden $G --mode both --decode-positions 16 < /dev/null > "$O/acc-$3.log" 2>&1
  echo "acc $3 exit $?"
}
acc 0 none w4a16 &
acc 1 fp8 w4a8 &
wait
for t in w4a16 w4a8; do echo "== $t"; grep -aiE "kl|verdict|pass|fail|top1|flip" "$O/acc-$t.log" | tail -14 | cut -c1-200; done
echo "== batched == solo (27B tp1 fp8 act)"
INTARSIA_REPO=$R INTARSIA_TARGET_VOL=$V HIP_VISIBLE_DEVICES=0 INTARSIA_MODELS=$H $D env INTARSIA_BATCH_TEST_BACKEND=hip INTARSIA_REQUIRE_GPU=1 \
  INTARSIA_TEST_MXFP4_ACT=fp8 INTARSIA_QWEN36_DIR=$M27 cargo test --release -p intarsia-models --test batch_cuda_real real_35b_batched_decode_is_bitwise -- --test-threads=1 --nocapture < /dev/null 2>&1 \
  | grep -aE "^test |test result|panicked|row-exact|FAIL" | head -20
