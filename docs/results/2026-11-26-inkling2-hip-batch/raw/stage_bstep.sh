#!/usr/bin/env bash
# Window 2 stage (lane ink-hip-batch): Model::forward_batch on the real Inkling-2-Mini (examples/batch_step.rs, real text prompts):
#  1. batched tokens == solo tokens (asserted every step) at B 1/4/8/16, rows at mixed short contexts (256 + 8 i: the conv ring wraps,
#     the SWA window does not), default env (main's prefill: chunk 256);
#  2. the same past the 2048-key line of the full layers (2040 + 3 i: the solo steps cross from op 75's fused kernel to its split path),
#     prefill at chunk 2048 + MoE WMMA + flash for time (both state sets prefill alike);
#  3. speed only at a 4k context (--batched-only), B 1/4/8/16: per-step ms median / p99.
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-batch
S=$REPO/.scratch
B=/target/release/examples/batch_step
A="--backend hip --model /models/Inkling-2-Mini --mem-budget 29 --prompts text"
PF="env INTARSIA_HIP_INK_MOE_WMMA=1 INTARSIA_HIP_INK_ATTN_FLASH=1"
now() { date '+%F %T'; }
echo "[stage] bitwise short (ctx 256 + 8 i) $(now)"
bash "$S/run.sh" $B $A --ctx 256 --ctx-spread 8 --rounds 4 --steps 16 --batches 1,4,8,16 --digest 2>&1 | grep -avE '^\s*$' | tail -14
echo "[stage] bitwise 2k (ctx 2040 + 3 i) $(now)"
bash "$S/run.sh" $PF $B $A --ctx 2040 --ctx-spread 3 --rounds 2 --steps 8 --batches 4,16 --prefill-chunk 2048 --digest 2>&1 | grep -avE '^\s*$' | tail -10
echo "[stage] speed 4k batched-only $(now)"
bash "$S/run.sh" $PF $B $A --ctx 4096 --rounds 3 --steps 16 --batches 1,4,8,16 --prefill-chunk 2048 --batched-only 2>&1 | grep -avE '^\s*$' | tail -10
BR=${BR:-b2}
for e in "" "$PF"; do
  echo "[stage] plan show --max-batch 4 (${e:-default env}) $(now)"
  bash "$S/run.sh" $e /work/.scratch/bin/intarsia-$BR plan show -m /models/Inkling-2-Mini --backend hip --max-batch 4 --max-seq 4352 --mem-budget 29 --spec off 2>&1 \
    | grep -aE 'kv cap|weights|prefill chunk|_rows|moe_experts|rel_sink|ink_attn|conv_mix|wmma|flash|Error|error' | head -30
done
echo "[stage] done $(now)"
