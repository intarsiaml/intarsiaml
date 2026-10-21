#!/usr/bin/env bash
# CPU (f32 accumulate, f64 NVFP4/MoE/GDN math) reference on chat_reuse_4k, prefill rows, top-64 dump. Runs under fleetq on endor.
set -u
D=/private/tmp/claude-501/-opt-code-intarsiaml/f31ccf91-6032-412f-aeab-eea38852047d/scratchpad/m3
B=/opt/code/intarsiaml/.claude/worktrees/agent-ad695ea935d5d712e/target/release/intarsia
echo "start $(date +%T)"
INTARSIA_CPU_FREE_MEM_GB=20 INTARSIA_ACC_POS_DUMP=$D/cpu-chat4k-fp8kv.pos.tsv $B accuracy --model /Users/josef/Models/Qwen3.6-35B-A3B-NVFP4 --backend cpu \
  --golden $D/chat4k.safetensors --mode prefill --max-seq 8192 --mem-budget 40 --prefill-chunk 256 --kv-dtype fp8_e4m3 \
  --allow-unstamped --allow-dirty --write-self-golden $D/dumps/cpu-chat4k-fp8kv.dump.safetensors > $D/cpu-chat4k-fp8kv.log 2>&1
echo "exit $? $(date +%T)"
