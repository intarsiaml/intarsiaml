#!/usr/bin/env bash
# A2 end-to-end of the noise-aware gate on the CPU backend, Qwen3-Reranker-0.6B. Run inside a fleetq slot (FLEETQ_JOB set).
set -u
cd /opt/code/intarsiaml/.claude/worktrees/agent-a8eb89d1ece4a7d80
G=/opt/code/intarsiaml/golden
E=/private/tmp/claude-501/-opt-code-intarsiaml/f31ccf91-6032-412f-aeab-eea38852047d/scratchpad/e2e
RR=/Users/josef/.cache/huggingface/hub/models--Qwen--Qwen3-Reranker-0.6B/snapshots/e61197ed45024b0ed8a2d74b80b4d909f1255473
BIN=./target/debug/intarsia
mkdir -p $E
GOLD=$G/qwen3-reranker-0.6b.safetensors
COMMON="--model $RR --backend cpu --golden $GOLD --mode prefill"
$BIN ref-noise --golden $GOLD --other $G/qwen3-reranker-0.6b-fp32.safetensors --method precision_spread --out $E/rec-all.json --how "e2e: bf16 golden against fp32" > $E/00-rec-all.log 2>&1
$BIN ref-noise --golden $GOLD --other $G/qwen3-reranker-0.6b-fp32.safetensors --method precision_spread --tokenizer-dir $RR --chat-turns chatml --out $E/rec-scored.json --how "e2e: bf16 golden against fp32" > $E/00-rec-scored.log 2>&1
echo "== (a) --no-ref-noise"
$BIN accuracy $COMMON --no-ref-noise --write-verdict $E/va.json > $E/a.log 2>&1; echo "exit=$?"
echo "== (b) all-rows record, factor 5"
$BIN accuracy $COMMON --ref-noise $E/rec-all.json --ref-noise-factor 5 --write-verdict $E/vb.json > $E/b.log 2>&1; echo "exit=$?"
echo "== (c) scored-rows record, factor 5"
$BIN accuracy $COMMON --ref-noise $E/rec-scored.json --ref-noise-factor 5 --write-verdict $E/vc.json > $E/c.log 2>&1; echo "exit=$?"
echo "== (d) write self-golden"
$BIN accuracy $COMMON --ref-noise $E/rec-scored.json --write-self-golden $E/S.safetensors --allow-unstamped --allow-dirty > $E/d0.log 2>&1; echo "exit=$?"
echo "== (d1) clean self vs ref noise"
$BIN accuracy --model $RR --backend cpu --mode prefill --self-golden $E/S.safetensors --self-vs-ref-noise --allow-unstamped --allow-same-code --allow-dirty --write-verdict $E/vd1.json > $E/d1.log 2>&1; echo "exit=$?"
echo "== (d2) faulted self vs ref noise"
INTARSIA_CPU_FAULT=matmul:1e-3 $BIN accuracy --model $RR --backend cpu --mode prefill --self-golden $E/S.safetensors --self-vs-ref-noise --allow-unstamped --allow-same-code --allow-dirty --write-verdict $E/vd2.json > $E/d2.log 2>&1; echo "exit=$?"
echo "== (d3) clean self vs ref noise with a tiny worst-prompt factor (must FAIL on the worst prompt only)"
$BIN accuracy --model $RR --backend cpu --mode prefill --self-golden $E/S.safetensors --self-vs-ref-noise --ref-noise-prompt-factor 1e-9 --allow-unstamped --allow-same-code --allow-dirty --write-verdict $E/vd3.json > $E/d3.log 2>&1; echo "exit=$?"
echo "== (e) self vs ref noise with no record (empty sidecar lookup)"
$BIN accuracy $COMMON --write-self-golden $E/S3.safetensors --allow-unstamped --allow-dirty > $E/e0.log 2>&1; echo "exit=$? (self-golden written without a record)"
$BIN accuracy --model $RR --backend cpu --mode prefill --self-golden $E/S3.safetensors --self-vs-ref-noise --allow-unstamped --allow-same-code --allow-dirty > $E/e1.log 2>&1; echo "exit=$? (no record: must fail with the message)"
$BIN accuracy --model $RR --backend cpu --mode prefill --self-golden $E/S.safetensors --self-vs-ref-noise --no-ref-noise --allow-unstamped --allow-same-code --allow-dirty > $E/e2.log 2>&1; echo "exit=$? (--no-ref-noise conflict)"
