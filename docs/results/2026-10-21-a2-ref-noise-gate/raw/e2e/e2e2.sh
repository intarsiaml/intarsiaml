#!/usr/bin/env bash
set -u
cd /opt/code/intarsiaml/.claude/worktrees/agent-a8eb89d1ece4a7d80
E=/private/tmp/claude-501/-opt-code-intarsiaml/f31ccf91-6032-412f-aeab-eea38852047d/scratchpad/e2e
RR=/Users/josef/.cache/huggingface/hub/models--Qwen--Qwen3-Reranker-0.6B/snapshots/e61197ed45024b0ed8a2d74b80b4d909f1255473
BIN=./target/debug/intarsia
F="--model $RR --backend cpu --mode prefill --self-golden $E/S.safetensors --self-vs-ref-noise --allow-unstamped --allow-same-code --allow-dirty"
echo "== (d4) fault 1e-2: mean KL over the noise"
INTARSIA_CPU_FAULT=matmul:1e-2 $BIN accuracy $F --write-verdict $E/vd4.json > $E/d4.log 2>&1; echo "exit=$?"
echo "== (d5) fault 1e-3 (mean inside the noise), prompt factor 1.0: worst prompt over its limit"
INTARSIA_CPU_FAULT=matmul:1e-3 $BIN accuracy $F --ref-noise-prompt-factor 1.0 --write-verdict $E/vd5.json > $E/d5.log 2>&1; echo "exit=$?"
for f in d4 d5; do
  echo "---- $f"
  grep -E "vs ref noise|KL threshold|non-finite|largest|max position|top-1 flips|tail diff|self-golden =>|self-golden (PASS|FAIL)" $E/$f.log | cut -c1-600
done
echo "---- e1/e2 errors"
grep -h "^Error" $E/e1.log $E/e2.log | cut -c1-500
