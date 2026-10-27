#!/usr/bin/env bash
# Lane H27P, ON tatooine INSIDE a GPU window (win.sh acc.sh): is the 27B's logit dump bitwise chunk-invariant? `intarsia accuracy --mode both`
# against the W4A16 golden's prompts at tp1 (W4A8, the recipe default), once at --prefill-chunk 256 (card 0) and once at 2048 (card 1; the recipe's
# peel 16 applies), each with --write-self-golden; then dumpeq.py compares every tensor (prefill rows and decode rows) byte for byte.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
O=${O:-$HOME/intarsiaml-h27p-logs/acc}; mkdir -p "$O" "$R/golden"
cp -n /data/models/intarsia/golden/qwen3.8-27b-mxfp4-hfquark-*top20.safetensors "$R/golden/" 2>/dev/null
GP=golden/qwen3.8-27b-mxfp4-hfquark-w4a16-top20.safetensors
acc() { # acc <gpu> <chunk>
  G=$1 gpu 900 /work/h27p-intarsia accuracy --backend hip --model $M27 --mem-budget 29 --max-seq 4096 --golden /work/$GOLD --mode both --decode-positions 16 \
    --prefill-chunk $2 --allow-unstamped --allow-dirty --write-self-golden /work/h27p-acc-c$2.safetensors > "$O/acc-c$2.log" 2>&1
  echo "acc chunk $2 exit $?"
}
GOLD=$GP acc 0 256 &
GOLD=$GP acc 1 2048 &
wait
for c in 256 2048; do echo "== chunk $c"; grep -aiE "chunk|verdict|PASS|FAIL|mean kl" "$O/acc-c$c.log" | tail -5 | cut -c1-200; done
python3 -I $R/scripts/mx/dumpeq.py $R/h27p-acc-c256.safetensors $R/h27p-acc-c2048.safetensors
