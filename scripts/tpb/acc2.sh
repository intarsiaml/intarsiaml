#!/usr/bin/env bash
# Lane TPB, ON tatooine INSIDE a GPU window (win.sh acc2.sh): KL of the engine against the 27B W4A16 golden at tp2 (W4A8, the recipe default),
# prefill + 16 decode positions per prompt, with f32 all-reduce partials (the default) and with bf16 partials (INTARSIA_TP_PARTIAL=bf16).
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
O=${O:-$HOME/intarsiaml-tpb-logs/acc2}; mkdir -p "$O" "$R/golden"
cp -n /data/models/intarsia/golden/qwen3.8-27b-mxfp4-hfquark-*top20.safetensors "$R/golden/" 2>/dev/null
GP=golden/qwen3.8-27b-mxfp4-hfquark-w4a16-top20.safetensors
# ARMS: space separated arm names (default "f32 bf16"); an arm's env is INTARSIA_TP_PARTIAL=<arm> for f32/bf16, else ENV_<arm> (e.g. ENV_hyb="A=1 B=2")
for arm in ${ARMS:-f32 bf16}; do
  case $arm in f32|bf16) e="env INTARSIA_TP_PARTIAL=$arm" ;; hyb) e="env INTARSIA_TP_PARTIAL=bf16-prefill" ;; rec) e="env" ;; *) v=ENV_$arm; e="env ${!v:-}" ;; esac
  gpu 230 $e /work/tpb-intarsia accuracy --backend hip --model $M27 --tp 2 --devices 0,1 --mem-budget 29 --max-seq 4096 --golden /work/$GP --mode both \
    --decode-positions 16 --allow-unstamped --allow-dirty > "$O/acc-$arm.log" 2>&1
  echo "== tp2 $arm: exit $?"; grep -aiE "21-way KL|golden gate|mean" "$O/acc-$arm.log" | tail -6 | cut -c1-220
done
