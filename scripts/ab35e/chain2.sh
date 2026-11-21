#!/usr/bin/env bash
# ab35e: remaining windows (w3 gsm8k A B A, w4 gsm8k B with fp8 KV, w5 1-stream replay control) inside ONE fleetq job (one wait in the queue).
cd "$(dirname "$0")/../.." || exit 1
bash scripts/ab35e/window.sh w3 gsm8k
BARGS="--kv-dtype fp8_e4m3" BLABEL=fp8 bash scripts/ab35e/window.sh w4 gsm8kb
CTL_SERVE="--max-batch 2 --prefix-store on" bash scripts/ab35e/window-ctl.sh w5 replay1
REPLAYS="A 3|B 3|B 2|A 2" bash scripts/ab35e/window.sh w6 replay
