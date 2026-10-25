#!/usr/bin/env bash
# Lane wg merge gates, ON tatooine inside wgm-win.sh (radiance down, card 0): 35B accuracy with option 1 ON vs the knob-off control.
#   short golden (acc-short, 10 prompts): floor verdict + the tracked HIP baseline and noise envelope (golden/baselines/qwen36-35b-a3b-nvfp4-vllm-top20.hip.*.json)
#   long golden (5 prompts, 16k): floor only (no HIP baseline exists for it)
# Both arms use the same flags as the recorded baseline (--mode both --decode-positions 16 --max-seq 4096 --mem-budget 29 for short). The baseline predates build
# stamps, so --allow-unstamped --allow-same-code. Logs in ~/wgm-logs/acc/<arm>-<golden>.log; exit codes are printed, never fatal.
set -uo pipefail
export HIP_VISIBLE_DEVICES=0
O=$HOME/wgm-logs/acc; mkdir -p "$O"
D=$HOME/intarsiaml-wgm/deploy/tatooine/dev.sh
OPT1="INTARSIA_HIP_WMMA_ROWS=1 INTARSIA_HIP_WMMA_TABLE=fp8a8=all,bf16=all"
M=/models/Qwen3.6-35B-A3B-NVFP4; G=/models/golden
B=/work/golden/baselines/qwen36-35b-a3b-nvfp4-vllm-top20.hip
T0=$(date +%s); BUDGET=${WG_BUDGET:-1200}
run() { # tag, env, args...
  local tag=$1 e=$2; shift 2
  if [ $(( $(date +%s) - T0 )) -ge "$BUDGET" ]; then echo "acc: skip $tag (budget)"; return; fi
  INTARSIA_REPO=$HOME/intarsiaml-wgm INTARSIA_TARGET_VOL=intarsia-cargo-target-wgm $D env $e /target/release/intarsia accuracy --backend hip --device 0 --model $M "$@" \
    < /dev/null > "$O/$tag.log" 2>&1
  echo "$tag exit=$? $(grep -aE 'verdict|PASS|FAIL' "$O/$tag.log" | tail -n 4 | tr '\n' '|' | cut -c1-300)"
}
SHORT="--golden $G/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors --mode both --decode-positions 16 --max-seq 4096 --mem-budget 29"
LONG="--golden $G/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors --mode both --decode-positions 64 --max-seq 16384 --mem-budget 29"
BASE="--baseline $B.baseline.json --noise-envelope $B.envelope.json --allow-unstamped --allow-same-code"
run opt1-short "$OPT1" $SHORT $BASE
run off-short "WG_ARM=off" $SHORT $BASE
run opt1-long "$OPT1" $LONG
run off-long "WG_ARM=off" $LONG
echo "ACC DONE $(( $(date +%s) - T0 )) s"
