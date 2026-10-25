#!/usr/bin/env bash
# Lane l5 gate window B: 35B accuracy with the knob ON (short golden: INTARSIA_HIP_MOE_FP8_MIN=16 so the 10 short prompts take the fp8 experts, against the
# tracked HIP baseline + noise envelope, floor 3.0e-2, k=2 envelope; long golden at the default threshold, floor only: no HIP baseline exists for it) and the
# knob-off controls on the same binary. Flags are those of the recorded baseline (scripts/wgm/acc.sh).
. "$HOME/intarsiaml-l5-$NAME/scripts/l5/lib.sh"
B=/work/golden/baselines/qwen36-35b-a3b-nvfp4-vllm-top20.hip
SHORT="--golden $GOLD/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors --mode both --decode-positions 16 --max-seq 4096 --mem-budget 29"
LONG="--golden $GOLD/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors --mode both --decode-positions 64 --max-seq 16384 --mem-budget 29"
BASE="--baseline $B.baseline.json --noise-envelope $B.envelope.json --allow-unstamped --allow-same-code"
T0=$(date +%s); BUDGET=${ACC_BUDGET:-1500}
run() { # tag, env, args...
  local tag=$1 e=$2; shift 2
  if [ $(( $(date +%s) - T0 )) -ge "$BUDGET" ]; then echo "acc: skip $tag (budget)"; return; fi
  dv env $e /target/release/intarsia accuracy --backend hip --device 0 --model $M35 "$@" > "$O/$tag.log" 2>&1
  echo "$tag exit=$? $(grep -aE 'verdict|PASS|FAIL' "$O/$tag.log" | tail -n 5 | tr '\n' '|' | cut -c1-420)"
}
run on-short "INTARSIA_HIP_MOE_FP8_WMMA=1 INTARSIA_HIP_MOE_FP8_MIN=16" $SHORT $BASE
run on-long "INTARSIA_HIP_MOE_FP8_WMMA=1" $LONG
run off-short "L5_ARM=off" $SHORT $BASE
run off-long "L5_ARM=off" $LONG
echo "ACC DONE $(( $(date +%s) - T0 )) s"
