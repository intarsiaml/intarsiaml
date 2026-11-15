#!/usr/bin/env bash
# Lane gdn-chunk, gate 2 (35B NVFP4, one card): the chunked delta rule (INTARSIA_GDN_CHUNK=1) against the default build's
# sequential kernels. ON tatooine, inside a fleetq lease with radiance down. Steps (STEPS selects, default all):
#   trace   4k-token prompt with INTARSIA_HIP_TRACE=1, knob off / on: per-kernel GPU time (and which delta-rule kernel ran)
#   acc     `accuracy --mode both` against the vLLM golden, knob off / on, each writing a self-golden
#   self    knob on against the knob-off self-golden (--self-vs-ref-noise: mean KL(base || change) vs the golden's ref noise)
#   gen     256 greedy ids, short and 4k prompt, knob off / on (the divergence point)
#   long    the same pair and A/B on the long golden (prompts up to 16k tokens: many chunks per pass)
#   qual    `qualify --quick-only`, knob on / off
set -u
cd "$(dirname "$0")/../.." && . scripts/gdn-chunk/lib.sh
STEPS=${STEPS:-"trace acc self gen"}
W=/work/${O#"$TREE"/}   # $O as the container sees it
ON="INTARSIA_GDN_CHUNK=1"
SHORT='Write a long story about a robot who learns to paint.'
P4K="$(cat scripts/hip-decode/prompt4k.txt)"
GEN="generate -m $M35 --backend hip --mem-budget 29 --max-seq 8192 --chat --no-think --profile-every 0 --ignore-eos --show-ids"
ACC="accuracy --backend hip --model $M35 --mode both --mem-budget 29 --max-seq 17408 --allow-unstamped"
LG=$G/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors
NS=/work/golden/ref_noise/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors.json
NL=/work/golden/ref_noise/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors.json
AB="--allow-same-code --allow-config-mismatch"
has() { [[ " $STEPS " == *" $1 "* ]]; }
if has trace; then
  E="INTARSIA_HIP_TRACE=1" run 0 240 trace-off $GEN -p "$P4K" --max-tokens 8
  E="INTARSIA_HIP_TRACE=1 $ON" run 0 240 trace-on $GEN -p "$P4K" --max-tokens 8
fi
if has acc; then
  E="" run 0 420 acc-off $ACC --golden $SG --write-self-golden $W/acc-off.safetensors
  E="$ON" run 0 420 acc-on $ACC --golden $SG --write-self-golden $W/acc-on.safetensors
fi
if has self; then
  E="$ON" run 0 420 self-on $ACC --self-golden $W/acc-off.safetensors $AB --self-vs-ref-noise --ref-noise $NS
  E="$ON" run 0 420 self-on-budgets $ACC --self-golden $W/acc-off.safetensors $AB
  E="" run 0 420 self-off $ACC --self-golden $W/acc-off.safetensors $AB   # control: the default build against itself
fi
if has long; then
  E="" run 0 600 accl-off $ACC --golden $LG --write-self-golden $W/accl-off.safetensors
  E="$ON" run 0 600 accl-on $ACC --golden $LG --write-self-golden $W/accl-on.safetensors
  E="$ON" run 0 600 selfl-on $ACC --self-golden $W/accl-off.safetensors $AB --self-vs-ref-noise --ref-noise $NL
  E="$ON" run 0 600 selfl-on-budgets $ACC --self-golden $W/accl-off.safetensors $AB
fi
if has gen; then
  for arm in off on; do
    e=""; [[ $arm == on ]] && e="$ON"
    E="$e" run 0 240 gen-short-$arm $GEN -p "$SHORT" --max-tokens 256
    E="$e" run 0 240 gen-4k-$arm $GEN -p "$P4K" --max-tokens 256
    grep -a '^ids:' "$O/gen-short-$arm.err" > "$O/gen-short-$arm.ids"; grep -a '^ids:' "$O/gen-4k-$arm.err" > "$O/gen-4k-$arm.ids"
  done
fi
if has qual; then
  QG=${QG:-$G/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors}
  for arm in on off; do   # off: the control (the recorded baselines against the default build)
    e=""; [[ $arm == on ]] && e="$ON"
    E="$e" run 0 900 qual-$arm qualify --quick-only --backend hip --model $M35 --golden $QG \
      --oracle-golden /work/golden/oracle/qwen36-35b-a3b-nvfp4-cpu-oracle.safetensors --model-name Qwen3.6-35B-A3B-NVFP4 --mem-budget 29 \
      --bench-sizes 128,1024,4096 --reps 1 --gen 16 --out $W/qual-$arm
  done
fi
log "done"
