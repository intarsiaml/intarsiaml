#!/usr/bin/env bash
# Lane l5r7 logits gate: self-golden (top-64 log-probs of every prefill position and of the teacher-forced decode window) written by the merge-base build and
# by the branch build (z-fast launch order default ON) on the 35B short and long goldens; the two files must be byte-identical tensor by tensor (sgcmp.py).
# Also the branch build with INTARSIA_HIP_MOE_ZFAST=0 on the short golden (the knob off is the old order). Flags as the recorded baseline (scripts/wgm/acc.sh).
. "$HOME/intarsiaml-l5-$NAME/scripts/l5r7/lib.sh"
SHORT="--golden $GOLD/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors --mode both --decode-positions 16 --max-seq 4096 --mem-budget 29"
LONG="--golden $GOLD/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors --mode both --decode-positions 64 --max-seq 16384 --mem-budget 29"
D=$HOME/intarsiaml-l5-$NAME/l5-out; B=$HOME/intarsiaml-l5-zbase/l5-out; mkdir -p "$D" "$B"
T0=$(date +%s); BUDGET=${SG_BUDGET:-2400}
acc() { # <fn: dv|bs> <tag> <env> <outdir-in-container-relative> args...
  local fn=$1 tag=$2 e=$3 out=$4; shift 4
  if [ $(( $(date +%s) - T0 )) -ge "$BUDGET" ]; then echo "sg: skip $tag (budget)"; return; fi
  $fn env $e /target/release/intarsia accuracy --backend hip --device 0 --model $M35 "$@" --write-self-golden /work/l5-out/$out > "$O/$tag.log" 2>&1
  echo "$tag exit=$? $(grep -aE 'verdict|PASS|FAIL|^Error|error' "$O/$tag.log" | tail -n 4 | tr '\n' '|' | cut -c1-300)"
}
cmpf() { python3 "$HOME/intarsiaml-l5-$NAME/scripts/l5r7/sgcmp.py" "$1" "$2"; }
for g in short long; do
  if [ $g = short ]; then A="$SHORT"; else A="$LONG"; fi
  acc bs base-$g "L5R7=base" sg-base-$g.safetensors $A
  acc dv dev-$g "L5R7=dev" sg-dev-$g.safetensors $A
  echo "== self-golden $g: merge-base vs branch (z-fast on): $(cmpf $B/sg-base-$g.safetensors $D/sg-dev-$g.safetensors 2>&1)"
done
acc dv devoff-short "INTARSIA_HIP_MOE_ZFAST=0" sg-devoff-short.safetensors $SHORT
echo "== self-golden short: merge-base vs branch with INTARSIA_HIP_MOE_ZFAST=0: $(cmpf $B/sg-base-short.safetensors $D/sg-devoff-short.safetensors 2>&1)"
echo "SG DONE $(( $(date +%s) - T0 )) s"
