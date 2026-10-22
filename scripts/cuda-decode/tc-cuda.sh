#!/usr/bin/env bash
# Lane tc: the tail-merge A/B on spark1 (CUDA, 35B NVFP4), started from endor INSIDE an exclusive fleetq slot (vLLM stays up):
#   fleetq run --host spark1 --exclusive --mem 30 --label "tc: CUDA tail merge" -- bash scripts/cuda-decode/tc-cuda.sh <out dir>
# Needs the trees built by `scripts/deploy.sh spark1 --backend cuda --tree tc-base|tc-dev --ref <sha> --smoke none` (tc-base = origin/main,
# tc-dev = the lane branch). Arms: base = tc-base; off = tc-dev, knob unset; on = tc-dev with INTARSIA_PREFILL_TAIL_MERGE=$TM.
# Stages: 256 greedy ids after the 16654- and 4408-token prompts (bitwise A/B), then PFREPS fresh-process prefill timings of each
# prompt per arm (generate --max-tokens 1; the arm order is reversed on even reps).
set -uo pipefail
OUT=${1:?out dir}; TM=${TM:-1024}; PFREPS=${PFREPS:-5}
mkdir -p "$OUT"
for p in 16654 4408; do
  ssh tatooine "cat ~/intarsiaml-tc/bench-out/tc/prompt$p.txt" < /dev/null | ssh spark1 "mkdir -p ~/tc-prompts && cat > ~/tc-prompts/prompt$p.txt"
done
ssh spark1 "TM=$TM PFREPS=$PFREPS bash -s" > "$OUT/cuda.log" 2>&1 <<'EOS'
M=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
gen() { # gen <arm> <prompt N> <max tokens> <out prefix>
  local arm=$1 n=$2 mt=$3 out=$4 tree=tc-dev envx=""
  [ "$arm" = base ] && tree=tc-base
  [ "$arm" = on ] && envx="INTARSIA_PREFILL_TAIL_MERGE=$TM"
  env $envx $HOME/intarsia-deploy/$tree/target/release/intarsia generate -m $M --backend cuda --mem-budget 28 --max-seq 18432 --ignore-eos --show-ids --profile-every 0 -p "$(cat $HOME/tc-prompts/prompt$n.txt)" --max-tokens $mt < /dev/null 2> "$out.err" > "$out.out"
  grep -a '^ids:' "$out.err" > "$out.ids"
  grep -aE "prompt:|finish=|steps=|rror|panick" "$out.err" | sed "s/^/$arm $n: /" | cut -c1-200
}
nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv
mkdir -p $HOME/tc-out
for n in 16654 4408; do
  echo "--- ids $n $(date +%T)"
  for a in base off on; do gen $a $n 256 $HOME/tc-out/ids-$a-$n; done
  for a in off on; do
    if [ -s $HOME/tc-out/ids-$a-$n.ids ] && cmp -s $HOME/tc-out/ids-base-$n.ids $HOME/tc-out/ids-$a-$n.ids; then echo "IDS $n: base == $a"; else echo "IDS $n: base != $a"; fi
  done
done
for n in 16654 4408; do
  for rep in $(seq 1 $PFREPS); do
    order="base off on"; [ $((rep % 2)) -eq 0 ] && order="on off base"
    for a in $order; do
      echo "pf $n rep $rep $a: $(gen $a $n 1 $HOME/tc-out/pf-$a-$n-r$rep | grep -a 'finish=' | sed 's/.*prefill \([0-9]*\) ms.*/\1 ms/')"
    done
  done
done
echo "done $(date +%T)"
EOS
cat "$OUT/cuda.log"
rsync -a spark1:tc-out/ "$OUT/" < /dev/null
