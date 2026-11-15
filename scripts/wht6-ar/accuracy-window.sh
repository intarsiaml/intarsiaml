#!/usr/bin/env bash
# Lane wht6-ar, gate 2 (35B NVFP4, tp=2 on both cards): the lossy wht6 all-reduce wire (INTARSIA_HIP_AR_WIRE=wht6) against the
# default exact wire, same build. ON tatooine, inside a fleetq lease with radiance down. Steps (STEPS selects):
#   acc     `accuracy --mode both` against the vLLM golden, wire off / on, each writing a self-golden
#   self    wire on against the wire-off self-golden: --self-vs-ref-noise (the change-vs-base KL rule), the plain self-golden
#           budgets, and the control (wire off against its own dump)
#   gen     256 greedy ids, short and 4k prompt, wire off / on (the divergence point; scripts/wht6-ar/diverge.py)
#   long    the acc pair and the change-vs-base rule on the long golden (prompts up to 16k tokens)
#   floor   the wire with a 1 MiB / 8 MiB floor (INTARSIA_HIP_AR_WIRE_MIN_KB) against the wire-off dump, short golden
set -u
cd "$(dirname "$0")/../.." && . scripts/wht6-ar/lib.sh
STEPS=${STEPS:-"acc self gen"}
W=/work/${O#"$TREE"/}   # $O as the container sees it
ON="INTARSIA_HIP_AR_WIRE=wht6${ON_EXTRA:+ $ON_EXTRA}"
TP2="--tp 2 --devices 0,1"
SHORT='Write a long story about a robot who learns to paint.'
P4K="$(cat scripts/hip-decode/prompt4k.txt)"
M=${M:-$M35}
SG=${SG:-$G/qwen36-35b-a3b-nvfp4-vllm-top20-meta.safetensors}
LG=${LG:-$G/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors}
NS=${NS:-/work/golden/ref_noise/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors.json}
NL=${NL:-/work/golden/ref_noise/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors.json}
GEN="$BIN generate -m $M --backend hip $TP2 --mem-budget 29 --max-seq 8192 --chat --no-think --profile-every 0 --ignore-eos --show-ids"
ACC="$BIN accuracy --backend hip --model $M $TP2 --mode both --mem-budget 29 --max-seq 17408 --allow-unstamped"
AB="--allow-same-code --allow-config-mismatch"
has() { [[ " $STEPS " == *" $1 "* ]]; }
sum() { log "  $1: $(grep -ahE '^golden gate|self-golden gate|SELF-STATS|ref-noise|^Error|LOSSY' "$O/$1.out" "$O/$1.err" | cut -c1-260 | tr '\n' '|')"; }
if has acc; then
  E="" run 0,1 420 acc-off $ACC --golden $SG --write-self-golden $W/acc-off.safetensors; sum acc-off
  E="$ON" run 0,1 420 acc-on $ACC --golden $SG --write-self-golden $W/acc-on.safetensors; sum acc-on
fi
if has self; then
  E="$ON" run 0,1 420 self-on $ACC --self-golden $W/acc-off.safetensors $AB --self-vs-ref-noise --ref-noise $NS; sum self-on
  E="$ON" run 0,1 420 self-on-budgets $ACC --self-golden $W/acc-off.safetensors $AB; sum self-on-budgets
  E="" run 0,1 420 self-off $ACC --self-golden $W/acc-off.safetensors $AB; sum self-off   # control: the exact wire against itself
fi
if has gen; then
  for arm in off on; do
    e=""; [[ $arm == on ]] && e="$ON"
    E="$e" run 0,1 240 gen-short-$arm $GEN -p "$SHORT" --max-tokens 256
    E="$e" run 0,1 240 gen-4k-$arm $GEN -p "$P4K" --max-tokens 256
    grep -a '^ids:' "$O/gen-short-$arm.err" > "$O/gen-short-$arm.ids"; grep -a '^ids:' "$O/gen-4k-$arm.err" > "$O/gen-4k-$arm.ids"
  done
  log "  divergence: $(python3 scripts/wht6-ar/diverge.py "$O"/gen-short-off.ids "$O"/gen-short-on.ids "$O"/gen-4k-off.ids "$O"/gen-4k-on.ids | tr '\n' '|')"
fi
if has long; then
  E="" run 0,1 600 accl-off $ACC --golden $LG --write-self-golden $W/accl-off.safetensors; sum accl-off
  E="$ON" run 0,1 600 accl-on $ACC --golden $LG --write-self-golden $W/accl-on.safetensors; sum accl-on
  E="$ON" run 0,1 600 selfl-on $ACC --self-golden $W/accl-off.safetensors $AB --self-vs-ref-noise --ref-noise $NL; sum selfl-on
fi
if has floor; then
  for kb in 1024 8192; do
    E="$ON INTARSIA_HIP_AR_WIRE_MIN_KB=$kb" run 0,1 420 self-on-$kb $ACC --self-golden $W/acc-off.safetensors $AB --self-vs-ref-noise --ref-noise $NS; sum self-on-$kb
  done
fi
log "done"
