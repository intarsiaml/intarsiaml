#!/usr/bin/env bash
# Lane wht6-8b, gate 2 (35B NVFP4, tp=2 on both cards): the lossy all-reduce wires (INTARSIA_HIP_AR_WIRE=wht8 and, for the
# comparison in the same build, wht6) against the default exact wire, same build. ON tatooine, inside a fleetq lease with radiance
# down. WIRES (default "wht8 wht6") are the arms; STEPS selects:
#   acc     `accuracy --mode both` against the vLLM golden, wire off / each wire, each writing a self-golden
#   self    each wire against the wire-off self-golden: --self-vs-ref-noise (the change-vs-base KL rule); the control (wire off
#           against its own dump); the plain self-golden budgets for the first wire
#   cross   wht8 against the wht6 self-golden and the reverse (the lossy wires against each other)
#   gen     256 greedy ids, short and 4k prompt, wire off / each wire (the divergence point; scripts/wht6-8b/diverge.py)
#   long    the acc dumps and the change-vs-base rule on the long golden (prompts up to 16k tokens)
#   gref    the vLLM golden with its reference-noise record (`--ref-noise`, the required tp=2 gate class), wire off / each wire
#           (GREF_LONG=1: the long golden too)
set -u
cd "$(dirname "$0")/../.." && . scripts/wht6-8b/lib.sh
STEPS=${STEPS:-"acc self gen"}
W=/work/${O#"$TREE"/}   # $O as the container sees it
WIRES=${WIRES:-"wht8 wht6"}
on() { echo "INTARSIA_HIP_AR_WIRE=$1${ON_EXTRA:+ $ON_EXTRA}"; }
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
PRE=$(sed -n "s/^head=\([0-9a-f]*\) .*/\1/p" "$TREE/.stamp")   # the build is its own pre-change commit (the wire is the only difference, a runtime switch)
AB="--allow-same-code --allow-config-mismatch --pre-change $PRE"
has() { [[ " $STEPS " == *" $1 "* ]]; }
sum() { log "  $1: $(grep -ahE '^golden gate|self-golden gate|SELF-STATS|ref-noise|^Error|LOSSY' "$O/$1.out" "$O/$1.err" | cut -c1-260 | tr '\n' '|')"; }
if has acc; then
  E="" run 0,1 420 acc-off $ACC --golden $SG --write-self-golden $W/acc-off.safetensors; sum acc-off
  for w in $WIRES; do E="$(on $w)" run 0,1 420 acc-$w $ACC --golden $SG --write-self-golden $W/acc-$w.safetensors; sum acc-$w; done
fi
if has self; then
  for w in $WIRES; do E="$(on $w)" run 0,1 420 self-$w $ACC --self-golden $W/acc-off.safetensors $AB --self-vs-ref-noise --ref-noise $NS; sum self-$w; done
  E="$(on ${WIRES%% *})" run 0,1 420 self-budgets-${WIRES%% *} $ACC --self-golden $W/acc-off.safetensors $AB; sum self-budgets-${WIRES%% *}
  E="" run 0,1 420 self-off $ACC --self-golden $W/acc-off.safetensors $AB; sum self-off   # control: the exact wire against itself
fi
if has cross; then   # wht8 against the wht6 dump: independent decorrelations (a floor) or a shared deviation of the lossy path (a bug)?
  E="$(on wht8)" run 0,1 420 cross-8v6 $ACC --self-golden $W/acc-wht6.safetensors $AB --self-vs-ref-noise --ref-noise $NS; sum cross-8v6
  E="$(on wht6)" run 0,1 420 cross-6v8 $ACC --self-golden $W/acc-wht8.safetensors $AB --self-vs-ref-noise --ref-noise $NS; sum cross-6v8
fi
if has gen; then
  for arm in off $WIRES; do
    e=""; [[ $arm != off ]] && e="$(on $arm)"
    E="$e" run 0,1 240 gen-short-$arm $GEN -p "$SHORT" --max-tokens 256
    E="$e" run 0,1 240 gen-4k-$arm $GEN -p "$P4K" --max-tokens 256
    grep -a '^ids:' "$O/gen-short-$arm.err" > "$O/gen-short-$arm.ids"; grep -a '^ids:' "$O/gen-4k-$arm.err" > "$O/gen-4k-$arm.ids"
  done
  for w in $WIRES; do
    log "  divergence $w: $(python3 scripts/wht6-8b/diverge.py "$O"/gen-short-off.ids "$O"/gen-short-$w.ids "$O"/gen-4k-off.ids "$O"/gen-4k-$w.ids | tr '\n' '|')"
  done
fi
if has long; then
  [[ -f $O/accl-off.safetensors ]] || { E="" run 0,1 600 accl-off $ACC --golden $LG --write-self-golden $W/accl-off.safetensors; sum accl-off; }
  for w in $WIRES; do
    E="$(on $w)" run 0,1 600 selfl-$w $ACC --self-golden $W/accl-off.safetensors $AB --self-vs-ref-noise --ref-noise $NL; sum selfl-$w
  done
fi
if has gref; then   # the required tp=2 class: the vLLM golden with its reference-noise record, wire off / each wire
  for arm in off $WIRES; do
    e=""; [[ $arm != off ]] && e="$(on $arm)"
    E="$e" run 0,1 420 gref-$arm $ACC --golden $SG --ref-noise $NS; sum gref-$arm
    [[ ${GREF_LONG:-0} == 1 ]] && { E="$e" run 0,1 600 grefl-$arm $ACC --golden $LG --ref-noise $NL; sum grefl-$arm; }
  done
fi
log "done"
