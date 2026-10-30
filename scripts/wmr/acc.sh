#!/usr/bin/env bash
# Lane wmr, ON tatooine inside a window (radiance down): 35B accuracy, HIP tp1, main default vs option 1 (INTARSIA_HIP_WMMA_ROWS=1
# INTARSIA_HIP_WMMA_TABLE=fp8a8=all,bf16=all), with the repo's same-weights gate (ADR-0007 section 6; docs/accuracy.md "Same-weights regression gate",
# "Change against base"): the knob-off dump is written with --write-self-golden, the knob-on run is compared with --self-golden (--self-vs-ref-noise
# and, as a second run, the default budgets). --mode both: prefill is the M > 16 GEMM path and cannot move, the decode window is what changes.
#   S (card 0) short golden (10 prompts): off write, on svrn, on default budgets, off control (off vs its own dump: must be 0)
#   L (card 1) long golden (5 prompts, 16k): off write, on svrn, on default budgets
# usage: acc.sh [S|L|SL]; logs ~/wmr-logs/acc/<tag>.log, dumps ~/intarsiaml-tp2-wmr/dumps/<short|long>-main.safetensors (kept on tatooine)
set -uo pipefail
W=${1:-SL}
O=$HOME/wmr-logs/acc; mkdir -p "$O"
TREE=$HOME/intarsiaml-tp2-wmr; mkdir -p "$TREE/dumps"; # dumps go under /work/dumps in the container (the tree is the only writable mount)
 DEVSH=$TREE/deploy/tatooine/dev.sh; REL=intarsia-cargo-target-tp2-wmr
M=/models/Qwen3.6-35B-A3B-NVFP4; G=/models/golden
OPT1="INTARSIA_HIP_WMMA_ROWS=1 INTARSIA_HIP_WMMA_TABLE=fp8a8=all,bf16=all"
now() { date '+%F %T'; }
run() { # card tag env args...
  local g=$1 tag=$2 e=$3; shift 3
  echo "== $tag ($e) $(now)"
  INTARSIA_CONTAINER_NAME=wmr-job-$g INTARSIA_REPO=$TREE INTARSIA_TARGET_VOL=$REL HIP_VISIBLE_DEVICES=$g $DEVSH env $e /target/release/intarsia accuracy --backend hip --device 0 --model $M "$@" \
    < /dev/null > "$O/$tag.log" 2>&1
  echo "$tag exit=$? $(grep -aE 'SELF-STATS|verdict|PASS|FAIL|top-1|flips|mean' "$O/$tag.log" | tail -n 8 | tr '\n' '|' | cut -c1-900)"
}
chain() { # card kind golden decodepos maxseq
  local g=$1 k=$2 gold=$3 dp=$4 ms=$5
  local C="--golden $G/$gold --mode both --decode-positions $dp --max-seq $ms --mem-budget 29"
  local S="--self-golden /work/dumps/$k-main.safetensors --mode both --decode-positions $dp --max-seq $ms --mem-budget 29 --allow-same-code --allow-unstamped"
  run $g $k-off-write NOKNOB=1 $C --write-self-golden /work/dumps/$k-main.safetensors --allow-unstamped --allow-same-code
  run $g $k-on-svrn "$OPT1" $S --self-vs-ref-noise
  run $g $k-on-default "$OPT1" $S
  [[ $k == short ]] && run $g $k-off-control NOKNOB=1 $S --self-vs-ref-noise
  echo "$k chain done $(now)"
}
pids=()
if [[ $W == *S* ]]; then chain 0 short qwen36-35b-a3b-nvfp4-vllm-top20.safetensors 16 4096 > "$O/S.out" 2>&1 & pids+=($!); fi
if [[ $W == *L* ]]; then chain 1 long qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors 64 16384 > "$O/L.out" 2>&1 & pids+=($!); fi
for p in "${pids[@]}"; do wait "$p"; done
for f in S L; do [[ -s $O/$f.out ]] && { echo "######## $f"; cat "$O/$f.out"; }; done
echo "acc done $(now)"
