#!/usr/bin/env bash
# Lane c8k, spark side: the accuracy gates that do not need the new oracles, at prefill chunk 8192 and (control, same build) 2048:
# the three KL goldens (short, long, 4k) and acc-short against main's CUDA baseline. Inside a fleetq slot (--quiet gpu,cpu).
# Usage: gatesA.sh <outname> <tree> [chunks, default "8192 2048"]
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}"
P=$HOME/intarsia-decfuse; O=$P/out/$1; T=$2; CH=${3:-"8192 2048"}; mkdir -p $O
export PATH=/usr/local/cuda/bin:$PATH
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
GD=$P/pf-golden-c8k; B=$P/$T/target/acc/release/intarsia
PRE=${PRE_CHANGE:-c37a8d3}   # the commit the baseline was recorded from (golden/baselines/...cuda.baseline.json code.head)
unset INTARSIA_CUDA_FP8_GEMM_LT
echo "== binary: $($B --version 2>&1 | head -1); stamp: $(strings $B 2>/dev/null | grep -m1 'head=[0-9a-f]* code-tree=' )"
for C in $CH; do
  for g in qwen36-35b-a3b-nvfp4-vllm-top20 qwen36-35b-a3b-nvfp4-vllm-long-top20 qwen36-35b-a3b-nvfp4-vllm-4k-top20; do
    echo "== KL golden $g chunk $C $(date +%T)"
    timeout 1800 $B accuracy --model $M35 --backend cuda --golden $GD/$g.safetensors --mode prefill --max-seq 16384 --mem-budget 40 --prefill-chunk $C \
      --allow-dirty < /dev/null > $O/kl-$g-c$C.log 2>&1
    echo "exit $?"; grep -aE "^all |KL\(21-way\)|floor verdict|golden gate|^prefill chunk" $O/kl-$g-c$C.log | tail -5 | cut -c1-220
  done
  # 4k golden, both modes (qualify's acc-4k accuracy part: no absolute top-1 floor; a CUDA 4k baseline does not exist)
  echo "== acc-4k accuracy (both modes, --min-top1 0) chunk $C $(date +%T)"
  timeout 1800 $B accuracy --model $M35 --backend cuda --golden $GD/qwen36-35b-a3b-nvfp4-vllm-4k-top20.safetensors --mode both --min-top1 0 --decode-positions 16 \
    --max-seq 16384 --mem-budget 40 --prefill-chunk $C --allow-dirty < /dev/null > $O/acc4k-c$C.log 2>&1
  echo "exit $?"; grep -aE "^all |floor verdict|golden gate|baseline verdict|Error" $O/acc4k-c$C.log | tail -6 | cut -c1-220
  # acc-short against main's recorded baseline. 8192: the chunk differs from the baseline's 2048, so --allow-config-mismatch (identity
  # check only, no threshold changes); every short-golden prompt is under 2048 tokens, so the run is one chunk either way.
  X=""; [[ $C != 2048 ]] && X="--prefill-chunk $C --allow-config-mismatch"
  echo "== acc-short vs baseline, chunk $C $(date +%T)"
  timeout 1800 $B accuracy --model $M35 --backend cuda --device 0 --mem-budget 40 --golden $GD/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors --mode both \
    --decode-positions 16 --max-seq 4096 --baseline $GD/baselines/qwen36-35b-a3b-nvfp4-vllm-top20.cuda.baseline.json --pre-change $PRE $X \
    < /dev/null > $O/accshort-c$C.log 2>&1
  echo "exit $?"; grep -aE "^  mean KL|per-prompt gate:|verdict|Error|WARNING: (config|baseline)|differs|baseline [0-9]" $O/accshort-c$C.log | cut -c1-220
done
echo "== done $(date +%T)"
