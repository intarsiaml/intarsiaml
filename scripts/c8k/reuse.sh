#!/usr/bin/env bash
# Lane c8k, spark side: the prefix-reuse gate (`accuracy --mode reuse`) at prefill chunk 8192 and (control, same build) 2048 on
#   L  the long golden (12k prompts; the 4k prompts are skipped at 8192; continue judged by the committed oracles when they match)
#   S  the make_split_golden.py derivative with the 12k prompts split at 9500 (more new text after the restored prefix)
#   F  the 4k golden at 2048, at 1024 (qualify's acc-4k reuse chunk cap) and at 8192 (nothing can cross a chunk: expected structural FAIL)
# Inside a fleetq slot (--quiet gpu,cpu). Usage: reuse.sh <outname> <tree> [arms, default "L8192 S8192 L2048 S2048 F2048 F1024 F8192"]
# Env WAIT_MIN: wait up to this long for the oracles listed below (the endor chain delivers them), then run with what exists and say so.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}"
P=$HOME/intarsia-decfuse; O=$P/out/$1; T=$2; ARMS=${3:-"L8192 S8192 L2048 S2048 F2048 F1024 F8192"}; mkdir -p $O
export PATH=/usr/local/cuda/bin:$PATH
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
GD=$P/pf-golden; B=$P/$T/target/acc/release/intarsia
unset INTARSIA_CUDA_FP8_GEMM_LT
OR=$GD/oracle; pre=qwen36-35b-a3b-nvfp4-continue
NEED="$pre-long_docs_12k-s12062 $pre-long_code_12k-s12062 $pre-long_docs_12k-s9500 $pre-long_code_12k-s9500 $pre-long_code_4k-s2000-g4k $pre-long_code_4k-s3072-g4k"
for i in $(seq 1 $(( ${WAIT_MIN:-0} * 2 + 1 ))); do
  miss=""; for n in $NEED; do [[ -s $OR/$n.continue-oracle.safetensors ]] || miss="$miss $n"; done
  [[ -z $miss ]] && break; [[ $i -le $(( ${WAIT_MIN:-0} * 2 )) ]] && sleep 30
done
echo "== oracles missing at start:${miss:- none}"
echo "== binary: $($B --version 2>&1 | head -1)"
run() { # run <name> <golden file> <chunk> [extra args]
  local name=$1 gold=$2 chunk=$3; shift 3
  echo "== reuse $name chunk=$chunk $(date +%T)"
  timeout 3000 $B accuracy --model $M35 --backend cuda --golden $gold --mode reuse --max-seq 16384 --mem-budget 40 --prefill-chunk $chunk --allow-dirty "$@" \
    < /dev/null > $O/reuse-$name.log 2>&1
  echo "exit $?"
  grep -aE "^\S+ +(continue|restore|aligned) |prefix reuse: |prefix-reuse gate|SKIPPED|committed continue oracle|Error" $O/reuse-$name.log | cut -c1-250
}
for a in $ARMS; do
  case $a in
    L*) run $a $GD/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors ${a#L};;
    S*) run $a $GD/reuse-splits/qwen36-35b-a3b-nvfp4-vllm-long-reuse-s9500.safetensors ${a#S} --reuse-oracle-dir $OR --reuse-prompt long_docs_12k --reuse-prompt long_code_12k;;
    F*) run $a $GD/qwen36-35b-a3b-nvfp4-vllm-4k-top20.safetensors ${a#F};;
  esac
done
echo "== done $(date +%T)"
