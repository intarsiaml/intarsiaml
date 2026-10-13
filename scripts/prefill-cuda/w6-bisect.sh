#!/usr/bin/env bash
# Window 6 (spark1, vLLM down, inside scripts/prefill-cuda-window.sh): which of the two new numerics is behind the 35B gate
# changes seen in W4/W5 (chat_multiturn / chat_system per-prompt KL, long prefix-reuse, aggregate prefill KL)? The d940701
# build (dev2) with one new kernel switched off at a time (INTARSIA_CUDA_DISABLE; the registry falls back to the HEAD kernel,
# so each run is also the gate of that reduced configuration). The disable list is a baseline config difference ->
# --allow-config-mismatch. s3 (both off) is the control: HEAD's kernels in the new binary, expected to reproduce the baseline.
set -u
P=$HOME/intarsia-perf
BIN=$P/dev2/target/acc/release/intarsia
M=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
GD=$HOME/intarsiaml-golden
O=$P/out/w6; mkdir -p $O
DEADLINE=$(( $(date +%s) + ${W6_BUDGET_MIN:-14} * 60 ))
left() { echo $(( (DEADLINE - $(date +%s)) / 60 )); }
need() { if (( $(left) < $1 )); then echo "SKIP $2: $(left) min left, needs $1"; return 1; fi; echo "== $2 ($(left) min left) $(date +%T)"; return 0; }
gb=$(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) / 1024 / 1024 )); echo "MemAvailable ${gb} GB"; (( gb >= 60 )) || { echo "ABORT: memory"; exit 1; }
PROV="--pre-change 7e65770 --revlist $P/revlist"
SHORT="--mode both --decode-positions 16 --max-seq 4096 --mem-budget 60"
LONG="--mode all --decode-positions 64 --max-seq 16384 --mem-budget 70"
G_SHORT=$GD/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors
G_LONG=$GD/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors
run() { # run <tag> <name> <golden> <flags> <disable> <min>
  local tag=$1 name=$2 golden=$3 flags=$4 dis=$5 mins=$6 A=$P/art/$2
  need $mins "$tag ($name, disabled: $dis)" || return 0
  INTARSIA_CUDA_DISABLE=$dis $BIN accuracy --model $M --backend cuda --golden $golden $flags --baseline $A.baseline.json --noise-envelope $A.envelope.json $PROV --allow-config-mismatch > $O/$tag.log 2>&1
  echo "exit $?"
  grep -E "^prefill floor|^decode \(teacher-forced\) floor" $O/$tag.log | sed -E 's/\| tail mass.*//; s/\(not gated for this class\) ok //' | cut -c1-120
  grep -E "(floor|baseline) verdict|prefix-reuse verdict|per-prompt gate: (FAIL|ok)" $O/$tag.log | cut -c1-230
}
run s3-both-off      35b-short $G_SHORT "$SHORT" gdn_delta_rule_bf16_lanes,matmul_fp8_gemm_f8 2
run s1-f8-off        35b-short $G_SHORT "$SHORT" matmul_fp8_gemm_f8 2
run s2-lanes-off     35b-short $G_SHORT "$SHORT" gdn_delta_rule_bf16_lanes 2
run l1-f8-off        35b-long  $G_LONG  "$LONG"  matmul_fp8_gemm_f8 4
run l2-lanes-off     35b-long  $G_LONG  "$LONG"  gdn_delta_rule_bf16_lanes 4
echo "== w6 done $(date +%T), $(left) min left"
