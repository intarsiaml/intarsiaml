#!/usr/bin/env bash
# Window 1 (spark1, vLLM down, inside scripts/acc35b/window.sh): HEAD CUDA dumps for the cross-backend comparison of
# docs/results/2026-10-14-35b-accuracy.md. Every run writes a baseline JSON (all aggregates and per-prompt stats against
# the vLLM golden), a self-golden dump (top-64 log-probs, prefill and decode rows) and a per-position TSV.
#   cuda        HEAD as shipped (gate flags of scripts/phase1b-acceptance.sh, chunk 256), also gated against the
#               recorded old baseline + noise envelope (the "before" verdict)
#   oldmoe      INTARSIA_CUDA_DISABLE=moe_experts_mma_bf16 (the warp-per-row MoE kernel for every T, i.e. the old numerics)
#   simt        SIMT attention instead of flash / split-KV
#   long        35b-long decode-only (64 positions x 5 prompts), HEAD and oldmoe
#   sweep       kernel diagnostic: MoE off-reference fraction over small T (quantmoe moe_experts_mma_small_t_sweep)
set -u
P=$HOME/intarsia-acc35b
BIN=$P/src/target/acc/release/intarsia
M=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
GD=$HOME/intarsiaml-golden
CAL=$P/src/docs/results/2026-10-13-phase1b-acceptance/gate-calibration/spark1/art
O=$P/out/${OUT:-w1}; mkdir -p $O
DEADLINE=$(( $(date +%s) + ${BUDGET_MIN:-16} * 60 ))
left() { echo $(( (DEADLINE - $(date +%s)) / 60 )); }
need() { if (( $(left) < $1 )); then echo "SKIP $2: $(left) min left, needs $1"; return 1; fi; echo "== $2 ($(left) min left) $(date +%T)"; return 0; }
gb=$(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) / 1024 / 1024 )); echo "MemAvailable ${gb} GB"; (( gb >= 60 )) || { echo "ABORT: memory"; exit 1; }
SIMT=attn_decode_bf16_splitkv,attn_prefill_bf16_splitkv,attn_prefill_bf16_flash
SHORT="--mode both --decode-positions 16 --max-seq 4096 --mem-budget 60 --prefill-chunk 256"
LONGD="--mode decode --decode-positions 64 --max-seq 16384 --mem-budget 70 --prefill-chunk 256"
G_SHORT=$GD/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors
G_LONG=$GD/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors

run() { # run <tag> <golden> <flags> <env> <extra>
  local tag=$1 golden=$2 flags=$3 envs=$4 extra=$5
  need ${MIN_PER_RUN:-2} "run $tag" || return 0
  rm -f $O/$tag.pos.tsv
  env $envs INTARSIA_ACC_POS_DUMP=$O/$tag.pos.tsv $BIN accuracy --model $M --backend cuda --golden $golden $flags $extra \
    --write-baseline $O/$tag.baseline.json --write-self-golden $O/$tag.dump.safetensors --allow-dirty > $O/$tag.log 2>&1
  echo "exit $?"; grep -E "verdict|golden gate|failing per-prompt|^prefill |^decode" $O/$tag.log | head -8 | cut -c1-230
}

for STAGE in ${STAGES:-sweep cuda oldmoe simt long}; do
  case $STAGE in
    sweep)
      need 2 "MoE small-T sweep" && {
        (cd $P/src && CARGO_TARGET_DIR=target/acc cargo test --release -p intarsia-backend-cuda --test quantmoe moe_experts_mma -- --include-ignored --nocapture --test-threads=1 > $O/sweep.log 2>&1)
        echo "exit $?"; grep -E "SWEEP|elements off|test result|panicked" $O/sweep.log | cut -c1-170
      } ;;
    cuda)   run cuda   $G_SHORT "$SHORT" "" "--baseline $CAL/35b-short.baseline.json --noise-envelope $CAL/35b-short.envelope.json --pre-change 4963b43 --revlist $P/revlist" ;;
    oldmoe) run oldmoe $G_SHORT "$SHORT" "INTARSIA_CUDA_DISABLE=moe_experts_mma_bf16" "" ;;
    simt)   run simt   $G_SHORT "$SHORT" "INTARSIA_CUDA_DISABLE=$SIMT" "" ;;
    long)   MIN_PER_RUN=3 run long-cuda $G_LONG "$LONGD" "" ""
            MIN_PER_RUN=3 run long-oldmoe $G_LONG "$LONGD" "INTARSIA_CUDA_DISABLE=moe_experts_mma_bf16" "" ;;
  esac
done
echo "== w1 done $(date +%T), $(left) min left"
