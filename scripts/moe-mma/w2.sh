#!/usr/bin/env bash
# W2 (spark1, vLLM down, inside scripts/cuda-decode/window.sh): acceptance gates of the grouped MoE accumulation fix
# (docs/results/2026-10-17-moe-mma-small-t.md), on the tree ~/intarsia-decfuse/${TREE:-moe-mma-dev}. Same commands and
# gate inputs as scripts/acc35b/w3.sh (baseline + envelope + provenance of the 35B gates, the HF fp32 goldens of the 8B).
#   tests    cargo test --release -p intarsia-backend-cuda (the CUDA kernel tests, includes the small-T sweep)
#   kernels  intarsia kernels test --backend cuda --model <35B>
#   short    35b-short against the vLLM golden (recorded old baseline + noise envelope, chunk 256)
#   long     35b-long --mode all against the recorded old baseline + envelope
#   8b       Qwen3-8B against the HF fp32 goldens (bf16 floor class), both sets
set -u
A=$HOME/intarsia-acc35b
T=$HOME/intarsia-decfuse/${TREE:-moe-mma-dev}
BIN=$T/target/acc/release/intarsia
M=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
M8=/home/nvidia/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
GD=$HOME/intarsiaml-golden
CAL=$A/src/docs/results/2026-10-13-phase1b-acceptance/gate-calibration/spark1/art
O=$T/out/${OUT:-w2}; mkdir -p $O
DEADLINE=$(( $(date +%s) + ${BUDGET_MIN:-16} * 60 ))
left() { echo $(( (DEADLINE - $(date +%s)) / 60 )); }
need() { if (( $(left) < $1 )); then echo "SKIP $2: $(left) min left, needs $1"; return 1; fi; echo "== $2 ($(left) min left) $(date +%T)"; return 0; }
gb=$(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) / 1024 / 1024 )); echo "MemAvailable ${gb} GB"; (( gb >= 60 )) || { echo "ABORT: memory"; exit 1; }
PROV="--pre-change 4963b43 --revlist $A/revlist"
SHORT="--mode both --decode-positions 16 --max-seq 4096 --mem-budget 60 --prefill-chunk 256"
LONG="--mode all --decode-positions 64 --max-seq 16384 --mem-budget 70 --prefill-chunk 256"
G_SHORT=$GD/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors
G_LONG=$GD/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors
show() { echo "exit $1"; grep -E "verdict|golden gate|failing per-prompt|SELF-STATS|self-golden =>|^prefill |^decode" $2 | head -12 | cut -c1-230; }
cd $T || exit 1
for STAGE in ${STAGES:-tests kernels short long 8b}; do
  case $STAGE in
    tests) need 3 "CUDA kernel tests" && {
        CARGO_BUILD_JOBS=4 CARGO_TARGET_DIR=target/acc INTARSIA_REQUIRE_GPU=1 timeout 6m cargo test --release -p intarsia-backend-cuda -- --test-threads=1 > $O/tests.log 2>&1
        echo "exit $?"; grep -E "^test result|FAILED|panicked|failed" $O/tests.log | head -40 | cut -c1-200; } ;;
    kernels) need 2 "kernels test" && {
        timeout 5m $BIN kernels test --backend cuda --model $M > $O/kernels.log 2>&1; echo "exit $?"
        grep -cE "ADMITTED" $O/kernels.log; grep -E "NOT ADMITTED|FAILED|moe_experts" $O/kernels.log | head -12 | cut -c1-200; tail -4 $O/kernels.log | cut -c1-200; } ;;
    short) need 2 "35b-short vs vLLM, baseline + envelope" && {
        $BIN accuracy --model $M --backend cuda --golden $G_SHORT $SHORT --baseline $CAL/35b-short.baseline.json \
          --noise-envelope $CAL/35b-short.envelope.json $PROV > $O/35b-short.log 2>&1; show $? $O/35b-short.log; } ;;
    long) need 5 "35b-long vs vLLM, baseline + envelope" && {
        $BIN accuracy --model $M --backend cuda --golden $G_LONG $LONG --baseline $CAL/35b-long.baseline.json \
          --noise-envelope $CAL/35b-long.envelope.json $PROV > $O/35b-long.log 2>&1
        show $? $O/35b-long.log; grep -E "prefix-reuse verdict" $O/35b-long.log | cut -c1-200; } ;;
    8b) for g in qwen3-8b-fp32 qwen3-8b-long-fp32; do
          need 2 "8B $g" && {
            $BIN accuracy --model $M8 --backend cuda --golden $GD/$g.safetensors --mode both --decode-positions 16 --max-seq 4096 \
              --mem-budget 40 --gate-class bf16 > $O/$g.log 2>&1; show $? $O/$g.log; }
        done ;;
  esac
done
echo "== w2 done $(date +%T), $(left) min left"
