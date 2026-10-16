#!/usr/bin/env bash
# Window 3 (spark1, vLLM down, inside scripts/acc35b/window.sh): the final gates of docs/results/2026-10-14-35b-accuracy.md
# on the stamped HEAD build.
#   short     35b-short against the vLLM golden: recorded old baseline + noise envelope (phase1b section 10), chunk 256
#   cpuref    35b-short against the CPU backend's dump (--self-golden, budgets of section 2.2): HEAD, and oldmoe / SIMT
#             with loose budgets (statistics only, to check the offline numbers of scripts/acc35b/cpuref_gate.py)
#   long      35b-long --mode all against the recorded old baseline + envelope, with a dump (CUDA vs HIP on chat_reuse_4k)
#   8b        Qwen3-8B against the HF fp32 goldens (bf16 floor class), both sets
set -u
P=$HOME/intarsia-acc35b
BIN=$P/src/target/acc/release/intarsia
M=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
M8=/home/nvidia/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
GD=$HOME/intarsiaml-golden
CAL=$P/src/docs/results/2026-10-13-phase1b-acceptance/gate-calibration/spark1/art
CPUREF=$P/cpu.dump.safetensors
O=$P/out/w3; mkdir -p $O
DEADLINE=$(( $(date +%s) + ${BUDGET_MIN:-16} * 60 ))
left() { echo $(( (DEADLINE - $(date +%s)) / 60 )); }
need() { if (( $(left) < $1 )); then echo "SKIP $2: $(left) min left, needs $1"; return 1; fi; echo "== $2 ($(left) min left) $(date +%T)"; return 0; }
gb=$(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) / 1024 / 1024 )); echo "MemAvailable ${gb} GB"; (( gb >= 60 )) || { echo "ABORT: memory"; exit 1; }
SIMT=attn_decode_bf16_splitkv,attn_prefill_bf16_splitkv,attn_prefill_bf16_flash
PROV="--pre-change 4963b43 --revlist $P/revlist"
SHORT="--mode both --decode-positions 16 --max-seq 4096 --mem-budget 60 --prefill-chunk 256"
LONG="--mode all --decode-positions 64 --max-seq 16384 --mem-budget 70 --prefill-chunk 256"
G_SHORT=$GD/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors
G_LONG=$GD/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors
# section 2.2 budgets (scripts/acc35b/cpuref_gate.py, envelope oldmoe / SIMT / HIP against the CPU dump, k = 2)
BUDGET=${CPUREF_BUDGET:?CPUREF_BUDGET must hold the --self-max-* flags}
LOOSE="--self-max-kl 1 --self-max-pos-kl 100 --self-max-flip-pct 100 --self-max-tail 1"
show() { echo "exit $1"; grep -E "verdict|golden gate|failing per-prompt|SELF-STATS|self-golden =>|^prefill |^decode" $2 | head -12 | cut -c1-230; }

for STAGE in ${STAGES:-short cpuref long 8b}; do
  case $STAGE in
    short) need 2 "35b-short vs vLLM, baseline + envelope" && {
        $BIN accuracy --model $M --backend cuda --golden $G_SHORT $SHORT --baseline $CAL/35b-short.baseline.json \
          --noise-envelope $CAL/35b-short.envelope.json $PROV > $O/35b-short.log 2>&1; show $? $O/35b-short.log; } ;;
    cpuref) need 2 "35b-short vs CPU dump, HEAD" && {
        $BIN accuracy --model $M --backend cuda --self-golden $CPUREF $SHORT $BUDGET --allow-config-mismatch --allow-same-code \
          --allow-unstamped > $O/cpuref-cuda.log 2>&1; show $? $O/cpuref-cuda.log; }
      for v in oldmoe:moe_experts_mma_bf16 simt:$SIMT; do
        need 2 "35b-short vs CPU dump, ${v%%:*} (statistics)" && {
          INTARSIA_CUDA_DISABLE=${v#*:} $BIN accuracy --model $M --backend cuda --self-golden $CPUREF $SHORT $LOOSE \
            --allow-config-mismatch --allow-same-code --allow-unstamped > $O/cpuref-${v%%:*}.log 2>&1; show $? $O/cpuref-${v%%:*}.log; }
      done ;;
    long) need 5 "35b-long vs vLLM, baseline + envelope, with dump" && {
        $BIN accuracy --model $M --backend cuda --golden $G_LONG $LONG --baseline $CAL/35b-long.baseline.json \
          --noise-envelope $CAL/35b-long.envelope.json $PROV --write-self-golden $O/35b-long.dump.safetensors > $O/35b-long.log 2>&1
        show $? $O/35b-long.log; grep -E "prefix-reuse verdict" $O/35b-long.log | cut -c1-200; } ;;
    8b) for g in qwen3-8b-fp32 qwen3-8b-long-fp32; do
          need 2 "8B $g" && {
            $BIN accuracy --model $M8 --backend cuda --golden $GD/$g.safetensors --mode both --decode-positions 16 --max-seq 4096 \
              --mem-budget 40 --gate-class bf16 > $O/$g.log 2>&1; show $? $O/$g.log; }
        done ;;
  esac
done
echo "== w3 done $(date +%T), $(left) min left"
