#!/usr/bin/env bash
# Window A (spark1, vLLM down, INSIDE scripts/cuda-decode/window.sh): accuracy of the two prefill-3 changes against the
# vLLM golden (sha b6b51e1b1699c66c), every arm from ONE binary (tree $TREE, default p3-dev), only the environment differs:
#   base  = INTARSIA_CUDA_FP8_GEMM_REG=0                      (the origin/main kernel selection)
#   lanes = INTARSIA_CUDA_FP8_GEMM_REG=0 INTARSIA_CUDA_GDN_LANES=1   (Part A alone)
#   reg   = default (matmul_fp8_gemm_reg on, lanes off)       (Part B alone; bitwise base by test)
#   both  = INTARSIA_CUDA_GDN_LANES=1
# lane 1: 35b-short KL21 at chunk 2048 / 1024 / 256 (base: the same-window range; lanes at 2048 and 256; reg at 2048),
#         greedy ids (256 tokens, ~1500-token prompt) base, lanes eager + graph, reg, both; prefix reuse lanes; 8B short
# lane 2: 35b-long KL21 (same arms), prefix reuse base / reg, 8B long
# Steps are skipped when the budget (W_BUDGET_MIN, default 13) is near. Output ~/intarsia-decfuse/out/p3A
set -u
P=$HOME/intarsia-decfuse
TREE=${TREE:-p3-dev}
BIN=$P/$TREE/target/acc/release/intarsia
O=$P/out/p3A; rm -rf $O; mkdir -p $O
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
M8=/home/nvidia/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
GD=$HOME/intarsiaml-golden
S=$GD/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors; L=$GD/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors
DEADLINE=$(( $(date +%s) + ${W_BUDGET_MIN:-13}*60 + 30 ))
left() { echo $(( DEADLINE - $(date +%s) )); }   # seconds
gb=$(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) / 1024 / 1024 )); echo "MemAvailable ${gb} GB"; (( gb >= 90 )) || { echo "ABORT: memory"; exit 1; }
echo "bin: $($BIN --version 2>&1 | head -1)"
nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv | tee $O/clocks.txt
sha256sum $S | cut -c1-16
echo "== start $(date +%T)"
BASE="INTARSIA_CUDA_FP8_GEMM_REG=0"; LANES="INTARSIA_CUDA_FP8_GEMM_REG=0 INTARSIA_CUDA_GDN_LANES=1"; REG="INTARSIA_X=0"; BOTH="INTARSIA_CUDA_GDN_LANES=1"

acc() { # acc <tag> <golden> <chunk> <env...>
  local tag=$1 g=$2 chunk=$3; shift 3; local t0=$(date +%s)
  (( $(left) > 70 )) || { echo "acc $tag SKIPPED: $(left) s left" >> $O/lanes.log; return; }
  env "$@" timeout $(( $(left) < 330 ? $(left) : 330 )) $BIN accuracy --model $M35 --backend cuda --golden $g --mode prefill --max-seq 16384 --mem-budget 40 --prefill-chunk $chunk --allow-dirty > $O/acc-$tag.log 2>&1
  echo "acc $tag exit $? $(( $(date +%s) - t0 )) s: $(grep -E '^prefill floor' $O/acc-$tag.log | head -1 | cut -c1-150)" >> $O/lanes.log
}
acc8() { # acc8 <tag> <golden file> <env...>
  local tag=$1 g=$2; shift 2; local t0=$(date +%s)
  (( $(left) > 90 )) || { echo "acc8 $tag SKIPPED: $(left) s left" >> $O/lanes.log; return; }
  env "$@" timeout $(( $(left) < 240 ? $(left) : 240 )) $BIN accuracy --model $M8 --backend cuda --golden $GD/$g --mode both --decode-positions 16 --max-seq 4096 --mem-budget 40 --gate-class bf16 --allow-dirty > $O/acc8-$tag.log 2>&1
  echo "acc8 $tag exit $? $(( $(date +%s) - t0 )) s: $(grep -E 'verdict|golden gate|^prefill |^decode|PASS|FAIL' $O/acc8-$tag.log | head -4 | cut -c1-160 | tr '\n' '|')" >> $O/lanes.log
}
reuse() { # reuse <tag> <env...>
  local tag=$1; shift; local t0=$(date +%s)
  (( $(left) > 120 )) || { echo "reuse $tag SKIPPED: $(left) s left" >> $O/lanes.log; return; }
  env "$@" timeout $(( $(left) < 240 ? $(left) : 240 )) $BIN accuracy --model $M35 --backend cuda --golden $L --mode reuse --max-seq 16384 --mem-budget 40 --prefill-chunk 2048 --allow-dirty > $O/reuse-$tag.log 2>&1
  echo "reuse $tag exit $? $(( $(date +%s) - t0 )) s: $(grep -aiE 'reuse' $O/reuse-$tag.log | grep -aiE 'pass|fail|skip|identical|diverg|ok' | head -3 | cut -c1-200 | tr '\n' '|')" >> $O/lanes.log
}
gen() { # gen <tag> <extra args or -> <env...>; ~1500-token prompt
  local tag=$1 extra=$2; shift 2; [[ $extra == - ]] && extra=""
  (( $(left) > 60 )) || { echo "gen $tag SKIPPED: $(left) s left" >> $O/lanes.log; return; }
  env "$@" timeout 150 $BIN generate -m $M35 --backend cuda --mem-budget 29 --max-seq 4096 --chat --no-think --profile-every 0 --ignore-eos --show-ids \
    -p "$(head -c 5000 $P/$TREE/tools/prompts/corpus_docs.txt)" --max-tokens 256 $extra > $O/gen-$tag.out 2> $O/gen-$tag.err
  grep -a '^ids:' $O/gen-$tag.err > $O/gen-$tag.ids
  echo "gen $tag: $(grep -aE 'finish=|steps=' $O/gen-$tag.err | head -2 | cut -c1-160 | tr '\n' '|')" >> $O/lanes.log
}
(
  acc base-short-c2048 $S 2048 $BASE
  acc lanes-short-c2048 $S 2048 $LANES
  gen base-graph - $BASE
  gen lanes-graph - $LANES
  gen lanes-eager --no-graphs $LANES
  acc reg-short-c2048 $S 2048 $REG
  acc base-short-c1024 $S 1024 $BASE
  acc base-short-c256 $S 256 $BASE
  acc lanes-short-c256 $S 256 $LANES
  gen reg-graph - $REG
  gen both-graph - $BOTH
  reuse lanes $LANES
  acc8 short qwen3-8b-fp32.safetensors $BOTH
  echo "== lane 1 done, $(left) s left $(date +%T)" >> $O/lanes.log
) &
(
  acc base-long-c2048 $L 2048 $BASE
  acc lanes-long-c2048 $L 2048 $LANES
  reuse base $BASE
  acc reg-long-c2048 $L 2048 $REG
  acc base-long-c1024 $L 1024 $BASE
  acc base-long-c256 $L 256 $BASE
  reuse reg $REG
  acc8 long qwen3-8b-long-fp32.safetensors $BOTH
  echo "== lane 2 done, $(left) s left $(date +%T)" >> $O/lanes.log
) &
wait
echo "== done, $(left) s left $(date +%T)"
cat $O/lanes.log
for a in base-graph lanes-graph lanes-eager reg-graph both-graph; do printf '%s: %s ids\n' $a "$(tr -cd ',' < $O/gen-$a.ids 2>/dev/null | wc -c)"; done
python3 - $O <<'PY'
import sys, os
o = sys.argv[1]
def ids(t):
    try:
        s = open(f"{o}/gen-{t}.ids").read().split(":", 1)[1]
        return [int(v) for v in s.replace("[", " ").replace("]", " ").replace(",", " ").split()]
    except Exception as e:
        return None
b = ids("base-graph")
for t in ["lanes-graph", "lanes-eager", "reg-graph", "both-graph"]:
    x = ids(t)
    if b is None or x is None: print(f"{t}: missing"); continue
    d = next((i for i, (p, q) in enumerate(zip(b, x)) if p != q), None)
    print(f"{t} vs base-graph: " + ("identical" if d is None and len(b) == len(x) else f"first difference at generated token {d} of {len(b)}"))
x, y = ids("lanes-graph"), ids("lanes-eager")
if x and y: print("lanes graph vs eager: " + ("identical" if x == y else "differ"))
PY
