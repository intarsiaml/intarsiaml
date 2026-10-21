#!/usr/bin/env bash
# Accuracy gates of round 7 against the vLLM goldens, ON spark1 inside ONE fleetq slot (vLLM up; one model process at a
# time, --mem-budget 40; the kernels are deterministic, so the numbers do not depend on what else runs):
#   fleetq run --host spark1 --mem 45 --label "..." -- scripts/cuda-round6/sync-scripts.sh scripts/cuda-round7/acc.sh
# (sync-scripts copies scripts/ to ~/intarsia-decfuse/r6-scripts/scripts; this file then runs from there.)
# Arms: base = tree r7-base (origin/main as is); dev = tree r7-dev, defaults; lt = dev with INTARSIA_CUDA_FP8_GEMM_LT=1
# (opt-in); lt4k = lt with --prefill-chunk 4096. Every KL run also dumps its own top-20 (--write-self-golden) so the
# offline comparison with vLLM's own re-run (scripts/acc35b/xcompare.py) uses one method for every pair.
# STEPS (space separated, default all) picks steps by name. Output ~/intarsia-decfuse/out/${OUTNAME:-r7A}.
set -u
P=$HOME/intarsia-decfuse
O=$P/out/${OUTNAME:-r7A}; mkdir -p $O
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
M8=/home/nvidia/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
GD=$HOME/intarsiaml-golden
S=$GD/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors; L=$GD/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors
CORPUS=$P/r7-dev/tools/prompts/corpus_docs.txt
LT=INTARSIA_CUDA_FP8_GEMM_LT=1
gb=$(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) / 1024 / 1024 )); echo "MemAvailable ${gb} GB"; (( gb >= 44 )) || { echo "ABORT: memory"; exit 1; }
for t in r7-base r7-dev; do echo "bin $t: $($P/$t/target/acc/release/intarsia --version 2>&1 | head -1); ik_matmul_fp8_gemm_lt symbols: $(grep -c ik_matmul_fp8_gemm_lt $P/$t/target/acc/release/intarsia)"; done
nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu --format=csv | tee $O/clocks.txt
echo "== start $(date +%T)"
want() { [[ -z "${STEPS:-}" || " $STEPS " == *" $1 "* ]]; }
bin() { echo $P/$1/target/acc/release/intarsia; }
acc() { # acc <tag> <tree> <env or -> <golden> <chunk> ; KL21 of the prefill pass + own top-20 dump
  local tag=$1 tree=$2 e=$3 g=$4 chunk=$5; [[ $e == - ]] && e=INTARSIA_X=0; want acc-$tag || return 0; local t0=$(date +%s)
  env $e timeout 900 $(bin $tree) accuracy --model $M35 --backend cuda --golden $g --mode prefill --max-seq 16384 --mem-budget 40 \
    --prefill-chunk $chunk --allow-dirty --write-self-golden $O/self-$tag.safetensors --self-topk 20 > $O/acc-$tag.log 2>&1
  echo "acc $tag exit $? $(( $(date +%s) - t0 )) s: $(grep -E '^prefill floor' $O/acc-$tag.log | head -1 | cut -c1-150)" | tee -a $O/steps.log
}
acc8() { # acc8 <tag> <tree> <env or -> <golden file>
  local tag=$1 tree=$2 e=$3 g=$4; [[ $e == - ]] && e=INTARSIA_X=0; want acc8-$tag || return 0; local t0=$(date +%s)
  env $e timeout 600 $(bin $tree) accuracy --model $M8 --backend cuda --golden $GD/$g --mode both --decode-positions 16 --max-seq 4096 --mem-budget 40 --gate-class bf16 --allow-dirty > $O/acc8-$tag.log 2>&1
  echo "acc8 $tag exit $? $(( $(date +%s) - t0 )) s: $(grep -E 'verdict|golden gate|PASS|FAIL' $O/acc8-$tag.log | head -4 | cut -c1-160 | tr '\n' '|')" | tee -a $O/steps.log
}
reuse() { # reuse <tag> <tree> <env or -> <chunk>
  local tag=$1 tree=$2 e=$3 chunk=$4; [[ $e == - ]] && e=INTARSIA_X=0; want reuse-$tag || return 0; local t0=$(date +%s)
  env $e timeout 900 $(bin $tree) accuracy --model $M35 --backend cuda --golden $L --mode reuse --max-seq 16384 --mem-budget 40 --prefill-chunk $chunk --allow-dirty > $O/reuse-$tag.log 2>&1
  echo "reuse $tag exit $? $(( $(date +%s) - t0 )) s: $(grep -aE '^chat_reuse_4k|prefix-reuse gate' $O/reuse-$tag.log | cut -c1-90 | tr -s ' ' | tr '\n' '|')" | tee -a $O/steps.log
}
gen() { # gen <tag> <tree> <env or -> <prompt chars> <tokens> <extra args or ->; chat template, greedy
  local tag=$1 tree=$2 e=$3 chars=$4 n=$5 extra=$6; [[ $extra == - ]] && extra=""; [[ $e == - ]] && e=INTARSIA_X=0; want gen-$tag || return 0
  env $e timeout 600 $(bin $tree) generate -m $M35 --backend cuda --mem-budget 29 --max-seq 20480 --chat --no-think --profile-every 0 --ignore-eos --show-ids \
    -p "$(head -c $chars $CORPUS)" --max-tokens $n $extra > $O/gen-$tag.out 2> $O/gen-$tag.err
  grep -a '^ids:' $O/gen-$tag.err > $O/gen-$tag.ids
  echo "gen $tag: $(grep -aE 'finish=' $O/gen-$tag.err | head -1 | cut -c1-160)" | tee -a $O/steps.log
}
plan8() { # plan8 <tag> <tree>
  local tag=$1 tree=$2; want plan8-$tag || return 0
  timeout 300 $(bin $tree) plan show --backend cuda --model $M8 --max-seq 4096 --mem-budget 40 --json > $O/plan8-$tag.json 2> $O/plan8-$tag.err
  echo "plan8 $tag exit $?: $(wc -c < $O/plan8-$tag.json) bytes" | tee -a $O/steps.log
}
# in priority order (the window's timeout cuts the tail). base KL digits are round 6's (main unchanged since: docs only);
# dev must reproduce them, and its ids must equal base's.
gen base-graph r7-base - 5000 256 -
gen dev-graph r7-dev - 5000 256 -
gen lt-graph r7-dev $LT 5000 256 -
acc dev-short-c2048 r7-dev - $S 2048
acc lt-short-c2048 r7-dev $LT $S 2048
acc dev-long-c2048 r7-dev - $L 2048
acc lt-long-c2048 r7-dev $LT $L 2048
reuse lt r7-dev $LT 2048
gen base-16k r7-base - 54000 64 -
gen lt-16k r7-dev $LT 54000 64 -
gen lt-eager r7-dev $LT 5000 256 --no-graphs
acc lt4k-long-c4096 r7-dev $LT $L 4096
reuse lt4k r7-dev $LT 4096
acc lt-short-c256 r7-dev $LT $S 256
acc lt-long-c256 r7-dev $LT $L 256
plan8 base r7-base
plan8 dev r7-dev
echo "== done $(date +%T)"
python3 - $O <<'PY'
import sys
o = sys.argv[1]
def ids(t):
    try:
        s = open(f"{o}/gen-{t}.ids").read().split(":", 1)[1]
        return [int(v) for v in s.replace("[", " ").replace("]", " ").replace(",", " ").split()]
    except Exception:
        return None
for a, b in [("base-graph", "dev-graph"), ("base-graph", "lt-graph"), ("lt-graph", "lt-eager"), ("base-16k", "lt-16k")]:
    x, y = ids(a), ids(b)
    if x is None or y is None or not x: print(f"{b} vs {a}: missing"); continue
    d = next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), None)
    print(f"{b} vs {a}: " + ("identical" if d is None and len(x) == len(y) else f"first difference at generated token {d} of {len(x)}") + f" ({len(x)} tokens)")
try:
    pa = open(f"{o}/plan8-base.json").read(); pb = open(f"{o}/plan8-dev.json").read()
    print("8B plan base vs dev: " + ("identical" if pa == pb else "DIFFER"))
except Exception as e:
    print(f"8B plan: {e}")
PY
