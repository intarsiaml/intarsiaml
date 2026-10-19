#!/usr/bin/env bash
# Accuracy of the round-5 arms against the vLLM goldens, ON spark1 inside ONE fleetq slot (vLLM up; one 35B process at a
# time, --mem-budget 40; the kernels are deterministic, so the numbers do not depend on what else runs):
#   fleetq run --host spark1 --mem 45 --label "..." -- scripts/cuda-prefill4/remote.sh r5-f8 scripts/cuda-round5/acc.sh
# Arms (copy of scripts/cuda-prefill4/acc.sh):
#   base = tree r5-base (origin/main, built as is): the same-window base range (chunk 2048 / 1024 / 256)
#   dev  = tree r5-f8, default bf16 cache: must equal base to every digit (the bf16 kernels' SASS is unchanged)
#   f8   = tree r5-f8, --kv-dtype fp8_e4m3 (the opt-in fp8 KV cache: the f8kv attention kernels)
# STEPS (space separated, default all) picks steps by name. Output ~/intarsia-decfuse/out/${OUTNAME:-r5A}.
set -u
P=$HOME/intarsia-decfuse
O=$P/out/${OUTNAME:-r5A}; mkdir -p $O
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
M8=/home/nvidia/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
GD=$HOME/intarsiaml-golden
S=$GD/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors; L=$GD/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors
gb=$(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) / 1024 / 1024 )); echo "MemAvailable ${gb} GB"; (( gb >= 44 )) || { echo "ABORT: memory"; exit 1; }
for t in r5-base r5-f8; do echo "bin $t: $($P/$t/target/acc/release/intarsia --version 2>&1 | head -1)"; done
nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu --format=csv | tee $O/clocks.txt
echo "== start $(date +%T)"
want() { [[ -z "${STEPS:-}" || " $STEPS " == *" $1 "* ]]; }
bin() { echo $P/$1/target/acc/release/intarsia; }
F8ARG="--kv-dtype fp8_e4m3"
acc() { # acc <tag> <tree> <golden> <chunk> <extra args or -> ; KL21 of the prefill pass
  local tag=$1 tree=$2 g=$3 chunk=$4 extra=$5; [[ $extra == - ]] && extra=""; want acc-$tag || return 0; local t0=$(date +%s)
  timeout 900 $(bin $tree) accuracy --model $M35 --backend cuda --golden $g --mode prefill --max-seq 16384 --mem-budget 40 --prefill-chunk $chunk --allow-dirty $extra > $O/acc-$tag.log 2>&1
  echo "acc $tag exit $? $(( $(date +%s) - t0 )) s: $(grep -E '^prefill floor' $O/acc-$tag.log | head -1 | cut -c1-150)" | tee -a $O/steps.log
}
acc8() { # acc8 <tag> <tree> <golden file>
  local tag=$1 tree=$2 g=$3; want acc8-$tag || return 0; local t0=$(date +%s)
  timeout 600 $(bin $tree) accuracy --model $M8 --backend cuda --golden $GD/$g --mode both --decode-positions 16 --max-seq 4096 --mem-budget 40 --gate-class bf16 --allow-dirty > $O/acc8-$tag.log 2>&1
  echo "acc8 $tag exit $? $(( $(date +%s) - t0 )) s: $(grep -E 'verdict|golden gate|PASS|FAIL' $O/acc8-$tag.log | head -4 | cut -c1-160 | tr '\n' '|')" | tee -a $O/steps.log
}
reuse() { # reuse <tag> <tree> <extra args or ->
  local tag=$1 tree=$2 extra=$3; [[ $extra == - ]] && extra=""; want reuse-$tag || return 0; local t0=$(date +%s)
  timeout 900 $(bin $tree) accuracy --model $M35 --backend cuda --golden $L --mode reuse --max-seq 16384 --mem-budget 40 --prefill-chunk 2048 --allow-dirty $extra > $O/reuse-$tag.log 2>&1
  echo "reuse $tag exit $? $(( $(date +%s) - t0 )) s: $(grep -aE '^chat_reuse_4k|prefix-reuse gate' $O/reuse-$tag.log | cut -c1-90 | tr -s ' ' | tr '\n' '|')" | tee -a $O/steps.log
}
gen() { # gen <tag> <tree> <extra args or ->; ~1500-token prompt, 256 greedy tokens
  local tag=$1 tree=$2 extra=$3; [[ $extra == - ]] && extra=""; want gen-$tag || return 0
  timeout 300 $(bin $tree) generate -m $M35 --backend cuda --mem-budget 29 --max-seq 4096 --chat --no-think --profile-every 0 --ignore-eos --show-ids \
    -p "$(head -c 5000 $P/r5-f8/tools/prompts/corpus_docs.txt)" --max-tokens 256 $extra > $O/gen-$tag.out 2> $O/gen-$tag.err
  grep -a '^ids:' $O/gen-$tag.err > $O/gen-$tag.ids
  echo "gen $tag: $(grep -aE 'finish=|steps=|prompt' $O/gen-$tag.err | head -2 | cut -c1-160 | tr '\n' '|')" | tee -a $O/steps.log
}
gen base-graph r5-base -
gen dev-graph r5-f8 -
gen dev-eager r5-f8 --no-graphs
gen f8-graph r5-f8 "$F8ARG"
gen f8-eager r5-f8 "$F8ARG --no-graphs"
acc base-short-c2048 r5-base $S 2048 -
acc dev-short-c2048 r5-f8 $S 2048 -
acc f8-short-c2048 r5-f8 $S 2048 "$F8ARG"
acc base-short-c1024 r5-base $S 1024 -
acc base-short-c256 r5-base $S 256 -
acc f8-short-c256 r5-f8 $S 256 "$F8ARG"
acc base-long-c2048 r5-base $L 2048 -
acc dev-long-c2048 r5-f8 $L 2048 -
acc f8-long-c2048 r5-f8 $L 2048 "$F8ARG"
acc base-long-c1024 r5-base $L 1024 -
acc base-long-c256 r5-base $L 256 -
acc f8-long-c256 r5-f8 $L 256 "$F8ARG"
reuse base r5-base -
reuse dev r5-f8 -
reuse f8 r5-f8 "$F8ARG"
acc8 short r5-f8 qwen3-8b-fp32.safetensors
acc8 long r5-f8 qwen3-8b-long-fp32.safetensors
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
b = ids("base-graph")
for t in ["dev-graph", "dev-eager", "f8-graph", "f8-eager"]:
    x = ids(t)
    if b is None or x is None: print(f"{t}: missing"); continue
    d = next((i for i, (p, q) in enumerate(zip(b, x)) if p != q), None)
    print(f"{t} vs base-graph: " + ("identical" if d is None and len(b) == len(x) else f"first difference at generated token {d} of {len(b)}"))
f, e = ids("f8-graph"), ids("f8-eager")
if f and e:
    d = next((i for i, (p, q) in enumerate(zip(f, e)) if p != q), None)
    print("f8-eager vs f8-graph: " + ("identical" if d is None else f"first difference at token {d}"))
PY
