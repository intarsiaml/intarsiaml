#!/usr/bin/env bash
# Accuracy gates of round 6 against the vLLM goldens, ON spark1 inside ONE fleetq slot (vLLM up; one model process at a
# time, --mem-budget 40; the kernels are deterministic, so the numbers do not depend on what else runs):
#   fleetq run --host spark1 --mem 45 --label "..." -- scripts/cuda-round6/sync-scripts.sh scripts/cuda-round6/acc.sh
# Arms: base = tree r6-base (origin/main 0df54872 as is); dev = tree r6-dev (round-6 head, defaults); c4096 = dev with
# --prefill-chunk 4096 (opt-in, not bitwise: a user decision). Copy of scripts/cuda-round5/acc.sh plus: greedy ids on a
# ~900-token prompt (the w4a16 threshold zone) and a ~16k-token prompt (the 128-row flash blocks at long KV), the 8B plans.
# STEPS (space separated, default all) picks steps by name. Output ~/intarsia-decfuse/out/${OUTNAME:-r6A}.
set -u
P=$HOME/intarsia-decfuse
O=$P/out/${OUTNAME:-r6A}; mkdir -p $O
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
M8=/home/nvidia/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
GD=$HOME/intarsiaml-golden
S=$GD/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors; L=$GD/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors
CORPUS=$P/r6-dev/tools/prompts/corpus_docs.txt
gb=$(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) / 1024 / 1024 )); echo "MemAvailable ${gb} GB"; (( gb >= 44 )) || { echo "ABORT: memory"; exit 1; }
for t in r6-base r6-dev; do echo "bin $t: $($P/$t/target/acc/release/intarsia --version 2>&1 | head -1)"; done
nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu --format=csv | tee $O/clocks.txt
echo "== start $(date +%T)"
want() { [[ -z "${STEPS:-}" || " $STEPS " == *" $1 "* ]]; }
bin() { echo $P/$1/target/acc/release/intarsia; }
acc() { # acc <tag> <tree> <golden> <chunk> ; KL21 of the prefill pass
  local tag=$1 tree=$2 g=$3 chunk=$4; want acc-$tag || return 0; local t0=$(date +%s)
  timeout 900 $(bin $tree) accuracy --model $M35 --backend cuda --golden $g --mode prefill --max-seq 16384 --mem-budget 40 --prefill-chunk $chunk --allow-dirty > $O/acc-$tag.log 2>&1
  echo "acc $tag exit $? $(( $(date +%s) - t0 )) s: $(grep -E '^prefill floor' $O/acc-$tag.log | head -1 | cut -c1-150)" | tee -a $O/steps.log
}
acc8() { # acc8 <tag> <tree> <golden file>
  local tag=$1 tree=$2 g=$3; want acc8-$tag || return 0; local t0=$(date +%s)
  timeout 600 $(bin $tree) accuracy --model $M8 --backend cuda --golden $GD/$g --mode both --decode-positions 16 --max-seq 4096 --mem-budget 40 --gate-class bf16 --allow-dirty > $O/acc8-$tag.log 2>&1
  echo "acc8 $tag exit $? $(( $(date +%s) - t0 )) s: $(grep -E 'verdict|golden gate|PASS|FAIL' $O/acc8-$tag.log | head -4 | cut -c1-160 | tr '\n' '|')" | tee -a $O/steps.log
}
reuse() { # reuse <tag> <tree> <chunk>
  local tag=$1 tree=$2 chunk=$3; want reuse-$tag || return 0; local t0=$(date +%s)
  timeout 900 $(bin $tree) accuracy --model $M35 --backend cuda --golden $L --mode reuse --max-seq 16384 --mem-budget 40 --prefill-chunk $chunk --allow-dirty > $O/reuse-$tag.log 2>&1
  echo "reuse $tag exit $? $(( $(date +%s) - t0 )) s: $(grep -aE '^chat_reuse_4k|prefix-reuse gate' $O/reuse-$tag.log | cut -c1-90 | tr -s ' ' | tr '\n' '|')" | tee -a $O/steps.log
}
gen() { # gen <tag> <tree> <prompt chars> <tokens> <extra args or ->; chat template, greedy
  local tag=$1 tree=$2 chars=$3 n=$4 extra=$5; [[ $extra == - ]] && extra=""; want gen-$tag || return 0
  timeout 600 $(bin $tree) generate -m $M35 --backend cuda --mem-budget 29 --max-seq 20480 --chat --no-think --profile-every 0 --ignore-eos --show-ids \
    -p "$(head -c $chars $CORPUS)" --max-tokens $n $extra > $O/gen-$tag.out 2> $O/gen-$tag.err
  grep -a '^ids:' $O/gen-$tag.err > $O/gen-$tag.ids
  echo "gen $tag: $(grep -aE 'finish=' $O/gen-$tag.err | head -1 | cut -c1-160)" | tee -a $O/steps.log
}
plan8() { # plan8 <tag> <tree>
  local tag=$1 tree=$2; want plan8-$tag || return 0
  timeout 300 $(bin $tree) plan show --backend cuda --model $M8 --max-seq 4096 --mem-budget 40 --json > $O/plan8-$tag.json 2> $O/plan8-$tag.err
  echo "plan8 $tag exit $?: $(wc -c < $O/plan8-$tag.json) bytes" | tee -a $O/steps.log
}
gen base-graph r6-base 5000 256 -
gen dev-graph r6-dev 5000 256 -
gen dev-eager r6-dev 5000 256 --no-graphs
gen base-900 r6-base 3000 128 -
gen dev-900 r6-dev 3000 128 -
gen base-16k r6-base 54000 64 -
gen dev-16k r6-dev 54000 64 -
gen c4096-16k r6-dev 54000 64 "--prefill-chunk 4096"
acc base-short-c2048 r6-base $S 2048
acc dev-short-c2048 r6-dev $S 2048
acc base-long-c2048 r6-base $L 2048
acc dev-long-c2048 r6-dev $L 2048
acc dev-short-c4096 r6-dev $S 4096
acc dev-long-c4096 r6-dev $L 4096
reuse base r6-base 2048
reuse dev r6-dev 2048
reuse c4096 r6-dev 4096
acc8 short-base r6-base qwen3-8b-fp32.safetensors
acc8 short-dev r6-dev qwen3-8b-fp32.safetensors
acc8 long-dev r6-dev qwen3-8b-long-fp32.safetensors
plan8 base r6-base
plan8 dev r6-dev
echo "== done $(date +%T)"
python3 - $O <<'PY'
import sys, json
o = sys.argv[1]
def ids(t):
    try:
        s = open(f"{o}/gen-{t}.ids").read().split(":", 1)[1]
        return [int(v) for v in s.replace("[", " ").replace("]", " ").replace(",", " ").split()]
    except Exception:
        return None
for a, b in [("base-graph", "dev-graph"), ("base-graph", "dev-eager"), ("base-900", "dev-900"), ("base-16k", "dev-16k"), ("base-16k", "c4096-16k")]:
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
