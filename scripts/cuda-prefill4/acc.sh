#!/usr/bin/env bash
# Accuracy of the prefill-4 arms against the vLLM goldens (sha b6b51e1b1699c66c), ON spark1 inside ONE fleetq slot
# (fleetq run ... --mem 45 -- ssh spark1 bash ~/intarsia-decfuse/p4-dev/scripts/cuda-prefill4/acc.sh). Every step is
# sequential (one 35B process at a time, --mem-budget 40), so it fits next to a running vLLM: the kernels are deterministic,
# so the numbers do not depend on what else runs (only the wall time does). All arms run in this one slot:
#   base  = tree p4-base (origin/main 8638db03, built as is)                 the same-slot base (chunk 2048 / 1024 / 256)
#   split = tree p4-dev, default (gdn_delta_rule_bf16_split; bitwise base by test)
#   ks    = tree p4-dev, INTARSIA_CUDA_GDN_SPLIT=0 (kill switch: must equal base)
#   f8    = tree p4-dev, INTARSIA_CUDA_GDN_SPLIT=0 INTARSIA_CUDA_FP8_GEMM_F8=1 (the e4m3 tensor-core GEMM alone vs base)
# STEPS (space separated, default all) picks steps by name. Output ~/intarsia-decfuse/out/p4A ($OUTNAME).
set -u
P=$HOME/intarsia-decfuse
O=$P/out/${OUTNAME:-p4A}; mkdir -p $O
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
M8=/home/nvidia/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
GD=$HOME/intarsiaml-golden
S=$GD/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors; L=$GD/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors
gb=$(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) / 1024 / 1024 )); echo "MemAvailable ${gb} GB"; (( gb >= 44 )) || { echo "ABORT: memory"; exit 1; }
for t in p4-base p4-dev; do echo "bin $t: $($P/$t/target/acc/release/intarsia --version 2>&1 | head -1)"; done
nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu --format=csv | tee $O/clocks.txt
sha256sum $S | cut -c1-16
echo "== start $(date +%T)"
want() { [[ -z "${STEPS:-}" || " $STEPS " == *" $1 "* ]]; }
bin() { echo $P/$1/target/acc/release/intarsia; }
acc() { # acc <tag> <tree> <golden> <chunk> <env...>
  local tag=$1 tree=$2 g=$3 chunk=$4; shift 4; want acc-$tag || return 0; local t0=$(date +%s)
  env "$@" timeout 900 $(bin $tree) accuracy --model $M35 --backend cuda --golden $g --mode prefill --max-seq 16384 --mem-budget 40 --prefill-chunk $chunk --allow-dirty > $O/acc-$tag.log 2>&1
  echo "acc $tag exit $? $(( $(date +%s) - t0 )) s: $(grep -E '^prefill floor' $O/acc-$tag.log | head -1 | cut -c1-150)" | tee -a $O/steps.log
}
acc8() { # acc8 <tag> <tree> <golden file> <env...>
  local tag=$1 tree=$2 g=$3; shift 3; want acc8-$tag || return 0; local t0=$(date +%s)
  env "$@" timeout 600 $(bin $tree) accuracy --model $M8 --backend cuda --golden $GD/$g --mode both --decode-positions 16 --max-seq 4096 --mem-budget 40 --gate-class bf16 --allow-dirty > $O/acc8-$tag.log 2>&1
  echo "acc8 $tag exit $? $(( $(date +%s) - t0 )) s: $(grep -E 'verdict|golden gate|PASS|FAIL' $O/acc8-$tag.log | head -4 | cut -c1-160 | tr '\n' '|')" | tee -a $O/steps.log
}
reuse() { # reuse <tag> <tree> <env...>
  local tag=$1 tree=$2; shift 2; want reuse-$tag || return 0; local t0=$(date +%s)
  env "$@" timeout 900 $(bin $tree) accuracy --model $M35 --backend cuda --golden $L --mode reuse --max-seq 16384 --mem-budget 40 --prefill-chunk 2048 --allow-dirty > $O/reuse-$tag.log 2>&1
  echo "reuse $tag exit $? $(( $(date +%s) - t0 )) s: $(grep -aE '^chat_reuse_4k|prefix-reuse gate' $O/reuse-$tag.log | cut -c1-90 | tr -s ' ' | tr '\n' '|')" | tee -a $O/steps.log
}
gen() { # gen <tag> <tree> <extra args or -> <env...>; ~1500-token prompt, 256 greedy tokens
  local tag=$1 tree=$2 extra=$3; shift 3; [[ $extra == - ]] && extra=""; want gen-$tag || return 0
  env "$@" timeout 300 $(bin $tree) generate -m $M35 --backend cuda --mem-budget 29 --max-seq 4096 --chat --no-think --profile-every 0 --ignore-eos --show-ids \
    -p "$(head -c 5000 $P/p4-dev/tools/prompts/corpus_docs.txt)" --max-tokens 256 $extra > $O/gen-$tag.out 2> $O/gen-$tag.err
  grep -a '^ids:' $O/gen-$tag.err > $O/gen-$tag.ids
  echo "gen $tag: $(grep -aE 'finish=|steps=' $O/gen-$tag.err | head -2 | cut -c1-160 | tr '\n' '|')" | tee -a $O/steps.log
}
KS="INTARSIA_CUDA_GDN_SPLIT=0"; F8="INTARSIA_CUDA_GDN_SPLIT=0 INTARSIA_CUDA_FP8_GEMM_F8=1"; DEF="INTARSIA_X=0"
gen base-graph p4-base - $DEF
gen split-graph p4-dev - $DEF
gen split-eager p4-dev --no-graphs $DEF
gen ks-graph p4-dev - $KS
gen f8-graph p4-dev - $F8
acc base-short-c2048 p4-base $S 2048 $DEF
acc split-short-c2048 p4-dev $S 2048 $DEF
acc ks-short-c2048 p4-dev $S 2048 $KS
acc f8-short-c2048 p4-dev $S 2048 $F8
acc base-short-c1024 p4-base $S 1024 $DEF
acc base-short-c256 p4-base $S 256 $DEF
acc split-short-c256 p4-dev $S 256 $DEF
acc f8-short-c256 p4-dev $S 256 $F8
acc base-long-c2048 p4-base $L 2048 $DEF
acc split-long-c2048 p4-dev $L 2048 $DEF
acc f8-long-c2048 p4-dev $L 2048 $F8
acc base-long-c1024 p4-base $L 1024 $DEF
acc base-long-c256 p4-base $L 256 $DEF
acc f8-long-c256 p4-dev $L 256 $F8
reuse base p4-base $DEF
reuse split p4-dev $DEF
reuse f8 p4-dev $F8
acc8 short p4-dev qwen3-8b-fp32.safetensors $DEF
acc8 long p4-dev qwen3-8b-long-fp32.safetensors $DEF
echo "== done $(date +%T)"
python3 - $O <<'PY'
import sys, glob, os
o = sys.argv[1]
def ids(t):
    try:
        s = open(f"{o}/gen-{t}.ids").read().split(":", 1)[1]
        return [int(v) for v in s.replace("[", " ").replace("]", " ").replace(",", " ").split()]
    except Exception:
        return None
b = ids("base-graph")
for t in ["split-graph", "split-eager", "ks-graph", "f8-graph"]:
    x = ids(t)
    if b is None or x is None: print(f"{t}: missing"); continue
    d = next((i for i, (p, q) in enumerate(zip(b, x)) if p != q), None)
    print(f"{t} vs base-graph: " + ("identical" if d is None and len(b) == len(x) else f"first difference at generated token {d} of {len(b)}"))
PY
