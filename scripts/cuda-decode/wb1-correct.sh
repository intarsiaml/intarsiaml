#!/usr/bin/env bash
# Window B1 of the CUDA norm + GEMV work (spark1, vLLM down, INSIDE scripts/cuda-decode/window.sh): launches, correctness, tests.
#   wb1-correct.sh <base tree> <dev tree>         (trees under ~/intarsia-decfuse; base = check/cuda-noop-norm-gemv, dev = this branch)
# 1. nsys decode profile of base and dev (clean GPU); aborts the window when dev has not shed the launches (the fused kernels
#    must be admitted: a refused kernel falls back silently and the ids would still match)
# 2. in parallel (deterministic, GPU sharing changes no bit): the unit tests of the backend (fused_decode, admission, quantmoe),
#    every mutation binary of mutbuild.sh against the tests of its kernel, the 35b-short accuracy of dev against the base dumps of
#    the no-op check (out/noop-hipnorm), the 256 ids of dev (graph, --no-graphs, INTARSIA_NO_FUSION=1; 35B and 8B) against the
#    base ids of the no-op check, and the two 8B fp32 goldens.
set -u
BASE=${1:?base tree}; DEV=${2:?dev tree}
P=$HOME/intarsia-decfuse
D=$(dirname "$0")
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
M8=/home/nvidia/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
GD=$HOME/intarsiaml-golden
REFD=$P/out/noop-hipnorm       # the no-op check's results: base (= origin/main = hipnorm) ids and dumps
O=$P/out/b1; rm -rf $O; mkdir -p $O
BD=$P/$DEV/target/acc/release/intarsia
gb=$(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) / 1024 / 1024 )); echo "MemAvailable ${gb} GB"; (( gb >= 90 )) || { echo "ABORT: memory"; exit 1; }
echo "dev: $($BD --version 2>&1 | head -1)"

echo "== nsys $(date +%T)"
OUT_PROF=$(bash $D/w-profile.sh base:$BASE dev:$DEV 2>&1); echo "$OUT_PROF" | tail -30 | cut -c1-200
lb=$(python3 -I -c "import json;print(int(json.load(open('$P/out/prof-base/steps.json'))['launches']))" 2>/dev/null || echo 0)
ld=$(python3 -I -c "import json;print(int(json.load(open('$P/out/prof-dev/steps.json'))['launches']))" 2>/dev/null || echo 0)
echo "launches/step: base $lb dev $ld (expected -99)"
if (( lb == 0 || ld == 0 || ld > lb - 90 )); then echo "ABORT: the fused kernels are not in effect (base $lb dev $ld)"; exit 1; fi

SHORT="--mode both --decode-positions 16 --max-seq 4096 --mem-budget 60 --prefill-chunk 256"
( # lane 1: 35b-short accuracy of dev, dump compared to the base dump of the no-op check
  INTARSIA_ACC_POS_DUMP=$O/dev.pos.tsv $BD accuracy --model $M35 --backend cuda --golden $GD/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors $SHORT \
    --write-self-golden $O/dev.dump.safetensors --allow-dirty > $O/acc35-dev.log 2>&1
  echo "acc35 dev exit $?" >> $O/lane1.log
  grep -E "floor verdict|golden gate|^prefill floor|^decode" $O/acc35-dev.log | head -6 | cut -c1-230 >> $O/lane1.log
  python3 -I $D/cmp_dump.py $REFD/B.dump.safetensors $O/dev.dump.safetensors >> $O/lane1.log 2>&1
  cmp $REFD/B.pos.tsv $O/dev.pos.tsv && echo "per-position TSV identical to base" >> $O/lane1.log
  echo "lane1 done $(date +%T)" >> $O/lane1.log
) &
(
  gen() { # gen <35b|8b> <tag> <envs-or-"-"> [extra args]
    local m=$1 tag=$2 envs=$3 mp mb; shift 3; [ "$envs" = - ] && envs=""
    if [ $m = 35b ]; then mp=$M35; mb=29; else mp=$M8; mb=25; fi
    env $envs $BD generate -m $mp --backend cuda --mem-budget $mb --max-seq 4096 --chat --no-think --profile-every 0 --ignore-eos --show-ids \
      -p 'Write a long story about a robot who learns to paint.' --max-tokens 256 "$@" > $O/gen$m-$tag.out 2> $O/gen$m-$tag.err
    grep -a '^ids:' $O/gen$m-$tag.err > $O/gen$m-$tag.ids
    echo "gen $m $tag: $(grep -aE 'finish=|steps=' $O/gen$m-$tag.err | cut -c1-170 | tr '\n' ' ')" >> $O/lane2.log
  }
  gen 35b dev-graph -; gen 35b dev-eager - --no-graphs; gen 35b dev-nofusion INTARSIA_NO_FUSION=1
  gen 8b dev-graph -; gen 8b dev-eager - --no-graphs; gen 8b dev-nofusion INTARSIA_NO_FUSION=1
  for g in qwen3-8b-fp32 qwen3-8b-long-fp32; do
    $BD accuracy --model $M8 --backend cuda --golden $GD/$g.safetensors --mode both --decode-positions 16 --max-seq 4096 --mem-budget 25 --gate-class bf16 --allow-dirty > $O/acc8-$g.log 2>&1
    echo "acc8 $g exit $?" >> $O/lane2.log
    grep -E "floor verdict|golden gate|^prefill floor|^decode" $O/acc8-$g.log | head -5 | cut -c1-230 >> $O/lane2.log
  done
  echo "lane2 done $(date +%T)" >> $O/lane2.log
) &
( # lane 3: unit tests of the backend, then the mutations
  cd $P/$DEV && CARGO_TARGET_DIR=target/acc INTARSIA_REQUIRE_GPU=1 timeout 600 cargo test --release -p intarsia-backend-cuda \
    --test fused_decode --test admission --test quantmoe -- --test-threads=2 > $O/tests.log 2>&1
  echo "tests exit $?" >> $O/lane3.log; grep -E "^test result|^test .*FAILED" $O/tests.log | cut -c1-200 >> $O/lane3.log
  run1() { # run1 <mutation>
    local m=$1 f=""
    case $m in norm-*) f=rmsnorm_add_matmul_pair ;; gn-*) f=gated_rmsnorm_matmul_fp8 ;; esac
    INTARSIA_REQUIRE_GPU=1 timeout 300 $P/mutbins/$m --test-threads=1 $f > $O/mut-$m.log 2>&1
    local rc=$?
    if [ $m = unmutated ]; then echo "unmutated: rc $rc $(grep -E '^test result' $O/mut-$m.log)"
    elif grep -q "test result: FAILED" $O/mut-$m.log; then echo "KILLED   $m: $(grep -E '^test .*FAILED' $O/mut-$m.log | sed 's/^test //; s/ \.\.\. FAILED//' | tr '\n' ' ')"
    else echo "SURVIVED $m: rc $rc $(grep -E '^test result' $O/mut-$m.log)"; fi
  }
  export -f run1; export P O
  ls $P/mutbins | grep -v '\.path$' | xargs -P 4 -I{} bash -c 'run1 {}' >> $O/mutations.log 2>&1
  echo "lane3 done $(date +%T)" >> $O/lane3.log
) &
wait
echo "== lanes done $(date +%T)"
cat $O/lane1.log $O/lane2.log $O/lane3.log; echo "-- mutations"; sort $O/mutations.log
for m in 35b 8b; do
  ref=$REFD/gen$m-A-graph.ids
  for t in dev-graph dev-eager dev-nofusion; do
    if [ -s $ref ] && cmp -s $ref $O/gen$m-$t.ids; then echo "IDS $m $t == base"; else echo "IDS $m $t DIFFERS from base"; cmp $ref $O/gen$m-$t.ids | head -2; fi
  done
done | tee $O/idproof.txt
echo "== wb1 done $(date +%T)"
