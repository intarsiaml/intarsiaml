#!/usr/bin/env bash
# Lane ck, CUDA 8B recipe change (tuning.cuda prefill_chunk 4096): main vs branch checks on spark2 inside one fleetq slot.
#   fleetq run --host spark2 --mem 30 --expect 30 --label "..." -- scripts/ck/o12-checks.sh <outdir> <main-rev> 
# Builds origin/main (MAIN) and this tree (BRANCH) CUDA CLIs on spark2, then: plan show --json 35B + 8B on both (diff), greedy 64 ids on
# two corpus prompts (8B, main vs branch), `accuracy` of the 8B against its fp32 short + 4k goldens on the branch.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run --host spark2}"
OUT=${1:?outdir}; MAINREV=${2:-origin/main}
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; H=spark2; mkdir -p "$OUT"
G=/opt/code/intarsiaml/golden
ssh -n -o BatchMode=yes $H "mkdir -p ~/ck9/main ~/ck9/br ~/ck9/golden ~/ck9/out"
git -C "$REPO" archive "$MAINREV" | ssh $H "tar -x -C ~/ck9/main"
rsync -a --delete --exclude target --exclude .git --exclude golden --exclude .claude --exclude docs/results "$REPO/" $H:ck9/br/
scp -q $G/qwen3-8b-fp32.safetensors $G/qwen3-8b-4k-fp32-top20.safetensors $H:ck9/golden/
ssh -o BatchMode=yes $H "export FLEETQ_JOB=$FLEETQ_JOB; bash -s" <<'EOS' 2>&1 | tee "$OUT/checks.log"
set -u
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH CUDA_PATH=/usr/local/cuda
cd ~/ck9
for t in main br; do
  ( cd $t && CARGO_TARGET_DIR=$HOME/ck9/target nice cargo build --release -q -p intarsia-cli --features cuda -j 10 < /dev/null 2>&1 | grep -E '^error' | head -3 )
  cp target/release/intarsia bin-$t
  echo "built $t: $(./bin-$t --version)"
done
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
M8=/home/nvidia/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
mkdir -p out; : > out/plan.txt
for m in 35 8; do
  eval M=\$M$m; B=40; [ $m = 8 ] && B=26
  for t in main br; do
    ./bin-$t plan show --backend cuda --model $M --mem-budget $B --max-seq 17408 --json < /dev/null > out/plan-$m-$t.json 2> out/plan-$m-$t.err
    echo "plan $m $t exit $?" >> out/plan.txt
  done
  diff out/plan-$m-main.json out/plan-$m-br.json > out/plan-$m.diff; echo "plan $m diff exit $? ($(wc -l < out/plan-$m.diff) lines)" | tee -a out/plan.txt
done
CORPUS=$HOME/ck9/br/tools/prompts/corpus_docs.txt
for chars in 20000 54000; do for t in main br; do
  ./bin-$t generate --backend cuda -m $M8 --mem-budget 26 --max-seq 17408 --profile-every 0 --ignore-eos --show-ids --max-tokens 64 -p "$(head -c $chars $CORPUS)" > out/gen-$chars-$t.out 2> out/gen-$chars-$t.err < /dev/null
  grep -a '^ids:' out/gen-$chars-$t.err > out/gen-$chars-$t.ids
  echo "gen $chars $t: $(grep -aE 'finish=' out/gen-$chars-$t.err | head -1 | cut -c1-90) ids md5 $(md5sum < out/gen-$chars-$t.ids | cut -c1-8) $(grep -a 'prefill chunk' out/gen-$chars-$t.err | head -1)"
done; cmp -s out/gen-$chars-main.ids out/gen-$chars-br.ids && echo "  $chars: ids identical" || echo "  $chars: ids DIFFER"; done
for g in qwen3-8b-fp32:short qwen3-8b-4k-fp32-top20:4k; do f=${g%%:*}; n=${g##*:}
  ./bin-br accuracy --backend cuda --model $M8 --golden golden/$f.safetensors --mode all --max-seq 8192 --mem-budget 26 --gate-class bf16 --allow-unstamped --allow-dirty < /dev/null > out/acc-$n-br.log 2>&1
  echo "accuracy $n branch exit $?: $(grep -a -E 'golden gate|chunk' out/acc-$n-br.log | head -3 | tr '\n' ' ')"
done
EOS
rsync -a $H:ck9/out/ "$OUT/"
