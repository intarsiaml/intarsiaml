#!/usr/bin/env bash
# ON spark1 inside `fleetq run --host spark1 --mem 24` (non-exclusive, vLLM stays up): the CUDA no-op of T1. Builds
# base (origin/main) and dev, then greedy 256 tokens (short and 4k prompt, Qwen3-8B BF16) on each, plus dev with
# INTARSIA_TOKEN_FEEDBACK=1 (CUDA does not support feedback: must change nothing), and the dev CPU-side core/models tests.
# usage: bash cuda-run.sh   (trees in ~/t1/base and ~/t1/dev)
set -uo pipefail
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH CUDA_PATH=/usr/local/cuda CARGO_BUILD_JOBS=5
OUT=$HOME/t1/out; mkdir -p $OUT
M=$HOME/models/Qwen3-8B
for T in base dev; do
  echo "== $T build $(date '+%F %T')"
  ( cd $HOME/t1/$T && CARGO_TARGET_DIR=$HOME/t1/$T-target cargo build --release -p intarsia-cli --features cuda > $OUT/build-$T.log 2>&1 ); echo "$T build exit $?"
done
( cd $HOME/t1/dev && CARGO_TARGET_DIR=$HOME/t1/dev-target cargo test --release -p intarsia-core -p intarsia-models > $OUT/test-dev-cpu.log 2>&1 ); echo "dev cpu tests exit $?"
grep -h "test result" $OUT/test-dev-cpu.log | awk '{p+=$4; f+=$6} END {print "total passed", p, "failed", f}'
gen() { # tag bin fb prompt-file
  INTARSIA_TOKEN_FEEDBACK=$3 $2 generate -m $M --backend cuda --max-seq 8192 --mem-budget 24 --profile-every 0 --temperature 0 \
    --max-tokens 256 --ignore-eos --show-ids -p "$(cat $4)" > $OUT/cuda-g256-$1.out 2> $OUT/cuda-g256-$1.err
  echo "$1 exit $? $(grep -ho 'decode [0-9]* ms ([0-9.]* tok/s)' $OUT/cuda-g256-$1.err | tail -1)"
}
for n in short 4k; do
  pf=$HOME/t1/dev/docs/results/raw/2026-10-15-metal-decode-d1/prompt-short.txt
  [ $n = 4k ] && pf=$HOME/t1/dev/docs/results/raw/2026-10-15-metal-longctx-profile/prompts/prompt-4096.txt
  gen $n-base $HOME/t1/base-target/release/intarsia 0 $pf
  gen $n-dev $HOME/t1/dev-target/release/intarsia 0 $pf
  gen $n-devfb $HOME/t1/dev-target/release/intarsia 1 $pf
  for a in dev devfb; do cmp -s $OUT/cuda-g256-$n-base.out $OUT/cuda-g256-$n-$a.out && echo "CUDA greedy 256 $n $a: byte-identical" || echo "CUDA greedy 256 $n $a: DIFFER"; done
done
grep -h "graph" $OUT/cuda-g256-4k-dev.err | tail -2
echo CUDA_DONE $(date '+%F %T')
