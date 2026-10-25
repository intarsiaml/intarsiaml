#!/usr/bin/env bash
# Lane sp, final CUDA / Metal gates on the PR head (dev) against main 0b84f499 (base).
#   final-gates.sh build <cuda|metal>   (--load cpu)   builds base and dev (cuda: on a Spark from rsynced sources; metal: on endor)
#   final-gates.sh run   <cuda|metal>   (--quiet gpu,cpu)
# Checks (each prints PASS or FAIL):
#   1. spec-off no-op: `plan show` (json and text) byte-identical, 256 greedy ids identical, 8B and 35B, short and 4k prompts
#   2. 35B natural-v1 128 / 1k: dev MTP k=2 ids == dev spec off; dev MTP acceptance counts == base MTP acceptance counts
#   3. 35B natural-v1 128: dev `--spec mtp --spec-k 2 --mtp-draft-vocab 65536` ids == spec off
set -uo pipefail
mode=${1:?}; BE=${2:?}
if [ "$BE" = cuda ]; then
  P=$HOME/intarsia-decfuse; O=$P/out/sp-final; mkdir -p $O
  export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
  H=/home/nvidia/.cache/huggingface/hub
  M35=$H/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/$(ls $H/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots | head -1)
  M8=$H/models--Qwen--Qwen3-8B/snapshots/$(ls $H/models--Qwen--Qwen3-8B/snapshots | head -1)
  SRC_BASE=$P/sp-src-base; SRC_DEV=$P/sp-src-dev; TREE_BASE=$P/sp-base; TREE_DEV=$P/sp-dev
  BIN_BASE=$TREE_BASE/target/acc/release/intarsia; BIN_DEV=$TREE_DEV/target/acc/release/intarsia
  FLAGS="--mem-budget 30"; NAT=$SRC_DEV/sp-out; PD=$TREE_DEV/scripts/hg/prompts
  md() { md5sum < "$1" | cut -c1-8; }
else
  R=/opt/code/intarsiaml/.worktrees; O=${OUT:-/opt/code/wt-sp/docs/results/2026-10-25-sp-hip-spec/raw/followup/metal}; mkdir -p $O
  M35=$HOME/models/Qwen3.6-35B-A3B-NVFP4; M8=$HOME/models/Qwen3-8B
  TREE_BASE=$R/sp-base; TREE_DEV=/opt/code/wt-sp
  BIN_BASE=$R/sp-targets/base-metal/release/intarsia; BIN_DEV=$R/sp-targets/dev-metal/release/intarsia
  FLAGS=""; NAT=/opt/code/deepeval/tests/data/prompts/natural-v1; PD=$TREE_DEV/scripts/hg/prompts
  md() { md5 -q "$1"; }
fi
case "$mode" in
build)
  if [ "$BE" = cuda ]; then
    for t in sp-base sp-dev; do [ -d $P/$t ] || cp -a $P/hg-dev $P/$t; done
    rsync -rlpgoD --no-times --checksum --delete --exclude target --exclude .git --exclude golden --exclude tools/.venv --exclude sp-out $SRC_BASE/ $TREE_BASE/
    rsync -rlpgoD --no-times --checksum --delete --exclude target --exclude .git --exclude golden --exclude tools/.venv --exclude sp-out $SRC_DEV/ $TREE_DEV/
    for t in $TREE_BASE $TREE_DEV; do
      echo "== build $t"; (cd $t && env CARGO_BUILD_JOBS=8 CARGO_TARGET_DIR=target/acc cargo build --release --features cuda -p intarsia-cli 2>&1 | grep -E '^(error)|Finished' | tail -4)
      [ $t/target/acc/release/intarsia -nt $t/crates/intarsia-core/src/plan.rs ] || { echo "BUILD FAILED for $t"; exit 1; }
    done
    (cd $TREE_DEV && env CARGO_TARGET_DIR=target/acc cargo test --release --no-run -p intarsia-core --lib -p intarsia-models --test batch_recurrent --test mtp 2>&1 | grep -E '^error|Finished' | tail -3)
  else
    for t in base dev; do
      if [ $t = base ]; then S=$TREE_BASE; else S=$TREE_DEV; fi
      echo "== build $t"; (cd $S && CARGO_TARGET_DIR=$R/sp-targets/$t-metal cargo build --release --features metal -p intarsia-cli 2>&1 | grep -E '^error|Finished' | tail -3)
    done
  fi
  echo "dev head: $(cd $TREE_DEV 2>/dev/null && git rev-parse --short=8 HEAD 2>/dev/null || cat $TREE_DEV/HEAD.txt 2>/dev/null)"; echo "dev binary costgate strings: $(strings $BIN_DEV | grep -c intarsia_spec_costgate)"; echo "BUILD DONE" ;;
run)
  echo "source head: $(cd $TREE_DEV 2>/dev/null && git rev-parse --short=8 HEAD 2>/dev/null || cat $TREE_DEV/HEAD.txt 2>/dev/null); dev binary costgate strings: $(strings $BIN_DEV | grep -c intarsia_spec_costgate) (base: $(strings $BIN_BASE | grep -c intarsia_spec_costgate)); dev binary $(ls -l $BIN_DEV | awk '{print $5, $6, $7, $8}')"
  fails=0; chk() { if [ "$1" = 0 ]; then echo "PASS $2"; else echo "FAIL $2"; fails=$((fails+1)); fi; }
  echo "== 1. spec-off no-op vs main"
  for t in base dev; do
    if [ $t = base ]; then B=$BIN_BASE; else B=$BIN_DEV; fi
    RUN="$B" bash $TREE_DEV/scripts/hg/noop.sh run $O $t $BE $M8 $M35 $PD $FLAGS | tail -4
  done
  bash $TREE_DEV/scripts/hg/noop.sh cmp $O base dev | tee $O/noop-cmp.txt
  grep -q "DIFFER" $O/noop-cmp.txt; chk $((1 - $?)) "spec-off no-op (plan json/text + 256 ids, 8B and 35B, short and 4k)"
  echo "== 2. MTP k=2 equality and acceptance, 35B natural-v1"
  gen() { # gen <bin> <tag> <n> <args...>
    b=$1; tag=$2; n=$3; shift 3
    $b generate -m $M35 --backend $BE $FLAGS --max-seq 4096 --profile-every 0 --ignore-eos --show-ids --max-tokens 256 "$@" \
      -p "$(cat $NAT/n$n.txt)" > $O/$tag-n$n.out 2> $O/$tag-n$n.err < /dev/null
    grep -a '^ids:' $O/$tag-n$n.err > $O/$tag-n$n.ids
    echo "$tag n$n md5=$(md $O/$tag-n$n.ids) $(grep -a 'finish=' $O/$tag-n$n.err | tail -1 | sed 's/.*| decode/decode/' | cut -c1-40) | $(grep -a '^spec:' $O/$tag-n$n.err | tail -1 | cut -c1-110)"
  }
  for n in 128 1024; do
    gen $BIN_DEV dev-off $n --spec off
    gen $BIN_DEV dev-mtp2 $n --spec mtp --spec-k 2
    gen $BIN_BASE base-mtp2 $n --spec mtp --spec-k 2
    INTARSIA_SPEC_COSTGATE=0 gen $BIN_DEV dev-cg0-mtp2 $n --spec mtp --spec-k 2
    cmp -s $O/dev-off-n$n.ids $O/dev-mtp2-n$n.ids && [ -s $O/dev-off-n$n.ids ]; chk $? "n$n: dev MTP k=2 ids == dev spec off"
    cmp -s $O/base-mtp2-n$n.ids $O/dev-mtp2-n$n.ids && [ -s $O/dev-mtp2-n$n.ids ]; chk $? "n$n: dev MTP k=2 ids == main MTP k=2"
    cmp -s $O/base-mtp2-n$n.ids $O/dev-cg0-mtp2-n$n.ids && [ -s $O/dev-cg0-mtp2-n$n.ids ]; chk $? "n$n: dev MTP k=2 cost gate off ids == main MTP k=2"
    a=$(grep -a '^spec:' $O/base-mtp2-n$n.err | tail -1 | sed 's/(acceptance.*//'); b=$(grep -a '^spec:' $O/dev-cg0-mtp2-n$n.err | tail -1 | sed 's/(acceptance.*//')
    [ -n "$a" ] && [ "$a" = "$b" ]; chk $? "n$n: acceptance equal to main with the cost gate off ($b)"
    echo "info n$n: cost gate on: $(grep -a '^spec:' $O/dev-mtp2-n$n.err | tail -1 | sed 's/(acceptance.*//') (the probes of the plain-step cost change the pass schedule)"
  done
  echo "== 3. --mtp-draft-vocab 65536, 35B natural-v1 128"
  gen $BIN_DEV dev-v65536 128 --spec mtp --spec-k 2 --mtp-draft-vocab 65536
  cmp -s $O/dev-off-n128.ids $O/dev-v65536-n128.ids && [ -s $O/dev-v65536-n128.ids ]; chk $? "n128: --mtp-draft-vocab 65536 ids == spec off"
  if [ "$BE" = cuda ]; then
    echo "== 4. tests on the dev tree"
    cd $TREE_DEV
    env CARGO_TARGET_DIR=target/acc cargo test --release -p intarsia-core --lib 2>&1 | grep -E "^test result|FAILED|panicked|^error"
    env CARGO_TARGET_DIR=target/acc cargo test --release -p intarsia-models --test mtp 2>&1 | grep -E "^test result|FAILED|panicked|^error"
    env INTARSIA_BATCH_TEST_BACKEND=cuda INTARSIA_REQUIRE_GPU=1 CARGO_TARGET_DIR=target/acc cargo test --release -p intarsia-models --test batch_recurrent -- --test-threads=1 2>&1 | grep -E "^test result|FAILED|panicked|^error"
  fi
  echo "== done: $fails failing check(s)"; exit $fails ;;
esac
