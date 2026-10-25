#!/usr/bin/env bash
# Lane sp, ON a Spark (spark1) inside a fleetq job. CUDA must not regress with the branch: MTP equality (spec off == spec mtp k=2,
# 256 greedy ids) and no-op (branch vs main at 2c2d9941) at 128 and 1k natural-v1 tokens, plus the core lib tests and the CUDA
# batching test on the branch. Trees: $P/sp-src-base, $P/sp-src-dev (rsynced sources from endor), built in $P/sp-base, $P/sp-dev
# (seeded from $SEED, a built tree of a recent main, for warm cargo artefacts).
#   cuda-gates.sh build   (--load cpu)      cuda-gates.sh run   (--quiet gpu,cpu)
set -uo pipefail
P=$HOME/intarsia-decfuse; O=$P/out/sp; mkdir -p $O
SEED=${SEED:-hg-dev}
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
H=/home/nvidia/.cache/huggingface/hub
M35=$H/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/$(ls $H/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots | head -1)
case "${1:-run}" in
  build)
    for t in sp-base sp-dev; do [ -d $P/$t ] || cp -a $P/$SEED $P/$t; done
    for t in base dev; do
      rsync -rlpgoD --no-times --checksum --delete --exclude target --exclude .git --exclude golden --exclude tools/.venv --exclude sp-out \
        $P/sp-src-$t/ $P/sp-$t/
      echo "== build $t"
      (cd $P/sp-$t && env CARGO_BUILD_JOBS=8 CARGO_TARGET_DIR=target/acc cargo build --release --features cuda -p intarsia-cli 2>&1 | grep -E '^(error|warning: unused)|Finished' | tail -8)
      [ $P/sp-$t/target/acc/release/intarsia -nt $P/sp-$t/crates/intarsia-core/src/plan.rs ] || { echo "BUILD FAILED for $t"; exit 1; }
    done
    echo "== compile branch tests"
    (cd $P/sp-dev && env CARGO_TARGET_DIR=target/acc cargo test --release --no-run -p intarsia-core --lib -p intarsia-models --test batch_recurrent --test mtp 2>&1 | grep -E '^error|Finished|Executable' | tail -6) ;;
  run)
    for t in base dev; do
      for n in 128 1024; do
        for a in "off:--spec off" "mtp2:--spec mtp --spec-k 2"; do
          tag=${a%%:*}; args=${a#*:}
          f=$O/$t-$tag-n$n
          $P/sp-$t/target/acc/release/intarsia generate -m $M35 --backend cuda --mem-budget 30 --max-seq 4096 --profile-every 0 --ignore-eos \
            --show-ids --max-tokens 256 $args -p "$(cat $P/sp-src-dev/sp-out/n$n.txt)" > $f.out 2> $f.err < /dev/null
          grep -a '^ids:' $f.err > $f.ids
          echo "$t $tag n$n md5=$(md5sum < $f.ids | cut -c1-8) | $(grep -a 'finish=' $f.err | tail -1 | sed 's/.*| decode/decode/' | cut -c1-60) | $(grep -a '^spec:' $f.err | tail -1 | cut -c1-110)"
        done
      done
    done | tee $O/summary.txt
    echo "== equality (spec off == mtp, per tree) and no-op (base == dev)"
    for n in 128 1024; do
      for t in base dev; do
        cmp -s $O/$t-off-n$n.ids $O/$t-mtp2-n$n.ids && echo "n$n $t: spec off == mtp k=2" || echo "n$n $t: DIFFER off vs mtp"
      done
      for a in off mtp2; do
        cmp -s $O/base-$a-n$n.ids $O/dev-$a-n$n.ids && echo "n$n $a: base == dev" || echo "n$n $a: DIFFER base vs dev"
      done
    done
    echo "== tests on the branch"
    cd $P/sp-dev
    env CARGO_TARGET_DIR=target/acc cargo test --release -p intarsia-core --lib 2>&1 | grep -E "^test result|FAILED|panicked|^error"
    env CARGO_TARGET_DIR=target/acc cargo test --release -p intarsia-models --test mtp 2>&1 | grep -E "^test result|FAILED|panicked|^error"
    env INTARSIA_BATCH_TEST_BACKEND=cuda INTARSIA_REQUIRE_GPU=1 CARGO_TARGET_DIR=target/acc cargo test --release -p intarsia-models --test batch_recurrent -- --test-threads=1 2>&1 | grep -E "^test result|FAILED|panicked|^error"
    echo "== done" ;;
esac
