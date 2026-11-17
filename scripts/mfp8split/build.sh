#!/bin/zsh
# Lane mfp8split: build NEW (this tree) and BASE (origin/main, from `git archive` into .base/src-main, same target dir so registry deps are shared)
# with --features metal and copy the binaries aside (.base/bin-new, .base/bin-main); also the rows_exact test binary. Run under fleetq --load cpu.
set -u
W=/opt/code/intarsiaml/.worktrees/mfp8split; cd $W
mkdir -p .base/bin-new .base/bin-main .base/bin-tests
cargo build --release -p intarsia-cli --features metal 2>&1 | tail -3
cp target/release/intarsia .base/bin-new/intarsia || exit 1
out=$(cargo test --release -p intarsia-backend-metal --test rows_exact --no-run 2>&1 | tee /dev/stderr | grep -aoE 'target/release/deps/rows_exact-[0-9a-f]+' | tail -1)
[ -n "$out" ] && cp $out .base/bin-tests/rows_exact
echo "## new built $(md5 -q .base/bin-new/intarsia)"
if [ ! -d .base/src-main ]; then mkdir -p .base/src-main && git archive origin/main | tar -x -C .base/src-main; fi
( cd .base/src-main && CARGO_TARGET_DIR=$W/target-main cargo build --release -p intarsia-cli --features metal 2>&1 | tail -3 )
cp target-main/release/intarsia .base/bin-main/intarsia || exit 1
echo "## main built $(git rev-parse --short origin/main) $(md5 -q .base/bin-main/intarsia)"
