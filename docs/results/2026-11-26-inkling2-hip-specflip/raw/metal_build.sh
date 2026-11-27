#!/usr/bin/env bash
# endor, inside `fleetq run --host endor --load cpu` (lane ink-hip-specflip): the Metal CLI of the worktree as it is (the caller checks the
# tree out) -> .scratch/bin/intarsia-$1 (+ .commit), and the Metal tiny test executables -> .scratch/bin/mt-$1-<test>.
set -uo pipefail
NAME=${1:?name}
W=/opt/code/intarsiaml/.worktrees/ink-hip-specflip; S=$W/.scratch
cd "$W"; export CARGO_TARGET_DIR=$W/target
for v in $(env | grep -oE '^INTARSIA_[A-Z0-9_]+'); do unset "$v"; done
cargo build --release --locked -p intarsia-cli --features metal > $S/build-metal-$NAME.log 2>&1 || { tail -20 $S/build-metal-$NAME.log; exit 1; }
cp target/release/intarsia $S/bin/intarsia-$NAME && { git rev-parse HEAD; git status --porcelain -- crates recipes kernels | head -3; } > $S/bin/intarsia-$NAME.commit
echo "cli $NAME $(md5 -q $S/bin/intarsia-$NAME) $(head -1 $S/bin/intarsia-$NAME.commit)"
for t in inkling2_sampled_default inkling2_dspark_sampled inkling2_dspark_sampled_adapt inkling2_dspark; do
  [ -f crates/intarsia-models/tests/$t.rs ] || { echo "no test $t in this tree"; continue; }
  cargo test --release --locked -p intarsia-models --test $t --no-run > $S/build-mt-$NAME-$t.log 2>&1 || { tail -20 $S/build-mt-$NAME-$t.log; exit 1; }
  exe=$(grep -aoE '[^ (]*target/release/deps/[A-Za-z0-9_]+-[0-9a-f]+' $S/build-mt-$NAME-$t.log | tail -1)
  [ -n "$exe" ] && cp "$exe" $S/bin/mt-$NAME-$t && echo "  mt-$NAME-$t"
done
