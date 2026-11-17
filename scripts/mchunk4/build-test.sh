#!/bin/zsh
# Lane mchunk4: build B (this branch, --features metal), copy the binary aside (a later `cargo test --workspace --release` rebuilds
# target/release/intarsia WITHOUT the metal feature), then the focused tests and the kernels test on the copy. Run under fleetq --load cpu.
set -u
W=/opt/code/intarsiaml/.worktrees/mchunk4; R=$W/docs/results/2026-11-16-mchunk4/raw; cd $W
cargo build --release -p intarsia-cli --features metal 2>&1 | tail -3 || exit 1
mkdir -p .base/bin-mchunk4 && cp target/release/intarsia .base/bin-mchunk4/intarsia || exit 1
echo "## build done $(md5 -q .base/bin-mchunk4/intarsia)"
cargo test --release -p intarsia-backend-metal --lib chunk_tests 2>&1 | tail -8 | tee $R/test-chunk-unit.log
cargo test --release -p intarsia-backend-metal --test rows_exact --test mxfp4_nax --test mxfp4_gemv -- --nocapture 2>&1 | grep -aE "^(test |running|test result|mxfp4|error|warning: unused)|FAIL|panicked" | tee $R/test-rows.log
.base/bin-mchunk4/intarsia kernels test --backend metal 2>&1 | tee $R/kernels-test.out | tail -5
echo "## done"
