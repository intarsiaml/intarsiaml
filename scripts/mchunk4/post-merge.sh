#!/bin/zsh
# After merging main (msm192's mxfp4.rs changes): rebuild with the metal feature, then the focused tests and the kernels test. Run under fleetq --load cpu.
set -u
W=/opt/code/intarsiaml/.worktrees/mchunk4; R=$W/docs/results/2026-11-16-mchunk4/raw; cd $W
cargo build --release -p intarsia-cli --features metal 2>&1 | tail -2 || exit 1
mkdir -p .base/bin-merged && cp target/release/intarsia .base/bin-merged/intarsia
cargo test --release -p intarsia-backend-metal --lib --test rows_exact --test mxfp4_nax --test mxfp4_gemv 2>&1 | grep -aE "^(test result|error)|FAILED|panicked" | tee $R/test-post-merge.log
cargo test --release -p intarsia-cli --test docs_env_vars 2>&1 | grep -aE "^(test result|error)|FAILED" | tee -a $R/test-post-merge.log
.base/bin-merged/intarsia kernels test --backend metal > $R/kernels-test-post-merge.out 2>&1; tail -1 $R/kernels-test-post-merge.out | tee -a $R/test-post-merge.log; grep -ac FAIL $R/kernels-test-post-merge.out
echo "## done"
