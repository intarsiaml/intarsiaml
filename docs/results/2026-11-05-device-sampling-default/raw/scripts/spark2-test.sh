#!/usr/bin/env bash
# Lane lane-ds: cargo tests on spark2 (Linux, CPU features only; endor's line is 12 jobs long). Inside a fleetq --load cpu slot on spark2.
# docs/results is synced (committed result files are test inputs, e.g. cuda-round7), except this lane's own folder.
# Step 1: the server crate (sampling defaults); step 2: cargo test --workspace --no-fail-fast without intarsia-backend-metal (its tests
# do not compile on Linux: `intarsia_backend_metal::kernels` is macOS-only; first run's error in spark2-test-workspace-first-metal-compile-error.txt).
set -uo pipefail
WT=/opt/code/intarsiaml/.worktrees/device-sampling; OUT=$WT/docs/results/2026-11-05-device-sampling-default/raw
ssh spark2 'mkdir -p ~/intarsiaml-ds/src'
rsync -rlpt --delete --exclude .git --exclude .scratch --exclude '/target*' --exclude .worktrees --exclude docs/results/2026-11-05-device-sampling-default "$WT"/ spark2:intarsiaml-ds/src/
ssh -n spark2 'cd ~/intarsiaml-ds/src && export CARGO_BUILD_JOBS=8 && ~/.cargo/bin/cargo test -p intarsia-server 2>&1 | grep -aE "^test |test result|error|warning: unused|FAILED|panicked" | tail -150' > $OUT/spark2-test-server.txt 2>&1
echo "server tests: $(grep -a 'test result' $OUT/spark2-test-server.txt | tr '\n' ' ')"
ssh -n spark2 'cd ~/intarsiaml-ds/src && export CARGO_BUILD_JOBS=8 && ~/.cargo/bin/cargo test --workspace --exclude intarsia-backend-metal --no-fail-fast 2>&1' > $OUT/spark2-test-workspace.txt 2>&1
echo "workspace exit: $(tail -1 $OUT/spark2-test-workspace.txt | cut -c1-100)"
grep -aE "^test result|FAILED|failed" $OUT/spark2-test-workspace.txt | awk '/^test result/{p+=$4; f+=$6; i+=$8} /FAILED|failed/ && !/^test result/{print} END{print "passed",p,"failed",f,"ignored",i}' | tail -15
