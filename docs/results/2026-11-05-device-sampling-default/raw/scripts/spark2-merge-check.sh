#!/usr/bin/env bash
# Lane lane-ds: after merging origin/main 1a2a246f6: everything compiles (all test targets, minus the macOS-only Metal crate) and the tests this lane touches pass.
# Inside a fleetq --load cpu slot on spark2; ~/intarsiaml-ds-mc is removed at the end. CUDA-device tests are NOT run (compile only: --no-run).
set -uo pipefail
WT=/opt/code/intarsiaml/.worktrees/device-sampling; OUT=$WT/docs/results/2026-11-05-device-sampling-default/raw
trap 'ssh -n spark2 "rm -rf ~/intarsiaml-ds-mc"' EXIT
ssh -n spark2 'rm -rf ~/intarsiaml-ds-mc; mkdir -p ~/intarsiaml-ds-mc'
(cd $WT && git archive HEAD ':(exclude)docs/results') | ssh spark2 'tar -x -C ~/intarsiaml-ds-mc'
ssh -n spark2 'cd ~/intarsiaml-ds-mc && export CARGO_BUILD_JOBS=8 && ~/.cargo/bin/cargo test --workspace --exclude intarsia-backend-metal --no-run 2>&1 | grep -aE "^error|warning: build failed|Finished|Executable" | tail -5' > $OUT/spark2-merge-norun.txt 2>&1
echo "no-run: $(grep -a -c '^error' $OUT/spark2-merge-norun.txt) errors; $(grep -a Finished $OUT/spark2-merge-norun.txt)"
ssh -n spark2 'cd ~/intarsiaml-ds-mc && export CARGO_BUILD_JOBS=8 && ~/.cargo/bin/cargo test -p intarsia-server 2>&1 | grep -aE "test result|FAILED|panicked"; ~/.cargo/bin/cargo test -p intarsia-core --lib 2>&1 | grep -aE "test result|FAILED|panicked"; ~/.cargo/bin/cargo test -p intarsia-cli --test docs_env_vars 2>&1 | grep -aE "test result|FAILED|panicked"' > $OUT/spark2-merge-tests.txt 2>&1
cat $OUT/spark2-merge-tests.txt | cut -c1-150
