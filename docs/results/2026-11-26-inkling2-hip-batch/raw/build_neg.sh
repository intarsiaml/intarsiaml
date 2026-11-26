#!/usr/bin/env bash
# ON tatooine inside a fleetq --load cpu job (lane ink-hip-batch): the negative control of the switch check. .scratch/negtree = this
# branch with ink_mix.hip as before the shared-switch fix (each TU its own g_mix_mask). Builds its tests/ink_rows.rs in its own target
# dir (/target/neg) and copies the test binary to .scratch/bin/ink_rows-neg: run in the window, its mode-0 assertion must FAIL.
set -uo pipefail
cd "$HOME/intarsiaml-ink-hip-batch"
mkdir -p logs .scratch/bin
export INTARSIA_REPO=$PWD INTARSIA_TARGET_VOL=intarsia-cargo-target-ink-hip-batch INTARSIA_CONTAINER_NAME=ink-hip-batch-build
trap 'docker rm -f ink-hip-batch-build >/dev/null 2>&1' EXIT
deploy/tatooine/dev.sh bash -c 'cd /work/.scratch/negtree && CARGO_TARGET_DIR=/target/neg cargo test --release -j8 -p intarsia-backend-hip --test ink_rows --no-run 2>&1 | tail -4 && b=$(ls -t /target/neg/release/deps/ink_rows-* | grep -v "\.d$" | head -1) && echo "test binary $b" && cp "$b" /work/.scratch/bin/ink_rows-neg' < /dev/null > logs/build-neg.log 2>&1
rc=$?
cat logs/build-neg.log | tail -6
echo "neg build exit $rc"
exit $rc
