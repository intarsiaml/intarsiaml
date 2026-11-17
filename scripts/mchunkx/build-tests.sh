#!/bin/zsh
# Lane mchunkx: build the metal crate's test binaries (release) and copy rows_exact aside as .base/bin-tests/rows_exact. Run under fleetq --load cpu.
set -u
W=/opt/code/intarsiaml/.worktrees/mchunkx; cd $W
out=$(cargo test --release -p intarsia-backend-metal --test rows_exact --no-run 2>&1 | tee /dev/stderr | grep -aoE 'target/release/deps/rows_exact-[0-9a-f]+' | tail -1)
[ -n "$out" ] || { echo "no test binary"; exit 1; }
mkdir -p .base/bin-tests && cp $out .base/bin-tests/rows_exact && echo "## rows_exact -> .base/bin-tests/rows_exact ($(md5 -q .base/bin-tests/rows_exact))"
