#!/usr/bin/env bash
# Lane dsv41/tp2: sync this worktree to spark2, build the CUDA CLI and the dsv41_runtime test binary there, copy both to
# ~/intarsia-dsv41-graphs/bin on both Sparks. Runs INSIDE a fleetq slot (vLLM stays up):
#   fleetq run --host spark2 --load cpu --mem 16 --expect 25 --label "dsv41 tp2 build" -- scripts/dsv41-tp2/build.sh
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
H=${BUILD_HOST:-spark2}
STAMP="$(git -C "$REPO" rev-parse --short=9 HEAD) dirty=$(git -C "$REPO" diff --quiet HEAD -- crates kernels && echo no || echo yes) $(date +%m%d-%H%M)"
RP='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH; export MAX_JOBS=4 CARGO_BUILD_JOBS=4 FLASHINFER_NVCC_THREADS=1;'
rsh() { ssh -n -o BatchMode=yes "$H" "$RP export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
rsh 'mkdir -p ~/intarsia-dsv41-graphs/src ~/intarsia-dsv41-graphs/bin; test -d ~/intarsia-dsv41-graphs/src/target || { test -d ~/intarsia-dsv41-tp2/src/target && cp -a ~/intarsia-dsv41-tp2/src/target ~/intarsia-dsv41-graphs/src/target; true; }'
rsync -a --delete --exclude target --exclude .git --exclude golden --exclude tools/.venv --exclude .claude --exclude .scratch "$REPO/" "$H:intarsia-dsv41-graphs/src/"
echo "stamp: $STAMP"
rsh "cd ~/intarsia-dsv41-graphs/src && INTARSIA_BUILD_STAMP='$STAMP' cargo build --release --features cuda -p intarsia-cli 2>&1 | tail -3"
T=$(rsh "cd ~/intarsia-dsv41-graphs/src && cargo test --release -p intarsia-models --test dsv41_runtime --no-run 2>&1 | grep -o 'target/release/deps/dsv41_runtime-[0-9a-f]*' | tail -1")
rsh "cd ~/intarsia-dsv41-graphs/src && cp target/release/intarsia ~/intarsia-dsv41-graphs/bin/intarsia && cp $T ~/intarsia-dsv41-graphs/bin/dsv41_runtime && echo '$STAMP' > ~/intarsia-dsv41-graphs/bin/STAMP && ls -la ~/intarsia-dsv41-graphs/bin"
O=$([[ $H == spark2 ]] && echo spark1 || echo spark2)
rsh "ssh -o BatchMode=yes $O 'mkdir -p ~/intarsia-dsv41-graphs/bin' && scp -q ~/intarsia-dsv41-graphs/bin/intarsia ~/intarsia-dsv41-graphs/bin/dsv41_runtime ~/intarsia-dsv41-graphs/bin/STAMP $O:intarsia-dsv41-graphs/bin/"
for h in spark1 spark2; do echo "$h: $(ssh -n -o BatchMode=yes $h 'cat ~/intarsia-dsv41-graphs/bin/STAMP; md5sum ~/intarsia-dsv41-graphs/bin/intarsia | cut -c1-12')"; done
