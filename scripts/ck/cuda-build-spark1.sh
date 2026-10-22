#!/usr/bin/env bash
# Lane ck: sync this tree to spark1:~/ck/src and build the CUDA release CLI (CPU only, vLLM stays up).
#   fleetq run --host spark1 --mem 8 --load cpu --expect 25 --label "ck: build" -- scripts/ck/cuda-build-spark1.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
H=${HOST:-spark1}
ssh -n -o BatchMode=yes $H "mkdir -p ~/ck/src ~/ck/out"
rsync -a --delete --exclude target --exclude .git --exclude golden --exclude tools/.venv --exclude .claude --exclude docs/results "$REPO/" $H:ck/src/
echo "tree HEAD $(git -C "$REPO" rev-parse --short HEAD)"
ssh -n -o BatchMode=yes $H "export FLEETQ_JOB=$FLEETQ_JOB PATH=\$HOME/.cargo/bin:/usr/local/cuda/bin:\$PATH CUDA_PATH=/usr/local/cuda; cd ~/ck/src && CARGO_TARGET_DIR=\$HOME/ck/target nice cargo build --release -q -p intarsia-cli --features cuda -j 10 2>&1 | grep -E '^error' | head; ls -la \$HOME/ck/target/release/intarsia && \$HOME/ck/target/release/intarsia --version"
