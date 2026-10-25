#!/usr/bin/env bash
# lane df: build <tree> from <srcdir> on spark2. Run inside fleetq slot (--load cpu).
set -euo pipefail
: "${FLEETQ_JOB:?}"
HOST=spark2; T=$1; SRC=$2; STAMP=$3; SEED=${SEED_TREE:-mp-dev}; R='~/intarsia-decfuse'
RP='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
rsh() { ssh -n -o BatchMode=yes "$HOST" "$RP FLEETQ_JOB=$FLEETQ_JOB; $*"; }
rsh "mkdir -p $R/$T; if [ ! -d $R/$T/target ] && [ -d $R/$SEED/target ]; then cp -a $R/$SEED/target $R/$T/target; fi"
rsync -a --delete --exclude target --exclude .git --exclude golden --exclude tools/.venv --exclude .claude --exclude docs/results "$SRC/" "$HOST:intarsia-decfuse/$T/"
rsh "cd $R/$T && find . -path ./target -prune -o -type f \( -name '*.rs' -o -name '*.cu' -o -name '*.cuh' -o -name '*.h' -o -name '*.toml' \) -print0 | xargs -0 touch"
rsh "cd $R/$T && env INTARSIA_BUILD_STAMP='$STAMP' CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-10} CARGO_TARGET_DIR=target/acc cargo build --release --features cuda -p intarsia-cli 2>&1 | grep -E '^(error|warning: unused)|Finished' | tail -8"
rsh "ls -la $R/$T/target/acc/release/intarsia"
