#!/usr/bin/env bash
# Lane lt (from lane pf's scripts/lt/build.sh): sync a source dir to a Spark and build the CLI (CUDA). Run inside a fleetq slot (--load cpu):
#   FLEETQ_OWNER=claude-f31ccf91-lt fleetq run --host spark1 --load cpu --expect 25 --label "pf build" -- scripts/lt/build.sh <tree> <srcdir> [stamp]
# Trees live in ~/intarsia-decfuse/<tree> on HOST (default spark1); the target dir is seeded from SEED_TREE.
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq}"
HOST=${HOST:-spark1}; T=$1; SRC=$2; STAMP=${3:-lt-$T}
SEED=${SEED_TREE:-pf-base}; R='~/intarsia-decfuse'
RP='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
rsh() { ssh -n -o BatchMode=yes "$HOST" "$RP FLEETQ_JOB=$FLEETQ_JOB; $*"; }
rsh "mkdir -p $R/$T; if [ ! -d $R/$T/target ] && [ -d $R/$SEED/target ]; then cp -a $R/$SEED/target $R/$T/target; fi"
rsync -a --delete --exclude target --exclude .git --exclude golden --exclude tools/.venv --exclude .claude --exclude docs/results "$SRC/" "$HOST:intarsia-decfuse/$T/"
rsh "cd $R/$T && find . -path ./target -prune -o -type f \( -name '*.rs' -o -name '*.cu' -o -name '*.cuh' -o -name '*.h' -o -name '*.toml' \) -print0 | xargs -0 touch"
rsh "cd $R/$T && env INTARSIA_BUILD_STAMP='$STAMP' CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-8} CARGO_TARGET_DIR=target/acc cargo build --release --features cuda -p intarsia-cli 2>&1 | grep -E '^(error|warning: unused)|Finished' | tail -8"
rsh "ls -la $R/$T/target/acc/release/intarsia"
# helper scripts and the 35B goldens/oracle for the gate runs
ssh -n -o BatchMode=yes "$HOST" "mkdir -p ~/intarsia-decfuse/pf-scripts ~/intarsia-decfuse/pf-golden"
rsync -a "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/" "$HOST:intarsia-decfuse/pf-scripts/lt/"
rsync -a --include='qwen36*' --include='oracle/***' --include='ref_noise/***' --include='baselines/***' --exclude='*' /opt/code/intarsiaml/golden/ "$HOST:intarsia-decfuse/pf-golden/"
