#!/usr/bin/env bash
set -euo pipefail
: "${FLEETQ_JOB:?}"
WT=/opt/code/intarsiaml/.worktrees/sb8
H=spark2
ssh -n -o BatchMode=yes $H "mkdir -p ~/intarsiaml-sb8/src ~/intarsiaml-sb8/bin ~/intarsiaml-sb8/out"
STAMP="$(cd $WT && scripts/tree-stamp.sh)"
rsync -a --delete --exclude .git --exclude target --exclude tools/.venv --exclude .worktrees --exclude .claude "$WT/" "$H:intarsiaml-sb8/src/"
ssh -n -o BatchMode=yes $H "export PATH=\$HOME/.cargo/bin:/usr/local/cuda/bin:\$PATH; cd ~/intarsiaml-sb8/src && env INTARSIA_BUILD_STAMP='$STAMP' CARGO_BUILD_JOBS=16 cargo build --release --features cuda -p intarsia-cli 2>&1 | grep -E '^(error|warning: unused)|Finished' | tail -8; cp target/release/intarsia ~/intarsiaml-sb8/bin/intarsia; ~/intarsiaml-sb8/bin/intarsia --version | head -1; echo $STAMP"
