#!/usr/bin/env bash
# M3: CPU-slot HIP build of origin/main (this worktree) on tatooine, tree ~/intarsiaml-m3, target volume seeded from r9's.
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq}"
W=/opt/code/intarsiaml/.claude/worktrees/agent-ad695ea935d5d712e
rsync -a --delete --exclude target --exclude .git --exclude golden --exclude tools/.venv --exclude .claude --exclude bench-out "$W/" tatooine:intarsiaml-m3/
ssh -n -o BatchMode=yes tatooine "export FLEETQ_JOB=$FLEETQ_JOB; set -e
IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
if ! docker volume inspect intarsia-cargo-target-m3 >/dev/null 2>&1; then
  docker volume create intarsia-cargo-target-m3 >/dev/null
  docker run --rm -v intarsia-cargo-target-r9:/a:ro -v intarsia-cargo-target-m3:/b --user 0 \$IMG bash -c 'cp -a /a/. /b/ && chmod 0777 /b'
fi
cd ~/intarsiaml-m3 && mkdir -p bench-out
INTARSIA_REPO=\$HOME/intarsiaml-m3 INTARSIA_TARGET_VOL=intarsia-cargo-target-m3 deploy/tatooine/dev.sh cargo build --release -p intarsia-cli --features hip 2>&1 | grep -E '^error|Finished' | head
INTARSIA_REPO=\$HOME/intarsiaml-m3 INTARSIA_TARGET_VOL=intarsia-cargo-target-m3 deploy/tatooine/dev.sh /target/release/intarsia --version"
