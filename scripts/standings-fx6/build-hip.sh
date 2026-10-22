#!/usr/bin/env bash
# fx6: sync this worktree to tatooine:~/intarsiaml-fx6 and build the HIP binary (CPU only; run in a non-exclusive slot):
#   fleetq run --host tatooine --mem 0 --label "fx6 build" -- bash scripts/standings-fx6/build-hip.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
rsync -a --delete --exclude .git --exclude target --exclude docs/results ./ tatooine:intarsiaml-fx6/
git rev-parse HEAD | ssh tatooine 'cat > intarsiaml-fx6/.fx6-commit'
ssh tatooine 'cd ~/intarsiaml-fx6 && INTARSIA_REPO=~/intarsiaml-fx6 INTARSIA_TARGET_VOL=intarsia-cargo-target-fx6 deploy/tatooine/dev.sh bash -c "cargo build --release -p intarsia-cli --features hip 2>&1 | tail -5"' </dev/null
