#!/usr/bin/env bash
# endor, inside `fleetq run --host endor --load cpu` (lane ink-hip-overlap2): main's Metal cli (origin/main, default 1bf40e29f) built in
# this worktree by a detached checkout (no new folder), copied to .scratch/intarsia-metal-main6, then back to the branch. Refuses on a
# dirty tracked tree. Run it only while no window driver of this lane runs and nothing edits the worktree.
set -uo pipefail
W=/opt/code/intarsiaml/.worktrees/ink-hip-overlap; cd "$W"
for v in $(env | grep -oE '^INTARSIA_[A-Z0-9_]+'); do unset "$v"; done
REV=${1:-1bf40e29f}
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "[main-metal] tracked tree dirty: stop"; git status --short --untracked-files=no; exit 3; }
br=$(git branch --show-current); [ -n "$br" ] || { echo "[main-metal] not on a branch: stop"; exit 3; }
trap 'git checkout -q "$br" && echo "[main-metal] back on $br $(git rev-parse --short HEAD) $(date +%T)"' EXIT
git checkout -q --detach "$REV" || exit 3
echo "[main-metal] at $(git rev-parse --short HEAD) $(date +%T)"
cargo build --release --locked -p intarsia-cli --features metal 2>&1 | tail -1
cp target/release/intarsia .scratch/intarsia-metal-main6 && md5 .scratch/intarsia-metal-main6
git rev-parse HEAD > .scratch/intarsia-metal-main6.commit
