#!/usr/bin/env bash
# endor, inside `fleetq run --host endor --load cpu` (lane ink-hip-overlap2): the full cpu suite of the merged head (cpu_full.sh), then
# the Metal cli of the branch -> .scratch/intarsia-metal-o6 and the Metal tiny test binaries (built only: they run in the exclusive slot).
set -uo pipefail
W=/opt/code/intarsiaml/.worktrees/ink-hip-overlap; cd "$W"
for v in $(env | grep -oE '^INTARSIA_[A-Z0-9_]+'); do unset "$v"; done
bash docs/results/2026-11-26-inkling2-hip-overlap/raw/cpu_full.sh "${1:-merged}"; rc=$?
echo "[endor] metal cli (branch $(git rev-parse --short HEAD)) $(date +%T)"
cargo build --release --locked -p intarsia-cli --features metal 2>&1 | tail -1
cp target/release/intarsia .scratch/intarsia-metal-o6 && md5 .scratch/intarsia-metal-o6
git rev-parse HEAD > .scratch/intarsia-metal-o6.commit
cargo test --release --locked -p intarsia-models --test inkling2_batch --test inkling2 --no-run 2>&1 | grep -aE "Executable|error" | head -4
echo "[endor] done $(date +%T) cpu exit $rc"
exit $rc
