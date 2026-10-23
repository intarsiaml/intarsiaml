#!/usr/bin/env bash
# The exclusive spark1 window of this lane (run on endor under `fleetq run --host spark1 --exclusive`): per-pass phase
# timings, then three interleaved localeval passes over the arms.
set -uo pipefail
: "${FLEETQ_JOB:?}"
D=docs/results/2026-10-22-mtp-spec-decode
cd "$(dirname "${BASH_SOURCE[0]}")/../../../.."
REF=${REF:?}
scripts/deploy.sh spark1 --backend cuda --ref "$REF" --tree mtp --smoke quick --no-build -- bash $D/scripts/phases.sh target/release/intarsia replay-out/mtp-phases
rsync -a spark1:intarsia-deploy/mtp/replay-out/mtp-phases/ $D/raw/phases/
ARMS="off mtp1 mtp2 mtp3" PASS=p1 bash $D/scripts/window.sh
ARMS="mtp3 mtp2 mtp1 off" PASS=p2 bash $D/scripts/window.sh
ARMS="mtp2 off mtp3 mtp1" PASS=p3 bash $D/scripts/window.sh
