#!/usr/bin/env bash
# Lane l7m, LOCAL, the command of ONE fleetq job that holds both tatooine hosts and takes radiance down (one outage for everything):
#   fleetq run --host tatooine-r9700,tatooine-r9700-2 --mem tatooine-r9700=4,tatooine-r9700-2=4 --quiet gpu,cpu --takes-down radiance-rank0,radiance-rank1 -- scripts/l7m/window.sh
# order: OLD (unchanged test on origin/main) -> BIT (new test, macro 4096/8192/16384 vs off) -> NEG (none, nomacro, ulp) -> IDS (256 ids) -> served A/B main vs branch x3
# (single-stream, then --max-batch 16), card 1. Radiance is stopped once at the start and restored once at the end.
set -uo pipefail
export CARD=${CARD:-1}
W='bash ~/intarsiaml-l7m-hip/scripts/l7m/win.sh'
trap "ssh tatooine '$W restore'" EXIT
ssh tatooine "$W stop"
ssh tatooine "NOWIN=1 CARD=$CARD bash ~/intarsiaml-l7m-hip/scripts/l7m/gates.sh OLD BIT NEG IDS" 2>&1
echo "[window] gates done $(date +%T)"
cd "$(dirname "$0")/../.." || exit 1
NOWIN=1 CARD=$CARD scripts/l7m/ab.sh "main|base||" "branch|dev||" "main|base||--max-batch 16" "branch|dev||--max-batch 16" \
  "main|base||" "branch|dev||" "main|base||--max-batch 16" "branch|dev||--max-batch 16" \
  "main|base||" "branch|dev||" "main|base||--max-batch 16" "branch|dev||--max-batch 16" @l7mf 2>&1
echo "[window] done $(date +%T)"
