#!/usr/bin/env bash
# Lane S1: the whole spark2 exclusive job, from this Mac:
#   fleetq run --host spark2 --exclusive --takes-down retro --mem 100 --expect 120 --label "..." -- scripts/s1/window-job.sh
# 1. build s1-base / s1-dev on spark2 (vLLM up);
# 2. the vLLM-down window (scripts/s1/window.sh: agent-harness check, uncapped to 07:00 EDT or 24 min by day, vLLM restored
#    in the EXIT trap) running scripts/s1/wALL.sh: the intarsia bench A/B, plus localeval per arm at night;
# 3. with vLLM back up (the box still exclusive): the gates (scripts/s1/remote.sh STEPS, default "ids prof acc ktest").
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$S/../../docs/results/2026-10-22-s1-spark-prefill/raw"; mkdir -p "$LOG"
echo "== build $(date +%T)"
HOST=spark2 bash "$S/build.sh" 2>&1 | tee "$LOG/w-build.out"
ssh -n spark2 'for t in s1-base s1-dev; do test -x ~/intarsia-decfuse/$t/target/acc/release/intarsia || exit 1; done' || { echo "build missing"; exit 1; }
echo "== window $(date +%T)"
mnow=$(( 10#$(TZ=America/New_York date +%H) * 60 + 10#$(TZ=America/New_York date +%M) ))
if (( mnow < 6 * 60 )); then  # an hour or more of the night window left: A/B and localeval
  LOCALEVAL_ARMS="base dev cub" bash "$S/window.sh" ${WIN_MIN:-55} "S1 spark2 A/B + localeval (base/dev/cub)" \
    '~/intarsia-decfuse/s1-dev/scripts/s1/wALL.sh "ab le"' 2>&1 | tee "$LOG/w-window.out"
else  # a 30-minute day window: the A/B only
  bash "$S/window.sh" 24 "S1 spark2 A/B base/dev/cub (30 min day window)" \
    '~/intarsia-decfuse/s1-dev/scripts/s1/wALL.sh ab' 2>&1 | tee "$LOG/w-window.out"
fi
if [[ "${GATES:-1}" == 1 ]]; then
  echo "== gates (vLLM up) $(date +%T)"
  ssh -n -o BatchMode=yes spark2 "export FLEETQ_JOB=$FLEETQ_JOB OUTNAME=s1G STEPS='${GSTEPS:-ids prof acc ktest}'; bash ~/intarsia-decfuse/s1-dev/scripts/s1/remote.sh" 2>&1 | tee "$LOG/w-gates.out"
fi
echo "== window-job done $(date +%T)"
