#!/usr/bin/env bash
# Mac side of one window of lane ink-hip-overlap2, run as the fleetq command: sync, check the binaries, then serve_window.sh with the
# real-prompt client on tatooine (REAL_REMOTE: its cores, the gate-4 amendment).
#   window_r.sh w0      checks + priming (visits-w0.txt, stage_w0bg.sh on card 1)
#   window_r.sh <N>     paired speed window S<N> (N = 6, 7, 8; visits-sN.txt, stage_sNbg.sh, REAL_OFF = N - 6)
set -uo pipefail
N=$1
R=/opt/code/intarsiaml/.worktrees/ink-hip-overlap/docs/results/2026-11-26-inkling2-hip-overlap/raw
CPUS=${CPUS:-10,11,22,23}
bash "$R/sync.sh" || { echo "sync failed: no window"; exit 3; }
ssh -o BatchMode=yes tatooine 'test -x ~/intarsiaml-ink-hip-bpf/.scratch/bin/intarsia-o6 && test -x ~/intarsiaml-ink-hip-bpf/.scratch/bin/intarsia-main6' < /dev/null || { echo "binaries missing: no window"; exit 3; }
if [ "$N" = w0 ]; then
  exec env VISITS=$R/visits-w0.txt OUT=$R/w0 CAP=255 REAL_REMOTE=$CPUS BG=stage_w0bg.sh BG_CAP=240 bash "$R/serve_window.sh"
fi
exec env VISITS=$R/visits-s$N.txt OUT=$R/s$N CAP=250 LE_MIN=20 LE_MARGIN=6 REAL_OFF=$(( N - 6 )) REAL_REMOTE=$CPUS BG=stage_s${N}bg.sh BG_CAP=235 bash "$R/serve_window.sh"
