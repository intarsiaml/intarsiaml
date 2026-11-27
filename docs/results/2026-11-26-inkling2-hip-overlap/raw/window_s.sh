#!/usr/bin/env bash
# Mac side of one paired-speed window (lane ink-hip-overlap), run as the fleetq command: sync, check the 3 binaries, then serve_window.sh.
#   window_s.sh <N>   (N = 2 or 3; visits-sN.txt, stage_sNbg.sh, REAL_OFF = N - 1)
set -uo pipefail
N=$1
R=/opt/code/intarsiaml/.worktrees/ink-hip-overlap/docs/results/2026-11-26-inkling2-hip-overlap/raw
bash "$R/sync.sh" || { echo "sync failed: no window"; exit 3; }
ssh -o BatchMode=yes tatooine 'test -x ~/intarsiaml-ink-hip-bpf/.scratch/bin/intarsia-o4 && test -x ~/intarsiaml-ink-hip-bpf/.scratch/bin/intarsia-main5 && test -x ~/intarsiaml-ink-hip-bpf/.scratch/bin/intarsia-o5' < /dev/null || { echo "binaries missing: no window"; exit 3; }
exec env VISITS=$R/visits-s$N.txt OUT=$R/s$N CAP=250 LE_MIN=20 LE_MARGIN=6 REAL_OFF=$(( N - 1 )) BG=stage_s${N}bg.sh BG_CAP=240 bash "$R/serve_window.sh"
