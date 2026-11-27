#!/usr/bin/env bash
# Mac side of window S4 (lane ink-hip-overlap, checks only), run as the fleetq command: sync, check the binaries, window.sh + stage_s4.sh.
set -uo pipefail
R=/opt/code/intarsiaml/.worktrees/ink-hip-overlap/docs/results/2026-11-26-inkling2-hip-overlap/raw
mkdir -p "$R/s4"
bash "$R/sync.sh" || { echo "sync failed: no window"; exit 3; }
ssh -o BatchMode=yes tatooine 'test -x ~/intarsiaml-ink-hip-bpf/.scratch/bin/intarsia-o4 && test -x ~/intarsiaml-ink-hip-bpf/.scratch/bin/intarsia-o5' < /dev/null || { echo "binaries missing: no window"; exit 3; }
ssh -o BatchMode=yes tatooine 'CAP=235 bash ~/intarsiaml-ink-hip-bpf/.scratch/window.sh ~/intarsiaml-ink-hip-bpf/.scratch/stage_s4.sh' < /dev/null > "$R/s4/win.out" 2>&1
rc=$?
rsync -a tatooine:intarsiaml-ink-hip-bpf/logs/s4/ "$R/s4/card/" < /dev/null
echo "window exit $rc"; grep -aE '^\[window\]' "$R/s4/win.out"
exit $rc
