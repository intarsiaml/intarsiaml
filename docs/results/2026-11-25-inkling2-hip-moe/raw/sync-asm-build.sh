#!/usr/bin/env bash
# From endor: copy the worktree to tatooine, dump the kernels' VGPR counts, build the test binary (CPU only).
set -uo pipefail
cd /opt/code/intarsiaml/.worktrees/ink-hip-moe
rsync -a --delete --exclude /target --exclude /.git --exclude /.worktrees --exclude /logs ./ tatooine:intarsiaml-ink-hip-moe/ < /dev/null || exit 90
ssh -o BatchMode=yes tatooine 'bash ~/intarsiaml-ink-hip-moe/.scratch/asm.sh' < /dev/null
echo "asm exit $?"
[ "${ASM_ONLY:-0}" = 1 ] && exit 0
exec ssh -o BatchMode=yes tatooine 'bash ~/intarsiaml-ink-hip-moe/.scratch/build.sh' < /dev/null
