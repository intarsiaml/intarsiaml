#!/usr/bin/env bash
# From the Mac, inside `fleetq run --host tatooine-r9700 --load cpu` (lane ink-hip-overlap2, step 2): sync the merged head to tatooine's
# lane tree, refresh .scratch/main-src to origin/main (files main deleted since the old COMMIT removed, the rest extracted over it),
# then build o6 (the merged head, hip) + its tiny hip test binaries, and main6 (origin/main) + main's.
set -uo pipefail
W=/opt/code/intarsiaml/.worktrees/ink-hip-overlap; cd "$W"
MAIN=${MAIN:-origin/main}; T=tatooine; MS=intarsiaml-ink-hip-bpf/.scratch/main-src
bash docs/results/2026-11-26-inkling2-hip-overlap/raw/sync.sh || exit 1
old=$(ssh -o BatchMode=yes $T "cat $MS/COMMIT" < /dev/null); new=$(git rev-parse --short=9 $MAIN)
echo "[build6] head $(git rev-parse --short=9 HEAD); main-src $old -> $new $(date +%T)"
git diff --name-only --diff-filter=D "$old" "$new" > .scratch/main-src-deleted.txt
[ -s .scratch/main-src-deleted.txt ] && ssh -o BatchMode=yes $T "cd $MS && xargs -r rm -f" < .scratch/main-src-deleted.txt
git archive "$new" | ssh -o BatchMode=yes $T "tar -x -C $MS && echo $new > $MS/COMMIT" || exit 1
ssh -o BatchMode=yes $T "bash intarsiaml-ink-hip-bpf/.scratch/build.sh o6 m:inkling2_batch m:inkling2_packed m:inkling2" < /dev/null; rc=$?
echo "[build6] o6 exit $rc $(date +%T)"; [ $rc -ne 0 ] && exit $rc
ssh -o BatchMode=yes $T "bash intarsiaml-ink-hip-bpf/.scratch/build_main_tests.sh main6" < /dev/null; rc=$?
echo "[build6] main6 exit $rc $(date +%T)"
exit $rc
