#!/usr/bin/env bash
# storeflip: sync this worktree (tracked + untracked, uncommitted edits included) to spark2 ~/intarsiaml-storeflip/src and build there.
# The free-memory watchdog runs on the Spark (rbuild.sh), so a dropped ssh does not stop it.
#   FLEETQ_OWNER=claude-storeflip fleetq run --host spark2 --load cpu --mem 55 --expect 40 --label "storeflip build" -- bash scripts/storeflip/build.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
L='~/intarsiaml-storeflip'; R="$L/src"; BH=spark2
rsh() { ssh -n -o BatchMode=yes "$BH" "$@"; }
rsh "mkdir -p $R $L/bin $L/out $L/stage && rm -rf $L/stage/*"
( cd "$REPO" && COPYFILE_DISABLE=1 git ls-files -z --cached --others --exclude-standard | grep -zv '^docs/results/' | COPYFILE_DISABLE=1 tar --null -T - -cf - ) | ssh -o BatchMode=yes "$BH" "tar -x -C $L/stage"
rsh "find $L/stage -name '._*' -delete"   # macOS AppleDouble files: the CUDA build.rs would compile ._foo.cu
rsh "rsync -a --delete --exclude target --exclude golden --exclude tools/.venv --exclude out $L/stage/ $R/ && rm -rf $L/stage"
scp -q -o BatchMode=yes "$REPO/scripts/storeflip/rbuild.sh" "$BH:intarsiaml-storeflip/out/rbuild.sh"
STAMP="storeflip-$(git -C "$REPO" rev-parse --short=12 HEAD)-$(date +%H%M)"
echo "build start $(date +%T) $STAMP"
rsh "nohup bash $L/out/rbuild.sh $STAMP > $L/out/rbuild.log 2>&1 < /dev/null & echo started"
sleep 5
while rsh "pgrep -f '[i]ntarsiaml-storeflip/out/rbuild.sh' > /dev/null"; do
  echo "$(date +%T) $(rsh 'tail -1 ~/intarsiaml-storeflip/out/build-mem.txt; tail -c 150 ~/intarsiaml-storeflip/out/build.log | tr "\r\n" "  "')"
  sleep 60
done
rsh "cat $L/out/rbuild.log"
# gate 5 (Linux): the prefix store unit tests on the Spark, in the same --load cpu slot (no server beside them)
scp -q -o BatchMode=yes "$REPO/scripts/storeflip/rtest.sh" "$BH:intarsiaml-storeflip/out/rtest.sh"
rsh "bash $L/out/rtest.sh"
