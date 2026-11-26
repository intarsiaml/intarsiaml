# sourced by the ink-busy2048 job scripts
W=/opt/code/intarsiaml/.worktrees/ink-batch; N=$W/.scratch/busy2048; R=$W/.scratch/base; TB=$W/.scratch/base-target; TN=$N/target; O=$W/.scratch/b2048
CK=/Users/josef/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/b85c3cfa615fc413cc9af3379f491b963f3b586a
MAIN=0b8fab60015238e3ecd162f60d8494217931d1be
for v in $(env | grep -oE '^INTARSIA_[A-Z0-9_]+'); do unset $v; done
# run a command in the background and wait, so a TERM trap fires at once and can kill the tree
CP=""
run() { "$@" & CP=$!; wait $CP; local rc=$?; CP=""; return $rc; }
runlog() { local lf=$1; shift; "$@" > "$lf" 2>&1 & CP=$!; wait $CP; local rc=$?; CP=""; echo "[$(date +%T)] exit $rc: $*" | cut -c1-200; return $rc; }
killtree() { local p=$1 c; for c in $(pgrep -P "$p" 2>/dev/null); do killtree "$c"; done; kill -TERM "$p" 2>/dev/null; }
on_term() { echo "TERM received: killing process tree"; [ -n "$CP" ] && killtree "$CP"; type stop_group >/dev/null 2>&1 && stop_group; exit 143; }
trap on_term TERM INT
