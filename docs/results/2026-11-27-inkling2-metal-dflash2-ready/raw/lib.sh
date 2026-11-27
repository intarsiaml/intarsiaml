# sourced by slotA.sh / slotB.sh (lane ink-metal-dflash2-ready)
kill_tree() { local c; for c in $(pgrep -P $1 2>/dev/null); do kill_tree $c; done; kill $1 2>/dev/null; }
cleanup() { for c in $(pgrep -P $$ 2>/dev/null); do kill_tree $c; done; }
trap cleanup EXIT; trap 'cleanup; exit 130' INT TERM
for v in $(env | grep -oE '^INTARSIA_[A-Z0-9_]+'); do unset $v; done
W=/opt/code/intarsiaml/.worktrees/ink-metal-dflash2-ready; S=$W/.scratch; R=$W/docs/results/2026-11-27-inkling2-metal-dflash2-ready/raw
M=$HOME/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/b85c3cfa615fc413cc9af3379f491b963f3b586a
D=$HOME/.cache/huggingface/hub/models--incoai--Inkling-2-Mini-DFlash2/snapshots/4f9088ea66c21f2b99ed657d1941c306b03e84d2
B=$S/bin/intarsia-br1
others=$(pgrep -fl 'intarsia[^ ]* (serve|generate)' | head -3)
[ -n "$others" ] && { echo "GUARD: another intarsia run on the box: $others"; exit 1; }
flags() { case $1 in off) echo "--spec off" ;; default) echo "" ;; dfl*) echo "--spec dflash --spec-draft $D --spec-k ${1#dfl}" ;; esac; }
# serve_start NAME ARM PORT "EXTRA FLAGS" "ENV ASSIGNMENTS"  -> sets SPID (the time wrapper); log $O/s-NAME.{out,err}
serve_start() {
  local name=$1 arm=$2 port=$3 extra=$4 envs=$5
  env $envs /usr/bin/time -l $B serve -m $M --backend metal --no-supervisor --no-ui --port $port $(flags $arm) $extra > $O/s-$name.out 2> $O/s-$name.err &
  SPID=$!
}
serve_stop() {
  local c n=0; for c in $(pgrep -P $SPID 2>/dev/null); do kill -TERM $c; done
  while kill -0 $SPID 2>/dev/null && [ $n -lt 40 ]; do sleep 0.5; n=$((n+1)); done
  kill -0 $SPID 2>/dev/null && { for c in $(pgrep -P $SPID 2>/dev/null); do kill -KILL $c; done; echo "serve_stop: KILL needed"; }
  wait $SPID 2>/dev/null
}
