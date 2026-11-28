# sourced by slotD.sh (lane ink-metal-dgemv; from lane ink-metal-flip2's lib.sh)
kill_tree() { local c; for c in $(pgrep -P $1 2>/dev/null); do kill_tree $c; done; kill $1 2>/dev/null; }
cleanup() { for c in $(pgrep -P $$ 2>/dev/null); do kill_tree $c; done; }
trap cleanup EXIT; trap 'cleanup; exit 130' INT TERM
for v in $(env | grep -oE '^INTARSIA_[A-Z0-9_]+'); do unset $v; done
W=/opt/code/intarsiaml/.worktrees/ink-metal-dgemv; S=$W/.scratch; R=$W/docs/results/2026-11-28-inkling2-metal-dgemv/raw
M=$HOME/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/b85c3cfa615fc413cc9af3379f491b963f3b586a
B=$S/bin/intarsia-dg
others=$(pgrep -fl '^[^ ]*/intarsia[^ /]* (serve|generate)' | head -3)
[ -n "$others" ] && { echo "GUARD: another intarsia run on the box: $others"; exit 1; }
[ -e "$M/../Inkling-2-Mini-DFlash2/config.json" ] || { echo "GUARD: no drafter link beside the checkpoint"; exit 1; }
# arms (one binary, main + the opt-in device top-k, which no arm turns on): dflt = the Metal default (DFlash2 k7 + cut 0.3; the draft's
# block pass and head on the row-exact `_rows` kernels); rk0 = INTARSIA_DFLASH_ROWS_KERNELS=0 (the draft on the ordinary kernel picks);
# off = --spec off
flags() { case $1 in off) echo "--spec off" ;; *) echo "" ;; esac; }
envs() { case $1 in rk0) echo "INTARSIA_DFLASH_ROWS_KERNELS=0" ;; *) echo "" ;; esac; }
serve_start() {
  local name=$1 arm=$2 port=$3 extra=$4
  env $(envs $arm) /usr/bin/time -l $B serve -m $M --backend metal --no-supervisor --no-ui --port $port $(flags $arm) $extra > $O/s-$name.out 2> $O/s-$name.err &
  SPID=$!
}
serve_stop() {
  local c n=0; for c in $(pgrep -P $SPID 2>/dev/null); do kill -TERM $c; done
  while kill -0 $SPID 2>/dev/null && [ $n -lt 40 ]; do sleep 0.5; n=$((n+1)); done
  kill -0 $SPID 2>/dev/null && { for c in $(pgrep -P $SPID 2>/dev/null); do kill -KILL $c; done; echo "serve_stop: KILL needed"; }
  wait $SPID 2>/dev/null
}
