# sourced by slotR.sh (lane ink-dflash-cut; from lane ink-metal-solo's lib.sh)
kill_tree() { local c; for c in $(pgrep -P $1 2>/dev/null); do kill_tree $c; done; kill $1 2>/dev/null; }
cleanup() { for c in $(pgrep -P $$ 2>/dev/null); do kill_tree $c; done; }
trap cleanup EXIT; trap 'cleanup; exit 130' INT TERM
for v in $(env | grep -oE '^INTARSIA_[A-Z0-9_]+'); do unset $v; done
W=/opt/code/intarsiaml/.worktrees/ink-dflash-cut; S=$W/.scratch; R=$W/docs/results/2026-11-27-inkling2-dflash-cut/raw
M=$HOME/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/b85c3cfa615fc413cc9af3379f491b963f3b586a
D=$HOME/.cache/huggingface/hub/models--incoai--Inkling-2-Mini-DFlash2/snapshots/4f9088ea66c21f2b99ed657d1941c306b03e84d2
B=$S/bin/intarsia-ct
others=$(pgrep -fl '^[^ ]*/intarsia[^ /]* (serve|generate)' | head -3)
[ -n "$others" ] && { echo "GUARD: another intarsia run on the box: $others"; exit 1; }
# arms: off = --spec off; dfl5 = the Metal default (no --spec: DFlash2 k5 alone); cut7 = DFlash2 k7 + INTARSIA_DFLASH_CUT=0.3
flags() { case $1 in off) echo "--spec off" ;; dfl5) echo "" ;; cut7) echo "--spec dflash --spec-draft $D --spec-k 7" ;; esac; }
envs() { case $1 in cut7) echo "INTARSIA_DFLASH_CUT=0.3" ;; *) echo "" ;; esac; }
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
