#!/bin/bash
# endor EXCLUSIVE: lane ink-mode confirmation. Same binary, no probe, `generate` 1k/4k/16k --max-tokens 1 --spec off.
# Arms per cycle: C = after 45 s of GPU idle (cold), H = right after C (hot). Logs battery InstantAmperage (ioreg, ~2 Hz) + macmon.
set -u
: "${FLEETQ_JOB:?run under fleetq}"
kill_tree() { local c; for c in $(pgrep -P $1 2>/dev/null); do kill_tree $c; done; kill $1 2>/dev/null; }
cleanup() { for c in $(pgrep -P $$ 2>/dev/null); do kill_tree $c; done; }
trap cleanup EXIT; trap 'cleanup; exit 130' INT TERM
for v in $(env | grep -oE '^INTARSIA_[A-Z0-9_]+'); do unset $v; done
S=/opt/code/intarsiaml/.worktrees/ink-mode/.scratch/mode; O=$S/run-${1:-conf}; mkdir -p $O
BUDGET_S=${BUDGET_S:-1440}; IDLE=${IDLE:-45}
B=$S/bin/intarsia-probe
CK=/Users/josef/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/b85c3cfa615fc413cc9af3379f491b963f3b586a
PIDS=/opt/code/intarsiaml/.worktrees/ink-spec-flip/.scratch/longp.json
others=$(pgrep -fl 'intarsia[^ ]* (serve|generate)' || true)
[ -n "$others" ] && { echo "ABORT: other engine processes"; echo "$others"; exit 5; }
echo "binary md5 $(md5 -q $B) out $O start $(date +%T) idle $IDLE"
macmon pipe -i 250 > $O/macmon.jsonl 2>/dev/null & MM=$!
( while :; do a=$(ioreg -rn AppleSmartBattery | grep -E '"InstantAmperage" =' | grep -oE '[-0-9]+$'); echo "$(python3 -c 'import time;print(f"{time.time():.2f}")') $a"; sleep 0.3; done > $O/batt.txt ) & BT=$!
T0=$(date +%s)
G="generate -m $CK --backend metal --max-seq 20480 --prompt-ids $PIDS --temperature 0 --max-tokens 1 --profile-every 0 --spec off"
run() { local f=$O/$2-$1
  local t0=$(python3 -c 'import time;print(f"{time.time():.3f}")')
  $B $G > $f.out 2> $f.err; local rc=$?
  echo "$t0 $(python3 -c 'import time;print(f"{time.time():.3f}")')" > $f.t
  echo "$2 $1 rc $rc | $(grep -hE '^prefill\[[0-9]\]' $f.err | sed -E 's/prefill\[([0-9])\]: ([0-9]+) tokens, ([0-9]+) ms/p\1 \3/' | tr '\n' ' ')"
}
run H warm
n=0
while [ $(( $(date +%s) - T0 + IDLE + 25 )) -lt $BUDGET_S ]; do
  n=$((n+1)); sleep $IDLE; run C $(printf %03d $n); run H $(printf %03d $n)
done
kill $MM $BT 2>/dev/null
echo "== done $(date +%T), $n cycles"
