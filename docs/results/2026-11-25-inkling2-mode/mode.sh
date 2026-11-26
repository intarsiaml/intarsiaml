#!/bin/bash
# endor, EXCLUSIVE fleetq slot: lane ink-mode. Many fresh `generate` processes (1k / 4k / 16k prefill + 32 greedy tokens each,
# --spec off), each with the diagnostic INTARSIA_METAL_MODE_PROBE (state + microbench) and, around it, the box state:
# foreign per-process GPU time (ioreg AGXDeviceUserClient), vm_stat counters, swap, thermal, and a macmon stream for the slot.
# Arms, interleaved per cycle  A N A L:
#   A  default (weights wrapped zero-copy), probe after the run
#   N  INTARSIA_METAL_NO_WRAP=1 (weights copied into newBufferWithLength buffers), probe after the run
#   L  default, probe after load AND after the run (does a GPU warm-up before the first prefill change the mode)
# Time-boxed: stops starting processes after $BUDGET_S seconds.
set -u
: "${FLEETQ_JOB:?run under fleetq}"
kill_tree() { local c; for c in $(pgrep -P $1 2>/dev/null); do kill_tree $c; done; kill $1 2>/dev/null; }
cleanup() { for c in $(pgrep -P $$ 2>/dev/null); do kill_tree $c; done; }
trap cleanup EXIT; trap 'cleanup; exit 130' INT TERM
for v in $(env | grep -oE '^INTARSIA_[A-Z0-9_]+'); do unset $v; done
W=/opt/code/intarsiaml/.worktrees/ink-mode; S=$W/.scratch/mode
O=$S/run-${1:-$(date +%m%d-%H%M)}; mkdir -p $O
BUDGET_S=${BUDGET_S:-1380}
B=$S/bin/intarsia-probe; [ -x $B ] || { echo "ABORT: no $B"; exit 4; }
CK=/Users/josef/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/b85c3cfa615fc413cc9af3379f491b963f3b586a
PIDS=/opt/code/intarsiaml/.worktrees/ink-spec-flip/.scratch/longp.json   # 1021 / 4108 / 16455 tokens
[ -f $PIDS ] || { echo "ABORT: no $PIDS"; exit 4; }
others=$(pgrep -fl 'intarsia[^ ]* (serve|generate)' || true)
[ -n "$others" ] && { echo "ABORT: other engine processes on the box:"; echo "$others"; exit 5; }
echo "binary $B md5 $(md5 -q $B) out $O start $(date +%T)"
macmon pipe -i 250 > $O/macmon.jsonl 2>/dev/null &
MM=$!
T0=$(date +%s)
G="generate -m $CK --backend metal --max-seq 20480 --prompt-ids $PIDS --temperature 0 --max-tokens 32 --ignore-eos --show-ids --profile-every 0 --spec off"
run() { # arm tag
  local arm=$1 tag=$2 f=$O/$2-$1
  python3 $S/gputime.py > $f.g0.json
  vm_stat > $f.vm0; sysctl -n vm.swapusage > $f.sys0; pmset -g therm >> $f.sys0 2>&1
  local t_start=$(python3 -c 'import time; print(f"{time.time():.3f}")')
  local pe=end pl=load,end; [ $PROBE_OK = 1 ] || { pe=0; pl=0; }
  case $arm in
    A) env INTARSIA_METAL_MODE_PROBE=$pe $B $G > $f.out 2> $f.err & ;;
    N) env INTARSIA_METAL_MODE_PROBE=$pe INTARSIA_METAL_NO_WRAP=1 $B $G > $f.out 2> $f.err & ;;
    L) env INTARSIA_METAL_MODE_PROBE=$pl INTARSIA_METAL_MODE_PROBE_SUSTAIN_MS=4000 $B $G > $f.out 2> $f.err & ;;
  esac
  local p=$!; wait $p; local rc=$?
  local t_end=$(python3 -c 'import time; print(f"{time.time():.3f}")')
  python3 $S/gputime.py > $f.g1.json
  vm_stat > $f.vm1
  echo "$t_start $t_end" > $f.t
  python3 $S/gputime.py $f.g0.json $f.g1.json $p > $f.foreign
  echo "$tag $arm rc $rc | $(grep -hE '^prefill\[[0-9]\]' $f.err | sed -E 's/prefill\[([0-9])\]: ([0-9]+) tokens, ([0-9]+) ms/p\1 \3/' | tr '\n' ' ')| $(grep -h 'decode overall' $f.err | grep -oE '[0-9.]+ tok/s') | foreign $(cut -c1-120 $f.foreign)"
}
PROBE_OK=1
run A warm
grep -q '^modeprobe end' $O/warm-A.err || { PROBE_OK=0; echo "WARN: no modeprobe line in the warm run, probe off from here"; tail -3 $O/warm-A.err; }
grep -q '^prefill\[2\]' $O/warm-A.err || { echo "ABORT: warm run did not finish"; tail -5 $O/warm-A.err; exit 9; }
ARMS="A N A L"
run N warmN
grep -q '^prefill\[2\]' $O/warmN-N.err || { ARMS="A A A L"; echo "WARN: NO_WRAP arm failed, dropped"; tail -3 $O/warmN-N.err; }
n=0
while :; do
  for arm in $ARMS; do
    [ $(( $(date +%s) - T0 )) -ge $BUDGET_S ] && break 2
    n=$((n+1)); run $arm $(printf %03d $n)
  done
done
kill $MM 2>/dev/null
echo "== done $(date +%T), $n processes"
