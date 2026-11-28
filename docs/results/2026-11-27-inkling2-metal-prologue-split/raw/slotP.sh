#!/bin/bash
# endor EXCLUSIVE fleetq slot P, lane ink-prologue2: gates P1-P3 of docs/results/2026-11-27-inkling2-metal-prologue-split.md on one
# binary (this branch; arm on = INTARSIA_INK_PROLOGUE_SPLIT=1, off = unset, today's default). Speed: ABBA blocks with 45 s idle
# gaps before every timed run (endor's GPU burst depends on its power history), macmon logged.
set -u
: "${FLEETQ_JOB:?run under fleetq}"
kill_tree() { local c; for c in $(pgrep -P $1 2>/dev/null); do kill_tree $c; done; kill $1 2>/dev/null; }
cleanup() { for c in $(pgrep -P $$ 2>/dev/null); do kill_tree $c; done; }
trap cleanup EXIT; trap 'cleanup; exit 130' INT TERM
for v in $(env | grep -oE '^INTARSIA_[A-Z0-9_]+'); do unset $v; done
W=/opt/code/intarsiaml/.worktrees/ink-prologue2; S=$W/.scratch; R=$W/docs/results/2026-11-27-inkling2-metal-prologue-split/raw
M=$HOME/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/b85c3cfa615fc413cc9af3379f491b963f3b586a
B=$S/bin/intarsia-ps
others=$(pgrep -fl '^[^ ]*/intarsia[^ /]* (serve|generate)' | head -3)
[ -n "$others" ] && { echo "GUARD: another intarsia run on the box: $others"; exit 1; }
O=$R/P; mkdir -p $O; T0=$(date +%s)
echo "slot P start $(date +%T) binary md5 $(md5 -q $B) commit $(cat $B.commit) git $(git -C $W rev-parse --short HEAD)"
macmon pipe -i 250 > $S/macmon-P.jsonl 2>/dev/null &
envs() { case $1 in on) echo "INTARSIA_INK_PROLOGUE_SPLIT=1" ;; *) echo "" ;; esac; }
G() { env $(envs $1) $B generate -m $M --backend metal --profile-every 0 --show-ids --spec off "${@:2}"; }
# traced (the env prefix stays inside env: `VAR=x func` can persist in a shell function call)
GT() { env $(envs $1) INTARSIA_INK_PREFILL_TRACE=1 $B generate -m $M --backend metal --profile-every 0 --show-ids --spec off "${@:2}"; }
echo "== P1: bitwise test (split vs ink_kv_prologue_bf16, 17..2048 rows)"
$S/bin/t-ps --test-threads=1 > $O/p1.txt 2>&1
echo "p1 rc $? | $(grep -aE '^test result' $O/p1.txt | tail -1) | $(grep -aE 'panicked' $O/p1.txt | head -2 | cut -c1-300)"
echo "== P2: greedy ids, --spec off, p1k + len4k + len16k, 64 tokens, --ignore-eos (also the warm-up of both arms' kernels)"
for arm in off on; do
  GT $arm --max-seq 20480 --max-tokens 64 --ignore-eos --prompt-ids $R/pf3.json > $O/g-$arm.out 2> $O/g-$arm.err
  echo "g-$arm rc $? t+$(( $(date +%s) - T0 ))s | $(grep -aE '^prefill\[' $O/g-$arm.err | tr '\n' ' ' | cut -c1-300) | split traces: $(grep -ac 'prologue split' $O/g-$arm.err)"
done
python3 $R/md5.py $O/g-off.err $O/g-on.err
echo "== P3: prefill speed, ABBA x 3 (off on on off / on off off on / off on on off), 45 s idle before each run, 1 token"
mkdir -p $O/t
n=0
for arm in off on on off  on off off on  off on on off; do
  n=$((n+1)); f=$O/t/r$(printf %02d $n)-$arm
  sleep 45
  echo "r$n $arm start $(date +%T)" >> $O/t/marks.txt
  G $arm --max-seq 20480 --max-tokens 1 --prompt-ids $R/pf3.json > $f.out 2> $f.err
  echo "r$n $arm rc $? t+$(( $(date +%s) - T0 ))s | $(grep -aE '^prefill\[' $f.err | tr '\n' ' ' | cut -c1-300)"
  echo "r$n $arm end $(date +%T)" >> $O/t/marks.txt
done
python3 $R/pfabba.py $O/t
echo "slot P end $(date +%T) t+$(( $(date +%s) - T0 ))s"
