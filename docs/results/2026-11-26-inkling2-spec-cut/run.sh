#!/bin/bash
# endor exclusive slot: lane ink-spec-cut. Within-process A/B of the DSpark draft rule: A = gate 0.8 (today), B = cut 0.5, both k3.
# One process per length class; the diag binary alternates the rule per request (INTARSIA_DIAG_ARMS), cycle 0 = warm-up.
set -u
: "${FLEETQ_JOB:?run under fleetq}"
kill_tree() { local c; for c in $(pgrep -P $1 2>/dev/null); do kill_tree $c; done; kill $1 2>/dev/null; }
cleanup() { for c in $(pgrep -P $$ 2>/dev/null); do kill_tree $c; done; }
trap cleanup EXIT; trap 'cleanup; exit 130' INT TERM
for v in $(env | grep -oE '^INTARSIA_[A-Z0-9_]+'); do unset $v; done
W=/opt/code/intarsiaml/.worktrees/ink-spec-cut; S=$W/.scratch; O=$S/out; mkdir -p $O
M=$HOME/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/b85c3cfa615fc413cc9af3379f491b963f3b586a
B=$S/bin/intarsia-diag
others=$(pgrep -fl 'intarsia[^ ]* (serve|generate)' | grep -v "^$$ " | head -3)
[ -n "$others" ] && { echo "GUARD: another intarsia run on the box: $others"; exit 1; }
echo "start $(date +%T) md5 $(md5 -q $B) load $(uptime | sed 's/.*averages: //')"
G="generate -m $M --backend metal --profile-every 0 --max-seq 20480 --show-ids --max-tokens 128"
D="--spec dflash --spec-draft $M --spec-k 3"
ARMS="g3=0.8,0;c5=0,0.5"
echo "== spec off reference $(date +%T)"
for cls in short len1k len4k len16k; do
  env $B $G --prompt-ids $S/$cls.json --spec off > /dev/null 2> $O/off.$cls.err || echo "  off $cls rc $?"
done
echo "== diag A/B $(date +%T)"
for pass in 1 2; do
  for spec in "short 10" "len1k 20" "len4k 14" "len16k 8"; do
    set -- $spec; cls=$1; n=$2
    env INTARSIA_DIAG_ARMS="$ARMS" INTARSIA_DIAG_CYCLES=$n $B $G --prompt-ids $S/$cls.json $D > /dev/null 2> $O/ab.p$pass.$cls.err || echo "  p$pass $cls rc $?"
    echo "  p$pass $cls done $(date +%T) load $(uptime | sed 's/.*averages: //')"
  done
done
echo "end $(date +%T)"
