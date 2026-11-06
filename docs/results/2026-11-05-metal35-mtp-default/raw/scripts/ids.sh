#!/bin/zsh
# greedy ids (256, --ignore-eos), --spec off vs the recipe default (no --spec flag) vs explicit --spec mtp --spec-k 2 (ARMS), 35B Metal, generate. Inside a fleetq slot.
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
W=/opt/code/intarsiaml/.worktrees/flip-metal-mtp; BIN=${BIN:-$W/.scratch/intarsia.branch}; O=$W/.scratch/logs/${OUT:-ids}; mkdir -p $O
M=/Users/josef/models/Qwen3.6-35B-A3B-NVFP4; P=$W/.scratch/ids-prompts; fails=0; n=0
for f in ${=NAMES:-$(cd $P; ls | sed 's/\.txt$//')}; do
  case $f in *32764*) MS=34000;; *16k*|*16384*|*12k*) MS=20480;; *4k*|*glong*) MS=8192;; *) MS=4096;; esac
  MODE=(); case $f in chat-*) MODE=(--chat --no-think);; esac
  for arm in ${=ARMS:-off default k2}; do
    case $arm in off) A=(--spec off);; default) A=();; k2) A=(--spec mtp --spec-k 2);; esac
    $BIN generate --model $M --backend metal --max-seq $MS $A $MODE --temperature 0 --profile-every 0 --ignore-eos --show-ids -p "$(cat $P/$f.txt)" --max-tokens 256 < /dev/null > $O/$f-$arm.out 2> $O/$f-$arm.err
    grep -a '^ids:' $O/$f-$arm.err > $O/$f-$arm.ids
  done
  pt=$(grep -a -o 'prompt[ _a-z]*[=:] *[0-9]*' $O/$f-off.err | head -1)
  for arm in ${=ARMS:-off default k2}; do
    [ $arm = off ] && continue
    st=$(grep -a '^spec:' $O/$f-$arm.err | tail -1 | cut -c1-120); extra=""
    if [ $arm = default ]; then grep -aq 'default of recipe qwen3_5_moe on metal at tp=1' $O/$f-$arm.err && [ -n "$st" ] || extra=" NO-DEFAULT-LINE-OR-NO-SPEC-STATS"; fi
    if cmp -s $O/$f-off.ids $O/$f-$arm.ids && [ -s $O/$f-off.ids ] && [ -z "$extra" ]; then echo "PASS $f $arm ($pt) $(tr ',' '\n' < $O/$f-off.ids | wc -l | tr -d ' ') ids identical | $st"; else echo "FAIL $f $arm ($pt) differ/empty$extra | $st"; fails=$((fails+1)); fi
  done
done
echo "ids fails=$fails"
