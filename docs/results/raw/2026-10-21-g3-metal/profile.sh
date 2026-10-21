#!/bin/zsh
# G3 step 2: decode-step profile + clock study on one binary, plain `generate` (no server), one exclusive slot.
#  (a) cb-mode trace (stream as normal) at 128 / 4k / 16k: step period, GPU busy, idle gap per step
#  (b) kernel-mode trace at 4k / 16k: per-op split (attention vs GEMV), serialised
#  (c) clock study: 4k prompt, 256 decode tokens, 8 runs, macmon 100 ms; idle 0 / 30 s alternating to sample clock states
# usage (inside slot): zsh profile.sh <tag> [bin]
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a slot" >&2; exit 2; }
TAG=$1
S=/private/tmp/claude-501/-opt-code-intarsiaml/f31ccf91-6032-412f-aeab-eea38852047d/scratchpad
BIN=${2:-$S/intarsia-base}
R=/opt/code/intarsiaml/.claude/worktrees/agent-a34836167f72ea687/docs/results/raw
OUT=$R/2026-10-21-g3-metal/$TAG; mkdir -p $OUT
P=$R/2026-10-15-metal-longctx-profile/prompts
PS=$R/2026-10-15-metal-decode-d1/prompt-short.txt
M=/Users/josef/models/Qwen3-8B
export INTARSIA_CPU_FREE_MEM_GB=60
MPID=""
trap '[[ -n "$MPID" ]] && kill $MPID 2>/dev/null' EXIT
echo "binary $(md5 -q $BIN) $(date -Iseconds)" > $OUT/info.txt
idle3() { for i in 1 2 3; do ioreg -r -c IOAccelerator -d 1 | grep -o '"Device Utilization %"=[0-9]*'; sleep 1; done | tr '\n' ' '; echo; }
gen() { # name prompt-file max-tokens [env...]
  local name=$1 pf=$2 mt=$3; shift 3
  echo "== $name $(date +%T) idle: $(idle3)" | tee -a $OUT/info.txt
  /Users/josef/bin/macmon pipe -i 100 > $OUT/$name-macmon.jsonl 2>/dev/null & MPID=$!
  env "$@" $BIN generate --model $M --backend metal --max-seq 20000 --profile-every 0 --temperature 0 --max-tokens $mt --ignore-eos \
    --show-ids --prompt "$(cat $pf)" > $OUT/$name.out 2> $OUT/$name.err
  local rc=$?
  echo "$name end_unix $(python3 -c 'import time; print(f"{time.time():.3f}")')" >> $OUT/times.txt
  echo "   exit $rc $(grep -h 'finish=' $OUT/$name.err)" | tee -a $OUT/info.txt
  kill $MPID 2>/dev/null; wait $MPID 2>/dev/null; MPID=""
}
# (a) cb-mode
for c in short:$PS 4k:$P/prompt-4096.txt 16k:$P/prompt-16384.txt; do
  n=${c%%:*}; f=${c#*:}
  gen cb-$n $f 129 INTARSIA_METAL_PROFILE=cb:$OUT/cb-$n.jsonl
done
# (b) kernel-mode
for c in 4k:$P/prompt-4096.txt 16k:$P/prompt-16384.txt; do
  n=${c%%:*}; f=${c#*:}
  gen kern-$n $f 65 INTARSIA_METAL_PROFILE=kernel:$OUT/kern-$n.jsonl
done
# (c) clock study, plain
for i in 1 2 3 4 5 6 7 8; do
  gen clk4k-$i $P/prompt-4096.txt 256 A=1
  (( i % 2 == 0 )) && sleep 30
done
# (d) old-vs-new: a2102bf (10-15 SAFE, 87.7 % of 40.6 at 4k on 10-15) vs this binary, cb-mode 4k, ABBA x2
if [ -n "${OLD:-}" ]; then
  NEW=$BIN
  j=0
  for o in old new new old old new new old; do
    j=$((j+1))
    [ $o = old ] && BIN=$OLD || BIN=$NEW
    gen ab$j-$o $P/prompt-4096.txt 129 INTARSIA_METAL_PROFILE=cb:$OUT/ab$j-$o.jsonl
  done
  BIN=$NEW
fi
gzip -f $OUT/*.jsonl
echo PROFILE_DONE $(date -Iseconds) | tee -a $OUT/info.txt
