#!/bin/zsh
# G3 decode experiment 2, one exclusive slot, dev2 binary: first-commit 0 / 4 and spin-wait on the last buffer,
# cb-mode 4k 129 tokens and short 129 tokens, rotated x3. macmon logs CPU load for the spin arm.
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a slot" >&2; exit 2; }
TAG=$1
S=/private/tmp/claude-501/-opt-code-intarsiaml/f31ccf91-6032-412f-aeab-eea38852047d/scratchpad
BIN=$S/intarsia-dev2
R=/opt/code/intarsiaml/.claude/worktrees/agent-a34836167f72ea687/docs/results/raw
OUT=$R/2026-10-21-g3-metal/$TAG; mkdir -p $OUT
P=$R/2026-10-15-metal-longctx-profile/prompts
PS=$R/2026-10-15-metal-decode-d1/prompt-short.txt
M=/Users/josef/models/Qwen3-8B
export INTARSIA_CPU_FREE_MEM_GB=60
MPID=""
trap '[[ -n "$MPID" ]] && kill $MPID 2>/dev/null' EXIT
echo "binary $(md5 -q $BIN) $(date -Iseconds)" > $OUT/info.txt
gen() { # name prompt-file max-tokens [env...]
  local name=$1 pf=$2 mt=$3; shift 3
  /Users/josef/bin/macmon pipe -i 100 > $OUT/$name-macmon.jsonl 2>/dev/null & MPID=$!
  env "$@" $BIN generate --model $M --backend metal --max-seq 20000 --profile-every 0 --temperature 0 --max-tokens $mt --ignore-eos \
    --show-ids --prompt "$(cat $pf)" > $OUT/$name.out 2> $OUT/$name.err
  local rc=$?
  echo "$name end_unix $(python3 -c 'import time; print(f"{time.time():.3f}")')" >> $OUT/times.txt
  echo "$name exit $rc $(grep -h 'finish=' $OUT/$name.err)" | tee -a $OUT/info.txt
  kill $MPID 2>/dev/null; wait $MPID 2>/dev/null; MPID=""
}
arm() { # name -> env
  case $1 in
    base) echo "INTARSIA_METAL_FIRST_COMMIT=0";;
    fc4) echo "INTARSIA_METAL_FIRST_COMMIT=4";;
    fc4spin) echo "INTARSIA_METAL_FIRST_COMMIT=4 INTARSIA_METAL_SPIN_WAIT=1";;
  esac
}
for ctx in 4k:$P/prompt-4096.txt short:$PS; do
  n=${ctx%%:*}; f=${ctx#*:}
  for r in 1 2 3; do
    case $r in 1) ord=(base fc4 fc4spin);; 2) ord=(fc4spin base fc4);; 3) ord=(fc4 fc4spin base);; esac
    for a in $ord; do
      gen $n-$a-$r $f 129 ${(z)$(arm $a)} INTARSIA_METAL_PROFILE=cb:$OUT/$n-$a-$r.jsonl
    done
  done
done
gzip -f $OUT/*.jsonl
echo EXP_DONE $(date -Iseconds) | tee -a $OUT/info.txt
