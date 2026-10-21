#!/bin/zsh
# G3 lever experiments, one exclusive slot, same binary (perf/metal-g3 dev build), env toggles:
#  (a) prefill: NAX flash rescale skip (default _rs) vs plain (INTARSIA_METAL_FLASH_NAX=attn_nax_d128_s4): greedy ids
#      identity (48 tokens) at 4k / 16k, then plain-prefill A/B (generate --max-tokens 1) 4k and 16k, ABBA x2, macmon
#  (b) flash_bench rs vs plain at pos0 0 / 8192 / 14336, chunk 2048 (random data)
#  (c) decode: INTARSIA_METAL_FIRST_COMMIT 0 / 4 / 16, cb-mode 4k 129 tokens, rotated x3
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a slot" >&2; exit 2; }
TAG=$1
S=/private/tmp/claude-501/-opt-code-intarsiaml/f31ccf91-6032-412f-aeab-eea38852047d/scratchpad
BIN=$S/intarsia-dev1
W=/opt/code/intarsiaml/.claude/worktrees/agent-a34836167f72ea687
R=$W/docs/results/raw
OUT=$R/2026-10-21-g3-metal/$TAG; mkdir -p $OUT
P=$R/2026-10-15-metal-longctx-profile/prompts
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
PLAIN=INTARSIA_METAL_FLASH_NAX=attn_nax_d128_s4
# (a) identity
for c in 4k:prompt-4096.txt 16k:prompt-16384.txt; do
  n=${c%%:*}; f=$P/${c#*:}
  gen id-$n-rs $f 48 A=1
  gen id-$n-plain $f 48 $PLAIN
  cmp -s $OUT/id-$n-rs.out $OUT/id-$n-plain.out && echo "id $n: stdout identical" | tee -a $OUT/info.txt || echo "id $n: STDOUT DIFFERS" | tee -a $OUT/info.txt
done
# (a) plain prefill ABBA x2 per size
for n in 4096 16384; do
  j=0
  for v in rs plain plain rs rs plain plain rs; do
    j=$((j+1))
    if [ $v = rs ]; then gen pf$n-$j-$v $P/prompt-$n.txt 1 A=1; else gen pf$n-$j-$v $P/prompt-$n.txt 1 $PLAIN; fi
  done
done
# (b) flash_bench
for v in attn_nax_d128_s4_rs attn_nax_d128_s4 attn_nax_d128_s4 attn_nax_d128_s4_rs; do
  /Users/josef/bin/macmon pipe -i 100 > $OUT/fb-$v-$(date +%H%M%S)-macmon.jsonl 2>/dev/null & MPID=$!
  $S/flash_bench --chunk 2048 --layers 36 --reps 6 $v 0 8192 14336 >> $OUT/flash-bench.txt 2>&1
  echo "flash_bench $v exit $?" | tee -a $OUT/info.txt
  kill $MPID 2>/dev/null; wait $MPID 2>/dev/null; MPID=""
done
# (c) decode first-commit
for r in 1 2 3; do
  case $r in 1) ord=(0 4 16);; 2) ord=(16 0 4);; 3) ord=(4 16 0);; esac
  for fc in $ord; do
    gen fc$fc-$r $P/prompt-4096.txt 129 INTARSIA_METAL_FIRST_COMMIT=$fc INTARSIA_METAL_PROFILE=cb:$OUT/fc$fc-$r.jsonl
  done
done
gzip -f $OUT/*.jsonl
echo EXP_DONE $(date -Iseconds) | tee -a $OUT/info.txt
