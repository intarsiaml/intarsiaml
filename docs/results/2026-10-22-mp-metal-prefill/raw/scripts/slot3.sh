#!/bin/zsh
# lane MP slot 3: flash NAX s4 vs s4_big (micro ABAB + engine 4k/16k)
set -u
zmodload zsh/datetime
: ${FLEETQ_JOB:?run inside a fleetq slot}
S=/private/tmp/claude-501/-opt-code-intarsiaml/f31ccf91-6032-412f-aeab-eea38852047d/scratchpad/mp
R=/opt/code/wt-mp/docs/results/2026-10-22-mp-metal-prefill/raw/s3
P=/opt/code/wt-mp/docs/results/raw/2026-10-15-metal-longctx-profile/prompts
M=/Users/josef/models/Qwen3-8B
B=$S/intarsia-base
export INTARSIA_CPU_FREE_MEM_GB=60
echo "fleetq job $FLEETQ_JOB $(date -Iseconds)" > $R/meta.txt
/Users/josef/bin/macmon pipe -i 100 > $R/macmon.jsonl 2>/dev/null &
MP=$!
trap 'kill $MP 2>/dev/null' EXIT
gpu_temp() { tail -n 3 $R/macmon.jsonl | python3 -c 'import sys,json
t=0
for l in sys.stdin:
  try: t=json.loads(l)["temp"]["gpu_temp_avg"]
  except Exception: pass
print(int(t))'; }
cool() { local n=0; while [ $(gpu_temp) -gt 70 ] && [ $n -lt 9 ]; do sleep 5; n=$((n+1)); done; }
gen() { # variant stem ctx chunk maxtok [extra args]
  local nv=$1 stem=$2 ctx=$3 ch=$4 mt=$5; shift 5
  cool
  local s=$EPOCHREALTIME
  env INTARSIA_METAL_FLASH_NAX=$nv INTARSIA_METAL_PROFILE=cb:$R/$stem.jsonl $B generate --model $M --backend metal --max-seq 20000 --temperature 0 \
    --max-tokens $mt --ignore-eos --profile-every 0 --prefill-chunk $ch "$@" --prompt "$(cat $P/prompt-$ctx.txt)" > $R/$stem.out 2> $R/$stem.err
  printf "%.3f\t%.3f\t%s\t%s\n" $s $EPOCHREALTIME $stem "$(grep -o 'prefill [0-9]* ms ([0-9.]* tok/s)' $R/$stem.err | tail -1)" | tee -a $R/runs.tsv
}
sleep 2
B=$S/intarsia-c2
micro() { # variant round
  cool
  local s=$EPOCHREALTIME
  $S/flash_bench-c2 --chunk 2048 --reps 3 $1 0 4096 14336 > $R/micro-$1-r$2.txt 2>&1
  printf "%.3f	%.3f	micro-%s-r%s	-
" $s $EPOCHREALTIME $1 $2 >> $R/runs.tsv
}
micro attn_nax_d128_s4 1; micro attn_nax_d128_s4_big 1; micro attn_nax_d128_s4_big 2; micro attn_nax_d128_s4 2; micro attn_nax_d128_s4 3; micro attn_nax_d128_s4_big 3
for r in 1 2; do
  if [ $r = 1 ]; then o=(attn_nax_d128_s4:s4 attn_nax_d128_s4_big:big); else o=(attn_nax_d128_s4_big:big attn_nax_d128_s4:s4); fi
  for x in $o; do gen ${x%%:*} ${x##*:}-4k-r$r 4096 2048 2; done
  for x in $o; do gen ${x%%:*} ${x##*:}-16k-r$r 16384 2048 2; done
done
gen attn_nax_d128_s4 ids-s4-4k 4096 2048 48 --show-ids
gen attn_nax_d128_s4_big ids-big-4k 4096 2048 48 --show-ids
date -Iseconds >> $R/meta.txt

