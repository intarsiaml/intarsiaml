#!/bin/zsh
# lane MP slot 2: prefill chunk 2048 / 4096 / 8192 at 4k and 16k (cb-mode GPU time), plus greedy ids per chunk
set -u
zmodload zsh/datetime
: ${FLEETQ_JOB:?run inside a fleetq slot}
S=/private/tmp/claude-501/-opt-code-intarsiaml/f31ccf91-6032-412f-aeab-eea38852047d/scratchpad/mp
R=/opt/code/wt-mp/docs/results/2026-10-22-mp-metal-prefill/raw/s2
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
gen() { # stem ctx chunk maxtok [extra args]
  local stem=$1 ctx=$2 ch=$3 mt=$4; shift 4
  cool
  local s=$EPOCHREALTIME
  env INTARSIA_METAL_PROFILE=cb:$R/$stem.jsonl $B generate --model $M --backend metal --max-seq 20000 --temperature 0 \
    --max-tokens $mt --ignore-eos --profile-every 0 --prefill-chunk $ch "$@" --prompt "$(cat $P/prompt-$ctx.txt)" > $R/$stem.out 2> $R/$stem.err
  printf "%.3f\t%.3f\t%s\t%s\n" $s $EPOCHREALTIME $stem "$(grep -o 'prefill [0-9]* ms ([0-9.]* tok/s)' $R/$stem.err | tail -1)" | tee -a $R/runs.tsv
}
sleep 2
for r in 1 2; do
  for c in 2048 4096 8192; do gen c$c-4k-r$r 4096 $c 2; done
  for c in 8192 4096 2048; do gen c$c-16k-r$r 16384 $c 2; done
done
for c in 2048 4096 8192; do gen ids-c$c-4k 4096 $c 48 --show-ids; done
date -Iseconds >> $R/meta.txt
