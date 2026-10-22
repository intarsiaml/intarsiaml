#!/bin/zsh
# lane MP slot 1: base 4k prefill profile (kernel + cb mode) and flash NAX micro diagnostics
set -u
zmodload zsh/datetime
: ${FLEETQ_JOB:?run inside a fleetq slot}
S=/private/tmp/claude-501/-opt-code-intarsiaml/f31ccf91-6032-412f-aeab-eea38852047d/scratchpad/mp
R=/opt/code/wt-mp/docs/results/2026-10-22-mp-metal-prefill/raw/s1
P=/opt/code/wt-mp/docs/results/raw/2026-10-15-metal-longctx-profile/prompts
M=/Users/josef/models/Qwen3-8B
export INTARSIA_CPU_FREE_MEM_GB=60
echo "fleetq job $FLEETQ_JOB $(date -Iseconds)" > $R/meta.txt
/Users/josef/bin/macmon pipe -i 100 > $R/macmon.jsonl 2>/dev/null &
MP=$!
trap 'kill $MP 2>/dev/null' EXIT
gen() { # bin stem ctx [env...]
  local bin=$1 stem=$2 ctx=$3; shift 3
  local s=$EPOCHREALTIME
  env "$@" $bin generate --model $M --backend metal --max-seq 20000 --temperature 0 --max-tokens 2 --ignore-eos \
    --profile-every 0 --prompt "$(cat $P/prompt-$ctx.txt)" > $R/$stem.out 2> $R/$stem.err
  local e=$EPOCHREALTIME
  printf "%.3f\t%.3f\t%s\t%s\n" $s $e $stem "$(grep -o 'prefill [0-9]* ms ([0-9.]* tok/s)' $R/$stem.err | tail -1)" | tee -a $R/runs.tsv
}
sleep 3
gen $S/intarsia-base warm-4k 4096
gen $S/intarsia-base plain-4k-a 4096
gen $S/intarsia-base kernel-4k 4096 INTARSIA_METAL_PROFILE=kernel:$R/kernel-4k.jsonl
gen $S/intarsia-base cb-4k 4096 INTARSIA_METAL_PROFILE=cb:$R/cb-4k.jsonl
gen $S/intarsia-base plain-4k-b 4096
# micro: chunk 2048 at pos0 0 / 2048 / 14336, interleaved, twice
for r in 1 2; do
  for v in attn_nax_d128_s4 attn_nax_d128_s4_skip attn_nax_d128_s4_diag_nosoftmax attn_nax_d128_s4_diag_kvfixed; do
    s=$EPOCHREALTIME
    $S/flash_bench-c1 --chunk 2048 --reps 5 $v 0 2048 14336 > $R/micro-$v-r$r.txt 2>&1
    printf "%.3f\t%.3f\tmicro-%s-r%s\n" $s $EPOCHREALTIME $v $r >> $R/runs.tsv
  done
done
date -Iseconds >> $R/meta.txt
