#!/bin/zsh
# GPU work of metal phase B1 (doctor, identity, profiles); run inside a fleetq slot.
S=/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad
R=/opt/code/intarsiaml/docs/results/raw/2026-10-14-metal-phase-b
NEW=/opt/code/intarsiaml/target/release/intarsia
PRE=$S/intarsia-pre
M=/Users/josef/models/Qwen3-8B
mkdir -p $R/doctor $R/identity $R/profile
$PRE doctor --backend metal --out $R/doctor/doctor-pre.json > $R/doctor/doctor-pre.out 2>&1
$NEW doctor --backend metal --out $R/doctor/doctor-after.json > $R/doctor/doctor-after.out 2>&1
for B in pre new; do
  BIN=$NEW; [ $B = pre ] && BIN=$PRE
  for P in short 4k; do
    $BIN generate --model $M --backend metal --max-seq 8192 --profile-every 0 --temperature 0 --max-tokens 48 --show-ids --prompt "$(cat $R/prompt-$P.txt)" > $R/identity/$B-$P.out 2> $R/identity/$B-$P.err
  done
done
for P in short 4k; do
  # plain decode speed, then cb timeline, then per-kernel trace; 129 forced tokens = 128 decode steps
  for MODE in plain cb kernel; do
    if [ $MODE = plain ]; then unset INTARSIA_METAL_PROFILE; else export INTARSIA_METAL_PROFILE=$MODE:$R/profile/$MODE-$P.jsonl; fi
    $NEW generate --model $M --backend metal --max-seq 8192 --profile-every 0 --temperature 0 --max-tokens 129 --ignore-eos --prompt "$(cat $R/prompt-$P.txt)" > $R/profile/$MODE-$P.out 2> $R/profile/$MODE-$P.err
  done
  unset INTARSIA_METAL_PROFILE
done
echo done
