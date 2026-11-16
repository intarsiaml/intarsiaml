#!/bin/zsh
# Cost of one short prefill pass (M rows through all layers) on the 27B and the 35B: generate with prompts of ~5..40 tokens, --spec off.
# M <= 8 body rows run the GEMV kernels, longer ones the NAX GEMMs (weights decoded to bf16 scratch per call). Inside a fleetq slot.
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
W=/opt/code/intarsiaml/.worktrees/mac-plan; BIN=$W/target/release/intarsia; R=$W/scripts/mac-plan/raw/shortpass; mkdir -p $R
for m in Qwen3.8-27B-Quark-AWQ-MXFP4 Qwen3.6-35B-A3B-NVFP4; do
  for words in 3 7 12 16 24 32 64; do
    P=$(python3 -c "print(' '.join(['alpha','river','stone','cloud','seven','green','table','north']*8)[:0] + ' '.join((['alpha','river','stone','cloud','seven','green','table','north']*8)[:$words]))")
    for rep in 1 2; do
      $BIN generate --model $HOME/models/$m --backend metal --max-seq 4096 --profile-every 0 --temperature 0 --max-tokens 9 --ignore-eos --spec off --prompt "$P" > $R/$m-$words-$rep.out 2> $R/$m-$words-$rep.err
      echo "$m words=$words rep=$rep $(grep -ao 'prompt=[0-9]*' $R/$m-$words-$rep.err | tail -1) $(grep -aoE '(prefill|decode) [0-9]+ ms \([0-9.]+ tok/s\)' $R/$m-$words-$rep.err | tr '\n' ' ')"
    done
  done
done
