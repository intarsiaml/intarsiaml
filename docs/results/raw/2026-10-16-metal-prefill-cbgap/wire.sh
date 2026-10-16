#!/bin/zsh
# Job C: first-use latency of the weight shards, with and without pre-wiring.
SC=/private/tmp/claude-501/-opt-code-intarsiaml/0e8026c5-6dd0-4b83-b375-5ff0a66894fc/scratchpad/cbgap
R=/opt/code/intarsiaml-cbgap/docs/results/raw/2026-10-16-metal-prefill-cbgap
O=$R/wire; mkdir -p $O
source $R/waitidle.sh
for rep in 1 2; do for mode in none residency touch; do
  wait_idle || touch $O/$mode-$rep.contaminated
  $SC/wire_probe /Users/josef/models/Qwen3-8B $mode > $O/$mode-$rep.txt 2>&1
done; done
echo done
