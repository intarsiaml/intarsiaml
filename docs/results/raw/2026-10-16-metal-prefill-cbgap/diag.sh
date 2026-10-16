#!/bin/zsh
# Job A: cold vs warm gap structure, base (+host timestamps only) vs staged upload, chunk 256 / 2048, AHEAD sweep.
SC=/private/tmp/claude-501/-opt-code-intarsiaml/0e8026c5-6dd0-4b83-b375-5ff0a66894fc/scratchpad/cbgap
R=/opt/code/intarsiaml-cbgap/docs/results/raw/2026-10-16-metal-prefill-cbgap
O=$R/diag; mkdir -p $O
for C in 256 2048; do
  $R/trace-run.sh $SC/cbgap-bench-ts $O/ts-c$C $C 4062,1024 3 0
  $R/trace-run.sh $SC/cbgap-bench-v1 $O/v1-c$C $C 4062,1024 3 0
done
for A in 0 2 4 6; do
  $R/trace-run.sh $SC/cbgap-bench-v1 $O/v1-c256-ahead$A 256 4062,1024 3 0 INTARSIA_METAL_AHEAD=$A
done
echo done
