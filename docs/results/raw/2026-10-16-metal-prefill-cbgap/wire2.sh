#!/bin/zsh
# Job D: first-use latency of the weight shards with the backend's own pre-wiring (INTARSIA_METAL_PREWIRE unset) vs off.
SC=/private/tmp/claude-501/-opt-code-intarsiaml/0e8026c5-6dd0-4b83-b375-5ff0a66894fc/scratchpad/cbgap
R=/opt/code/intarsiaml-cbgap/docs/results/raw/2026-10-16-metal-prefill-cbgap
O=$R/wire2; mkdir -p $O
source $R/waitidle.sh
for rep in 1 2; do for cfg in "off 0" "on 0" "on 100" "on 300"; do
  set -- ${=cfg}
  wait_idle || touch $O/$1-$2-$rep.contaminated
  if [ $1 = off ]; then export INTARSIA_METAL_PREWIRE=0; else unset INTARSIA_METAL_PREWIRE; fi
  $SC/wire_probe2 /Users/josef/models/Qwen3-8B none $2 > $O/prewire-$1-sleep$2-$rep.txt 2>&1
done; done
echo done
