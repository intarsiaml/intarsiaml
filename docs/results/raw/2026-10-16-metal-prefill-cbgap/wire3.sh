#!/bin/zsh
# Does a pre-wire survive an idle gap of server length? wire_probe3: INTARSIA_METAL_PREWIRE=1 pre-wires; sleep ms between wrap and first use.
SC=/private/tmp/claude-501/-opt-code-intarsiaml/0e8026c5-6dd0-4b83-b375-5ff0a66894fc/scratchpad/cbgap
R=/opt/code/intarsiaml-cbgap/docs/results/raw/2026-10-16-metal-prefill-cbgap
O=$R/wire3; mkdir -p $O
source $R/waitidle.sh
for rep in 1 2; do for cfg in "0 300" "1 300" "0 3000" "1 3000" "1 10000" "1 40000" "0 40000"; do
  set -- ${=cfg}
  wait_idle || touch $O/pw$1-s$2-$rep.contaminated
  INTARSIA_METAL_PREWIRE=$1 $SC/wire_probe3 /Users/josef/models/Qwen3-8B none $2 > $O/pw$1-s$2-$rep.txt 2>&1
done; done
echo done
