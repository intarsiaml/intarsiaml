#!/bin/bash
S=/private/tmp/claude-501/-opt-code-intarsiaml/f31ccf91-6032-412f-aeab-eea38852047d/scratchpad/mac
cd $S
# envelope 2: another realization
INTARSIA_CPU_FAULT=matmul:5.5e-3 ./acc.sh --noise-ref $S/base.json --write-noise-envelope $S/env2.json --noise-perturb "external:INTARSIA_CPU_FAULT=matmul:5.5e-3" > env2.log 2>&1; echo env2 rc=$?
for t in "ok-4.2e-3|INTARSIA_CPU_FAULT=matmul:4.2e-3" "ok-3.5e-3|INTARSIA_CPU_FAULT=matmul:3.5e-3"; do
  echo "== ${t%%|*}"; ./gate.sh "${t%%|*}" $S/env2.json "${t##*|}"
done
