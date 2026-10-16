#!/bin/bash
S=/private/tmp/claude-501/-opt-code-intarsiaml/f31ccf91-6032-412f-aeab-eea38852047d/scratchpad/mac
cd $S
add() { # add <envfile> <p>
  INTARSIA_CPU_FAULT=matmul:$2 ./acc.sh --noise-ref $S/base.json --write-noise-envelope $1 --noise-perturb "external:INTARSIA_CPU_FAULT=matmul:$2" > env-add-$2.log 2>&1; echo "added $2 to $1 rc=$?"
}
cp env2.json env4.json; add $S/env4.json 3e-3; add $S/env4.json 6e-3
cp env4.json env7.json; add $S/env7.json 3.8e-3; add $S/env7.json 4.4e-3; add $S/env7.json 5e-3
for e in env4 env7; do
  for p in 3.5e-3 4.0e-3 4.2e-3 5.2e-3; do
    echo "== $e trial matmul:$p"; ./gate.sh $e-trial-$p $S/$e.json INTARSIA_CPU_FAULT=matmul:$p > /dev/null
  done
done
echo finished
