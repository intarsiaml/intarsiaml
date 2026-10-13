#!/bin/bash
S=/private/tmp/claude-501/-opt-code-intarsiaml/f31ccf91-6032-412f-aeab-eea38852047d/scratchpad/mac
cd $S
while IFS='|' read -r name envf var; do
  echo "== $name  [$var]  envelope: $envf"; ./gate.sh "$name" "$envf" "$var" | grep -E "FAIL|verdict|gate:|failing"
done <<LIST
mut-attn-1.05|$S/env2.json|INTARSIA_CPU_MUTATE=attn_prefill,attn_decode:scale=1.05
mut-attn-1.01|$S/env2.json|INTARSIA_CPU_MUTATE=attn_prefill,attn_decode:scale=1.01
mut-attn-1.002|$S/env2.json|INTARSIA_CPU_MUTATE=attn_prefill,attn_decode:scale=1.002
fault-matmul-1.0|$S/env2.json|INTARSIA_CPU_FAULT=matmul:1.0
fault-matmul-2e-2|$S/env2.json|INTARSIA_CPU_FAULT=matmul:2e-2
fixed-mut-attn-1.05|none|INTARSIA_CPU_MUTATE=attn_prefill,attn_decode:scale=1.05
fixed-mut-attn-1.01|none|INTARSIA_CPU_MUTATE=attn_prefill,attn_decode:scale=1.01
fixed-mut-attn-1.002|none|INTARSIA_CPU_MUTATE=attn_prefill,attn_decode:scale=1.002
LIST
