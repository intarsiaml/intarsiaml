#!/usr/bin/env bash
# Stage: first contact. moe_route admission, the kernel list, and a load attempt of the real model (expected: NoKernel naming the
# ops the lanes have not landed yet).
set -uo pipefail
R=$HOME/intarsiaml-ink-hip/docs/results/2026-11-25-inkling2-hip/raw
bash "$R/stage_admit.sh" moe_route
a=$?
echo "== kernels list (ink / route)"
bash "$R/run.sh" /target/release/intarsia kernels list --backend hip 2>&1 | grep -iE "route|ink|conv_mix|res_mix|sink|situ|mxfp8" | head -20
echo "== generate --help"
bash "$R/run.sh" /target/release/intarsia generate --help 2>&1 | head -60
echo "== load attempt"
bash "$R/run.sh" /target/release/intarsia generate /models/Inkling-2-Mini --backend hip --prompt "The capital of France is" --max-tokens 8 2>&1 | tail -15
echo "== load exit ${PIPESTATUS[0]}"
exit $a
