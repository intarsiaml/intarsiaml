#!/usr/bin/env bash
# Build ~/intarsia-decfuse/r7-base from BASE_SRC (a plain export of origin/main, files touched so cargo sees them as new),
# dropping the CUDA backend's build-script outputs first so the kernel library is rebuilt from the base sources, then
# check that the binary has no `ik_matmul_fp8_gemm_lt` (round-7 kernel) symbol. Small slot, vLLM up:
#   BASE_SRC=<dir> fleetq run --host spark1 --mem 8 --label "..." -- scripts/cuda-round7/build-base.sh
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
: "${BASE_SRC:?export dir}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
rsh() { ssh -n -o BatchMode=yes spark1 "export PATH=\$HOME/.cargo/bin:/usr/local/cuda/bin:\$PATH FLEETQ_JOB=$FLEETQ_JOB; $*"; }
rsh "rm -rf ~/intarsia-decfuse/r7-base/target/acc/release/build/intarsia-backend-cuda-* ~/intarsia-decfuse/r7-base/target/acc/release/.fingerprint/intarsia-backend-cuda-*"
SRC_DIR="$BASE_SRC" INTARSIA_STAMP="head=4be7b7d4 (origin/main export) code-tree=base" "$REPO/scripts/cuda-decode/build.sh" r7-base 2>&1 | grep -E "^error|Finished|intarsia$|stamp"
n=$(rsh "grep -c ik_matmul_fp8_gemm_lt ~/intarsia-decfuse/r7-base/target/acc/release/intarsia || true")
echo "r7-base ik_matmul_fp8_gemm_lt symbols: $n"
[[ "$n" == 0 ]]
