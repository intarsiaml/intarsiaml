#!/usr/bin/env bash
# Lane fp8kv, GPU window 2 (one fleetq slot on spark2, quiet on GPU and CPU):
#   1. served-token identity on the bf16 default, base (main 867b97145, bin/base) vs dev (bin/dev): ident.sh, 21 cells, both
#      `serve --max-batch 16`; stops before the A/B if it fails;
#   2. the interleaved serve A/B bf16 vs fp8 KV on dev, 3 rounds: SRV16 shape (1k prompts, 256 tokens, 1 / 4 / 8 / 16 streams,
#      --max-batch 16) and single stream at 1k / 4k / 16k (MTP default), le-window.sh.
#   fleetq run --host spark2 --quiet gpu,cpu --mem 50 --expect 45 --label "fp8kv: W2 bf16 identity + serve A/B bf16 vs fp8 KV" -- scripts/fp8kv/w2.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
H="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$H/../.."
echo "commit $(git rev-parse --short HEAD) $(date +%T)"
"$H/ident.sh" base dev; ok=$?
echo "identity exit $ok"; [[ $ok == 0 ]] || { echo "ABORT: base vs dev served tokens differ"; exit 2; }
date +%T
"$H/le-window.sh" w2 3 srv-bf16:dev::--max-batch_16 srv-fp8:dev::--max-batch_16_--kv-dtype_fp8 single-bf16:dev single-fp8:dev::--kv-dtype_fp8
date +%T
