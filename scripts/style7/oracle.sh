#!/usr/bin/env bash
# style7 step 4: first-token top-20 logprobs from the f64 CPU oracle (`serve --backend cpu`, same NVFP4 weights, chunk 256) on endor, for a subset of the pinned prompts.
#   FLEETQ_OWNER=claude-style7 fleetq run --host endor --load cpu --mem 45 --expect 120 --label "style7 cpu oracle first-token (18 prompts)" -- bash scripts/style7/oracle.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
OUT=$REPO/docs/results/2026-11-20-style7/raw; BIN=$REPO/scratch/target/release/intarsia; M=$REPO/scratch/models/nvfp4; PORT=8195
$BIN serve --backend cpu --model $M --port $PORT --host 127.0.0.1 --served-model-name oracle --max-seq 2048 --mem-budget 40 --profile-every 0 --prefill-chunk 256 --spec off --no-supervisor > $OUT/serve-oracle.log 2>&1 &
SP=$!; trap 'kill $SP 2>/dev/null; wait $SP 2>/dev/null; echo "oracle end $(date +%T)"' EXIT
for i in $(seq 1 300); do curl -sf http://127.0.0.1:$PORT/health >/dev/null && break; kill -0 $SP 2>/dev/null || { echo "serve died"; tail -5 $OUT/serve-oracle.log; exit 1; }; sleep 2; done
echo "oracle up $(date +%T)"
python3 $REPO/scripts/style7/firsttok.py --url http://127.0.0.1:$PORT/v1 --model oracle --tokens $OUT/${SUBSET:-subset20-tokens}.json --out $OUT/ft-${OUTNAME:-oracle}.json
