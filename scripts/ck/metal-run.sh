#!/usr/bin/env bash
# Lane ck, Metal: chunk-job.sh on endor (this Mac) inside this fleetq slot, outputs straight into the results dir.
#   fleetq run --host endor --exclusive --mem 45 --expect 40 --label "ck: ..." -- scripts/ck/metal-run.sh "gen accdec"
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run --host endor}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RAW=$REPO/docs/results/2026-10-22-ck-prefill-chunk/raw/metal; mkdir -p "$RAW"
CORPUS=$REPO/tools/prompts/corpus_docs.txt bash "$REPO/scripts/ck/chunk-job.sh" metal "$RAW" "${1:-acc speed}" < /dev/null 2>&1
