#!/usr/bin/env bash
# Lane stream16, GPU window 3 (one fleetq slot on spark1, quiet on GPU and CPU), dev rebuilt with the final defaults:
#   1. the two streaming rows_exact tests again (final defaults);
#   2. served-token identity base vs dev (ident.sh, both `serve --max-batch 16`); stops before the A/B if it fails;
#   3. interleaved serve A/B base vs dev, 3 rounds at 1 / 4 / 8 / 16 streams (1k prompt, 256 tokens);
#   4. interleaved single-stream decode A/B base vs dev, 3 rounds at 1k / 4k / 16k prompts.
#   fleetq run --host spark1 --quiet gpu,cpu --mem 50 --expect 40 --label "stream16: W3 identity + serve A/B" -- scripts/stream16/w3.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
H="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; RAW=$H/../../docs/results/2026-10-29-stream16-gemv-stream/raw; mkdir -p $RAW
L='~/intarsiaml-stream16'; T0=$SECONDS
cd "$H/../.."
date +%T; ssh -n -o BatchMode=yes spark1 "export FLEETQ_JOB=$FLEETQ_JOB; cd $L/src/crates/intarsia-backend-cuda && INTARSIA_REQUIRE_GPU=1 timeout 230 $L/bin/rows_exact-dev --test-threads=1 qpipe rows_pipe fp8_rows_q fp8_pair 2>&1 | grep -aE '^test |test result|panicked|G-B'" | tee $RAW/gates-w3-rows.txt
grep -q "test result: ok" $RAW/gates-w3-rows.txt || { echo "ABORT: rows_exact"; exit 3; }
date +%T; "$H/ident.sh" base dev; ok=$?
echo "identity exit $ok"; [[ $ok == 0 ]] || { echo "ABORT: base vs dev served tokens differ"; exit 2; }
date +%T; "$H/le-window.sh" w3-conc 3 base:base::--max-batch_16 dev:dev::--max-batch_16
date +%T; SIZES=1024,4096,16384 CONC=1 MS=17408 "$H/le-window.sh" w3-single 3 base:base::--max-batch_16 dev:dev::--max-batch_16
echo "window used $(( (SECONDS - T0) / 60 )) min"; date +%T
