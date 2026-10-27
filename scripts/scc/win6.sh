#!/usr/bin/env bash
# Lane scc window 6 (the gate window): (A) base vs dev vs spark2-vLLM at 1/4/8/16 streams (1024-token prompts, 256 generated, --max-batch 16),
# 3 interleaved rounds; (B) the recipe-default single-stream engine (no --max-batch, MTP on) of base vs dev at 1k / 4k / 16k, 3 rounds.
#   fleetq run --host spark2 --quiet gpu,cpu --mem 50 --expect 42 --label "scc: gate window" -- scripts/scc/win6.sh
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
scripts/scc/le-window.sh gateA 3 base:base::--max-batch_16 dev:dev3::--max-batch_16 vllm
env CONC=1 GEN=128 SIZES=1024,4096,16384 MS=20480 scripts/scc/le-window.sh gateB 3 base:base dev:dev3
