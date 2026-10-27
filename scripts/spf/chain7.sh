#!/usr/bin/env bash
# Lane spf: 35B 1k request steady-state nsys with --spec off (is the post-prefill idle gap MTP's?).
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
RAW=docs/results/2026-10-27-spf-cuda-prefill/raw
M35=$(ssh -n spark1 'ls -d /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/* | head -1')
fleetq run --host spark1 --quiet gpu,cpu --mem 50 --expect 10 --label "spf: 35B 1k request nsys, spec off" -- env MODEL=$M35 CHARS=3400 N=5 "XARGS=--spec off" scripts/spf/prof-serve.sh rope serve35-1k-off > $RAW/serve35-1k-off.driver.log 2>&1
