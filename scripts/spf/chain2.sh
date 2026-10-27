#!/usr/bin/env bash
# Lane spf: 35B NVFP4 prefill window on spark1: vLLM (the box's production server) against intarsia (recipe default = MTP) and intarsia --spec off.
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
RAW=docs/results/2026-10-27-spf-cuda-prefill/raw
M35=$(ssh -n spark1 'ls -d /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/* | head -1')
fleetq run --host spark1 --quiet gpu,cpu --mem 50 --expect 45 --label "spf: 35B prefill vLLM vs intarsia (mtp default) vs spec off x3" -- env MODEL=$M35 SN=intarsia-35b MB=45 SIZES=1024,4096,16384 scripts/spf/le-window.sh ab35b 3 vllm def:rope off:rope:::--spec_off > $RAW/ab35b.driver.log 2>&1
