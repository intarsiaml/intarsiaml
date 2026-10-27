#!/usr/bin/env bash
# Lane spf: the two gates that were open when spark1 dropped out of fleetq: rope_multi GPU test + planted mutants.
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
RAW=docs/results/2026-10-27-spf-cuda-prefill/raw
fleetq run --host spark1 --quiet gpu,cpu --mem 20 --expect 10 --label "spf: rope_multi test + quick admission, rope2" -- ssh spark1 'export PATH=/usr/local/cuda/bin:$PATH; cd ~/intarsiaml-spf/rope2; B=$(ls -t target/release/deps/rope_multi-* | grep -v "\.d$" | head -1); INTARSIA_REQUIRE_GPU=1 $B --nocapture 2>&1 | tail -6; target/release/intarsia kernels test --backend cuda --quick 2>&1 | tail -3' > $RAW/rope2-gpu.log 2>&1
fleetq run --host spark1 --quiet gpu,cpu --mem 20 --expect 25 --label "spf: rope planted mutants (5 builds)" -- ssh spark1 'M8=$(ls -d ~/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/* | head -1); KFLAGS="--op rope -m $M8" python3 - rope' < scripts/spf/mutants.py > $RAW/mutants.log 2>&1
