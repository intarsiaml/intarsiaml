#!/usr/bin/env bash
# Lane spf: final tree (rope2: per-launch env read + tests/rope_multi.rs): build (cpu), CUDA test binary build (cpu), then the GPU test + quick kernels test + short ids (quiet).
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
RAW=docs/results/2026-10-27-spf-cuda-prefill/raw
fleetq run --host spark1 --load cpu --mem 16 --expect 20 --label "spf: build rope2 + rope_multi test binary" -- bash -c 'scripts/spf/build.sh rope2 && ssh -n spark1 "export PATH=\$HOME/.cargo/bin:/usr/local/cuda/bin:\$PATH; cd ~/intarsiaml-spf/rope2 && CARGO_BUILD_JOBS=12 cargo test --release -p intarsia-backend-cuda --test rope_multi --no-run 2>&1 | tail -3"' > $RAW/build-rope2.log 2>&1
fleetq run --host spark1 --quiet gpu,cpu --mem 20 --expect 12 --label "spf: rope_multi test + kernels test quick, rope2" -- ssh spark1 'export PATH=/usr/local/cuda/bin:$PATH; cd ~/intarsiaml-spf/rope2; B=$(ls -t target/release/deps/rope_multi-* | grep -v "\.d$" | head -1); INTARSIA_REQUIRE_GPU=1 $B --nocapture 2>&1 | tail -6; target/release/intarsia kernels test --backend cuda --quick 2>&1 | tail -3' > $RAW/rope2-gpu.log 2>&1
