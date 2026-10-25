#!/bin/bash
cd "$(git rev-parse --show-toplevel)"
OUT=docs/results/2026-10-25-l8-fp8-gemm/raw/w3
fleetq run --host tatooine-r9700 --expect 6 --load cpu --label "l8: build f8bench ablation (CPU only)" -- ssh tatooine 'bash ~/intarsiaml-l8-src/scripts/l8/build.sh f8bench' > $OUT/build.log 2>&1 || exit 1
fleetq run --host tatooine-r9700 --expect 8 --quiet gpu,cpu --takes-down radiance-rank0 --label "l8: w3 f8bench ablation (what bounds the kernel), card 0 (radiance down ~5 min)" -- ssh tatooine 'bash ~/intarsiaml-l8-src/scripts/l8/win.sh bench 3 f8bench' > $OUT/window.log 2>&1
