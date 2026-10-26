#!/bin/bash
cd "$(git rev-parse --show-toplevel)"
OUT=docs/results/2026-10-25-l8-fp8-gemm/raw/w4
fleetq run --host tatooine-r9700 --expect 6 --load cpu --label "l8: build f8bench cl (CPU only)" -- ssh tatooine 'bash ~/intarsiaml-l8-src/scripts/l8/build.sh f8bench' > $OUT/build.log 2>&1 || exit 1
fleetq run --host tatooine-r9700 --expect 8 --quiet gpu,cpu --takes-down radiance-rank0 --label "l8: w4 f8bench clamped loads, card 0 (radiance down ~5 min)" -- ssh tatooine 'bash ~/intarsiaml-l8-src/scripts/l8/win.sh bench 4 f8bench' > $OUT/window.log 2>&1
