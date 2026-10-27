#!/usr/bin/env bash
# Lane LDF, on the Mac from the worktree root: sync, ISA dump as a fleetq cpu job, fetch and print the line/fold kernels' resource table.
set -uo pipefail
bash scripts/ldf/sync.sh || exit 1
fleetq run --host tatooine-r9700 --load cpu --expect 6 --label "GLR isa dump a8 gemm" -- ssh -o BatchMode=yes tatooine 'bash ~/intarsiaml-ldf/scripts/ldf/isa.sh mxfp4_a8_gemm' 2>&1 | tail -2
scp -q tatooine:intarsiaml-ldf/bench-out/isa/mxfp4_a8_gemm.res.txt ldf-out/
grep -a "error" ldf-out/mxfp4_a8_gemm.res.txt | head
scp -q tatooine:intarsiaml-ldf/bench-out/isa/mxfp4_a8_gemm.s ldf-out/
python3 scripts/ldf/resusage.py ldf-out/mxfp4_a8_gemm.res.txt "$@"
