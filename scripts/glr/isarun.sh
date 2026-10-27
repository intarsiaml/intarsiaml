#!/usr/bin/env bash
# Lane GLR, on the Mac from the worktree root: sync, ISA dump as a fleetq cpu job, fetch and print the line/fold kernels' resource table.
set -uo pipefail
bash scripts/glr/sync.sh || exit 1
fleetq run --host tatooine-r9700 --load cpu --expect 6 --label "GLR isa dump a8 gemm" -- ssh -o BatchMode=yes tatooine 'bash ~/intarsiaml-glr/scripts/glr/isa.sh mxfp4_a8_gemm' 2>&1 | tail -2
scp -q tatooine:intarsiaml-glr/bench-out/isa/mxfp4_a8_gemm.res.txt glr-out/
grep -a "error" glr-out/mxfp4_a8_gemm.res.txt | head
scp -q tatooine:intarsiaml-glr/bench-out/isa/mxfp4_a8_gemm.s glr-out/
python3 scripts/glr/resusage.py glr-out/mxfp4_a8_gemm.res.txt "$@"
