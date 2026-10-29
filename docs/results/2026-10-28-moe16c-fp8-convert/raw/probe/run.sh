#!/usr/bin/env bash
set -uo pipefail
ssh -o BatchMode=yes spark1 'mkdir -p ~/intarsiaml-moe16c/probe' && scp -q "$(dirname "$0")/p.cu" spark1:intarsiaml-moe16c/probe/
ssh -n -o BatchMode=yes spark1 'cd ~/intarsiaml-moe16c/probe && for a in sm_121 sm_121a; do echo "== $a"; /usr/local/cuda/bin/nvcc -O3 -arch=$a -cubin -o p_$a.cubin p.cu 2>&1 | head -5 && /usr/local/cuda/bin/cuobjdump -sass p_$a.cubin | grep -E "Function|FFMA|HADD2|F2FP|HFMA|FMA|F2F" ; done'
