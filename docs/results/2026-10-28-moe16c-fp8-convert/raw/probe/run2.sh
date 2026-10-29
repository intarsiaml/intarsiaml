#!/usr/bin/env bash
set -uo pipefail
scp -q "$(dirname "$0")/q.cu" spark1:intarsiaml-moe16c/probe/
ssh -n -o BatchMode=yes spark1 'cd ~/intarsiaml-moe16c/probe && /usr/local/cuda/bin/nvcc -O3 -arch=sm_121 -o q q.cu 2>&1 | head -5; /usr/local/cuda/bin/cuobjdump -sass q | grep -cE "FHFMA"; ./q'
