#!/bin/bash
cd /opt/code/intarsiaml/.worktrees/dsv41-pdl
export SPARK_DIR='~/intarsia-dsv41-pdl'
source scripts/dsv41-tp2/spark-dir.sh
: "${FLEETQ_JOB:?}"
ssh -n -o BatchMode=yes spark2 "export FLEETQ_JOB=$FLEETQ_JOB PATH=/usr/local/cuda/bin:\$PATH MAX_JOBS=4; cd $SPARK_DIR/src/kernels/cuda; L=$SPARK_DIR/logs/step1; nvcc -arch=sm_121 --std=c++17 -O3 --expt-relaxed-constexpr -I. -ptx exl3_moe_cp.cu -o \$L/cp.ptx 2>&1 | tail -3; ls -la \$L/cp.ptx; python3 -c \"
import re,sys
s=open(sys.argv[1]).read()
print('griddepcontrol in file', s.count('griddepcontrol'), 'funcs', len(re.findall(r'\.func', s)))
for m in re.finditer(r'\.entry\s+(\S+?)\s*\(.*?(?=\.entry|\Z)', s, re.S):
    print(' ', m[1][-40:] if 0 else m[1][40:80], 'wait', m[0].count('griddepcontrol.wait'), 'launch_dependents', m[0].count('griddepcontrol.launch_dependents'))
\" \$L/cp.ptx"
