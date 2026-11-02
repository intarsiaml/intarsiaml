#!/bin/bash
# step 1, the rest: which kernels hold the wait (PTX), and a second nsys pair of the 3-layer run
cd /opt/code/intarsiaml/.worktrees/dsv41-pdl
export SPARK_DIR='~/intarsia-dsv41-pdl'
source scripts/dsv41-tp2/spark-dir.sh
: "${FLEETQ_JOB:?}"
OUT=$PWD/.scratch/step1; mkdir -p $OUT
ssh -o BatchMode=yes spark2 bash -s > $OUT/remote2.log 2>&1 <<EOS
export FLEETQ_JOB=$FLEETQ_JOB PATH=\$HOME/.cargo/bin:/usr/local/cuda/bin:\$PATH MAX_JOBS=4
D=$SPARK_DIR; B=\$D/bin/intarsia; L=\$D/logs/step1; C=\$HOME/intarsia-dsv41-tp2/cache/l3-r0of1
cat \$D/bin/STAMP
cd \$D/src/kernels/cuda
for f in exl3_moe.cu exl3_moe_cp.cu; do
  nvcc -arch=sm_121 -std=c++17 -I. -ptx \$f -o \$L/\$f.ptx < /dev/null 2>&1 | tail -3
  python3 -c "
import re,sys
s=open(sys.argv[1]).read()
print(sys.argv[1].split('/')[-1], 'griddepcontrol in file', s.count('griddepcontrol'), '.func', len(re.findall(r'\.func', s)))
for m in re.finditer(r'\.entry\s+(\S+?)\s*\(.*?(?=\.entry|\Z)', s, re.S):
    n=m[1]
    print(' ', n[:64], 'wait', m[0].count('griddepcontrol.wait'), 'launch_dependents', m[0].count('griddepcontrol.launch_dependents'))
" \$L/\$f.ptx < /dev/null
done
cd \$L
P="The history of the lighthouse is long and full of storms. The keeper wrote it all in a book that nobody ever read."
export INTARSIA_DSV41_ENGRAM=off INTARSIA_DSV41_CAP=8192 INTARSIA_DSV41_CHUNK=512 INTARSIA_DSV41_SPEC=1
for arm in 1 0; do
  INTARSIA_DSV41_MOE_PDL=\$arm nsys profile --trace=cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none -o m\$arm --force-overwrite=true \$B generate -m \$C --backend cuda -p "\$P" --max-tokens 96 --ignore-eos < /dev/null 2>&1 | grep -aE 'finish=' | cut -c1-160
  nsys stats --report cuda_gpu_trace --format csv --output m\$arm m\$arm.nsys-rep > /dev/null 2>&1 < /dev/null
done
echo done-remote
EOS
scp -q "spark2:$SPARK_REL/logs/step1/m*_cuda_gpu_trace.csv" $OUT/
python3 scripts/dsv41-tp2/pdlchain.py $OUT/m0_cuda_gpu_trace.csv $OUT/m1_cuda_gpu_trace.csv | tee $OUT/chain2.txt
