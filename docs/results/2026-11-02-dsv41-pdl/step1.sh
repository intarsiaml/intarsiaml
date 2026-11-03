#!/bin/bash
# step 1 on spark2 next to vLLM: PTX check, tests with PDL on, 3-layer generate A/B with edges and nsys
cd /opt/code/intarsiaml/.worktrees/dsv41-pdl
export SPARK_DIR='~/intarsia-dsv41-pdl'
source scripts/dsv41-tp2/spark-dir.sh
: "${FLEETQ_JOB:?}"
OUT=$PWD/.scratch/step1; mkdir -p $OUT
ssh -o BatchMode=yes spark2 bash -s > $OUT/remote.log 2>&1 <<EOS
export FLEETQ_JOB=$FLEETQ_JOB PATH=\$HOME/.cargo/bin:/usr/local/cuda/bin:\$PATH MAX_JOBS=4 CARGO_BUILD_JOBS=4
D=$SPARK_DIR; B=\$D/bin/intarsia; L=\$D/logs/step1; mkdir -p \$L; C=\$HOME/intarsia-dsv41-tp2/cache/l3-r0of1
cat \$D/bin/STAMP
echo "== ptx"
cd \$D/src/kernels/cuda
for f in exl3_moe.cu exl3_moe_cp.cu; do
  nvcc -arch=sm_121 -std=c++17 -I. --cuda-device-only -ptx \$f -o \$L/\$f.ptx < /dev/null 2>&1 | tail -3
done
for f in exl3_moe.cu exl3_moe_cp.cu; do
python3 -c "
import re,sys
s=open(sys.argv[1]).read()
for m in re.finditer(r'\.entry\s+(\S+)\s*\(.*?(?=\.entry|\Z)', s, re.S):
    n=m[1]
    if 'epilogue' in n or 'grouped_cp' in n or 'dense_cp' in n:
        print(n[:70], 'wait', m[0].count('griddepcontrol.wait'), 'launch_dependents', m[0].count('griddepcontrol.launch_dependents'))
" \$L/\$f.ptx < /dev/null
done
cd \$D/src
echo "== kernel tests, PDL on"
T=\$(cargo test --release -p intarsia-backend-cuda --test exl3_moe --no-run 2>&1 < /dev/null | grep -o 'target/release/deps/exl3_moe-[0-9a-f]*' | tail -1)
echo "test bin \$T"
INTARSIA_DSV41_MOE_PDL=1 INTARSIA_REQUIRE_GPU=1 timeout 300 ./\$T --test-threads 1 < /dev/null 2>&1 | tail -12
cd \$L
echo "== dsv41_runtime, graphs vs eager, PDL on"
INTARSIA_DSV41_MOE_PDL=1 INTARSIA_CUDA_GRAPH_EDGES=1 INTARSIA_DSV41_BACKEND=cuda INTARSIA_DSV41_CACHE=\$C INTARSIA_DSV41_DPOS=1 INTARSIA_DSV41_DPOS_GRAPHS=1 INTARSIA_DSV41_DP_SPEC=1 timeout 300 \$D/bin/dsv41_runtime device_position_passes_equal_host_passes --nocapture --test-threads 1 < /dev/null > dp.log 2>&1; echo rc \$?; grep -aE 'test result|panicked|cuda graph' dp.log | sort | uniq -c | sort -rn | head -12
P="The history of the lighthouse is long and full of storms. The keeper wrote it all in a book that nobody ever read."
export INTARSIA_DSV41_ENGRAM=off INTARSIA_DSV41_CAP=8192 INTARSIA_DSV41_CHUNK=512 INTARSIA_DSV41_SPEC=1 INTARSIA_CUDA_GRAPH_EDGES=1
echo "== generate A/B"
for arm in 0 1 0b 1b; do
  INTARSIA_DSV41_MOE_PDL=\${arm:0:1} timeout 240 \$B generate -m \$C --backend cuda -p "\$P" --max-tokens 256 --ignore-eos --show-ids < /dev/null > g\$arm.out 2> g\$arm.err; echo "arm \$arm rc \$? md5 \$(grep -av 'finish=' g\$arm.out | md5sum | cut -c1-12) ids \$(grep -a '^ids' g\$arm.err | md5sum | cut -c1-12)"; grep -a 'finish=' g\$arm.out g\$arm.err | cut -c1-160; grep -a 'cuda graph' g\$arm.err | sed 's/([^)]*)//' | sort | uniq -c | sort -rn | head -8
done
echo "== nsys"
for arm in 0 1; do
  INTARSIA_DSV41_MOE_PDL=\$arm nsys profile --trace=cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none -o n\$arm --force-overwrite=true \$B generate -m \$C --backend cuda -p "\$P" --max-tokens 96 --ignore-eos < /dev/null 2>&1 | grep -aE 'finish=|rror' | cut -c1-160
  nsys stats --report cuda_gpu_trace --format csv --output n\$arm n\$arm.nsys-rep > /dev/null 2>&1 < /dev/null; ls -la n\$arm*
done
echo done-remote
EOS
scp -q "spark2:$SPARK_REL/logs/step1/n*_cuda_gpu_trace.csv" $OUT/ 
python3 scripts/dsv41-tp2/pdlchain.py $OUT/n0*_cuda_gpu_trace.csv $OUT/n1*_cuda_gpu_trace.csv > $OUT/chain.txt 2>&1
cat $OUT/chain.txt
