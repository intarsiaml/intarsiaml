#!/usr/bin/env bash
# Lane fp8kv: the attention bucket of the B = 16 batched decode step, bf16 vs fp8 KV. nsys over timing_35b_batched_step
# (INTARSIA_STEP_BATCH=16, prompt 1024, graphs on: 20 + 3 x 60 = 200 steps) per cache dtype, in the order given (default
# bf16 fp8_e4m3); sums the decode-attention kernels (row-table split-KV stage 1, its combine, the row-table KV append) per
# step. Inside a fleetq slot on spark2 quiet on GPU and CPU (W3 runs it first):
#   scripts/fp8kv/stepprof.sh [kv ...]
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
H="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW=$H/../../docs/results/2026-10-29-fp8kv-opt-in/raw/stepprof; mkdir -p $RAW
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
for kv in ${@:-bf16 fp8_e4m3}; do
  echo "== stepprof kv=$kv $(date +%T)"
  ssh -n -o BatchMode=yes spark2 "export FLEETQ_JOB=$FLEETQ_JOB PATH=/usr/local/cuda/bin:\$PATH; O=~/intarsiaml-fp8kv/out/stepprof-$kv; mkdir -p \$O
    cd ~/intarsiaml-fp8kv/src/crates/intarsia-models && INTARSIA_STEP_BATCH=16 INTARSIA_STEP_PROMPT=1024 INTARSIA_TEST_KV_DTYPE=$kv INTARSIA_QWEN36_DIR=$M35 INTARSIA_REQUIRE_GPU=1 \
      timeout 200 nsys profile -t cuda --cuda-graph-trace=node --force-overwrite true -o \$O/nsys ~/intarsiaml-fp8kv/bin/batch_cuda_real-dev --ignored --nocapture --test-threads=1 --exact timing_35b_batched_step 2>&1 | grep -aE '^STEP|panicked'
    nsys stats -r cuda_gpu_kern_sum -f csv --force-export=true -o \$O/kern \$O/nsys.nsys-rep > /dev/null 2>&1
    python3 - \$O/kern_cuda_gpu_kern_sum.csv <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
tot = {}
for r in rows:
    n = r['Name']; ns = float(r['Total Time (ns)'])
    b = 'decode attention (rows split-KV + combine + KV append)' if ('splitkv_rows_persist' in n or ('combine_kernel' in n and 'true' in n) or 'kv_append_rows' in n) else None
    if b: tot[b] = tot.get(b, 0) + ns
    if b: print(f'  {ns / 200 / 1e6:7.3f} ms/step  {r[\"Instances\"]:>6} calls  {n[:110]}')
for b, ns in tot.items(): print(f'BUCKET {b}: {ns / 200 / 1e6:.3f} ms per step (200 steps)')
PY
    rm -f \$O/nsys.sqlite" 2>&1 | tee $RAW/$kv.txt
done
echo "== stepprof done $(date +%T)"
