#!/usr/bin/env bash
# Admission of the prefill-4 kernels on spark1, vLLM UP (small slot), tree p4-dev, output ~/intarsia-decfuse/out/p4adm
# (spark1 side; run through remote.sh):
#   1. the shared sweeps `intarsia kernels test --op gdn_delta_rule` (every registered kernel of the op, the split kernel
#      included) and `--op matmul_fp8` (matmul_fp8_gemm_f8 included) on the 35B shapes
#   2. plan-time admission on the 35B plan's exact shapes (`plan show --admit --no-admission-cache --json`, prefill chunk
#      2048) for base (INTARSIA_CUDA_GDN_SPLIT=0), dev (default) and f8 (INTARSIA_CUDA_FP8_GEMM_F8=1): the planned kernels
#      run on the clamped cases against the cpu-ref oracle, as `serve` does at load
#   fleetq run --host spark1 --mem 30 --label "..." -- scripts/cuda-prefill4/remote.sh p4-dev scripts/cuda-prefill4/admission.sh
set -u
P=$HOME/intarsia-decfuse; T=$P/p4-dev; BIN=$T/target/acc/release/intarsia; O=$P/out/p4adm; mkdir -p $O
M=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
for op in gdn_delta_rule matmul_fp8; do
  echo "== sweep $op $(date +%T)"
  $BIN kernels test --backend cuda --model $M --op $op -v > $O/sweep-$op.out 2>&1; echo "exit $?" >> $O/sweep-$op.out
  grep -aE "ADMITTED|REJECTED|FAIL|fail|exit" $O/sweep-$op.out | tail -14 | cut -c1-200
done
for arm in base:INTARSIA_CUDA_GDN_SPLIT=0 dev:INTARSIA_X=0 f8:INTARSIA_CUDA_FP8_GEMM_F8=1; do
  n=${arm%%:*}; e=${arm#*:}
  echo "== plan admit $n ($e) $(date +%T)"
  env $e $BIN plan show -m $M --backend cuda --mem-budget 24 --max-seq 8192 --prefill-chunk 2048 --admit --no-admission-cache --json > $O/plan-$n.json 2> $O/plan-$n.err; echo "exit $?" >> $O/plan-$n.err
  tail -3 $O/plan-$n.err | cut -c1-200
  python3 - $O/plan-$n.json <<'PY'
import json, sys
try:
    p = json.load(open(sys.argv[1]))
except Exception as e:
    print("no json:", e); sys.exit(0)
ops = p.get("ops") or p.get("plan", {}).get("ops") or []
seen = {}
for o in ops:
    k = str(o.get("kernel"))
    if "matmul_fp8" in k or "gdn_delta" in k:
        seen.setdefault((o.get("op"), k, bool(o.get("admitted"))), 0)
        seen[(o.get("op"), k, bool(o.get("admitted")))] += 1
for (op, k, ad), c in sorted(seen.items()):
    print(f"{op} {k} {'admitted' if ad else 'NOT ADMITTED'} x{c}")
print(f"kernels {len(ops)}, not admitted:", [o.get("kernel") for o in ops if not o.get("admitted")])
PY
done
echo "== done $(date +%T)"
