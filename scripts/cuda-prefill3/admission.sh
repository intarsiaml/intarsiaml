#!/usr/bin/env bash
# Admission of the prefill-3 kernels on spark1, vLLM UP (small slot), tree p3-dev, output ~/intarsia-decfuse/out/p3adm:
#   1. the shared sweep `intarsia kernels test --op matmul_fp8` (35B shapes; its cases stop at M = 40, below the 256-token
#      threshold of matmul_fp8_gemm_reg, so the sweep grades the kernels around it, not it) and `--op gdn_delta_rule`
#   2. plan-time admission on the 35B plan's exact shapes (`plan show --admit --no-admission-cache --json`, prefill chunk
#      2048): the planned kernels, matmul_fp8_gemm_reg included, run on the clamped cases (rows <= 32, real N and K)
#      against the cpu-ref oracle, as `serve` does at load
#   3. the 8B fp32 long golden (no fp8 projections and no GatedDeltaNet in the 8B: a sanity check of the shared path)
#   fleetq run --host spark1 --mem 30 --label "prefill-3 admission" --wait-timeout 60 -- scripts/cuda-prefill3/admission.sh
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
HOST=${HOST:-spark1}
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
L=$S/.adm_remote_$FLEETQ_JOB.sh
cat > "$L" <<'EOS'
set -u
P=$HOME/intarsia-decfuse; T=$P/p3-dev; BIN=$T/target/acc/release/intarsia; O=$P/out/p3adm; mkdir -p $O
M=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
M8=/home/nvidia/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
echo "== sweep matmul_fp8 $(date +%T)"
$BIN kernels test --backend cuda --model $M --op matmul_fp8 -v > $O/sweep-matmul_fp8.out 2>&1; echo "exit $?" >> $O/sweep-matmul_fp8.out
tail -25 $O/sweep-matmul_fp8.out
echo "== sweep gdn_delta_rule $(date +%T)"
$BIN kernels test --backend cuda --model $M --op gdn_delta_rule -v > $O/sweep-gdn.out 2>&1; echo "exit $?" >> $O/sweep-gdn.out
tail -12 $O/sweep-gdn.out
for arm in base:INTARSIA_CUDA_FP8_GEMM_REG=0 dev:INTARSIA_X=0; do
  n=${arm%%:*}; e=${arm#*:}
  echo "== plan admit $n ($e) $(date +%T)"
  env $e $BIN plan show -m $M --backend cuda --mem-budget 24 --max-seq 8192 --prefill-chunk 2048 --admit --no-admission-cache --json > $O/plan-$n.json 2> $O/plan-$n.err; echo "exit $?" >> $O/plan-$n.err
  tail -5 $O/plan-$n.err | cut -c1-200
  python3 - $O/plan-$n.json <<'PY'
import json, sys
try:
    p = json.load(open(sys.argv[1]))
except Exception as e:
    print("no json:", e); sys.exit(0)
ops = p.get("ops") or p.get("plan", {}).get("ops") or []
for o in ops:
    if "matmul_fp8" in str(o.get("kernel")) or "gdn_delta" in str(o.get("kernel")):
        print(o.get("op"), o.get("kernel"), "admitted" if o.get("admitted") else "NOT ADMITTED", o.get("calls_per_decode_step"))
print("not admitted:", [o.get("kernel") for o in ops if not o.get("admitted")])
PY
done
echo "== 8B long golden $(date +%T)"
$BIN accuracy --model $M8 --backend cuda --golden $HOME/intarsiaml-golden/qwen3-8b-long-fp32.safetensors --mode both --decode-positions 16 --max-seq 4096 --mem-budget 24 --gate-class bf16 --allow-dirty > $O/acc8-long.log 2>&1; echo "exit $?" >> $O/acc8-long.log
grep -aE "prefill |decode |verdict|PASS|FAIL|exit" $O/acc8-long.log | tail -8 | cut -c1-200
echo "== done $(date +%T)"
EOS
scp -q "$L" "$HOST:/tmp/p3adm_$FLEETQ_JOB.sh"
rm -f "$L"
ssh -n -o BatchMode=yes "$HOST" "export PATH=\$HOME/.cargo/bin:/usr/local/cuda/bin:\$PATH; export FLEETQ_JOB=$FLEETQ_JOB; bash /tmp/p3adm_$FLEETQ_JOB.sh; rc=\$?; rm -f /tmp/p3adm_$FLEETQ_JOB.sh; exit \$rc"
