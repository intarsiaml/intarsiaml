#!/usr/bin/env bash
# Admission of the round-5 kernels on spark1, vLLM UP (small slot), tree r5-f8, output ~/intarsia-decfuse/out/r5adm
# (spark1 side; through scripts/cuda-prefill4/remote.sh r5-f8 scripts/cuda-round5/admission.sh). Copy of
# scripts/cuda-prefill4/admission.sh:
#   1. the shared sweeps `intarsia kernels test --op attn_decode` and `--op attn_prefill` on the 35B shapes (every registered
#      attention kernel: the f8kv kernels on the sweep's fp8_e4m3 cache cases, the bf16 ones on bf16 / f32), and the whole
#      `--quick` sweep (exit status: SKIPPED kernels no longer fail it)
#   2. plan-time admission on the 35B plan's exact shapes (`plan show --admit --no-admission-cache --json`, prefill chunk
#      2048) for dev (default bf16 cache) and f8 (`--kv-dtype fp8_e4m3`)
set -u
P=$HOME/intarsia-decfuse; T=$P/r5-f8; BIN=$T/target/acc/release/intarsia; O=$P/out/r5adm; mkdir -p $O
M=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
for op in attn_decode attn_prefill; do
  echo "== sweep $op $(date +%T)"
  $BIN kernels test --backend cuda --model $M --op $op -v > $O/sweep-$op.out 2>&1; echo "exit $?" >> $O/sweep-$op.out
  grep -aE " ADMITTED| REJECTED|NO CASES|SKIPPED|kernels, |^exit|Error" $O/sweep-$op.out | tail -14 | cut -c1-200
done
echo "== quick sweep $(date +%T)"
$BIN kernels test --backend cuda --model $M --quick > $O/sweep-quick.out 2>&1; echo "exit $?" >> $O/sweep-quick.out
grep -aE "REJECTED|incomplete|kernels, |^exit|Error" $O/sweep-quick.out | tail -6 | cut -c1-200
for arm in "dev:" "f8:--kv-dtype fp8_e4m3"; do
  n=${arm%%:*}; a=${arm#*:}
  echo "== plan admit $n ($a) $(date +%T)"
  $BIN plan show -m $M --backend cuda --mem-budget 24 --max-seq 8192 --prefill-chunk 2048 $a --admit --no-admission-cache --json > $O/plan-$n.json 2> $O/plan-$n.err; echo "exit $?" >> $O/plan-$n.err
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
    if "attn" in k or "kv_append" in k:
        key = (o.get("op"), k, bool(o.get("admitted")))
        seen[key] = seen.get(key, 0) + 1
for (op, k, ad), c in sorted(seen.items()):
    print(f"{op} {k} {'admitted' if ad else 'NOT ADMITTED'} x{c}")
print(f"kernels {len(ops)}, kv_dtype {p.get('kv_dtype')}, not admitted:", [o.get("kernel") for o in ops if not o.get("admitted")])
PY
done
echo "== done $(date +%T)"
