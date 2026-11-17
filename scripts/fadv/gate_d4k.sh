#!/usr/bin/env bash
# fadv gate (d), 4k prompt only (the first gates.sh run used a 10k prompt that exceeded --max-seq 8192). Under fleetq on the Spark.
set -uo pipefail
L=$HOME/intarsiaml-fadv; BIN=$L/bin/intarsia; OUT=$L/out
M=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
python3 - > "$OUT/prompt4k.txt" <<'PY'
import sys
sys.stdout.write("".join(f"Paragraph {i}: the quick brown fox number {i} jumps over lazy dog {i*7} while the river flows past mill {i*3}. " for i in range(120)))
PY
for arm in 1 0; do
  INTARSIA_LOAD_DROP_CACHE=$arm "$BIN" generate --model "$M" --backend cuda --max-seq 8192 -p "$(cat "$OUT/prompt4k.txt")" --max-tokens 64 --show-ids > "$OUT/gen-4k-drop$arm.log" 2>&1 < /dev/null
done
a=$(grep -a '^ids' "$OUT/gen-4k-drop1.log"); b=$(grep -a '^ids' "$OUT/gen-4k-drop0.log")
[[ -n $a && $a == "$b" ]] && echo "greedy ids 4k: identical" || echo "greedy ids 4k: DIFFER or missing"
grep -a '^prompt:' "$OUT/gen-4k-drop1.log"; echo "$a" | cut -c1-120
