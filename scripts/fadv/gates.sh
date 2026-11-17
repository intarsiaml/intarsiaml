#!/usr/bin/env bash
# fadv gates (a)-(d) on a Spark: page cache after a load with INTARSIA_LOAD_DROP_CACHE=1 vs 0, load time, plan show, greedy ids.
# Run on the Spark under: fleetq run --host spark2 --quiet gpu,cpu --mem 32 -- ~/intarsiaml-fadv/gates.sh
set -uo pipefail
L=$HOME/intarsiaml-fadv; BIN=$L/bin/intarsia; OUT=$L/out; mkdir -p "$OUT"
M=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
cache_mb() { free -m | awk '/^Mem:/ {print $6}'; }      # buff/cache
avail_mb() { free -m | awk '/^Mem:/ {print $7}'; }
evict() { for f in "$M"/*.safetensors; do dd if="$f" iflag=nocache count=0 status=none; done; }   # no root: drops only these files
warm()  { cat "$M"/*.safetensors > /dev/null; }
SHORT="The capital of France is"
python3 - > "$OUT/prompt4k.txt" <<'PY'
import sys
s = "".join(f"Paragraph {i}: the quick brown fox number {i} jumps over lazy dog {i*7} while the river flows past mill {i*3}. " for i in range(120))
sys.stdout.write(s)
PY
run() { # run <arm 0|1> <cold|warm> <tag>
  local arm=$1 mode=$2 tag=$3 t0 t1 b0 b1 peak
  [[ $mode == cold ]] && evict || warm
  sleep 2; b0=$(cache_mb)
  t0=$(date +%s.%N)
  INTARSIA_LOAD_DROP_CACHE=$arm "$BIN" generate --model "$M" --backend cuda -p "$SHORT" --max-tokens 1 --show-ids > "$OUT/$tag.log" 2>&1 < /dev/null
  t1=$(date +%s.%N); sleep 1; b1=$(cache_mb)
  echo "$tag drop=$arm $mode cache_before_mb=$b0 cache_after_mb=$b1 delta_mb=$((b1-b0)) avail_after_mb=$(avail_mb) wall_s=$(echo "$t1 - $t0" | bc) rc=$?" | tee -a "$OUT/gate_ab.txt"
}
echo "== gates a,b $(date +%T) earlyoom=$(systemctl is-active earlyoom) vllm_cache_noise: $(cache_mb) $(sleep 5; cache_mb)" | tee -a "$OUT/gate_ab.txt"
# ABBA, cold-ish then warm
n=0; for arm in 1 0 0 1; do n=$((n+1)); run $arm cold cold$n-drop$arm; done
n=0; for arm in 1 0 0 1; do n=$((n+1)); run $arm warm warm$n-drop$arm; done
echo "== gate c plan show $(date +%T)"
for arm in 1 0; do INTARSIA_LOAD_DROP_CACHE=$arm "$BIN" plan show --model "$M" --backend cuda --json > "$OUT/plan-drop$arm.json" 2> "$OUT/plan-drop$arm.err" < /dev/null; done
cmp "$OUT/plan-drop1.json" "$OUT/plan-drop0.json" && echo "plan show: byte-identical ($(wc -c < "$OUT/plan-drop1.json") bytes)" | tee -a "$OUT/gate_c.txt"
for arm in 1 0; do INTARSIA_LOAD_DROP_CACHE=$arm "$BIN" plan show --model "$M" --backend cuda > "$OUT/plantxt-drop$arm.txt" 2>&1 < /dev/null; done
cmp "$OUT/plantxt-drop1.txt" "$OUT/plantxt-drop0.txt" && echo "plan show (text): byte-identical" | tee -a "$OUT/gate_c.txt"
echo "== gate d greedy ids $(date +%T)"
for arm in 1 0; do
  INTARSIA_LOAD_DROP_CACHE=$arm "$BIN" generate --model "$M" --backend cuda -p "$SHORT" --max-tokens 64 --show-ids > "$OUT/gen-short-drop$arm.log" 2>&1 < /dev/null
  INTARSIA_LOAD_DROP_CACHE=$arm "$BIN" generate --model "$M" --backend cuda --max-seq 8192 -p "$(cat "$OUT/prompt4k.txt")" --max-tokens 64 --show-ids > "$OUT/gen-4k-drop$arm.log" 2>&1 < /dev/null
done
for k in short 4k; do
  a=$(grep -a '^ids' "$OUT/gen-$k-drop1.log"); b=$(grep -a '^ids' "$OUT/gen-$k-drop0.log")
  [[ -n $a && $a == "$b" ]] && echo "greedy ids $k: identical" | tee -a "$OUT/gate_d.txt" || echo "greedy ids $k: DIFFER or missing" | tee -a "$OUT/gate_d.txt"
  grep -a '^prompt:' "$OUT/gen-$k-drop1.log" | tee -a "$OUT/gate_d.txt"
done
