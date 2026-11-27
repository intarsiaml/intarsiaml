#!/usr/bin/env bash
# Background stage W9 (lane ink-hip-bpf) on card 1, beside card 0's paired 4-stream filler reps (visits-w9.txt):
#   (1) prime main4's admission (its batching server, the speed visits' config), then touch logs/w9/main4-primed (card 0's A
#       visits wait for it, so they start cached);
#   (2) hip tiny `inkling2_packed` (no env: main's default PF2, #537 in the base) and `inkling2_batch` on the b12 test builds;
#   (3) the knob-off gate: b12 `--max-batch 16` with packing off (no INTARSIA_INK_* set), the 16 real prompts at once, replies
#       compared later with main's solo replies;
#   (4) the B16 defaults check, if time is left: b12 with the recommended packing knobs at the recipe's defaults (`--max-batch 16`,
#       no --max-seq, no --kv-fraction): the context and the KV capacity (/metrics), to compare with 16 x the default 4096.
# All within BG_CAP s.
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-bpf
S=$REPO/.scratch
L=$REPO/logs/w9
mkdir -p "$L"
rm -f "$L/main4-primed"
M=/models/Inkling-2-Mini
BR=${BR:-b12}; MA=${MA:-main4}
C=ink-hip-bpf-run2
now() { date '+%F %T'; }
T0=$(date +%s); CAPT=${BG_CAP:-240}
left() { echo $(( CAPT - ($(date +%s) - T0) )); }
srv() {  # name binary serve-args... (leading NAME=value = env): start on card 1, port 8133, wait until up (at most $1 s via WAITS)
  local name=$1 b=$2; shift 2
  docker rm -f $C >/dev/null 2>&1
  local envs=""; while [[ "${1:-}" =~ ^[A-Z_][A-Z0-9_]*= ]]; do envs="$envs $1"; shift; done
  HIP_VISIBLE_DEVICES=1 INK_CONTAINER=$C nohup bash "$S/run.sh" bash -c "HOME=/work/.scratch/home exec env $envs \
    /work/.scratch/bin/intarsia-$b serve -m $M --backend hip --host 0.0.0.0 --port 8133 --model-id ink $*" > "$L/$name.log" 2>&1 < /dev/null &
  UP=0; IP=""
  local i
  for i in $(seq 1 "${WAITS:-150}"); do
    IP=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' $C 2>/dev/null)
    if [ -n "$IP" ] && curl -sf -m 2 "http://$IP:8133/health" > /dev/null 2>&1; then UP=$i; break; fi
    sleep 1
  done
  echo "[bg] $name ($b) up after ${UP}s (0 = not up) $(now): $(grep -aE 'admission on plan|context|packed prefill' "$L/$name.log" | tr '\n' ' ' | cut -c1-300)"
}
# (1)
echo "[bg] (1) prime $MA $(now)"
WAITS=150 srv prime-$MA $MA --max-batch 16 --max-seq 4608 --kv-fraction 0.9
docker rm -f $C >/dev/null 2>&1
touch "$L/main4-primed"
# (2)
T="cargo test --release -j8 -p intarsia-models --no-fail-fast"
for t in packed batch; do
  (( $(left) < 45 )) && { echo "[bg] (2) tiny $t SKIP ($(left) s left)"; continue; }
  echo "[bg] (2) tiny inkling2_$t, no env ($(left) s left) $(now)"
  docker rm -f $C >/dev/null 2>&1
  HIP_VISIBLE_DEVICES=1 INK_CONTAINER=$C timeout 50 bash "$S/run.sh" env INTARSIA_REQUIRE_GPU=1 INTARSIA_INKLING2_BACKEND=hip $T --test inkling2_$t -- \
    --nocapture --test-threads=1 > "$L/tiny-$t.log" 2>&1
  echo "[bg] tiny $t exit $? $(now)"
  grep -aE "^test .*(ok|FAILED)$|test result|packed == solo|panicked" "$L/tiny-$t.log" | cut -c1-200 | head -16 | sed "s/^/[bg tiny $t] /"
done
docker rm -f $C >/dev/null 2>&1
# (3)
if (( $(left) > 60 )); then
  echo "[bg] (3) knob-off $BR --max-batch 16 ($(left) s left) $(now)"
  WAITS=$(( $(left) - 30 )) srv off-$BR $BR --max-batch 16 --max-seq 4608 --kv-fraction 0.9
  if [ "$UP" != 0 ]; then
    curl -sf -m 5 "http://$IP:8133/metrics" > "$L/off-metrics0.txt" 2>&1
    timeout $(( $(left) - 5 )) python3 "$S/serve_bitwise.py" "http://$IP:8133/v1" conc "$L/off-$BR-bitwise-conc.json" > "$L/off-$BR-bitwise-conc.out" 2>&1
    echo "[bg] knob-off conc exit $? $(now): $(head -n 1 "$L/off-$BR-bitwise-conc.out")"
    curl -sf -m 5 "http://$IP:8133/metrics" > "$L/off-metrics1.txt" 2>&1
    echo "[bg] knob-off pack passes: $(grep -h '^intarsia_batch_pack_passes_total ' "$L/off-metrics1.txt")"
  fi
  docker rm -f $C >/dev/null 2>&1
else
  echo "[bg] (3) knob-off SKIP ($(left) s left)"
fi
# (4)
if (( $(left) > 35 )); then
  echo "[bg] (4) B16 defaults with packing 4096 ($(left) s left) $(now)"
  WAITS=$(( $(left) - 8 )) srv dflt-$BR $BR INTARSIA_INK_PACK=1 INTARSIA_INK_PACK_ROWS=4096 --max-batch 16
  [ "$UP" != 0 ] && curl -sf -m 5 "http://$IP:8133/metrics" | grep -E '^intarsia_kv_(bytes_total|tokens_capacity) ' | sed 's/^/[bg dflt] /'
  docker rm -f $C >/dev/null 2>&1
else
  echo "[bg] (4) B16 defaults SKIP ($(left) s left)"
fi
docker rm -f $C >/dev/null 2>&1
echo "[bg] done $(now)"
