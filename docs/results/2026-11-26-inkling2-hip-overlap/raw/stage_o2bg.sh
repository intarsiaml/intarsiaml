#!/usr/bin/env bash
# Background stage O2 (lane ink-hip-overlap) on card 1, beside card 0's visits (visits-o2.txt):
#   (1) prime o2's admission with the overlap on (INTARSIA_OVERLAP=1, stream 1 at high priority); touch logs/o2/o2-primed (card 0's
#       P0 waits for it);
#   (2) gate 1's poisoned run: o2 with INTARSIA_HIP_POISON=ff (every new device allocation 0xff: NaN in bf16 and f32), overlap on,
#       pace 0, priority high: the 16 real prompts at once and at 4 (4 waves); the replies compare offline with W4's solo replies.
# All within BG_CAP s.
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-bpf
S=$REPO/.scratch
L=$REPO/logs/o2
mkdir -p "$L"
rm -f "$L/o2-primed"
M=/models/Inkling-2-Mini
BR=${BR:-o2}
C=ink-hip-bpf-run2
now() { date '+%F %T'; }
T0=$(date +%s); CAPT=${BG_CAP:-245}
left() { echo $(( CAPT - ($(date +%s) - T0) )); }
srv() {  # name binary serve-args...: start on card 1, port 8133, wait until up (at most WAITS s)
  local name=$1 b=$2; shift 2
  docker rm -f $C >/dev/null 2>&1
  HIP_VISIBLE_DEVICES=1 INK_CONTAINER=$C nohup bash "$S/run.sh" bash -c "HOME=/work/.scratch/home exec env $SENV \
    /work/.scratch/bin/intarsia-$b serve -m $M --backend hip --host 0.0.0.0 --port 8133 --model-id ink $*" > "$L/$name.log" 2>&1 < /dev/null &
  UP=0; IP=""
  local i
  for i in $(seq 1 "${WAITS:-150}"); do
    IP=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' $C 2>/dev/null)
    if [ -n "$IP" ] && curl -sf -m 2 "http://$IP:8133/health" > /dev/null 2>&1; then UP=$i; break; fi
    sleep 1
  done
  echo "[bg] $name ($b) up after ${UP}s (0 = not up) $(now): $(grep -aE 'admission on plan|context|packed prefill|overlap|continuous batching' "$L/$name.log" | tr '\n' ' ' | cut -c1-500)"
}
echo "[bg] (1) prime $BR, overlap on (X1's config) $(now)"
SENV="INTARSIA_OVERLAP=1 INTARSIA_HIP_STREAM1_PRIORITY=high" WAITS=170 srv prime-$BR $BR --max-batch 16 --max-seq 4608 --kv-fraction 0.9
docker rm -f $C >/dev/null 2>&1
touch "$L/o2-primed"
if (( $(left) > 50 )); then
  echo "[bg] (2) poisoned o2, overlap on ($(left) s left) $(now)"
  SENV="INTARSIA_OVERLAP=1 INTARSIA_HIP_STREAM1_PRIORITY=high INTARSIA_HIP_POISON=ff" WAITS=$(( $(left) - 30 )) srv poison-$BR $BR --max-batch 16 --max-seq 4608 --kv-fraction 0.9
  if [ "$UP" != 0 ]; then
    for m in conc conc:4:0 conc:4:1 conc:4:2 conc:4:3; do
      (( $(left) < 12 )) && { echo "[bg] poison $m SKIP ($(left) s left)"; continue; }
      ms=${m//:/_}
      timeout $(( $(left) - 3 )) python3 "$S/serve_bitwise.py" "http://$IP:8133/v1" $m "$L/poison-bitwise-$ms.json" > "$L/poison-bitwise-$ms.out" 2>&1
      echo "[bg] poison $m exit $? $(now): $(head -n 1 "$L/poison-bitwise-$ms.out")"
    done
  fi
else
  echo "[bg] (2) poison SKIP ($(left) s left)"
fi
docker rm -f $C >/dev/null 2>&1
echo "[bg] done $(now)"
