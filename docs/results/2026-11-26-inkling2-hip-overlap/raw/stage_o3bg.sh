#!/usr/bin/env bash
# Background stage O3 (lane ink-hip-overlap) on card 1, beside card 0's visits (visits-o1.txt):
#   (1) prime o1's admission with the overlap on (INTARSIA_OVERLAP=1: the decode twin's plan too), card 0's X1 config; touch
#       logs/o3/o3-primed (card 0's X1 waits for it); the load log is the admission record of the touched ops;
#   (2) hip tiny `inkling2_batch` (the decode twin test: twin rows == solo, a prefill on the model beside the twin's steps) and
#       `inkling2_packed` (the overlap engine test) on the o1 test builds.
# All within BG_CAP s.
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-bpf
S=$REPO/.scratch
L=$REPO/logs/o3
mkdir -p "$L"
rm -f "$L/o3-primed"
M=/models/Inkling-2-Mini
BR=${BR:-o3}
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
echo "[bg] (1) prime $BR, overlap on, priority high $(now)"
SENV="INTARSIA_OVERLAP=1 INTARSIA_HIP_STREAM1_PRIORITY=high" WAITS=170 srv prime-$BR $BR --max-batch 16 --max-seq 4608 --kv-fraction 0.9
docker rm -f $C >/dev/null 2>&1
touch "$L/o3-primed"
T="cargo test --release -j8 -p intarsia-models --no-fail-fast"
for t in batch packed ""; do
  tt=inkling2${t:+_$t}
  (( $(left) < 40 )) && { echo "[bg] (2) tiny $tt SKIP ($(left) s left)"; continue; }
  tt=inkling2${t:+_$t}
  echo "[bg] (2) tiny $tt on $BR ($(left) s left) $(now)"
  docker rm -f $C >/dev/null 2>&1
  HIP_VISIBLE_DEVICES=1 INK_CONTAINER=$C timeout 45 bash "$S/run.sh" env INTARSIA_REQUIRE_GPU=1 INTARSIA_INKLING2_BACKEND=hip $T --test $tt -- \
    --nocapture --test-threads=1 > "$L/tiny-$tt.log" 2>&1
  echo "[bg] tiny $tt exit $? $(now)"
  grep -aE "^test .*(ok|FAILED)$|test result|panicked|overlap, pace" "$L/tiny-$tt.log" | cut -c1-220 | head -24 | sed "s/^/[bg tiny $tt] /"
done
docker rm -f $C >/dev/null 2>&1
echo "[bg] done $(now)"
