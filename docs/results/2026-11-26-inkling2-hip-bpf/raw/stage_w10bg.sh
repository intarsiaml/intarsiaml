#!/usr/bin/env bash
# Background stage W10 (lane ink-hip-bpf, #538's merge condition) on card 1, beside card 0's visits (visits-w10.txt):
#   (1) prime b13's admission in card 0's B1 config (no INTARSIA_* env: packing default-on, 2048 rows), then touch
#       logs/w10/b13-primed (card 0's A1 visit waits for it, so B1 starts cached);
#   (2) hip tiny `inkling2_packed` and `inkling2_batch` on the b13 test builds, no env (packing default-on now: the batch suite's
#       engine tests pack);
#   (3) b13's solo server (no --max-batch, --spec off, --max-seq 4608): the 16 real prompts one at a time, the reference for B1's
#       replies on the merged base.
# All within BG_CAP s.
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-bpf
S=$REPO/.scratch
L=$REPO/logs/w10
mkdir -p "$L"
rm -f "$L/b13-primed"
M=/models/Inkling-2-Mini
BR=${BR:-b13}
C=ink-hip-bpf-run2
now() { date '+%F %T'; }
T0=$(date +%s); CAPT=${BG_CAP:-245}
left() { echo $(( CAPT - ($(date +%s) - T0) )); }
srv() {  # name binary serve-args...: start on card 1, port 8133, wait until up (at most WAITS s)
  local name=$1 b=$2; shift 2
  docker rm -f $C >/dev/null 2>&1
  HIP_VISIBLE_DEVICES=1 INK_CONTAINER=$C nohup bash "$S/run.sh" bash -c "HOME=/work/.scratch/home exec \
    /work/.scratch/bin/intarsia-$b serve -m $M --backend hip --host 0.0.0.0 --port 8133 --model-id ink $*" > "$L/$name.log" 2>&1 < /dev/null &
  UP=0; IP=""
  local i
  for i in $(seq 1 "${WAITS:-150}"); do
    IP=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' $C 2>/dev/null)
    if [ -n "$IP" ] && curl -sf -m 2 "http://$IP:8133/health" > /dev/null 2>&1; then UP=$i; break; fi
    sleep 1
  done
  echo "[bg] $name ($b) up after ${UP}s (0 = not up) $(now): $(grep -aE 'admission on plan|context|packed prefill|continuous batching' "$L/$name.log" | tr '\n' ' ' | cut -c1-360)"
}
# (1)
echo "[bg] (1) prime $BR (B1's config) $(now)"
WAITS=150 srv prime-$BR $BR --max-batch 16 --max-seq 4608 --kv-fraction 0.9
docker rm -f $C >/dev/null 2>&1
touch "$L/b13-primed"
# (2)
T="cargo test --release -j8 -p intarsia-models --no-fail-fast"
for t in packed batch; do
  (( $(left) < 45 )) && { echo "[bg] (2) tiny $t SKIP ($(left) s left)"; continue; }
  echo "[bg] (2) tiny inkling2_$t on $BR, no env ($(left) s left) $(now)"
  docker rm -f $C >/dev/null 2>&1
  HIP_VISIBLE_DEVICES=1 INK_CONTAINER=$C timeout 50 bash "$S/run.sh" env INTARSIA_REQUIRE_GPU=1 INTARSIA_INKLING2_BACKEND=hip $T --test inkling2_$t -- \
    --nocapture --test-threads=1 > "$L/tiny-$t.log" 2>&1
  echo "[bg] tiny $t exit $? $(now)"
  grep -aE "^test .*(ok|FAILED)$|test result|packed == solo|panicked" "$L/tiny-$t.log" | cut -c1-200 | head -20 | sed "s/^/[bg tiny $t] /"
done
docker rm -f $C >/dev/null 2>&1
# (3)
if (( $(left) > 60 )); then
  echo "[bg] (3) solo $BR ($(left) s left) $(now)"
  WAITS=$(( $(left) - 45 )) srv solo-$BR $BR --max-seq 4608 --spec off
  if [ "$UP" != 0 ]; then
    timeout $(( $(left) - 3 )) python3 "$S/serve_bitwise.py" "http://$IP:8133/v1" solo "$L/solo-$BR-bitwise-solo.json" > "$L/solo-$BR-bitwise-solo.out" 2>&1
    echo "[bg] solo $BR exit $? $(now): $(head -n 1 "$L/solo-$BR-bitwise-solo.out")"
  fi
  docker rm -f $C >/dev/null 2>&1
else
  echo "[bg] (3) solo SKIP ($(left) s left)"
fi
docker rm -f $C >/dev/null 2>&1
echo "[bg] done $(now)"
