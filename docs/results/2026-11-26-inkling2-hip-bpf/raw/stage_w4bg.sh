#!/usr/bin/env bash
# Background stage W4 (lane ink-hip-bpf) on card 1 during the gate-2 serve window (card 0 runs the branch's solo and packed servers):
#   (1) gate 3, server level: main's solo server (no --max-batch, --spec off), the 16 real prompts one at a time (serve_bitwise.py
#       solo, run here on tatooine against the container), compared later with the branch's solo replies (card 0);
#   (2) the branch's hip tiny suites (b9 test builds): inkling2_batch + inkling2, inkling2_packed at the defaults and with
#       INTARSIA_HIP_INK_PF2=0;
#   (3) the real-shape inkling2_packed test (2 x 2048 in the 4096-row workspace, 6 and 16 segments, resumed);
#   (4) G2 short / long of the branch if time is left. All within BG_CAP s.
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-bpf
S=$REPO/.scratch
L=$REPO/logs/w4
mkdir -p "$L"
M=/models/Inkling-2-Mini
MA=${MA:-main3}; BR=${BR:-b9}
C=ink-hip-bpf-run2
now() { date '+%F %T'; }
T0=$(date +%s); CAPT=${BG_CAP:-230}
left() { echo $(( CAPT - ($(date +%s) - T0) )); }
T="cargo test --release -j8 -p intarsia-models --no-fail-fast"
r1() {  # seconds, command...
  local s=$1; shift
  docker rm -f $C >/dev/null 2>&1
  HIP_VISIBLE_DEVICES=1 INK_CONTAINER=$C timeout "$s" bash "$S/run.sh" "$@"
}
# (1) main's solo server
echo "[bg] (1) solo server $MA $(now)"
docker rm -f $C >/dev/null 2>&1
HIP_VISIBLE_DEVICES=1 INK_CONTAINER=$C nohup bash "$S/run.sh" bash -c "HOME=/work/.scratch/home exec /work/.scratch/bin/intarsia-$MA serve \
  -m $M --backend hip --host 0.0.0.0 --port 8133 --model-id ink --max-seq 4608 --spec off" > "$L/solo-$MA.log" 2>&1 < /dev/null &
up=0
for i in $(seq 1 120); do
  ip=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' $C 2>/dev/null)
  if [ -n "$ip" ] && curl -sf -m 2 "http://$ip:8133/health" > /dev/null 2>&1; then up=$i; break; fi
  sleep 1
done
echo "[bg] solo $MA up after ${up}s (0 = not up) $(now)"
if [ "$up" != 0 ]; then
  timeout 100 python3 "$S/serve_bitwise.py" "http://$ip:8133/v1" solo "$L/solo-$MA-bitwise-solo.json" > "$L/solo-$MA-bitwise-solo.out" 2>&1
  echo "[bg] solo $MA bitwise solo exit $? $(now): $(head -n 1 "$L/solo-$MA-bitwise-solo.out")"
fi
grep -aE 'continuous batching|packed prefill|admission|admitted|Error|error' "$L/solo-$MA.log" | head -5 | sed 's/^/[bg solo] /'
docker rm -f $C >/dev/null 2>&1
# (2) tiny suites
echo "[bg] (2) tiny inkling2_batch + inkling2 ($(left) s left) $(now)"
r1 50 env INTARSIA_REQUIRE_GPU=1 INTARSIA_INKLING2_BACKEND=hip $T --test inkling2_batch --test inkling2 -- --test-threads=1 > "$L/tiny-suites.log" 2>&1
grep -aE "FAILED|test result" "$L/tiny-suites.log" | head -10 | sed 's/^/[bg tiny] /'
for arm in default pf2off; do
  e=""; [ $arm = pf2off ] && e="INTARSIA_HIP_INK_PF2=0"
  echo "[bg] tiny inkling2_packed $arm ($(left) s left) $(now)"
  r1 40 env $e INTARSIA_REQUIRE_GPU=1 INTARSIA_INKLING2_BACKEND=hip $T --test inkling2_packed -- --nocapture --test-threads=1 \
    inkling2_packed_equals_solo_tiny inkling2_packed_engine inkling2_packed_refuses inkling2_packed_intermediates > "$L/tiny-packed-$arm.log" 2>&1
  grep -aE "^test |test result|packed == solo|panicked|refused|compared" "$L/tiny-packed-$arm.log" | cut -c1-200 | head -16 | sed "s/^/[bg packed $arm] /"
done
# (3) real shapes
s=$(( $(left) - 5 )); (( s > 110 )) && s=110
if (( s > 40 )); then
  echo "[bg] (3) real-shape inkling2_packed, cap $s s $(now)"
  r1 $s env INTARSIA_REQUIRE_GPU=1 INTARSIA_INKLING2_BACKEND=hip INTARSIA_INK_PACK_REAL=$M $T --test inkling2_packed -- --nocapture --test-threads=1 \
    inkling2_packed_equals_solo_real > "$L/real-packed.log" 2>&1
  echo "[bg] real exit $? $(now)"
  grep -aE "^test |test result|packed == solo|panicked|refused|moe_split_rows" "$L/real-packed.log" | cut -c1-200 | head -14 | sed "s/^/[bg real] /"
else
  echo "[bg] (3) real-shape SKIP ($(left) s left)"
fi
# (4) G2 of the branch
for g in short long; do
  (( $(left) < 35 )) && { echo "[bg] G2 $g $BR SKIP ($(left) s left)"; continue; }
  echo "[bg] G2 $g $BR [$(now)] $(r1 $(( $(left) - 5 )) /work/.scratch/bin/intarsia-$BR accuracy -m $M --backend hip --max-seq 4096 --mode prefill \
    --golden golden/oracle/inkling2-mini-oracle-w4a16-$g-top20.safetensors 2>&1 | grep -aE "prefill floor|floor verdict|Error|panicked" | cut -c1-300 | tr '\n' ' ')"
done
docker rm -f $C >/dev/null 2>&1
echo "[bg] done $(now)"
