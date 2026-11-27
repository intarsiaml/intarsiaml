#!/usr/bin/env bash
# Background stage W5 (lane ink-hip-bpf) on card 1 during the gate-2 serve window (card 0 runs the packed servers):
#   (1) the real-shape inkling2_packed test first, on a card with nothing else loaded (2 x 2048 in the 4096-row workspace, 6 and 16
#       segments, resumed segments, 2 x 1024; W4's ran beside a server's freed memory and got a 1121-token KV cap);
#   (2) G2 long of the branch (W4 ran short: 4.491e-3, = main);
#   (3) admission of the touched ops on the branch binary (`kernels test --backend hip --op ...`). All within BG_CAP s.
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-bpf
S=$REPO/.scratch
L=$REPO/logs/w5
mkdir -p "$L"
M=/models/Inkling-2-Mini
BR=${BR:-b10}
C=ink-hip-bpf-run2
now() { date '+%F %T'; }
T0=$(date +%s); CAPT=${BG_CAP:-180}
left() { echo $(( CAPT - ($(date +%s) - T0) )); }
T="cargo test --release -j8 -p intarsia-models --no-fail-fast"
r1() {  # seconds, command...
  local s=$1; shift
  docker rm -f $C >/dev/null 2>&1
  HIP_VISIBLE_DEVICES=1 INK_CONTAINER=$C timeout "$s" bash "$S/run.sh" "$@"
}
echo "[bg] card 1 VRAM before: $(cat /sys/class/drm/card*/device/mem_info_vram_used 2>/dev/null | tr '\n' ' ')"
s=$(( $(left) - 60 )); (( s > 120 )) && s=120
echo "[bg] (1) real-shape inkling2_packed, cap $s s $(now)"
r1 $s env INTARSIA_REQUIRE_GPU=1 INTARSIA_INKLING2_BACKEND=hip INTARSIA_INK_PACK_REAL=$M $T --test inkling2_packed -- --nocapture --test-threads=1 \
  inkling2_packed_equals_solo_real > "$L/real-packed.log" 2>&1
echo "[bg] real exit $? $(now)"
grep -aE "^test |test result|packed == solo|panicked|refused|moe_split_rows|left:|right:" "$L/real-packed.log" | cut -c1-220 | head -16 | sed "s/^/[bg real] /"
if (( $(left) > 35 )); then
  echo "[bg] (2) G2 long $BR [$(now)] $(r1 $(( $(left) - 5 )) /work/.scratch/bin/intarsia-$BR accuracy -m $M --backend hip --max-seq 4096 --mode prefill \
    --golden golden/oracle/inkling2-mini-oracle-w4a16-long-top20.safetensors 2>&1 | grep -aE "prefill floor|floor verdict|Error|panicked" | cut -c1-300 | tr '\n' ' ')"
else
  echo "[bg] (2) G2 long SKIP ($(left) s left)"
fi
for op in conv_mix moe_experts; do
  (( $(left) < 20 )) && { echo "[bg] ADMIT $op SKIP ($(left) s left)"; continue; }
  echo "[bg] (3) ADMIT $op [$(now)]"
  r1 $(( $(left) - 3 )) /work/.scratch/bin/intarsia-$BR kernels test --backend hip --op $op 2>&1 | grep -aE "ADMITTED|REJECT|FAIL|NO CASES|kernels,|Error|panicked" | head -8 | sed "s/^/[bg admit $op] /"
done
docker rm -f $C >/dev/null 2>&1
echo "[bg] done $(now)"
