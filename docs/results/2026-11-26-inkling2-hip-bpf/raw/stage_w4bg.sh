#!/usr/bin/env bash
# Background stage W4 (lane ink-hip-bpf) on card 1 during the gate-2 serve window: the branch's hip tiny suites (b8 test builds:
# inkling2_batch, inkling2, inkling2_packed at the defaults and with INTARSIA_HIP_INK_PF2=0), then stage_g2.sh (G2 short / long,
# main2 and b8; admission of the touched ops), all within BG_CAP s.
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-bpf
S=$REPO/.scratch
L=$REPO/logs/w4
mkdir -p "$L"
now() { date '+%F %T'; }
T0=$(date +%s); CAPT=${BG_CAP:-220}
T="cargo test --release -j8 -p intarsia-models --no-fail-fast"
r1() { HIP_VISIBLE_DEVICES=1 INK_CONTAINER=ink-hip-bpf-run2 timeout 45 bash "$S/run.sh" env INTARSIA_REQUIRE_GPU=1 INTARSIA_INKLING2_BACKEND=hip "$@"; }
echo "[bg] tiny inkling2_batch + inkling2 $(now)"
r1 $T --test inkling2_batch --test inkling2 -- --test-threads=1 > "$L/tiny-suites.log" 2>&1
grep -aE "FAILED|test result" "$L/tiny-suites.log" | head -10 | sed 's/^/[bg tiny] /'
for arm in default pf2off; do
  e=""; [ $arm = pf2off ] && e="INTARSIA_HIP_INK_PF2=0"
  echo "[bg] tiny inkling2_packed $arm $(now)"
  r1 env $e $T --test inkling2_packed -- --nocapture --test-threads=1 inkling2_packed_equals_solo_tiny inkling2_packed_engine inkling2_packed_refuses > "$L/tiny-packed-$arm.log" 2>&1
  grep -aE "^test |test result|packed == solo|panicked|refused" "$L/tiny-packed-$arm.log" | cut -c1-200 | head -14 | sed "s/^/[bg packed $arm] /"
done
echo "[bg] real-shape inkling2_packed (2 x 2048 in the 4096-row workspace) $(now)"
r1 env INTARSIA_INK_PACK_REAL=/models/Inkling-2-Mini $T --test inkling2_packed -- --nocapture --test-threads=1 inkling2_packed_equals_solo_real > "$L/real-packed.log" 2>&1
grep -aE "^test |test result|packed == solo|panicked|refused|moe_split_rows" "$L/real-packed.log" | cut -c1-200 | head -12 | sed "s/^/[bg real] /"
left=$(( CAPT - ($(date +%s) - T0) ))
echo "[bg] G2 + admission, $left s left $(now)"
MA=main2 BR=b8 BG_CAP=$left bash "$S/stage_g2.sh"
