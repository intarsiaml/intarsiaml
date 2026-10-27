#!/usr/bin/env bash
# Lane LDF variant sweep, ON tatooine INSIDE a fleetq job that holds one R9700 and takes radiance down (restored by the EXIT trap):
# the mxfp4_a8 test binary with INTARSIA_HIP_MXFP4_A8_LDV = 0..31 (a8_gemm_kernel's V bits): timing_a8_tp2 at M = 2048 for every value, then the best three
# against 0 in interleaved rounds at M = 1024, 2048, 4096, then the 48-shape output hash of 0 and of the best three (must be equal).
#   BIN=/work/ldf-bin/mxfp4_a8-v1 W=w2 bash ~/intarsiaml-ldf/scripts/ldf/w2.sh
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
BIN=${BIN:-/work/ldf-bin/mxfp4_a8-v1}; W=${W:-w2}; VS=${VS:-$(seq 0 31)}; ROUNDS=${ROUNDS:-3}; TOP=${TOP:-3}
O=$HOME/intarsiaml-ldf-logs/$W; mkdir -p "$O"; T0=$(date +%s)
left() { echo $(( ${WINDOW_S:-2400} - ($(date +%s) - T0) )); }
trap restore EXIT
echo "[ldf] $W stopping radiance $(now)"; systemctl --user stop radiance < /dev/null; sleep 3
if [[ -n "${LDSBENCH:-}" ]]; then G=0 gpu 500 /work/ldf-bin/ldsbench > "$O/lds.txt" 2>&1; echo "[ldf] lds: $(wc -l < "$O/lds.txt") lines"; fi
sample() { # sample <tag> <seconds>: sclk (MHz) and power (W) of both cards every 0.1 s into $O/clk-<tag>.txt
  ( end=$(( $(date +%s) + $2 )); while [[ $(date +%s) -lt $end ]]; do
      for c in 0 1; do h=$(ls -d /sys/class/drm/card$c/device/hwmon/hwmon* | head -1); echo "$c $(( $(cat $h/freq1_input) / 1000000 )) $(( $(cat $h/power1_average) / 1000000 ))"; done; sleep 0.1; done ) > "$O/clk-$1.txt" &
}
tm() { # tm <tag> <M list> <V>
  G=0 gpu 300 env GLR_TIMING_M=$2 INTARSIA_HIP_MXFP4_A8_LDV=$3 $BIN --ignored --nocapture timing_a8_tp2 2>&1 | grep -aE "^TIME|panicked|error" > "$O/t-$1.txt"
}
sumus() { awk '/^TIME/ {s += $(NF-3)} END {printf "%.1f", s}' "$1"; }  # sum of the "us" column (the number before "us")
for v in $VS; do
  [[ $(left) -lt 600 ]] && { echo "[ldf] skip V=$v (window)"; continue; }
  tm "v$v" 2048 $v; echo "V=$v sum=$(sumus "$O/t-v$v.txt") us $(now)"
done
tm "v0-end" 2048 0; echo "V=0 (end) sum=$(sumus "$O/t-v0-end.txt") us"
# the best three (by the sum) against 0
best=$(for v in $VS; do [[ -s "$O/t-v$v.txt" && $v != 0 ]] && echo "$(sumus "$O/t-v$v.txt") $v"; done | sort -n | head -$TOP | awk '{print $2}' | tr '\n' ' ')
echo "[ldf] best: $best"
for r in $(seq 1 $ROUNDS); do
  for v in 0 $best; do
    [[ $(left) -lt 400 ]] && break
    tm "r$r-v$v" 1024,2048,4096 $v
  done
  echo "[ldf] round $r done $(now)"
done
for v in 0 $best; do
  [[ $(left) -lt 240 ]] && { echo "[ldf] skip hash V=$v (window)"; continue; }
  G=0 gpu 900 env INTARSIA_HIP_MXFP4_A8_LDV=$v $BIN --ignored --nocapture a8_mma_hashes 2>&1 | grep -a "^HASH\|panicked" > "$O/hash-v$v.txt"
  echo "[ldf] hash V=$v: $(wc -l < "$O/hash-v$v.txt") lines; differing from V=0: $(diff "$O/hash-v0.txt" "$O/hash-v$v.txt" | grep -c '^<')"
  G=0 gpu 600 env INTARSIA_HIP_MXFP4_A8_LDV=$v $BIN a8_ 2>&1 | grep -aE "^test |test result|panicked" > "$O/corr-v$v.txt"
  echo "[ldf] corr V=$v: $(grep -a 'test result' "$O/corr-v$v.txt" | head -1)"
done
if [[ -n "${SAMPLE:-}" ]]; then  # clock and power under the kernel: V=0 and the best variant, M = 4096, 600 launches
  for v in 0 $(echo $best | awk '{print $1}'); do
    sample "v$v" 40; G=0 gpu 300 env GLR_TIMING_ITERS=600 GLR_TIMING_M=4096 INTARSIA_HIP_MXFP4_A8_LDV=$v $BIN --ignored --nocapture timing_a8_tp2 2>&1 | grep -aE "^TIME" > "$O/ts-v$v.txt"; wait
    echo "[ldf] clock V=$v: $(awk '$3 > 150 {n++; f += $2; p += $3} END {printf "%d samples busy, mean sclk %.0f MHz, mean %.0f W", n, f / n, p / n}' "$O/clk-v$v.txt")"
  done
fi
echo "[ldf] $W done $(now), $(left) s of the window left"
