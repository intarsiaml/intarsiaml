#!/usr/bin/env bash
# Window 3 stage (lane ink-hip-pf2): pf2c binary, new prefill settings. (1) 4k / 16k prefill, one process per configuration:
# INTARSIA_HIP_INK_PF2 0 / 3 / 7 / 15 / 19 / 51 / 67, ZP 16 and the 32-pair tile at mask 3, then 3 and 0 again (after the
# experts' bitwise test); (2) rocprofv3 kernel trace of the 4k prefill at mask 3. Card 0. Radiance down <= 5 min.
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-pf2
S=$REPO/.scratch
OUT=$REPO/logs/w3
rm -rf "$OUT"; mkdir -p "$OUT"
M=/models/Inkling-2-Mini
B=/work/.scratch/bin/intarsia-${BIN:-pf2c}
P4="$(cat "$REPO/scripts/cmx/ids-prompts/4k-0.txt")"
P16="$P4
$P4
$P4
$P4"
NEW="INTARSIA_HIP_INK_MOE_WMMA=1 INTARSIA_HIP_INK_ATTN_FLASH=1"
G="generate -m $M --backend hip --max-seq 20480 --spec off --ignore-eos --profile-every 0 --prefill-chunk 2048"
now() { date '+%T'; }
clean() { docker rm -f ink-hip-pf2-run >/dev/null 2>&1; }
line() { grep -aoE "prefill [0-9]+ ms \([0-9.]+ tok/s\)|Error.*|panicked.*" | head -2 | tr '\n' ' '; }
echo "[stage] moe pieces / OCC bitwise test $(now)"
timeout 40 bash "$S/run.sh" env INTARSIA_REQUIRE_GPU=1 cargo test --release -j8 -p intarsia-backend-hip --test ink_moe -- --nocapture --test-threads=1 ink_moe_pf2_pieces 2>&1 | grep -aE "bitwise|test result|panicked|differs|error" | cut -c1-300
clean
for cfg in "INTARSIA_HIP_INK_PF2=0" "INTARSIA_HIP_INK_PF2=3" "INTARSIA_HIP_INK_PF2=7" "INTARSIA_HIP_INK_PF2=15" "INTARSIA_HIP_INK_PF2=19" "INTARSIA_HIP_INK_PF2=51" "INTARSIA_HIP_INK_PF2=67" "INTARSIA_HIP_INK_PF2=3 INTARSIA_HIP_INK_PF2_ZP=16" "INTARSIA_HIP_INK_PF2=3 INTARSIA_HIP_INK_MOE_WMMA=32" "INTARSIA_HIP_INK_PF2=3" "INTARSIA_HIP_INK_PF2=0"; do
  for p in 4k 16k; do
    [ $p = 4k ] && PR="$P4" || PR="$P16"
    echo "GEN $p $cfg | $(bash "$S/run.sh" env $NEW $cfg $B $G --max-tokens 2 -p "$PR" 2>&1 | line) [$(now)]"
  done
done
echo "[stage] rocprof 4k mask 3 $(now)"
mkdir -p "$OUT/rp-4k"
INTARSIA_IMAGE=intarsia-hip-prof:rocm7.2.4-rust1.90.0 bash "$S/run.sh" env $NEW INTARSIA_HIP_INK_PF2=3 rocprofv3 --kernel-trace --output-format csv -d /work/logs/w3/rp-4k -o run -- $B $G --max-tokens 2 -p "$P4" 2>&1 | line; echo
echo "[stage] done $(now)"
