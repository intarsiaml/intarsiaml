#!/usr/bin/env bash
# Window 3 stage (lane ink-hip-pf2): pf2b binary, new prefill settings. (1) 4k / 16k prefill, one process per configuration:
# INTARSIA_HIP_INK_PF2 0 / 3 / 7 / 15 / 19 / 51 and ZP 16 at mask 3, then 0 and 3 again (after the OCC bitwise test); (2) the experts bench at T = 2048 (tiles 32 /
# 64); (3) rocprofv3 kernel trace of the 4k prefill at mask 3. Card 0. Radiance down <= 5 min.
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-pf2
S=$REPO/.scratch
OUT=$REPO/logs/w3
rm -rf "$OUT"; mkdir -p "$OUT"
M=/models/Inkling-2-Mini
B=/work/.scratch/bin/intarsia-${BIN:-pf2b}
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
for cfg in "INTARSIA_HIP_INK_PF2=0" "INTARSIA_HIP_INK_PF2=3" "INTARSIA_HIP_INK_PF2=7" "INTARSIA_HIP_INK_PF2=15" "INTARSIA_HIP_INK_PF2=19" "INTARSIA_HIP_INK_PF2=51" "INTARSIA_HIP_INK_PF2=3 INTARSIA_HIP_INK_PF2_ZP=16" "INTARSIA_HIP_INK_PF2=0" "INTARSIA_HIP_INK_PF2=3"; do
  for p in 4k 16k; do
    [ $p = 4k ] && PR="$P4" || PR="$P16"
    echo "GEN $p $cfg | $(bash "$S/run.sh" env $NEW $cfg $B $G --max-tokens 2 -p "$PR" 2>&1 | line) [$(now)]"
  done
done
echo "[stage] experts bench $(now)"
timeout 60 bash "$S/run.sh" env INK_MOE_BENCH=1 INK_MOE_BENCH_T=2048 INK_MOE_BENCH_NOSWEEP=1 INTARSIA_REQUIRE_GPU=1 cargo test --release -j8 -p intarsia-backend-hip --test ink_moe -- --nocapture --test-threads=1 ink_moe_bench 2>&1 | grep -aE "INKMOEBENCH|test result|panicked" | cut -c1-260
clean
echo "[stage] rocprof 4k mask 3 $(now)"
mkdir -p "$OUT/rp-4k"
INTARSIA_IMAGE=intarsia-hip-prof:rocm7.2.4-rust1.90.0 bash "$S/run.sh" env $NEW INTARSIA_HIP_INK_PF2=3 rocprofv3 --kernel-trace --output-format csv -d /work/logs/w3/rp-4k -o run -- $B $G --max-tokens 2 -p "$P4" 2>&1 | line; echo
echo "[stage] done $(now)"
