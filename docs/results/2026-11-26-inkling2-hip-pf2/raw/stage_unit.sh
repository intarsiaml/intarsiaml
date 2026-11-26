#!/usr/bin/env bash
# Window 2 stage (lane ink-hip-pf2): pf2a binary. (1) attention bench at 2048 rows (prologue, flash; PF2 0 vs 3); (2) 4k prefill
# end to end for INTARSIA_HIP_INK_PF2 = 0 / 1 / 3 / 7 / 15 with the new settings; (3) rocprofv3 kernel trace of the 4k prefill at
# the default mask; (4) the bitwise tests (pf2_bitwise_prefill, ink_moe_pf2_pieces), capped. Budget: radiance down <= 5 min.
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-pf2
S=$REPO/.scratch
OUT=$REPO/logs/w2
rm -rf "$OUT"; mkdir -p "$OUT"
M=/models/Inkling-2-Mini
B=/work/.scratch/bin/intarsia-${BIN:-pf2a}
P4="$(cat "$REPO/scripts/cmx/ids-prompts/4k-0.txt")"
NEW="INTARSIA_HIP_INK_MOE_WMMA=1 INTARSIA_HIP_INK_ATTN_FLASH=1"
G="generate -m $M --backend hip --max-seq 20480 --spec off --ignore-eos --profile-every 0 --prefill-chunk 2048"
now() { date '+%T'; }
clean() { docker rm -f ink-hip-pf2-run >/dev/null 2>&1; }  # a capped run leaves its container behind (timeout kills only the client)
line() { grep -aoE "prefill [0-9]+ ms \([0-9.]+ tok/s\)|Error.*|panicked.*" | head -2 | tr '\n' ' '; }
echo "[stage] bench $(now)"
timeout 60 bash "$S/run.sh" env INK_PF2_BENCH=1 INTARSIA_REQUIRE_GPU=1 cargo test --release -j8 -p intarsia-backend-hip --test ink_attn -- --nocapture --test-threads=1 bench_pf2_prefill 2>&1 | grep -aE "PF2BENCH|test result|panicked|error" | cut -c1-240
clean
for m in 0 1 3 7 15; do
  echo "GEN4k pf2=$m | $(bash "$S/run.sh" env $NEW INTARSIA_HIP_INK_PF2=$m $B $G --max-tokens 2 -p "$P4" 2>&1 | line) [$(now)]"
done
echo "[stage] rocprof 4k default mask $(now)"
mkdir -p "$OUT/rp-4k"
INTARSIA_IMAGE=intarsia-hip-prof:rocm7.2.4-rust1.90.0 bash "$S/run.sh" env $NEW rocprofv3 --kernel-trace --output-format csv -d /work/logs/w2/rp-4k -o run -- $B $G --max-tokens 2 -p "$P4" 2>&1 | line; echo
echo "[stage] attn bitwise test $(now)"
timeout 85 bash "$S/run.sh" env INTARSIA_REQUIRE_GPU=1 cargo test --release -j8 -p intarsia-backend-hip --test ink_attn -- --nocapture --test-threads=1 pf2_bitwise_prefill 2>&1 | grep -aE "bitwise|test result|panicked|differs|error" | cut -c1-300
rc=${PIPESTATUS[0]}; clean; echo "attn test exit $rc (124 = cap) [$(now)]"
echo "[stage] moe pieces test $(now)"
timeout 75 bash "$S/run.sh" env INTARSIA_REQUIRE_GPU=1 cargo test --release -j8 -p intarsia-backend-hip --test ink_moe -- --nocapture --test-threads=1 ink_moe_pf2_pieces 2>&1 | grep -aE "bitwise|test result|panicked|differs|error" | cut -c1-300
rc=${PIPESTATUS[0]}; clean; echo "moe test exit $rc (124 = cap) [$(now)]"
echo "[stage] done $(now)"
