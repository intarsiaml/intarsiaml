#!/usr/bin/env bash
# Pre-stage W8 (lane ink-hip-bpf), before the speed visits: the real-shape inkling2_packed test on card 1 (nothing else loaded; the
# test now loads its models as the serve loader does with --max-batch, `LoadOptions::batched`, so no P12 copies shrink the KV):
# 17 + 2048, 4 segments around the expert split, blend tails, 2 x 2048 in the 4096-row workspace, 6 segments (4029 rows, resumed),
# 16 segments (resumed), 2 x 1024; head, every state byte, restorable_len and the decode step after, against the solo prefill.
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-bpf
S=$REPO/.scratch
L=$REPO/logs/w8
mkdir -p "$L"
C=ink-hip-bpf-run2
now() { date '+%F %T'; }
echo "[pre] real-shape inkling2_packed (card 1) $(now)"
docker rm -f $C >/dev/null 2>&1
HIP_VISIBLE_DEVICES=1 INK_CONTAINER=$C timeout ${PRE_CAP:-70} bash "$S/run.sh" env INTARSIA_REQUIRE_GPU=1 INTARSIA_INKLING2_BACKEND=hip \
  INTARSIA_INK_PACK_REAL=/models/Inkling-2-Mini cargo test --release -j8 -p intarsia-models --no-fail-fast --test inkling2_packed -- \
  --nocapture --test-threads=1 inkling2_packed_equals_solo_real > "$L/real-packed.log" 2>&1
echo "[pre] real exit $? $(now)"
grep -aE "^test |test result|packed == solo|panicked|refused|moe_split_rows|left:|right:" "$L/real-packed.log" | cut -c1-220 | head -16 | sed "s/^/[pre real] /"
docker rm -f $C >/dev/null 2>&1
echo "[pre] done $(now)"
