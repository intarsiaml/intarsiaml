#!/usr/bin/env bash
# Window 11 stage (lane ink-hip-batch, the last window, <= 1:30 down): branch ink/hip-batch-mixmode rebased onto main after #529.
# (1) the coordinator's command, with INTARSIA_REQUIRE_GPU=1 (a missing GPU fails instead of skipping) and --nocapture (the switch print):
#     cargo test --release -p intarsia-backend-hip --test ink_rows -- --test-threads=1
#     its conv_mix test asserts, per leg, that the row-table TU reads the mix mode just set (ik_ink_mix_rows_switches);
# (2) the negative control: the same test built without the shared switch (.scratch/bin/ink_rows-neg): its mode-0 leg must FAIL.
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-batch
S=$REPO/.scratch
now() { date '+%F %T'; }
echo "[stage] ink_rows (fix) $(now)"
bash "$S/run.sh" env INTARSIA_REQUIRE_GPU=1 cargo test --release -p intarsia-backend-hip --test ink_rows -- --test-threads=1 --nocapture 2>&1 \
  | grep -aE "^test |test result|row-table calls|switches|panicked|assert|error(\[|:)" | head -20
echo "[stage] ink_rows-neg conv_mix (control: must fail at mode 0) $(now)"
bash "$S/run.sh" env INTARSIA_REQUIRE_GPU=1 /work/.scratch/bin/ink_rows-neg --test-threads=1 --nocapture conv_mix_rows_bitwise_solo 2>&1 \
  | grep -aE "^test |test result|switches|panicked|assert|left|right" | head -12
echo "[stage] done $(now)"
