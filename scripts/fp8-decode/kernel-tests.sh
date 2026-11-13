#!/usr/bin/env bash
# fp8-decode: on spark1 (under fleetq; FLEETQ_JOB passed through), the attention gates and the 64k decode-attention micro-timing
# of the build in ~/intarsia-fp8-decode/src. MODE=gates | timing | both (default both).
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
ssh -o BatchMode=yes spark1 "export FLEETQ_JOB=$FLEETQ_JOB" 'PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH; export CARGO_BUILD_JOBS=4 MAX_JOBS=4 INTARSIA_REQUIRE_GPU=1
cd ~/intarsia-fp8-decode/src || exit 1
T="cargo test --release --locked -p intarsia-backend-cuda"
if [ "'"${MODE:-both}"'" != timing ]; then
  for t in attn_f8kv rows_exact row_invariance attention row_tables; do
    echo "== gate $t"; $T --test $t 2>&1 | grep -E -i "^test result|FAILED|panicked|error(\[|:)|skip" | head -20
  done
fi
if [ "'"${MODE:-both}"'" != gates ]; then
  for pos in 4096 32768 65536; do
    for kv in bf16 fp8_e4m3; do
      for pf in 2 1; do
        [ $kv = bf16 ] && [ $pf = 1 ] && continue
        echo "== timing pos=$pos kv=$kv pf=$pf"
        INTARSIA_ATTN_F8_PF=$pf ATTNT_KV=$kv ATTNT_POS=$pos ATTNT_M=1,2 $T --test row_tables -- --ignored --nocapture --exact timing_attn_rows 2>&1 | grep ATTNT
      done
    done
  done
fi'
