#!/usr/bin/env bash
# Lane mr, ON tatooine inside a fleetq slot that takes radiance down (both cards held): the HIP gates of lane-mr/hip-mxfp4-rows.
# Builds come first, from scripts/mr/build.sh (cpu job). Logs go to ~/mr-logs/gates. The trap starts radiance again.
#   S1 card 0 alone : timing_mxfp4_rows (old gemv vs the rows kernel, M = 1..16, 27B shapes)
#   S2 parallel     : card 0 base no-op, card 1 dev no-op (batching off: 8B, 35B, 27B plan show text/json + 256 greedy ids, short and 4k)
#   S3 parallel     : card 0 rows_exact (mxfp4) + negative control + rows_registry; card 1 kernels test --quick
#   S4 parallel     : 27B batched decode == solo step (M = 2..16) with the dev tree (card 0) and the base tree (card 1)
set -uo pipefail
O=$HOME/mr-logs/gates; mkdir -p "$O"
S=$HOME/mr-hip/scripts
D=$HOME/mr-hip/deploy/tatooine/dev.sh
VD=intarsia-cargo-target-mr; VB=intarsia-cargo-target-mrbase
RD=$HOME/mr-hip; RB=$HOME/mr-hip/.base
H=/data/models/radiance/hf-cache/hub
M8=/models/Qwen3-8B; M35=/models/Qwen3.6-35B-A3B-NVFP4
M27=/models/models--amd--Qwen3.8-27B-Quark-AWQ-MXFP4/snapshots/5233554c5fa56afda40150556b95573c2d7d29c0
now() { date '+%F %T'; }
trap 'bash $S/mr/win.sh restore' EXIT
bash $S/mr/win.sh stop
dev() { # dev <repo> <vol> <gpu> <cmd...>
  r=$1; v=$2; g=$3; shift 3
  INTARSIA_REPO=$r INTARSIA_TARGET_VOL=$v HIP_VISIBLE_DEVICES=$g $D "$@" < /dev/null
}
F='^test |^test result|FAILED|panicked|^error|G-B|SKIPPED'

echo "[gates] S1 timing $(now)"
dev $RD $VD 0 env INTARSIA_REQUIRE_GPU=1 cargo test --release -p intarsia-backend-hip --test rows_exact -- --ignored --nocapture timing_mxfp4_rows 2>&1 \
  | grep -aE "^TIME mxfp4|^test |panicked|^error" | tee "$O/timing-mxfp4.log"

echo "[gates] S2 no-op $(now)"
noop() { # noop <tag> <repo> <vol> <gpu>
  RUN="env INTARSIA_REPO=$2 INTARSIA_TARGET_VOL=$3 HIP_VISIBLE_DEVICES=$4 $D /target/release/intarsia" \
    bash $RD/scripts/hg/noop.sh run "$O" "$1" hip $M8 $M35 $RD/scripts/hg/prompts < /dev/null
}
noop27() { # noop27 <tag> <repo> <vol> <gpu>
  for p in short 4k; do
    INTARSIA_MODELS=$H INTARSIA_REPO=$2 INTARSIA_TARGET_VOL=$3 HIP_VISIBLE_DEVICES=$4 $D /target/release/intarsia generate -m "$M27" --backend hip \
      --max-seq 8192 --chat --no-think --temperature 0 --profile-every 0 --ignore-eos --show-ids -p "$(cat $RD/scripts/hg/prompts/$p.txt)" \
      --max-tokens 256 < /dev/null > /dev/null 2> "$O/gen-27-$p-$1.err"
    grep -a '^ids:' "$O/gen-27-$p-$1.err" > "$O/gen-27-$p-$1.ids"
    echo "[noop27] $1 $p: $(tr ',' '\n' < "$O/gen-27-$p-$1.ids" | wc -l) ids"
  done
  for j in "--json:json" ":txt"; do
    INTARSIA_MODELS=$H INTARSIA_REPO=$2 INTARSIA_TARGET_VOL=$3 HIP_VISIBLE_DEVICES=$4 $D /target/release/intarsia plan show --backend hip --model "$M27" \
      --max-seq 8192 ${j%%:*} < /dev/null > "$O/plan-27-$1.${j##*:}" 2> "$O/plan-27-$1.${j##*:}.err"
  done
}
( noop base $RB $VB 0; noop27 base $RB $VB 0 ) > "$O/noop-base.log" 2>&1 &
( noop dev $RD $VD 1; noop27 dev $RD $VD 1 ) > "$O/noop-dev.log" 2>&1 &
wait
cat "$O/noop-base.log" "$O/noop-dev.log"
bash $RD/scripts/hg/noop.sh cmp "$O" base dev
for p in short 4k; do cmp -s "$O/gen-27-$p-base.ids" "$O/gen-27-$p-dev.ids" && [ -s "$O/gen-27-$p-dev.ids" ] && echo "ids 27 $p: identical" || echo "ids 27 $p: DIFFER or empty"; done
for k in json txt; do cmp -s "$O/plan-27-base.$k" "$O/plan-27-dev.$k" && [ -s "$O/plan-27-dev.$k" ] && echo "plan 27 $k: byte-identical" || echo "plan 27 $k: DIFFER or empty"; done

echo "[gates] S3 rows_exact + quick $(now)"
(
  echo "-- rows_exact mxfp4 (dev)"
  dev $RD $VD 0 env INTARSIA_REQUIRE_GPU=1 cargo test --release -p intarsia-backend-hip --test rows_exact mxfp4 -- --test-threads=1 --nocapture 2>&1 | grep -aE "$F"
  echo "-- NEGATIVE CONTROL: the mma kernel where the rows kernel runs (this MUST FAIL)"
  dev $RD $VD 0 env INTARSIA_REQUIRE_GPU=1 INTARSIA_HIP_GB_CONTROL=matmul_mxfp4_gemv_rows=matmul_mxfp4_mma \
    cargo test --release -p intarsia-backend-hip --test rows_exact mxfp4_rows -- --test-threads=1 --nocapture 2>&1 | grep -aE "$F"
  echo "-- rows_registry"
  dev $RD $VD 0 env INTARSIA_REQUIRE_GPU=1 cargo test --release -p intarsia-backend-hip --test rows_registry -- --test-threads=1 --nocapture 2>&1 | grep -aE "$F"
) > "$O/g-rows.log" 2>&1 &
( echo "-- kernels test --quick"; dev $RD $VD 1 /target/release/intarsia kernels test --backend hip --quick 2>&1 | tail -4 ) > "$O/g-quick.log" 2>&1 &
wait
cat "$O/g-rows.log" "$O/g-quick.log"

echo "[gates] S4 27B batched decode == solo step $(now)"
T="env INTARSIA_BATCH_TEST_BACKEND=hip INTARSIA_REQUIRE_GPU=1 INTARSIA_QWEN36_DIR=$M27"
( INTARSIA_MODELS=$H dev $RD $VD 0 $T cargo test --release -p intarsia-models --test batch_cuda_real real_35b_batched_decode_is_bitwise -- --test-threads=1 --nocapture 2>&1 \
    | grep -aE "$F|batch plan notes|bitwise in|not row-exact|row-exact" ) > "$O/g-27-dev.log" 2>&1 &
( INTARSIA_MODELS=$H dev $RB $VB 1 $T cargo test --release -p intarsia-models --test batch_cuda_real real_35b_batched_decode_is_bitwise -- --test-threads=1 --nocapture 2>&1 \
    | grep -aE "$F|batch plan notes|bitwise in|not row-exact|row-exact" ) > "$O/g-27-base.log" 2>&1 &
wait
echo "== 27B batched decode, dev"; cat "$O/g-27-dev.log"; echo "== 27B batched decode, base"; cat "$O/g-27-base.log"
echo "[gates] done $(now)"
