#!/usr/bin/env bash
# Lane mxr, ON tatooine inside ONE fleetq slot that takes radiance down (--takes-down radiance --quiet gpu,cpu; both cards):
# the HIP gates of lane-mxr/hip-mxfp4-rows (window w4). Builds come first (cbuild.sh both, mutants.sh: cpu jobs).
# Logs: ~/intarsiaml-mxr-logs/w4. The trap starts radiance again and waits until it answers.
#   T  card 0: scripts/mxr/mxr3 timing (production int8 path vs the bf16 rows kernel, tp1 + tp2 shapes; W4A16)
#      card 1: kernels test --quick, admission, quick_coverage (dev)
#   B  card 0: rows_exact mxfp4 (rows, pair, edge) dev; I8=0 (the bf16 kernel) dev; negative control; rows_registry
#      card 1: the mutants (each MUST fail rows_exact)
#   N  card 0 base / card 1 dev: no-op 8B + 35B (plan show + 256 ids short / 4k), 27B 256 ids short / 4k / 16k + plan show
#   S  card 0 dev / card 1 base: 27B batched decode == solo step (M = 2..16)
#   P  both cards: 27B tp2 256 ids short + 4k, dev vs base (sequential)
#   R  optional: rocprofv3 kernel trace of radiance's 8-row step (skipped unless asked)
#   gates.sh [sections, default "T B N S P"]
set -uo pipefail
SECTIONS=${*:-T B N S P}
want() { [[ " $SECTIONS " == *" $1 "* ]]; }
O=$HOME/intarsiaml-mxr-logs/w4; mkdir -p "$O"
RD=$HOME/intarsiaml-mxr-dev; RB=$RD/.base
VD=intarsia-mxr-target-dev; VB=intarsia-mxr-target-base
D=$RD/deploy/tatooine/dev.sh
H=/data/models/radiance/hf-cache/hub
M8=/models/Qwen3-8B; M35=/models/Qwen3.6-35B-A3B-NVFP4
M27=/models/models--amd--Qwen3.8-27B-Quark-AWQ-MXFP4/snapshots/5233554c5fa56afda40150556b95573c2d7d29c0
now() { date '+%F %T'; }
restore() {
  systemctl --user start radiance
  for i in $(seq 1 48); do curl -s -m 3 http://localhost:8080/v1/models | grep -q '"data":\[' && break; sleep 5; done
  echo "[mxr] radiance $(systemctl --user is-active radiance) after $((i * 5)) s; models: $(curl -s -m 3 http://localhost:8080/v1/models | head -c 100)"
}
trap restore EXIT
systemctl --user stop radiance; sleep 3; echo "[mxr] radiance $(systemctl --user is-active radiance) $(now)"
dev() { # dev <repo> <vol> <gpu> <cmd...>
  local r=$1 v=$2 g=$3; shift 3
  INTARSIA_REPO=$r INTARSIA_TARGET_VOL=$v HIP_VISIBLE_DEVICES=$g $D "$@" < /dev/null
}
F='^test |^test result|FAILED|panicked|^error|G-B|SKIPPED'
dr() { # dr <gpu> <cmd...>: the dev image with the repo at /work (harness binaries)
  local g=$1; shift
  docker run --rm --device /dev/kfd --device /dev/dri --group-add "$(stat -c %g /dev/kfd)" --group-add "$(stat -c %g /dev/dri/renderD128)" \
    --group-add video --ipc host --security-opt seccomp=unconfined --user "$(id -u):$(id -g)" -e HOME=/tmp -e HIP_VISIBLE_DEVICES=$g \
    -v "$RD":/work -w /work intarsia-hip-dev:rocm7.2.4-rust1.90.0 "$@" < /dev/null
}

if want T; then
  echo "[gates] T timing + quick/admission $(now)"
  ( for sh in 17408x5120 5120x17408 12288x5120 10240x5120 6144x5120 5120x6144 1024x5120 48x5120 \
              8704x5120 5120x8704 5120x5120 3072x5120 5120x3072 512x5120 24x5120; do
      dr 0 /work/.mxrbin/mxr3 ${sh%x*} ${sh#*x} 200 3 1
    done
    for sh in 17408x5120 5120x17408 5120x6144 1024x5120; do dr 0 /work/.mxrbin/mxr3 ${sh%x*} ${sh#*x} 100 2 0; done ) > "$O/t-timing.log" 2>&1 &
  ( echo "-- kernels test --quick (dev)"; dev $RD $VD 1 /target/release/intarsia kernels test --backend hip --quick 2>&1 | tail -6
    echo "-- admission + quick_coverage (dev)"
    dev $RD $VD 1 env INTARSIA_REQUIRE_GPU=1 cargo test --release -p intarsia-backend-hip --test admission --test quick_coverage -- --test-threads=1 2>&1 | grep -aE "$F" ) > "$O/t-quick.log" 2>&1 &
  wait
  grep -aE "^## |total mismatches" "$O/t-timing.log"; cat "$O/t-quick.log"
fi

if want B; then
  echo "[gates] B bitwise + mutants $(now)"
  ( echo "-- rows_exact mxfp4 (dev, int8 path default)"
    dev $RD $VD 0 env INTARSIA_REQUIRE_GPU=1 cargo test --release -p intarsia-backend-hip --test rows_exact mxfp4 -- --test-threads=1 --nocapture 2>&1 | grep -aE "$F"
    echo "-- rows_exact mxfp4 with INTARSIA_HIP_MXFP4_I8=0 (the bf16 rows kernel)"
    dev $RD $VD 0 env INTARSIA_REQUIRE_GPU=1 INTARSIA_HIP_MXFP4_I8=0 cargo test --release -p intarsia-backend-hip --test rows_exact mxfp4 -- --test-threads=1 --nocapture 2>&1 | grep -aE "$F"
    echo "-- NEGATIVE CONTROL: INTARSIA_HIP_GB_CONTROL=matmul_mxfp4_gemv_rows=matmul_mxfp4_mma (MUST FAIL)"
    dev $RD $VD 0 env INTARSIA_REQUIRE_GPU=1 INTARSIA_HIP_GB_CONTROL=matmul_mxfp4_gemv_rows=matmul_mxfp4_mma \
      cargo test --release -p intarsia-backend-hip --test rows_exact mxfp4_rows_are -- --test-threads=1 --nocapture 2>&1 | grep -aE "$F" | head -8
    echo "-- rows_registry"
    dev $RD $VD 0 env INTARSIA_REQUIRE_GPU=1 cargo test --release -p intarsia-backend-hip --test rows_registry -- --test-threads=1 2>&1 | grep -aE "$F" ) > "$O/b-rows.log" 2>&1 &
  ( for m in clean m1-noflag m2-xscale m3-pairing m4-chainorder; do
      echo "-- mutant $m: $(sha256sum /data/intarsiaml-mxr-target-dev/mutants/$m/rows_exact | cut -c1-16) (clean must pass, every mutant must FAIL)"
      dev $RD $VD 1 env INTARSIA_REQUIRE_GPU=1 /target/mutants/$m/rows_exact mxfp4_rows_edge_inputs_are_bitwise --test-threads=1 2>&1 \
        | grep -aE "^test |test result|G-B FAIL" | head -3
    done ) > "$O/b-mutants.log" 2>&1 &
  wait
  cat "$O/b-rows.log" "$O/b-mutants.log"
fi

noop27() { # noop27 <tag> <repo> <vol> <gpu> [extra generate flags]
  local tag=$1 r=$2 v=$3 g=$4; shift 4
  for p in short 4k 16k; do
    INTARSIA_MODELS=$H dev $r $v $g /target/release/intarsia generate -m "$M27" --backend hip "$@" --max-seq 20480 --chat --no-think \
      --temperature 0 --profile-every 0 --ignore-eos --show-ids -p "$(cat $RD/scripts/mxr/prompts/$p.txt)" --max-tokens 256 \
      > /dev/null 2> "$O/gen-27-$p-$tag.err"
    grep -a '^ids:' "$O/gen-27-$p-$tag.err" > "$O/gen-27-$p-$tag.ids"
    echo "[noop27] $tag $p: $(tr ',' '\n' < "$O/gen-27-$p-$tag.ids" | wc -l) ids"
  done
}
plan27() { # plan27 <tag> <repo> <vol> <gpu>
  for j in "--json:json" ":txt"; do
    INTARSIA_MODELS=$H dev $2 $3 $4 /target/release/intarsia plan show --backend hip --model "$M27" --max-seq 8192 ${j%%:*} \
      > "$O/plan-27-$1.${j##*:}" 2> "$O/plan-27-$1.${j##*:}.err"
  done
}
if want N; then
  echo "[gates] N no-op $(now)"
  noop() { RUN="env INTARSIA_REPO=$2 INTARSIA_TARGET_VOL=$3 HIP_VISIBLE_DEVICES=$4 $D /target/release/intarsia" \
             bash $RD/scripts/hg/noop.sh run "$O" "$1" hip $M8 $M35 $RD/scripts/hg/prompts < /dev/null; }
  ( noop base $RB $VB 0; noop27 base $RB $VB 0; plan27 base $RB $VB 0 ) > "$O/n-base.log" 2>&1 &
  ( noop dev $RD $VD 1; noop27 dev $RD $VD 1; plan27 dev $RD $VD 1 ) > "$O/n-dev.log" 2>&1 &
  wait
  cat "$O/n-base.log" "$O/n-dev.log"
  bash $RD/scripts/hg/noop.sh cmp "$O" base dev
  for p in short 4k 16k; do
    if cmp -s "$O/gen-27-$p-base.ids" "$O/gen-27-$p-dev.ids" && [ -s "$O/gen-27-$p-dev.ids" ]; then echo "ids 27 $p: identical ($(tr ',' '\n' < "$O/gen-27-$p-dev.ids" | wc -l) ids)"
    else echo "ids 27 $p: DIFFER or empty"; fi
  done
  for k in json txt; do
    if cmp -s "$O/plan-27-base.$k" "$O/plan-27-dev.$k"; then echo "plan 27 $k: byte-identical"; else echo "plan 27 $k: DIFFER (diff below)"; diff "$O/plan-27-base.$k" "$O/plan-27-dev.$k" | head -20; fi
  done
fi

if want S; then
  echo "[gates] S 27B batched decode == solo step $(now)"
  T="env INTARSIA_BATCH_TEST_BACKEND=hip INTARSIA_REQUIRE_GPU=1 INTARSIA_QWEN36_DIR=$M27"
  INTARSIA_MODELS=$H dev $RD $VD 0 $T cargo test --release -p intarsia-models --test batch_cuda_real real_35b_batched_decode_is_bitwise -- --test-threads=1 --nocapture 2>&1 \
    | grep -aE "$F|row-exact" > "$O/s-27-dev.log" 2>&1 &
  INTARSIA_MODELS=$H dev $RB $VB 1 $T cargo test --release -p intarsia-models --test batch_cuda_real real_35b_batched_decode_is_bitwise -- --test-threads=1 --nocapture 2>&1 \
    | grep -aE "$F|row-exact" > "$O/s-27-base.log" 2>&1 &
  wait
  echo "-- dev"; cat "$O/s-27-dev.log"; echo "-- base"; cat "$O/s-27-base.log"
fi

if want P; then
  echo "[gates] P 27B tp2 ids $(now)"
  for tag in base dev; do
    if [ $tag = base ]; then r=$RB; v=$VB; else r=$RD; v=$VD; fi
    for p in short 4k; do
      INTARSIA_MODELS=$H dev $r $v 0,1 /target/release/intarsia generate -m "$M27" --backend hip --tp 2 --max-seq 8192 --chat --no-think \
        --temperature 0 --profile-every 0 --ignore-eos --show-ids -p "$(cat $RD/scripts/mxr/prompts/$p.txt)" --max-tokens 256 \
        > /dev/null 2> "$O/gen-27tp2-$p-$tag.err"
      grep -a '^ids:' "$O/gen-27tp2-$p-$tag.err" > "$O/gen-27tp2-$p-$tag.ids"
    done
  done
  for p in short 4k; do
    if cmp -s "$O/gen-27tp2-$p-base.ids" "$O/gen-27tp2-$p-dev.ids" && [ -s "$O/gen-27tp2-$p-dev.ids" ]; then echo "ids 27 tp2 $p: identical ($(tr ',' '\n' < "$O/gen-27tp2-$p-dev.ids" | wc -l) ids)"
    else echo "ids 27 tp2 $p: DIFFER or empty"; tail -3 "$O/gen-27tp2-$p-dev.err"; fi
  done
fi
echo "[gates] done $(now)"
