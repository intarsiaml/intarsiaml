#!/usr/bin/env bash
# Lane wg merge gates, ON tatooine inside ONE fleetq job that takes radiance down (both cards), run through wgm-win.sh (restores radiance):
#   A  batching-off / knob-unset no-op: main build (card 0) vs merged-head build (card 1), plan show text+json and 256 greedy ids (8B, 35B, short + 4k)
#   B1 card 1: batch_recurrent --test-threads=1, kernels test --quick (hip), rows_exact + wmma_folds + frag_layout (knob off and opt1 builds)
#   B0 card 0: 35B batch_cuda_real batched==solo ("batched" tests) knob off, then the same with option 1 on, then the --ignored long-prompt test with option 1 on
#   C  card 1 (after B1): 35B MTP k=2 identity with option 1 ON: ids == spec off (same build, option 1), acceptance counts == knob-off MTP;
#      natural-v1 128 / 1k. Info: option 1 spec-off ids vs knob-off spec-off ids (solo output may change once).
# usage: gates.sh [A|B|C|ABC]   (default ABC). Logs: ~/wgm-logs/.
set -uo pipefail
W=${1:-ABC}
O=$HOME/wgm-logs; mkdir -p "$O"
D=$HOME/intarsiaml-wgm/deploy/tatooine/dev.sh
M8=/models/Qwen3-8B; M35=/models/Qwen3.6-35B-A3B-NVFP4
now() { date '+%F %T'; }
OPT1="INTARSIA_HIP_WMMA_ROWS=1 INTARSIA_HIP_WMMA_TABLE=fp8a8=all,bf16=all"
if [[ $W == *A* ]]; then
  noop() { # tag tree vol gpu
    RUN="env INTARSIA_REPO=$HOME/intarsiaml-$2 INTARSIA_TARGET_VOL=$3 HIP_VISIBLE_DEVICES=$4 $D /target/release/intarsia" \
      bash $HOME/intarsiaml-wgm/scripts/hg/noop.sh run "$O/noop" "$1" hip $M8 $M35 $HOME/intarsiaml-wgm/scripts/hg/prompts < /dev/null
  }
  echo "[gates] A no-op $(now)"
  noop base wgm-base intarsia-cargo-target-wgmbase 0 > "$O/noop-base.log" 2>&1 &
  noop dev wgm intarsia-cargo-target-wgm 1 > "$O/noop-dev.log" 2>&1 &
  wait
  cat "$O/noop-base.log" "$O/noop-dev.log"
  bash $HOME/intarsiaml-wgm/scripts/hg/noop.sh cmp "$O/noop" base dev | tee "$O/noop-cmp.txt"
fi
if [[ $W == *B* ]]; then
  dev() { g=$1; shift; INTARSIA_REPO=$HOME/intarsiaml-wgm INTARSIA_TARGET_VOL=intarsia-cargo-target-wgm HIP_VISIBLE_DEVICES=$g $D "$@" < /dev/null; }
  T="env INTARSIA_BATCH_TEST_BACKEND=hip INTARSIA_REQUIRE_GPU=1 INTARSIA_QWEN36_DIR=$M35"
  mtp() { # card 1: MTP k=2 identity with option 1 on, natural-v1 128 / 1k
    gen() { # tag n envs... -- generate args
      local tag=$1 n=$2; shift 2; local envs=(); while [ "$1" != -- ]; do envs+=("$1"); shift; done; shift
      dev 1 env INTARSIA_SPEC_COSTGATE= "${envs[@]}" /target/release/intarsia generate -m $M35 --backend hip --max-seq 4096 --profile-every 0 --ignore-eos --show-ids \
        --max-tokens 256 "$@" -p "$(cat $HOME/intarsiaml-wgm/scripts/wgm/prompts/n$n.txt)" > "$O/mtp-$tag-n$n.out" 2> "$O/mtp-$tag-n$n.err"
      grep -a '^ids:' "$O/mtp-$tag-n$n.err" > "$O/mtp-$tag-n$n.ids"
      echo "$tag n$n md5=$(md5sum < "$O/mtp-$tag-n$n.ids" | cut -c1-8) $(grep -a 'finish=' "$O/mtp-$tag-n$n.err" | tail -n 1 | sed 's/.*| decode/decode/' | cut -c1-40) | $(grep -a '^spec:' "$O/mtp-$tag-n$n.err" | tail -n 1 | cut -c1-110)"
    }
    local n a b
    echo "-- C: 35B MTP k=2 identity, option 1 on ($(now))"
    for n in 128 1024; do
      gen off-spec-off $n NOKNOB=1 -- --spec off
      gen off-mtp2 $n NOKNOB=1 -- --spec mtp --spec-k 2
      gen opt1-spec-off $n INTARSIA_HIP_WMMA_ROWS=1 INTARSIA_HIP_WMMA_TABLE=fp8a8=all,bf16=all -- --spec off
      gen opt1-mtp2 $n INTARSIA_HIP_WMMA_ROWS=1 INTARSIA_HIP_WMMA_TABLE=fp8a8=all,bf16=all -- --spec mtp --spec-k 2
      cmp -s "$O/mtp-opt1-spec-off-n$n.ids" "$O/mtp-opt1-mtp2-n$n.ids" && [ -s "$O/mtp-opt1-mtp2-n$n.ids" ] && echo "PASS n$n: option 1 MTP k=2 ids == option 1 spec off" || echo "FAIL n$n: option 1 MTP k=2 ids != option 1 spec off"
      cmp -s "$O/mtp-off-spec-off-n$n.ids" "$O/mtp-off-mtp2-n$n.ids" && [ -s "$O/mtp-off-mtp2-n$n.ids" ] && echo "PASS n$n (control): knob-off MTP k=2 ids == knob-off spec off" || echo "FAIL n$n (control): knob-off MTP k=2 ids != knob-off spec off"
      a=$(grep -a '^spec:' "$O/mtp-off-mtp2-n$n.err" | tail -n 1 | sed 's/(acceptance.*//'); b=$(grep -a '^spec:' "$O/mtp-opt1-mtp2-n$n.err" | tail -n 1 | sed 's/(acceptance.*//')
      [ -n "$a" ] && [ "$a" = "$b" ] && echo "PASS n$n: option 1 acceptance counts == knob-off ($b)" || echo "FAIL n$n: acceptance counts differ: knob-off [$a] option 1 [$b]"
      cmp -s "$O/mtp-off-spec-off-n$n.ids" "$O/mtp-opt1-spec-off-n$n.ids" && echo "info n$n: option 1 spec-off ids == knob-off spec-off ids" || echo "info n$n: option 1 spec-off ids differ from knob-off (solo output changes once, expected)"
    done
  }
  
  echo "[gates] B $(now)"
  (
    echo "-- batch_recurrent"
    dev 1 $T cargo test --release --locked -p intarsia-models --test batch_recurrent -- --test-threads=1 --nocapture 2>&1 | grep -aE "^test |^test result|FAILED|panicked|^error"
    echo "-- kernels test --quick (hip)"
    dev 1 /target/release/intarsia kernels test --backend hip --quick 2>&1 | tail -n 4
    echo "-- kernels test --quick (hip), option 1 ON (the WMMA kernels are exercised only with the knob)"
    dev 1 env $OPT1 /target/release/intarsia kernels test --backend hip --quick 2>&1 | grep -aE "wmma|FAIL|NO CASES|kernels,|cases in" | tail -n 24
    echo "-- rows_exact + wmma_folds + frag_layout, knob unset"
    dev 1 env INTARSIA_REQUIRE_GPU=1 INTARSIA_HIP_DEVICE=0 cargo test --release --locked -p intarsia-backend-hip --test rows_exact --test wmma_folds --test frag_layout --no-fail-fast -- --test-threads=1 2>&1 | grep -aE "^test result|FAILED|panicked"
    echo "-- same, option 1 on"
    dev 1 env $OPT1 INTARSIA_REQUIRE_GPU=1 INTARSIA_HIP_DEVICE=0 cargo test --release --locked -p intarsia-backend-hip --test rows_exact --test wmma_folds --test frag_layout --no-fail-fast -- --test-threads=1 2>&1 | grep -aE "^test result|FAILED|panicked"
    [[ $W == *C* ]] && mtp
  ) > "$O/g-r.log" 2>&1 &
  (
    echo "-- 35B/8B batched==solo, knob unset"
    dev 0 $T cargo test --release --locked -p intarsia-models --test batch_cuda_real -- --test-threads=1 --nocapture batched 2>&1 | grep -aE "^test |^test result|FAILED|panicked|^error"
    echo "-- 35B/8B batched==solo, option 1 ON ($OPT1)"
    dev 0 env $OPT1 $T cargo test --release --locked -p intarsia-models --test batch_cuda_real -- --test-threads=1 --nocapture batched 2>&1 | grep -aE "^test |^test result|FAILED|panicked|^error"
    echo "-- ignored long-prompt, option 1 ON"
    dev 0 env $OPT1 $T cargo test --release --locked -p intarsia-models --test batch_cuda_real -- --ignored real_35b_batched_engine_long_prompts_equal_the_local_engine --test-threads=1 --nocapture 2>&1 | grep -aE "^test |^test result|FAILED|panicked|^error"
  ) > "$O/g-c.log" 2>&1
  wait
  echo "== batch_cuda_real"; cat "$O/g-c.log"
  echo "== batch_recurrent + kernels quick + rows_exact + MTP"; cat "$O/g-r.log"
fi
echo "[gates] done $(now)"
