#!/usr/bin/env bash
# Lane HDC gates, ON tatooine INSIDE a fleetq job that holds both R9700s with radiance down (drive.sh `gates` item, NOSTOP=1).
# Sections (env SEC): fuse (mxfp4_a8_fuse test, output hash with the knob 0 / 1 / default), rows (rows_exact: every row count equals the
# one-row kernel, hip quantmoe/mxfp4 decode tests), adm (kernels test --quick, base and dev CLI), plan (plan show of 8B, 35B and the
# 27B tp2, base CLI vs dev CLI), time (rows kernel micro timing, knob 0 vs 1 vs default). Logs in ~/intarsiaml-hdc-logs/$W.
#   NOSTOP=1 W=w3 SEC='fuse rows adm plan' bash ~/intarsiaml-hdc/scripts/hdc/gates.sh
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
W=${W:-w3}; SEC=${SEC:-fuse rows adm plan time}
O=$HOME/intarsiaml-hdc-logs/$W; mkdir -p "$O"
has() { [[ " $SEC " == *" $1 "* ]]; }
[[ -n "${NOSTOP:-}" ]] || trap restore EXIT
[[ -n "${NOSTOP:-}" ]] || { systemctl --user stop radiance < /dev/null; sleep 3; }
CT="cargo test --release -p intarsia-backend-hip"

if has fuse; then   # same output hash in every mode; all rel-l2 / batch-invariance asserts pass
  for m in "off|INTARSIA_HIP_MXFP4_A8_FUSE=0" "all|INTARSIA_HIP_MXFP4_A8_FUSE=1" "default|A8_FUSE_TEST_DEFAULT=1"; do
    n=${m%%|*}; e=${m#*|}
    G=0 gpu 900 env INTARSIA_REQUIRE_GPU=1 $e $CT --test mxfp4_a8_fuse -- --test-threads=1 --nocapture > "$O/fuse-$n.txt" 2>&1
    echo "== fuse $n rc=$? $(grep -a 'FUSEMODE\|FUSEHASH\|test result' "$O/fuse-$n.txt" | tr '\n' ' ' | cut -c1-200)"
  done
  h0=$(grep -ah FUSEHASH "$O/fuse-off.txt"); h1=$(grep -ah FUSEHASH "$O/fuse-all.txt"); h2=$(grep -ah FUSEHASH "$O/fuse-default.txt")
  [[ -n "$h0" && "$h0" == "$h1" && "$h0" == "$h2" ]] && echo "== fuse hashes EQUAL in all three modes ($h0)" || echo "== fuse hashes DIFFER: $h0 / $h1 / $h2"
fi

if has rows; then   # the row-exactness suite (every M equals the one-row kernel) and the mxfp4 decode tests on the default path
  G=0 gpu 1500 env INTARSIA_REQUIRE_GPU=1 $CT --test rows_exact -- --test-threads=1 > "$O/rows_exact.txt" 2>&1
  echo "== rows_exact rc=$? $(grep -a 'test result' "$O/rows_exact.txt" | tail -1)"
  for t in mxfp4_decode mxfp4_a8 a8_served; do
    G=0 gpu 900 env INTARSIA_REQUIRE_GPU=1 $CT --test $t -- --test-threads=1 > "$O/$t.txt" 2>&1
    echo "== $t rc=$? $(grep -a 'test result' "$O/$t.txt" | tail -1)"
  done
fi

if has adm; then   # kernel admission (every HIP kernel, quick cases)
  G=0 gpu 600 /work/hdc-dev-cli kernels test --backend hip --quick > "$O/adm.log" 2>&1; echo "== admission rc=$?: $(grep -ac ' ADMITTED' "$O/adm.log") admitted rows; not admitted:"
  grep -av " ADMITTED" "$O/adm.log" | grep -av "^SKIPPED" | tail -6
fi

if has plan; then  # `plan show` base CLI (card 0) vs dev CLI (card 1) in parallel: byte-identical
  for m in 8B:/models/Qwen3-8B 35B:/models/Qwen3.6-35B-A3B-NVFP4; do
    tag=${m%%:*}; path=${m#*:}
    G=0 MODELS=/data/models/intarsia gpu 150 /work/hdc-base-cli plan show --backend hip --model $path --max-seq 8192 > "$O/plan-$tag-base.txt" 2> "$O/plan-$tag-base.err" &
    G=1 MODELS=/data/models/intarsia gpu 150 /work/hdc-dev-cli plan show --backend hip --model $path --max-seq 8192 > "$O/plan-$tag-dev.txt" 2> "$O/plan-$tag-dev.err" &
    wait
    if cmp -s "$O/plan-$tag-base.txt" "$O/plan-$tag-dev.txt" && [ -s "$O/plan-$tag-base.txt" ]; then echo "== plan $tag: byte-identical ($(wc -c < "$O/plan-$tag-base.txt") B)"; else echo "== plan $tag: DIFFERS or empty"; fi
  done
  C27="plan show --backend hip --model $M27 --tp 2 --devices 0,1 --max-seq 4096 --spec off"
  gpu 150 /work/hdc-base-cli $C27 > "$O/plan-27B-base.txt" 2> "$O/plan-27B-base.err"
  gpu 150 /work/hdc-dev-cli $C27 > "$O/plan-27B-dev.txt" 2> "$O/plan-27B-dev.err"
  if cmp -s "$O/plan-27B-base.txt" "$O/plan-27B-dev.txt" && [ -s "$O/plan-27B-base.txt" ]; then echo "== plan 27B tp2: byte-identical ($(wc -c < "$O/plan-27B-base.txt") B)"; else echo "== plan 27B tp2: DIFFERS or empty"; fi
fi

if has time; then  # rows kernel micro timing at M = 1..16 (the shapes of timing_mxfp4_rows), knob 0 / 1 / default
  for m in "off|INTARSIA_HIP_MXFP4_A8_FUSE=0" "all|INTARSIA_HIP_MXFP4_A8_FUSE=1" "default|A=1"; do
    n=${m%%|*}; e=${m#*|}
    G=0 gpu 600 env INTARSIA_REQUIRE_GPU=1 $e $CT --test rows_exact timing_mxfp4_rows -- --ignored --nocapture > "$O/time-$n.txt" 2>&1
    echo "== time $n rc=$? $(grep -ac '^TIME' "$O/time-$n.txt") TIME lines"
  done
fi
echo "[hdc] gates $W done $(now)"
