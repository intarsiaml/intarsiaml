#!/usr/bin/env bash
# Lane PF27b micro job, ON tatooine inside one short fleetq job (< 3 min, takes radiance-rank0/1 down): stop radiance, run the sections of
# SEC (in this order), restore radiance and wait until :8080 answers (EXIT trap). Output in ~/intarsiaml-pf27b/pf27b-out/$W.
#   link   tests/comm.rs p2p_link_rate (dev): P2P copy-engine rate, push / reduce phases per block count, all-reduce per wide setting
#   pair   tests/mxfp4_a8.rs a8_pair_is_bitwise_two_single_calls (dev)
#   wide   tests/comm.rs wide-path gates (dev): bitwise vs the old path, delayed-rank stress, the no-wait negative control
#   arrate tests/comm.rs all_reduce_rate, base and dev
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
: "${W:?W=<name>}"; SEC=${SEC:-link pair}
O=$R/pf27b-out/$W; mkdir -p "$O"; T0=$(date +%s)
has() { [[ " $SEC " == *" $1 "* ]]; }
if [[ -z "${NORAD:-}" ]]; then  # NORAD=1: inside a window whose driver (drive.sh's guard) owns radiance
  trap restore EXIT
  stop_radiance
fi
if has link; then
  gpu 110 env ${LENV:-} /work/pf27b-bin/comm-dev --ignored --nocapture --test-threads 1 p2p_link_rate > "$O/link.txt" 2>&1
  echo "== link rc=$? $(grep -c '^P2P\|^AR' "$O/link.txt") lines $(now)"
fi
if has pair; then
  G=0 gpu 60 /work/pf27b-bin/mxfp4_a8-dev --test-threads 1 a8_pair > "$O/pair.txt" 2>&1
  echo "== pair rc=$? $(grep -a '^test result' "$O/pair.txt")"
fi
if has wide; then
  gpu 100 /work/pf27b-bin/comm-dev --test-threads 1 wide_path > "$O/wide.txt" 2>&1
  echo "== wide rc=$? $(grep -a '^test result' "$O/wide.txt") $(grep -a 'negative control' "$O/wide.txt")"
fi
#   gen    generate 4k prefill (--max-tokens 1, spec off) per arm of GARMS (bin[:ENV=V,ENV=V[:chunk]]), GREPS rounds, skipped once 140 s are gone
if has gen; then
  P=$(cat $R/scripts/cmtp/ids-prompts/4k-0.txt)
  for arm in $(for i in $(seq 1 ${GREPS:-1}); do echo ${GARMS:-pf27b-base-cli pf27b-dev-cli}; done); do
    (( $(date +%s) - T0 > 140 )) && { echo "== gen $arm: skipped (time)"; continue; }
    IFS=: read -r b e ch <<< "$arm"; e=${e//,/ }; ch=${ch:+--prefill-chunk $ch}; tg=$(echo $arm | tr ':=,' '___')
    tg=$tg-$(date +%s%N | cut -c9-13)
    gpu 120 env $e /work/$b generate -m $M27 --backend hip --tp 2 --devices 0,1 --mem-budget 29 --max-seq ${GMS:-8192} --spec off --ignore-eos --max-tokens 1 --profile-every 0 $ch -p "$P" > "$O/gen-$tg.out" 2> "$O/gen-$tg.err"
    echo "== gen $arm: $(grep -a 'prefill ' "$O/gen-$tg.err" | tail -1 | cut -c1-160) ($(( $(date +%s) - T0 )) s)"
  done
fi
#   prof   rocprofv3 kernel trace of the 4k generate for each bin of PBINS -> $O/p-<bin>.csv.gz
if has prof; then
  P=$(cat $R/scripts/cmtp/ids-prompts/4k-0.txt)
  for b in ${PBINS:-pf27b-dev-cli}; do
    rm -rf "$R/pf27b-prof"; mkdir -p "$R/pf27b-prof"
    gpup 100 rocprofv3 --kernel-trace --output-format csv -d /work/pf27b-prof -o run -- /work/$b generate -m $M27 --backend hip --tp 2 --devices 0,1 --mem-budget 29 --max-seq 8192 --spec off --ignore-eos --max-tokens 1 --profile-every 0 -p "$P" > "$O/p-$b.out" 2> "$O/p-$b.err"
    csv=$(find "$R/pf27b-prof" -name '*kernel_trace.csv' | head -1); [[ -n "$csv" ]] && gzip -c "$csv" > "$O/p-$b.csv.gz"; rm -rf "$R/pf27b-prof"
    echo "== prof $b: $(grep -a 'prefill ' "$O/p-$b.err" | tail -1 | cut -c1-120) csv=$(ls -s $O/p-$b.csv.gz 2>/dev/null) ($(( $(date +%s) - T0 )) s)"
  done
fi
#   corr   every a8_ test of the dev mxfp4_a8 binary under CENVS (default, the per-group path, the staged kernel)
if has corr; then
  for e in ${CENVS:-A=1 INTARSIA_HIP_MXFP4_A8_PERGROUP=1 INTARSIA_HIP_MXFP4_A8_DIRECT=0}; do
    G=0 gpu 100 env $e /work/pf27b-bin/mxfp4_a8-dev --test-threads 1 a8_ > "$O/corr-$e.txt" 2>&1
    echo "== corr [$e] rc=$?: $(grep -a '^test result' "$O/corr-$e.txt") $(grep -a 'FAILED\|panicked' "$O/corr-$e.txt" | head -3 | tr '\n' ' ') ($(( $(date +%s) - T0 )) s)"
  done
fi
#   time   timing_a8_tp2 (dev) with TENV, once per DTILE value of DTILES (forced direct tile; "-" = the default table)
if has time; then
  for t in ${DTILES:--}; do
    e=""; [[ $t != - ]] && e="INTARSIA_HIP_MXFP4_A8_DTILE=${t:0:1},${t:1:1},${t:2:1},${t:3:1}"
    G=0 gpu 60 env $e ${TENV:-} /work/pf27b-bin/mxfp4_a8-dev --ignored --nocapture timing_a8_tp2 2>&1 | grep -aE "^TIME|panicked" > "$O/time-$t.txt"
    echo "== time $t: $(awk '{printf "%s M=%s %s | ", $3, $5, $6}' "$O/time-$t.txt" | sed 's/M=M=/M=/g')"
  done
fi
#   hash   a8_mma_hashes of the single W4A8 GEMM, base vs dev binary (the refactored launcher must give the same bits)
if has hash; then
  for b in base dev; do G=0 gpu 60 /work/pf27b-bin/mxfp4_a8-$b --ignored --nocapture a8_mma_hashes 2>&1 | grep -a "^HASH\|panicked" > "$O/hash-$b.txt"; done
  echo "== hash: base $(wc -l < "$O/hash-base.txt") lines, dev $(wc -l < "$O/hash-dev.txt"), differing: $(diff "$O/hash-base.txt" "$O/hash-dev.txt" | grep -c '^<')"
fi
#   neg    the wide-path negative control with its wrong-sum count (stderr)
if has neg; then
  gpu 60 /work/pf27b-bin/comm-dev --nocapture --test-threads 1 wide_path_without_the_wait > "$O/neg.txt" 2>&1
  echo "== neg rc=$? $(grep -a 'negative control\|test result' "$O/neg.txt" | tr '\n' ' ')"
fi
#   plan   plan show 8B / 35B, base vs dev CLI, one card each (byte-identical)
if has plan; then
  for m in 8B:/models/Qwen3-8B 35B:/models/Qwen3.6-35B-A3B-NVFP4; do
    tag=${m%%:*}; path=${m#*:}
    G=0 MODELS=/data/models/intarsia gpu 90 /work/pf27b-base-cli plan show --backend hip --model $path --max-seq 8192 > "$O/plan-$tag-base.txt" 2> "$O/plan-$tag-base.err" &
    G=1 MODELS=/data/models/intarsia gpu 90 /work/pf27b-dev-cli plan show --backend hip --model $path --max-seq 8192 > "$O/plan-$tag-dev.txt" 2> "$O/plan-$tag-dev.err" &
    wait
    if cmp -s "$O/plan-$tag-base.txt" "$O/plan-$tag-dev.txt" && [ -s "$O/plan-$tag-base.txt" ]; then echo "== plan $tag: byte-identical ($(wc -c < "$O/plan-$tag-base.txt") B)"; else echo "== plan $tag: DIFFER or empty"; fi
  done
fi
# ---- window sections (W2: longer than a micro job) ----
#   tests  the whole comm and ar_norm test files (dev) under the default env, INTARSIA_HIP_AR_DRAIN=1 and INTARSIA_HIP_AR_WIDE=0 (the LL
#          stress, the mode 8 control, the wide-path gates and the no-wait control run in each pass)
if has tests; then
  for e in A=1 INTARSIA_HIP_AR_DRAIN=1 INTARSIA_HIP_AR_WIDE=0; do
    for t in comm ar_norm; do
      gpu 600 env $e /work/pf27b-bin/$t-dev --test-threads 1 --nocapture > "$O/tests-$t-$e.txt" 2>&1
      echo "== tests $t [$e] rc=$?: $(grep -a '^test result' "$O/tests-$t-$e.txt" | tr '\n' ' ') $(grep -a 'negative control' "$O/tests-$t-$e.txt" | tr '\n' ' ')"
    done
  done
fi
#   adm    kernels test --backend hip --quick of the dev CLI
if has adm; then
  gpu 1500 /work/pf27b-dev-cli kernels test --backend hip --quick > "$O/adm.log" 2>&1
  echo "== adm exit $?: $(grep -ac ' ADMITTED' "$O/adm.log") admitted rows; pair_a8_mma: $(grep -a 'pair_a8_mma' "$O/adm.log" | head -3 | tr '\n' ' ' | cut -c1-300); tail: $(tail -3 "$O/adm.log" | tr '\n' ' ' | cut -c1-300)"
fi
#   ids    the det / ars / pf27 ids gate: greedy 256 ids of the 18 prompts in scripts/dfh/ids-prompts, tp2, base vs dev, spec off + MTP default
if has ids; then
  IG="generate -m $M27 --backend hip --tp 2 --devices 0,1 --mem-budget 29 --max-seq 20480 --chat --no-think --profile-every 0 --ignore-eos --show-ids --max-tokens 256"
  for pr in $(ls $R/scripts/dfh/ids-prompts | sed 's/\.txt$//'); do
    for sp in off mtp; do
      sa=""; [[ $sp == off ]] && sa="--spec off"
      for b in base dev; do
        gpu 420 /work/pf27b-$b-cli $IG $sa -p "$(cat $R/scripts/dfh/ids-prompts/$pr.txt)" > /dev/null 2> "$O/ids-$pr-$sp-$b.err"
        grep -a '^ids' "$O/ids-$pr-$sp-$b.err" > "$O/ids-$pr-$sp-$b.ids"
      done
      if [[ -s "$O/ids-$pr-$sp-base.ids" ]] && cmp -s "$O/ids-$pr-$sp-base.ids" "$O/ids-$pr-$sp-dev.ids"; then r=SAME; else r=DIFF; fi
      echo "== ids $pr $sp: $r ($(wc -w < "$O/ids-$pr-$sp-base.ids") words; $(grep -aoE 'prompt=[0-9]+' "$O/ids-$pr-$sp-base.err" | tail -1))"
    done
  done
fi
#   soak   SOAKN (120) fresh tapped tp=2 processes of the dev CLI in one container, prompt SOAKP (4k-0: both prefill chunks take the 20 MiB
#          wide launches), --spec mtp --spec-k 2, 16 tokens, the env not setting any knob; then SOAKB (3) base processes. Taps (per-row layer
#          hashes) and id lines of every process; the verdict: all dev taps byte-identical to each other and to the base taps
if has soak; then
  sp=${SOAKP:-4k-0}
  for v in "dev:${SOAKN:-120}" "base:${SOAKB:-3}"; do
    b=${v%%:*}; n=${v#*:}; rm -f "$O/soak-$b.ids"
    gpu 3300 bash -c '
      b=$1 n=$2 o=$3 sp=$4 m=$5
      for i in $(seq 1 "$n"); do
        INTARSIA_DET_TAP=$o/soak-$b-$i.tap /work/pf27b-$b-cli generate -m "$m" --backend hip --tp 2 --devices 0,1 --mem-budget 29 --max-seq 20480 --chat --no-think \
          --profile-every 0 --ignore-eos --show-ids -p "$(cat /work/scripts/dfh/ids-prompts/$sp.txt)" --max-tokens 16 --spec mtp --spec-k 2 > /dev/null 2> "$o/soak-$b.tmp"
        echo "rc=$? $(grep -a "^ids" "$o/soak-$b.tmp")" >> "$o/soak-$b.ids"
      done; rm -f "$o/soak-$b.tmp"' _ "$b" "$n" "/work/pf27b-out/$W" "$sp" "$M27"
    echo "== soak $b: $(wc -l < "$O/soak-$b.ids") processes; distinct id lines $(sort -u "$O/soak-$b.ids" | wc -l); rc!=0: $(grep -vc '^rc=0' "$O/soak-$b.ids") ($(now))"
  done
  ref=$(md5sum "$O/soak-base-1.tap" 2>/dev/null | cut -d' ' -f1)
  echo "== soak taps: dev $(ls "$O"/soak-dev-*.tap 2>/dev/null | wc -l) files, distinct $(md5sum "$O"/soak-dev-*.tap | cut -d' ' -f1 | sort -u | wc -l), equal to base-1 ($ref): $(md5sum "$O"/soak-dev-*.tap | grep -c "^$ref"); base distinct $(md5sum "$O"/soak-base-*.tap | cut -d' ' -f1 | sort -u | wc -l); ids dev==base: $(cmp -s <(sort -u "$O/soak-dev.ids") <(sort -u "$O/soak-base.ids") && echo yes || echo NO)"
fi
if has arrate; then
  for b in base dev; do gpu 60 /work/pf27b-bin/comm-$b --ignored --nocapture all_reduce_rate > "$O/arrate-$b.txt" 2>&1; done
fi
echo "[pf27b] micro done $(now), $(( $(date +%s) - T0 )) s"
