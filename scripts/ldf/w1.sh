#!/usr/bin/env bash
# Lane LDF, ON tatooine INSIDE a fleetq job that holds one R9700 and takes radiance down (radiance is stopped here, restored by the EXIT trap):
# kernel-level window on the mxfp4_a8 test binary (no cargo, no model). Sections (env SEC, default all): list (rocprof counters), corr (correctness
# tests), hash (output hashes LINES=0 vs 1), time (timing variants), prof (counter passes). Logs in ~/intarsiaml-ldf-logs/$W.
#   BIN=/work/ldf-bin/mxfp4_a8-v1 W=w1 bash ~/intarsiaml-ldf/scripts/ldf/w1.sh
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
BIN=${BIN:-/work/ldf-bin/mxfp4_a8-v1}; W=${W:-w1}; SEC=${SEC:-list corr hash mut adm plan time hashv prof ktrace}; MS=${MS:-256,512,1024,1536,2048,4096}
O=$HOME/intarsiaml-ldf-logs/$W; mkdir -p "$O"; T0=$(date +%s)
has() { [[ " $SEC " == *" $1 "* ]]; }
left() { echo $(( ${WINDOW_S:-2400} - ($(date +%s) - T0) )); }
[[ -n "${NOSTOP:-}" ]] || trap restore EXIT  # NOSTOP=1: run inside a window whose caller owns radiance (drive.sh)
[[ -n "${NOSTOP:-}" ]] || { echo "[ldf] $W stopping radiance $(now)"; systemctl --user stop radiance < /dev/null; sleep 3; }
echo "[ldf] radiance stopped $(now); GPUs: $(docker ps -q --filter name=radiance | wc -l) radiance containers"

if has list; then
  gpup 200 rocprofv3 --list-avail > "$O/list-avail.txt" 2>&1; echo "[ldf] list-avail: $(wc -l < "$O/list-avail.txt") lines"
fi
if has lds; then  # LDS read + WMMA inner-loop micro-benchmark (scripts/ldf/ldsbench.hip), one card
  G=0 gpu 500 /work/ldf-bin/ldsbench ${LDSF:-} > "$O/lds.txt" 2>&1; echo "[ldf] lds: $(wc -l < "$O/lds.txt") lines $(now)"
fi
if has corr; then  # the a8_ tests (reference, rows == solo, M-independence, position) on the default kernel, the padded layout (LDS=0), distance 1 (PD=1) and the per-group path
  for e in "A=1" "INTARSIA_HIP_MXFP4_A8_LDS=0" "INTARSIA_HIP_MXFP4_A8_PD=1" "INTARSIA_HIP_MXFP4_A8_PERGROUP=1" "INTARSIA_HIP_MXFP4_A8_LDS=0 INTARSIA_HIP_MXFP4_A8_PERGROUP=1"; do
    echo "== corr $e"; G=0 gpu 600 env $e $BIN a8_ 2>&1 | grep -aE "^test |test result|panicked|differ|n=" | head -20 > "$O/corr-$(echo $e | tr ' =' '__').txt"
    tail -3 "$O/corr-$(echo $e | tr ' =' '__').txt"
  done
fi
if has hash; then  # the 48-shape output hash: main's binary (BASEBIN), the default, the padded layout, distance 1: all bitwise equal
  hh() { G=0 gpu 900 env ${3:-A=1} $1 --ignored --nocapture a8_mma_hashes 2>&1 | grep -a "^HASH\|panicked" > "$O/hash-$2.txt"; }
  hh ${BASEBIN:-/work/ldf-bin/mxfp4_a8-base} main; hh $BIN dev; hh $BIN lds0 INTARSIA_HIP_MXFP4_A8_LDS=0; hh $BIN pd1 INTARSIA_HIP_MXFP4_A8_PD=1
  echo "== hashes: $(wc -l < "$O/hash-main.txt") lines; differing from main: dev $(diff "$O/hash-main.txt" "$O/hash-dev.txt" | grep -c '^<'), lds0 $(diff "$O/hash-main.txt" "$O/hash-lds0.txt" | grep -c '^<'), pd1 $(diff "$O/hash-main.txt" "$O/hash-pd1.txt" | grep -c '^<')"
fi
t() { # t <tag> <env...>: timing_a8_tp2 over the M list
  local tag=$1; shift
  [[ $(left) -lt 120 ]] && { echo "[ldf] skip $tag (window)"; return; }
  G=0 gpu 300 env GLR_TIMING_M=$MS "$@" $BIN --ignored --nocapture timing_a8_tp2 2>&1 | grep -aE "^TIME|panicked|error" > "$O/t-$tag.txt"
  echo "== $tag: $(grep -ac '^TIME' "$O/t-$tag.txt") lines $(now)"
}
if has time; then
  IFS=';' read -ra TV <<< "${TVARS:-new|A=1;old|INTARSIA_HIP_MXFP4_A8_PD=1}"
  for v in "${TV[@]}"; do t "${v%%|*}" "${v#*|}"; done
fi
if has mut; then  # every planted mutant binary must FAIL the a8_ tests (the clean one passes); which tests caught it is logged
  for b in clean $(ls "$HOME/intarsiaml-ldf/ldf-bin" | sed -n 's/^mut-\(m[0-9].*\)$/\1/p' | sort); do
    [[ $(left) -lt 90 ]] && { echo "[ldf] skip mutant $b (window)"; continue; }
    G=0 gpu 300 /work/ldf-bin/mut-$b a8_ > "$O/mut-$b.txt" 2>&1; rc=$?
    echo "== mutant $b: exit $rc; failed: $(grep -aE '^test .* FAILED' "$O/mut-$b.txt" | sed 's/^test //; s/ \.\.\. FAILED//' | tr '\n' ' ')"
  done
fi
if has adm; then  # kernel admission of the dev CLI (every HIP kernel, quick cases)
  G=0 gpu 600 /work/ldf-dev-cli kernels test --backend hip --quick > "$O/adm.log" 2>&1; echo "== admission exit $?: $(grep -ac ' ADMITTED' "$O/adm.log") admitted rows; not admitted:"
  grep -av " ADMITTED" "$O/adm.log" | grep -av "^SKIPPED" | tail -8
fi
if has plan; then  # `plan show` of the 8B and the 35B, base CLI on card 0, dev CLI on card 1, in parallel: byte-identical
  for m in 8B:/models/Qwen3-8B 35B:/models/Qwen3.6-35B-A3B-NVFP4; do
    tag=${m%%:*}; path=${m#*:}
    G=0 MODELS=/data/models/intarsia gpu 150 /work/ldf-base-cli plan show --backend hip --model $path --max-seq 8192 > "$O/plan-$tag-base.txt" 2> "$O/plan-$tag-base.err" &
    G=1 MODELS=/data/models/intarsia gpu 150 /work/ldf-dev-cli plan show --backend hip --model $path --max-seq 8192 > "$O/plan-$tag-dev.txt" 2> "$O/plan-$tag-dev.err" &
    wait
    if cmp -s "$O/plan-$tag-base.txt" "$O/plan-$tag-dev.txt" && [ -s "$O/plan-$tag-base.txt" ]; then echo "== plan $tag: byte-identical ($(wc -c < "$O/plan-$tag-base.txt") B)"; else echo "== plan $tag: DIFFER or empty"; fi
  done
fi
if has hashv; then  # hashes under forced tile / schedule variants (HVARS "tag|ENV;..."), diffed against the default-loader hash-0
  [[ -f "$O/hash-dev.txt" ]] || G=0 gpu 900 env A=1 $BIN --ignored --nocapture a8_mma_hashes 2>&1 | grep -a "^HASH\|panicked" > "$O/hash-dev.txt"
  IFS=';' read -ra HV <<< "${HVARS:-}"
  for v in "${HV[@]}"; do
    [[ $(left) -lt 200 ]] && break
    tag=${v%%|*}; G=0 gpu 900 env ${v#*|} $BIN --ignored --nocapture a8_mma_hashes 2>&1 | grep -a "^HASH\|panicked" > "$O/hash-$tag.txt"
    echo "== hash $tag: $(wc -l < "$O/hash-$tag.txt") lines; differing from default: $(diff "$O/hash-dev.txt" "$O/hash-$tag.txt" | grep -c '^<')"
  done
fi
if has prof; then
  # counter passes: candidates present in the avail list, four per pass, on q+gate (tile 4x4x2x4) and gate+up (4x4x4x2) at M=2048
  CANDS="${CANDS:-GRBM_GUI_ACTIVE SQ_WAVES SQ_BUSY_CYCLES SQ_WAVE_CYCLES SQ_WAIT_ANY SQ_WAIT_INST_ANY SQ_ACTIVE_INST_ANY SQ_ACTIVE_INST_VALU SQ_ACTIVE_INST_LDS SQ_ACTIVE_INST_VMEM SQ_ACTIVE_INST_MISC SQ_INSTS_VALU SQ_INSTS_LDS SQ_INSTS_VMEM SQ_INSTS_WMMA SQ_INST_CYCLES_VMEM SQ_LDS_BANK_CONFLICT SQ_LDS_IDX_ACTIVE SQ_INSTS_VMEM_RD SQ_INSTS_FLAT SQ_WAIT_INST_LDS SQ_INST_LEVEL_VMEM SQ_INST_LEVEL_LDS SQ_INSTS_SALU TCP_TOTAL_CACHE_ACCESSES_sum TCP_TCC_READ_REQ_sum TCP_TOTAL_READ_sum TCP_PENDING_STALL_CYCLES_sum TCP_READ_TAGCONFLICT_STALL_CYCLES_sum TCP_TCC_READ_REQ_LATENCY_sum TCP_TOTAL_CACHE_ACCESSES TCP_TCC_READ_REQ TCP_PENDING_STALL_CYCLES TCC_HIT_sum TCC_MISS_sum TCC_REQ_sum TCC_EA0_RDREQ_sum TCC_EA_RDREQ_sum TCC_HIT TCC_MISS TCC_REQ TCC_EA0_RDREQ TCC_EA_RDREQ TCC_READ_sum TCC_TAG_STALL_sum}"
  PRESENT=""; for c in $CANDS; do grep -qaE "(^|[^A-Za-z0-9_])$c([^A-Za-z0-9_]|$)" "$O/list-avail.txt" 2>/dev/null && PRESENT="$PRESENT $c"; done
  echo "[ldf] counters present:$PRESENT"; echo "$PRESENT" > "$O/counters-present.txt"
  set -- $PRESENT; n=0
  while [[ $# -gt 0 ]]; do
    G_=${GROUP:-4}; grp="${*:1:$G_}"; shift $(( $# < G_ ? $# : G_ )); n=$((n + 1))
    IFS=';' read -ra PV <<< "${PVARS:-new|A=1;old|INTARSIA_HIP_MXFP4_A8_PD=1}"
    for v in "${PV[@]}"; do
      [[ $(left) -lt 150 ]] && break 2
      tag=${v%%|*}; e=${v#*|}; d=/work/ldf-prof/$W-$tag-$n; rm -rf "$R/ldf-prof/$W-$tag-$n"; mkdir -p "$R/ldf-prof"
      G=0 gpup 150 rocprofv3 --pmc $grp --output-format csv -d $d -o p -- env GLR_TIMING_M=2048 GLR_TIMING_ONLY=q+gate,gate+up GLR_TIMING_ITERS=6 $e $BIN --ignored --nocapture timing_a8_tp2 > "$O/prof-$tag-$n.out" 2>&1
      f=$(find "$R/ldf-prof/$W-$tag-$n" -name '*counter_collection.csv' | head -1)
      if [[ -n "$f" ]]; then gzip -c "$f" > "$O/prof-$tag-$n.csv.gz"; echo "[ldf] pass $n $tag: $grp -> ok"; else echo "[ldf] pass $n $tag: $grp -> no csv: $(grep -a -i 'error\|invalid\|not found' "$O/prof-$tag-$n.out" | head -2 | tr '\n' ' ')"; fi
      rm -rf "$R/ldf-prof/$W-$tag-$n"
    done
  done
fi
if has ktrace; then  # kernel durations (rocprofv3 --kernel-trace) of KVARS "tag|ENV;..." on the 27B tp2 shapes at every M of KMS (default 2048)
  IFS=';' read -ra KV <<< "${KVARS:-new|A=1;old|INTARSIA_HIP_MXFP4_A8_PD=1}"
  for km in ${KMS:-2048}; do for v in "${KV[@]}"; do
    [[ $(left) -lt 150 ]] && break
    tag=${v%%|*}-m$km; e=${v#*|}; d=/work/ldf-prof/$W-k-$tag; rm -rf "$R/ldf-prof/$W-k-$tag"; mkdir -p "$R/ldf-prof"
    G=0 gpup 200 rocprofv3 --kernel-trace --output-format csv -d $d -o k -- env GLR_TIMING_M=$km GLR_TIMING_ITERS=20 $e $BIN --ignored --nocapture timing_a8_tp2 > "$O/kt-$tag.out" 2>&1
    f=$(find "$R/ldf-prof/$W-k-$tag" -name '*kernel_trace.csv' | head -1)
    if [[ -n "$f" ]]; then gzip -c "$f" > "$O/kt-$tag.csv.gz"; echo "[ldf] ktrace $tag ok"; else echo "[ldf] ktrace $tag: no csv"; fi
    rm -rf "$R/ldf-prof/$W-k-$tag"
  done; done
fi
echo "[ldf] $W done $(now), $(left) s of the window left"
