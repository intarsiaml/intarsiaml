#!/usr/bin/env bash
# Lane pf (2026-10-22, docs/results/2026-10-22-pf-prefill-ctx.md): GPU stages run ON tatooine, inside a radiance-down window
# opened by pf-window.sh. GPU 0 only. Needs, built beforehand in CPU slots:
#   base binary   volume intarsia-cargo-target-pf       (origin/main, INTARSIA_HIP_FLASH_VP does not exist there)
#   dev binary    volume intarsia-cargo-target-pf-dev   (this tree)
#   test binaries volume intarsia-cargo-target-pf-test  (cargo test -p intarsia-backend-hip --no-run, this tree)
# and bench-out/pf/prompt{4096,16384,4089,16654}.txt (N-token raw prompts cut from prompt4k.txt; 4089 / 16654 = localeval's 4k / 16k).
#   pf-stages.sh <stage>...   stages: prof opvp bitwise ab ids chunk
# Output: bench-out/pf/out/. Every inner command has stdin from /dev/null (a docker run would eat the rest of this script).
set -uo pipefail
export INTARSIA_REPO=${INTARSIA_REPO:-$HOME/intarsiaml-pf}
cd "$INTARSIA_REPO" || exit 2
export HIP_VISIBLE_DEVICES=0
D=deploy/tatooine/dev.sh
M35=/models/Qwen3.6-35B-A3B-NVFP4
PF=bench-out/pf
OUT=$PF/out
mkdir -p "$OUT"
VOL_BASE=intarsia-cargo-target-pf
VOL_DEV=intarsia-cargo-target-pf-dev
VOL_TEST=intarsia-cargo-target-pf-test
PROFIMG=intarsia-hip-prof:rocm7.2.4-rust1.90.0
ts() { date +%T; }
T0=$(date +%s)
BUDGET=${BUDGET:-2700}  # seconds from start; stages that need more than what is left are skipped and logged
left() { echo $((BUDGET - ($(date +%s) - T0))); }
need() { if (($(left) < $1)); then echo "SKIPPED $2: $(left) s left, needs $1"; return 1; fi; return 0; }

vpenv() { [ "$1" = x ] && echo x || echo "INTARSIA_HIP_FLASH_VP=$1"; }  # x = leave the knob unset (the build's default)
# gen <tag> <vol> <prompt N> <max tokens> <ENV=val|x> [extra generate args]: writes bench-out/pf/gen-<tag>.sh
mkgen() {
  local tag=$1 vol=$2 n=$3 mt=$4 envv=$5; shift 5
  local ex=""; [ "$envv" != x ] && ex="export $envv;"
  printf '%s\n' "$ex /target/release/intarsia generate -m $M35 --backend hip --mem-budget 29 --max-seq 18432 --ignore-eos $* -p \"\$(cat /work/$PF/prompt$n.txt)\" --max-tokens $mt" > "$PF/gen-$tag.sh"
}
# run <tag> <vol>: unprofiled run of gen-<tag>.sh, prints the prompt / finish lines
run() {
  INTARSIA_TARGET_VOL=$2 $D bash -c "bash /work/$PF/gen-$1.sh" < /dev/null 2>&1 | grep -aE "prompt:|finish=|rror|panick"
}
# prof <tag> <vol>: rocprofv3 kernel trace of gen-<tag>.sh, csv in out/prof/<tag>
prof() {
  rm -rf "$OUT/prof/$1"; mkdir -p "$OUT/prof/$1"
  INTARSIA_IMAGE=$PROFIMG INTARSIA_TARGET_VOL=$2 $D bash -c "timeout -s KILL 300 rocprofv3 --kernel-trace --output-format csv -d /work/$OUT/prof/$1 -o run -- bash /work/$PF/gen-$1.sh" < /dev/null 2>&1 | grep -aE "prompt:|finish=|rror|panick"
  local csv; csv=$(find "$OUT/prof/$1" -name '*kernel_trace.csv' | head -1)
  [ -n "$csv" ] && python3 -I $PF/agg.py "$csv" > "$OUT/prof/$1/agg.txt" 2>&1
}
# flash_ms <tag>: summed flash_hpb kernel ms of out/prof/<tag>
flash_ms() {
  python3 -I - "$(find "$OUT/prof/$1" -name '*kernel_trace.csv' | head -1)" <<'EOF'
import csv, sys
t = 0.0
for r in csv.DictReader(open(sys.argv[1])):
    if "flash_hpb" in r["Kernel_Name"]:
        t += (int(r["End_Timestamp"]) - int(r["Start_Timestamp"])) / 1e6
print(f"{t:.2f}")
EOF
}

prof_run() { # prof_run <vol> <tag prefix>: per-op profile (--profile-every 1 --op-table, the engine's own profiler) + rocprofv3 traces
  local vol=$1 pre=$2 n
  for n in 4096 16384; do
    need 120 "op_table $n" || continue
    echo "--- op_table $n ($vol, --profile-every 1) $(ts)"
    mkgen "${pre}optab$n" $vol $n 1 x --profile-every 1 --op-table
    INTARSIA_TARGET_VOL=$vol $D bash -c "bash /work/$PF/gen-${pre}optab$n.sh" < /dev/null > "$OUT/${pre}optab$n.out" 2> "$OUT/${pre}optab$n.err"
    grep -aE "prompt:|finish=" "$OUT/${pre}optab$n.err"
  done
  for n in 4096 16384 4089 16654; do
    need 150 "rocprofv3 $n" || continue
    echo "--- rocprofv3 $vol $n $(ts)"
    mkgen "${pre}tb$n" $vol $n 1 x --profile-every 0
    prof "${pre}tb$n" $vol
  done
}
stage_prof() { prof_run $VOL_BASE ""; }        # the BASE binary (origin/main): out/optab*, out/prof/tb*
stage_profdev() { prof_run $VOL_DEV "nd-"; }   # the DEV binary with the build's default knobs: out/nd-optab*, out/prof/nd-tb*

stage_opvp() { # the new V staging, kernel time only: rocprofv3 trace of the DEV binary at 16384, VP=1 and VP=0
  need 300 opvp || return 0
  local v
  for v in 1 0; do
    echo "--- rocprofv3 dev VP=$v 16384 $(ts)"
    mkgen "td16384-vp$v" $VOL_DEV 16384 1 "INTARSIA_HIP_FLASH_VP=$v" --profile-every 0
    prof "td16384-vp$v" $VOL_DEV
    echo "flash_hpb total ms VP=$v: $(flash_ms td16384-vp$v)"
  done
}

stage_hpb() { # flash kernel time by prompt size and INTARSIA_HIP_FLASH_HPB (4 vs 1), dev binary, VP=1: grid fill of short / tail chunks
  local n h
  for n in 512 1024 2048 4096 16654; do
    for h in 4 1; do
      need 150 "hpb $n/$h" || continue
      mkgen "hp$n-h$h" $VOL_DEV $n 1 "INTARSIA_HIP_FLASH_VP=1; export INTARSIA_HIP_FLASH_HPB=$h" --profile-every 0
      echo "--- hpb n=$n HPB=$h $(ts)"
      prof "hp$n-h$h" $VOL_DEV
      echo "flash_hpb total ms n=$n HPB=$h: $(flash_ms hp$n-h$h)"
    done
  done
}

stage_bitwise() { # the hip crate's flash_v2 (v2 == v1 bytes, incl. 35B long-context shapes) and attention tests per VP value (x = unset)
  local v t b h
  for t in flash_v2 attention; do
    b=$(INTARSIA_TARGET_VOL=$VOL_TEST $D bash -c "ls -t /target/release/deps/$t-* | grep -v '\\.d\$' | head -1" < /dev/null 2>/dev/null | tr -d '\r')
    for v in ${VARMS:-0 1}; do
      need 240 "$t VP=$v" || continue
      echo "--- test $t VP=$v ($b) $(ts)"
      INTARSIA_TARGET_VOL=$VOL_TEST $D bash -c "[ $v != x ] && export INTARSIA_HIP_FLASH_VP=$v; timeout 600 $b --nocapture --test-threads=1" < /dev/null 2>&1 | grep -aE "^test |test result|INTARSIA_HIP_FLASH|bitwise|panicked|FAILED" | head -40
    done
  done
  b=$(INTARSIA_TARGET_VOL=$VOL_TEST $D bash -c "ls -t /target/release/deps/flash_v2-* | grep -v '\\.d\$' | head -1" < /dev/null 2>/dev/null | tr -d '\r')
  for h in 2 1; do
    for v in ${VARMS:-0 1}; do
      need 120 "flash_v2 HPB=$h VP=$v" || continue
      echo "--- test flash_v2 HPB=$h VP=$v $(ts)"
      INTARSIA_TARGET_VOL=$VOL_TEST $D bash -c "[ $v != x ] && export INTARSIA_HIP_FLASH_VP=$v; export INTARSIA_HIP_FLASH_HPB=$h; timeout 600 $b --nocapture --test-threads=1" < /dev/null 2>&1 | grep -aE "^test |test result|INTARSIA_HIP_FLASH|bitwise|panicked|FAILED" | head -10
    done
  done
}

stage_ab() { # interleaved unprofiled prefill, n=3: 16384 / 16654 / 4096, base binary + dev with INTARSIA_HIP_FLASH_VP=<v> for v in VARMS (default "0 1"; x = unset)
  local rep n v
  for n in 4096 16384 16654; do
    mkgen "ab-base-$n" $VOL_BASE $n 1 x --profile-every 0
    for v in ${VARMS:-0 1}; do mkgen "ab-d$v-$n" $VOL_DEV $n 1 "$(vpenv $v)" --profile-every 0; done
  done
  for rep in 1 2 3; do
    need 400 "ab rep $rep" || return 0
    echo "--- ab rep $rep $(ts)"
    for n in 4096 16384 16654; do
      echo "base-$n: $(run ab-base-$n $VOL_BASE)"
      for v in ${VARMS:-0 1}; do echo "dev$v-$n: $(run ab-d$v-$n $VOL_DEV)"; done
    done
  done
}

stage_ab8() { # the same for the 8B (D = 128 kernels): 4096 / 16384 prompts, base vs dev with VARMS
  local rep n v
  for n in 4096 16384; do
    mkgen "ab8-base-$n" $VOL_BASE $n 1 x --profile-every 0
    sed -i "s#-m $M35 --backend hip --mem-budget 29#-m /models/Qwen3-8B --backend hip#" "$PF/gen-ab8-base-$n.sh"
    for v in ${VARMS:-0 1}; do
      mkgen "ab8-d$v-$n" $VOL_DEV $n 1 "$(vpenv $v)" --profile-every 0
      sed -i "s#-m $M35 --backend hip --mem-budget 29#-m /models/Qwen3-8B --backend hip#" "$PF/gen-ab8-d$v-$n.sh"
    done
  done
  for rep in 1 2 3; do
    need 300 "ab8 rep $rep" || return 0
    echo "--- ab8 rep $rep $(ts)"
    for n in 4096 16384; do
      echo "8b base-$n: $(run ab8-base-$n $VOL_BASE)"
      for v in ${VARMS:-0 1}; do echo "8b dev$v-$n: $(run ab8-d$v-$n $VOL_DEV)"; done
    done
  done
}

stage_vpm() { # flash kernel time (rocprofv3) of each INTARSIA_HIP_FLASH_VP value in VARMS at 16384, dev binary
  local v
  for v in ${VARMS:-0 1 3 5 7}; do
    need 150 "vpm $v" || continue
    mkgen "vpm$v" $VOL_DEV 16384 1 "$(vpenv $v)" --profile-every 0
    echo "--- vpm VP=$v $(ts)"
    prof "vpm$v" $VOL_DEV
    echo "flash_hpb total ms VP=$v: $(flash_ms vpm$v)"
  done
}

stage_ids() { # 256 greedy ids at 4096 and 16384: base binary vs dev VP=0 vs dev VP=1
  local n
  for n in 4096 16384 16654; do
    need 400 "ids $n" || continue
    echo "--- ids $n $(ts)"
    mkgen "ids-base-$n" $VOL_BASE $n 256 x --show-ids --profile-every 0
    mkgen "ids-d0-$n" $VOL_DEV $n 256 "$(vpenv 0)" --show-ids --profile-every 0
    mkgen "ids-d1-$n" $VOL_DEV $n 256 "$(vpenv ${IDV:-1})" --show-ids --profile-every 0
    INTARSIA_TARGET_VOL=$VOL_BASE $D bash -c "bash /work/$PF/gen-ids-base-$n.sh" < /dev/null 2> "$OUT/ids-base-$n.err" > /dev/null
    INTARSIA_TARGET_VOL=$VOL_DEV $D bash -c "bash /work/$PF/gen-ids-d0-$n.sh" < /dev/null 2> "$OUT/ids-d0-$n.err" > /dev/null
    INTARSIA_TARGET_VOL=$VOL_DEV $D bash -c "bash /work/$PF/gen-ids-d1-$n.sh" < /dev/null 2> "$OUT/ids-d1-$n.err" > /dev/null
    local a
    for a in base d0 d1; do grep -a '^ids:' "$OUT/ids-$a-$n.err" > "$OUT/ids-$a-$n.ids"; grep -aE "prompt:|finish=" "$OUT/ids-$a-$n.err" | sed "s/^/$a: /"; done
    echo "ids $n: $(tr -cd ',' < "$OUT/ids-base-$n.ids" | wc -c) commas in base"
    cmp -s "$OUT/ids-base-$n.ids" "$OUT/ids-d0-$n.ids" && echo "IDS $n: base == dev VP=0" || echo "IDS $n: base != dev VP=0"
    cmp -s "$OUT/ids-base-$n.ids" "$OUT/ids-d1-$n.ids" && echo "IDS $n: base == dev VP=${IDV:-1}" || echo "IDS $n: base != dev VP=${IDV:-1}"
  done
}

stage_ids8() { # 256 greedy ids of the 8B after the 4096 prompt: base vs dev (VP = IDV, x = default)
  local a vol
  mkgen ids8-base $VOL_BASE 4096 256 x --show-ids --profile-every 0
  mkgen ids8-dev $VOL_DEV 4096 256 "$(vpenv ${IDV:-1})" --show-ids --profile-every 0
  sed -i "s#-m $M35 --backend hip --mem-budget 29#-m /models/Qwen3-8B --backend hip#" "$PF/gen-ids8-base.sh" "$PF/gen-ids8-dev.sh"
  echo "--- ids8 $(ts)"
  INTARSIA_TARGET_VOL=$VOL_BASE $D bash -c "bash /work/$PF/gen-ids8-base.sh" < /dev/null 2> "$OUT/ids8-base.err" > /dev/null
  INTARSIA_TARGET_VOL=$VOL_DEV $D bash -c "bash /work/$PF/gen-ids8-dev.sh" < /dev/null 2> "$OUT/ids8-dev.err" > /dev/null
  for a in base dev; do grep -a '^ids:' "$OUT/ids8-$a.err" > "$OUT/ids8-$a.ids"; grep -aE "prompt:|finish=" "$OUT/ids8-$a.err" | sed "s/^/8b $a: /"; done
  echo "ids8: $(tr -cd ',' < "$OUT/ids8-base.ids" | wc -c) commas in base"
  cmp -s "$OUT/ids8-base.ids" "$OUT/ids8-dev.ids" && echo "IDS8: base == dev VP=${IDV:-1}" || echo "IDS8: base != dev VP=${IDV:-1}"
}

stage_chunk() { # finding only (not a lever): --prefill-chunk 8192 / 4096 on the 16384 prompt, base binary, ids + prefill time
  local c
  for c in 4096 8192; do
    need 200 "chunk $c" || continue
    echo "--- chunk $c 16384 $(ts)"
    mkgen "ch-$c" $VOL_BASE 16384 256 x --show-ids --profile-every 0 --prefill-chunk $c
    INTARSIA_TARGET_VOL=$VOL_BASE $D bash -c "bash /work/$PF/gen-ch-$c.sh" < /dev/null 2> "$OUT/ch-$c.err" > /dev/null
    grep -aE "prompt:|finish=|rror" "$OUT/ch-$c.err"
    grep -a '^ids:' "$OUT/ch-$c.err" > "$OUT/ch-$c.ids"
  done
  cmp -s "$OUT/ch-4096.ids" "$OUT/ch-8192.ids" && echo "CHUNK 8192 ids == chunk 4096 ids" || echo "CHUNK 8192 ids != chunk 4096 ids"
}

for s in "$@"; do
  echo "################ stage $s start $(ts) (budget left $(left) s)"
  "stage_$s"
  echo "################ stage $s end $(ts)"
done
