#!/usr/bin/env bash
# Window 7 (lane ink-hip-vrow's first; also #534's leftover gates), both cards. Full logs in logs/w7/.
#   card 0, the vrow tree (binary vrow1, its test targets in the volume): (a) tests/ink_mix.rs rows-exact tests (every blend limit
#           against the first forms); (b) the fuse bench (LoopBlend / ConvMix + blend, T 1..8, 16, limits r:n); (c) the verify cost
#           probe at the real 1k prompt, rows 1..8, INTARSIA_HIP_INK_BLEND_FUSE 2:2 / 16:16 / 2:16 / 8:16 / 2:2 again; (d) the bitwise
#           verify probe (rows 2..8 at 17 / 1024 / 4000 + 16 greedy windows) at 16:16.
#   card 1, #534's k5 build (verify2, its test executables saved in .scratch/bin/t-verify2): (e) the bitwise verify probe rows 2..6;
#           (f) greedy md5 of verify2, --spec off and the spec default, short-0 / 4k-0 x 64 / 256 (main2's from window 4);
#           (g) the hip tiny suites; (h) rocprofv3 kernel trace of the timing probe at the 4108-id prompt (rows 1..8, 8 reps).
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-verify
S=$REPO/.scratch
O=$REPO/logs/w7
rm -rf "$O"; mkdir -p "$O/md5"
M=/models/Inkling-2-Mini
now() { date '+%T'; }
r0() { HIP_VISIBLE_DEVICES=0 bash "$S/run.sh" "$@"; }
r1() { HIP_VISIBLE_DEVICES=1 INK_CONTAINER=ink-hip-verify-run2 bash "$S/run.sh" "$@"; }
VEXE=$(grep -aoE '/target/release/deps/verify_rows_real-[0-9a-f]+' "$REPO/logs/build-mtest-verify_rows_real.log" | tail -1)
XEXE=$(grep -aoE '/target/release/deps/ink_mix-[0-9a-f]+' "$REPO/logs/build-htest-ink_mix.log" | tail -1)
T2=/work/.scratch/bin/t-verify2
for f in verify_rows_real inkling2 inkling2_verify_softcap inkling2_dspark inkling2_dspark_cut; do
  [ -x "$S/bin/t-verify2/$f" ] || { echo "[stage] guard: t-verify2/$f missing"; exit 3; }
done
[ -x "$S/bin/intarsia-vrow1" ] && [ -x "$S/bin/intarsia-verify2" ] && [ -n "$VEXE" ] && [ -n "$XEXE" ] || { echo "[stage] guard: a binary is missing"; exit 3; }
PT="INTARSIA_VERIFY_DIR=$M INTARSIA_BATCH_TEST_BACKEND=hip INTARSIA_VERIFY_ROWS=8 INTARSIA_VERIFY_TIME_ONLY=1"
card0() {
  echo "[c0] exes $VEXE $XEXE"
  echo "[c0] (b) fuse bench $(now)"
  r0 $XEXE ink_blend_fuse_bench --ignored --nocapture --test-threads=1 > $O/fuse-bench.log 2>&1
  echo "rc $?"; grep -aE "FUSEBENCH|panicked" $O/fuse-bench.log | cut -c1-260
  # (c) blend fuse r:n, and fk0 = INTARSIA_INK_DEC_FUSED_KEYS=0 (op 75's split path at every visit, bitwise the fused kernel)
  for f in 2:2 16:16 2:16 fk0-2:2 fk0-16:16 2:2b; do
    bf=${f#fk0-}; bf=${bf%b}; fk=""; [ "${f#fk0-}" != "$f" ] && fk="INTARSIA_INK_DEC_FUSED_KEYS=0"
    echo "[c0] (c) probe 1k rows 1..8, blend fuse $bf ${fk} $(now)"
    r0 env INTARSIA_HIP_INK_BLEND_FUSE=$bf $fk $PT INTARSIA_VERIFY_PROMPT=/work/.scratch/prompts/len1k.json $VEXE --nocapture > "$O/probe-${f/:/_}.log" 2>&1
    echo "rc $?"; grep -aE "PROBE_TIME |PROBE_TIME_MIN|panicked" "$O/probe-${f/:/_}.log" | cut -c1-400
  done
  echo "[c0] (a) ink_mix rows-exact tests $(now)"
  r0 $XEXE ink_attn_res_mix_and_blend_rows_exact ink_conv_mix_rows_exact --test-threads=1 > $O/mix-tests.log 2>&1
  echo "rc $?"; grep -aE "^test |test result|panicked" $O/mix-tests.log | cut -c1-300
  echo "[c0] (d) bitwise verify probe rows 2..8, blend fuse 16:16 $(now)"
  r0 env INTARSIA_HIP_INK_BLEND_FUSE=16:16 INTARSIA_VERIFY_DIR=$M INTARSIA_BATCH_TEST_BACKEND=hip INTARSIA_VERIFY_ROWS=8 INTARSIA_VERIFY_LENS=17,1024,4000 $VEXE --nocapture > $O/probe-bitwise-16_16.log 2>&1
  echo "rc $?"; grep -aE "PROBE_RESULT|PROBE_GRAPHS|panicked|test result" $O/probe-bitwise-16_16.log | cut -c1-300
  echo "[c0] done $(now)"
}
PS="$(cat "$REPO/scripts/cmx/ids-prompts/short-0.txt")"
P4="$(cat "$REPO/scripts/cmx/ids-prompts/4k-0.txt")"
gen() {  # binary mode(off|dflt) prompt-name tokens
  local f=$O/md5/$1-$2-$3-$4 X=""
  [ "$2" = off ] && X="--spec off"
  local PR="$PS"; [ "$3" = 4k ] && PR="$P4"
  r1 /work/.scratch/bin/intarsia-$1 generate -m $M --backend hip --max-seq 8192 --ignore-eos --profile-every 0 --max-tokens "$4" $X -p "$PR" > "$f.out" 2> "$f.err"
  local rc=$? md5; md5=$(sed -e 's/finish=length.*//' -e '/steps=/d' -e '/tok\/s/d' "$f.out" | md5sum | cut -c1-32)
  echo "MD5 $1 $2 $3 $4 $md5 rc $rc | $(cat "$f.out" "$f.err" | grep -aoE 'decode [0-9]+ ms \([0-9.]+ tok/s\)' | head -1) | $(grep -aE '^spec: verify' "$f.err" | head -1 | cut -c1-90) | $(grep -aE '^--spec dflash|DSpark default' "$f.err" | head -1 | cut -c1-120)"
}
card1() {
  echo "[c1] (e) bitwise verify probe rows 2..6, verify2 $(now)"
  r1 env INTARSIA_VERIFY_DIR=$M INTARSIA_BATCH_TEST_BACKEND=hip INTARSIA_VERIFY_ROWS=6 INTARSIA_VERIFY_LENS=17,1024,4000 $T2/verify_rows_real --nocapture > $O/probe-bitwise-verify2.log 2>&1
  echo "rc $?"; grep -aE "PROBE_RESULT|PROBE_GRAPHS|panicked|test result" $O/probe-bitwise-verify2.log | cut -c1-300
  echo "[c1] (f) md5 verify2 $(now)"
  for p in short 4k; do for n in 64 256; do for m in off dflt; do gen verify2 $m $p $n; done; done; done
  echo "[c1] (g) hip tiny suites, verify2 $(now)"
  for t in inkling2 inkling2_verify_softcap inkling2_dspark inkling2_dspark_cut; do
    r1 env INTARSIA_INKLING2_BACKEND=hip $T2/$t --test-threads 1 > $O/tiny-$t.log 2>&1
    echo "$t rc $? | $(grep -aE '^test result' $O/tiny-$t.log | head -1)"; grep -aE "^test .*FAILED" $O/tiny-$t.log | head -5
  done
  echo "[c1] (h) rocprofv3 kernel trace, timing probe 4k rows 1..8, verify2 $(now)"
  mkdir -p "$O/rp4k"
  INTARSIA_IMAGE=intarsia-hip-prof:rocm7.2.4-rust1.90.0 r1 env $PT INTARSIA_VERIFY_PROMPT=/work/.scratch/prompts/len4k.json INTARSIA_VERIFY_TIME_REPS=8 \
    rocprofv3 --kernel-trace --output-format csv -d /work/logs/w7/rp4k -o run -- $T2/verify_rows_real --nocapture > $O/probe-rp4k.log 2>&1
  echo "rc $?"; grep -aE "PROBE_TIME |panicked" $O/probe-rp4k.log | cut -c1-400
  csv=$(find "$O/rp4k" -name '*kernel_trace.csv' | head -1)
  if [ -n "$csv" ]; then
    python3 "$S/rowtrace.py" "$csv" 8 8 2 --kernels > $O/rowtrace4k.txt 2>&1; head -12 $O/rowtrace4k.txt
    gzip -f "$csv"
  fi
  echo "[c1] done $(now)"
}
echo "[stage] start $(now)"
card0 > "$O/c0.log" 2>&1 &
card1 > "$O/c1.log" 2>&1 &
wait
echo "---- card 0"; cat "$O/c0.log"
echo "---- card 1"; cat "$O/c1.log"
echo "[stage] done $(now)"
