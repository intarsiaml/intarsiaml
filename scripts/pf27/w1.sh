#!/usr/bin/env bash
# Lane PF27 window script, ON tatooine INSIDE one fleetq job that holds both R9700s and takes radiance down (stopped here, restored by
# the EXIT trap before the job ends). Sections (env SEC, run in this order):
#   P     rocprofv3 kernel trace of one 4k prefill (generate --max-tokens 1, tp2, spec off) per CLI in PBINS -> $O/p-<bin>.csv.gz
#   T     generate 4k prefill timing, base / dev interleaved, REPS rounds (quick end-to-end signal, not the gate)
#   corr  the a8_ correctness tests of the dev test binary (default env, DIRECT=0, PERGROUP=1)
#   hash  a8_mma_hashes: base binary, dev binary (default, DIRECT=0, each DTILE of DTILES): all must equal the base
#   time  timing_a8_tp2 at M in MS: base binary, dev binary per DTILE
#   ktrace rocprofv3 kernel trace of timing_a8_tp2 (M=2048), base and dev
#   adm   kernels test --backend hip --quick of the dev CLI
#   plan  plan show of the 8B and the 35B, base vs dev CLI (byte-identical)
#   O=<out dir> SEC="..." bash ~/intarsiaml-pf27/scripts/pf27/w1.sh
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
SEC=${SEC:-P T corr hash time ktrace adm plan}; MS=${MS:-1024,2048,4096}; REPS=${REPS:-3}
PBINS=${PBINS:-pf27-base-cli pf27-dev-cli}; DTILES=${DTILES:-4424 4224 2444 2244 4442}
TB=/work/pf27-bin/mxfp4_a8-base; TD=${TD:-/work/pf27-bin/mxfp4_a8-dev}
O=${O:-$HOME/intarsiaml-pf27/pf27-out/w1}; mkdir -p "$O"; T0=$(date +%s)
has() { [[ " $SEC " == *" $1 "* ]]; }
left() { echo $(( ${WINDOW_S:-3600} - ($(date +%s) - T0) )); }
if [[ -z "${NOSTOP:-}" ]]; then  # NOSTOP=1: inside a window whose caller (drive.sh's guard) owns radiance
  trap restore EXIT
  echo "[pf27] stopping radiance $(now)"; systemctl --user stop radiance < /dev/null; sleep 3
fi
echo "[pf27] radiance $(systemctl --user is-active radiance) $(now)"
P=$(cat $R/scripts/cmtp/ids-prompts/4k-0.txt)
GEN="generate -m $M27 --backend hip --tp 2 --devices 0,1 --mem-budget 29 --max-seq 8192 --spec off --ignore-eos --max-tokens 1"

if has P; then
  for b in $PBINS; do
    d=/work/pf27-prof; rm -rf "$R/pf27-prof"; mkdir -p "$R/pf27-prof"
    gpup 235 rocprofv3 --kernel-trace --output-format csv -d $d -o run -- /work/$b $GEN --profile-every 0 -p "$P" > "$O/p-$b.out" 2> "$O/p-$b.err"
    csv=$(find "$R/pf27-prof" -name '*kernel_trace.csv' | head -1)
    [[ -n "$csv" ]] && gzip -c "$csv" > "$O/p-$b.csv.gz"
    rm -rf "$R/pf27-prof"
    echo "P $b: $(grep -a 'prefill ' "$O/p-$b.err" | tail -1 | cut -c1-160) csv=$(ls -s $O/p-$b.csv.gz 2>/dev/null || echo NO) $(now)"
  done
fi
if has T; then
  for rep in $(seq 1 $REPS); do
    for arm in ${TARMS:-pf27-base-cli pf27-dev-cli pf27-dev-cli:4096}; do
      b=${arm%%:*}; ch=""; [[ $arm == *:* ]] && ch="--prefill-chunk ${arm#*:}"; tg=$(echo $arm | tr ':' '_')
      gpu 230 /work/$b $GEN $ch --profile-every 0 -p "$P" > "$O/t-$tg-$rep.out" 2> "$O/t-$tg-$rep.err"
      echo "T $arm rep$rep: $(grep -a 'prefill ' "$O/t-$tg-$rep.err" | tail -1 | cut -c1-160)"
    done
  done
fi
if has corr; then
  for e in "A=1" "INTARSIA_HIP_MXFP4_A8_DIRECT=0" "INTARSIA_HIP_MXFP4_A8_PERGROUP=1"; do
    f="$O/corr-$(echo $e | tr ' =' '__').txt"
    G=0 gpu 900 env $e $TD a8_ > "$f" 2>&1
    echo "== corr [$e]: $(grep -aE '^test result' "$f" | tail -1) $(grep -aE '^test .*FAILED' "$f" | tr '\n' ' ')"
  done
fi
if has hash; then
  hrun() { G=0 gpu 900 env "${@:3}" $2 --ignored --nocapture a8_mma_hashes 2>&1 | grep -a "^HASH\|panicked" > "$O/hash-$1.txt"; }
  hrun base $TB A=1
  hrun dev $TD A=1
  hrun dev-direct0 $TD INTARSIA_HIP_MXFP4_A8_DIRECT=0
  hrun dev-pergroup $TD INTARSIA_HIP_MXFP4_A8_PERGROUP=1
  hrun base-pergroup $TB INTARSIA_HIP_MXFP4_A8_PERGROUP=1
  for t in $DTILES; do [[ $(left) -lt 600 ]] && break; hrun dev-$t $TD INTARSIA_HIP_MXFP4_A8_DTILE=${t:0:1},${t:1:1},${t:2:1},${t:3:1}; done
  for f in "$O"/hash-dev*.txt; do
    ref="$O/hash-base.txt"; [[ $f == *pergroup* ]] && ref="$O/hash-base-pergroup.txt"
    echo "== $(basename $f): $(wc -l < $f) lines, differing from $(basename $ref) ($(wc -l < $ref) lines): $(diff "$ref" "$f" | grep -c '^<')"
  done
fi
if has hq; then  # quick: dev hashes of each DTILE against W1's base hashes (HREF)
  HREF=${HREF:-$HOME/intarsiaml-pf27/pf27-out/w1/hash-base.txt}
  for t in $DTILES; do
    G=0 gpu 300 env INTARSIA_HIP_MXFP4_A8_DTILE=${t:0:1},${t:1:1},${t:2:1},${t:3:1} $TD --ignored --nocapture a8_mma_hashes 2>&1 | grep -a "^HASH\|panicked" > "$O/hash-dev-$t.txt"
    echo "== hq $t: $(wc -l < "$O/hash-dev-$t.txt") lines, differing from base: $(diff "$HREF" "$O/hash-dev-$t.txt" | grep -c '^<')"
  done
fi
if has time; then
  t() { local tag=$1 bin=$2; shift 2
    [[ $(left) -lt 200 ]] && { echo "[pf27] skip time $tag (window)"; return; }
    G=0 gpu 300 env GLR_TIMING_M=$MS "$@" $bin --ignored --nocapture timing_a8_tp2 2>&1 | grep -aE "^TIME|panicked|error" > "$O/t-$tag.txt"
    echo "== time $tag: $(awk '/^TIME/ {s += $(NF-3)} END {print s " us summed"}' "$O/t-$tag.txt") $(now)"
  }
  [[ -z "${TQUICK:-}" ]] && t base $TB A=1
  t dev $TD A=1
  for tt in $DTILES; do t dev-$tt $TD INTARSIA_HIP_MXFP4_A8_DTILE=${tt:0:1},${tt:1:1},${tt:2:1},${tt:3:1}; done
  t base2 $TB A=1
  [[ -z "${TQUICK:-}" ]] && t dev2 $TD A=1
fi
if has diag; then  # PF27-DIAG timing-only variants of the direct kernel (TD = the diag test binary), M = 2048
  for d in ${DIAGS:-0 1 2 3 4 5 6 0}; do
    G=0 gpu 300 env GLR_TIMING_M=2048 INTARSIA_HIP_MXFP4_A8_DIAG=$d $TD --ignored --nocapture timing_a8_tp2 2>&1 | grep -aE "^TIME|panicked|error" > "$O/diag-$d.txt"
    echo "== diag $d: $(awk '/^TIME/ {s += $(NF-3)} END {print s " us summed"}' "$O/diag-$d.txt")"
  done
  for d in ${DHASH:-}; do  # correct-result diag variants: hashes against W1's base
    G=0 gpu 300 env INTARSIA_HIP_MXFP4_A8_DIAG=$d $TD --ignored --nocapture a8_mma_hashes 2>&1 | grep -a "^HASH\|panicked" > "$O/hash-diag-$d.txt"
    echo "== hash diag $d: $(wc -l < "$O/hash-diag-$d.txt") lines; bf16 lines differing from base: $(grep -a Bf16 "$O/hash-diag-$d.txt" | diff <(grep -a Bf16 $HOME/intarsiaml-pf27/pf27-out/w1/hash-base.txt) - | grep -c '^<')"
  done
fi
if has ktrace; then
  for v in "base|$TB" "dev|$TD"; do
    [[ $(left) -lt 200 ]] && break
    tag=${v%%|*}; bin=${v#*|}; d=/work/pf27-prof/k-$tag; rm -rf "$R/pf27-prof/k-$tag"; mkdir -p "$R/pf27-prof"
    G=0 gpup 200 rocprofv3 --kernel-trace --output-format csv -d $d -o k -- env GLR_TIMING_M=2048 GLR_TIMING_ITERS=20 $bin --ignored --nocapture timing_a8_tp2 > "$O/kt-$tag.out" 2>&1
    f=$(find "$R/pf27-prof/k-$tag" -name '*kernel_trace.csv' | head -1)
    if [[ -n "$f" ]]; then gzip -c "$f" > "$O/kt-$tag.csv.gz"; echo "[pf27] ktrace $tag ok"; else echo "[pf27] ktrace $tag: no csv"; fi
    rm -rf "$R/pf27-prof/k-$tag"
  done
fi
if has adm; then
  G=0 gpu 900 /work/pf27-dev-cli kernels test --backend hip --quick > "$O/adm.log" 2>&1
  echo "== admission exit $?: $(grep -ac ' ADMITTED' "$O/adm.log") admitted rows; tail:"; tail -4 "$O/adm.log"
fi
if has plan; then
  for m in 8B:/models/Qwen3-8B 35B:/models/Qwen3.6-35B-A3B-NVFP4; do
    tag=${m%%:*}; path=${m#*:}
    G=0 MODELS=/data/models/intarsia gpu 150 /work/pf27-base-cli plan show --backend hip --model $path --max-seq 8192 > "$O/plan-$tag-base.txt" 2> "$O/plan-$tag-base.err" &
    G=1 MODELS=/data/models/intarsia gpu 150 /work/pf27-dev-cli plan show --backend hip --model $path --max-seq 8192 > "$O/plan-$tag-dev.txt" 2> "$O/plan-$tag-dev.err" &
    wait
    if cmp -s "$O/plan-$tag-base.txt" "$O/plan-$tag-dev.txt" && [ -s "$O/plan-$tag-base.txt" ]; then echo "== plan $tag: byte-identical ($(wc -c < "$O/plan-$tag-base.txt") B)"; else echo "== plan $tag: DIFFER or empty"; fi
  done
fi
echo "[pf27] window done $(now), $(left) s of the budget left"
