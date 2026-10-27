#!/usr/bin/env bash
# Lane HDC attribution, ON tatooine INSIDE a fleetq job that holds both R9700s and has radiance down (drive.sh `attr` item, NOSTOP=1; or
# standalone, then it stops and restores radiance itself). Cards must be free (no hdc-serve running).
# Sections (env SEC, default all): plan (plan show of the 27B under the knob arms), op (generate --op-table, synchronised per op),
# prof (rocprofv3 --kernel-trace of an unprofiled MTP-off decode, both ranks), gen (generate timing per arm, no profiler).
# Logs in ~/intarsiaml-hdc-logs/$W. BIN is the CLI inside /work.
#   W=w1 SEC='plan op prof' bash ~/intarsiaml-hdc/scripts/hdc/attr.sh
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
BIN=${BIN:-/work/hdc-base-cli}; W=${W:-w1}; SEC=${SEC:-plan op prof gen}; NTOK=${NTOK:-130}
O=$HOME/intarsiaml-hdc-logs/$W; mkdir -p "$O"; P27=$M27
has() { [[ " $SEC " == *" $1 "* ]]; }
[[ -n "${NOSTOP:-}" ]] || trap restore EXIT
[[ -n "${NOSTOP:-}" ]] || { echo "[hdc] $W stopping radiance $(now)"; systemctl --user stop radiance < /dev/null; sleep 3; }
# a ~1k-token raw prompt (3300 chars of the gate prompt set; the load line prints the token count)
PROMPT=$(head -c 3300 "$HOME/intarsiaml-hdc/scripts/cmtp/ids-prompts/4k-0.txt")
COMMON="--backend hip --model $P27 --tp 2 --devices 0,1 --max-seq 4096"

if has plan; then
  # arm name | env
  for a in "base|A=1" "arnorm|INTARSIA_TP_ARNORM=1" "wmma|INTARSIA_HIP_WMMA_ROWS=1" "wmmabf|INTARSIA_HIP_WMMA_ROWS=1 INTARSIA_HIP_WMMA_TABLE=bf16=all,fp8a16=1-16" "wmmaall|INTARSIA_HIP_WMMA_ROWS=1 INTARSIA_HIP_WMMA_TABLE=fp8a8=all,fp8a16=all,bf16=all"; do
    n=${a%%|*}; e=${a#*|}
    gpu 240 env $e $BIN plan show $COMMON --spec off > "$O/plan-$n.txt" 2> "$O/plan-$n.err"; echo "[hdc] plan $n rc=$? $(wc -c < "$O/plan-$n.txt") B $(now)"
  done
  for n in arnorm wmma wmmabf wmmaall; do echo "== plan diff base vs $n: $(diff "$O/plan-base.txt" "$O/plan-$n.txt" | grep -c '^[<>]') lines"; done
fi

if has op; then   # per-op table, one synchronised step at a time (inflates launch cost: use as a cross-check only)
  gpu 300 $BIN generate $COMMON --spec off --profile-every 1 --op-table --ignore-eos --max-tokens 48 -p "$PROMPT" > "$O/op.out" 2> "$O/op.err"
  echo "[hdc] op rc=$? $(now): $(grep -a 'loaded' "$O/op.err" | cut -c1-300)"
fi

if has gen; then  # decode rate per arm without the profiler (generate prints tok/s); 3 rounds interleaved
  for r in 1 2 3; do
    for a in "base|A=1" "arnorm|INTARSIA_TP_ARNORM=1" "fuse|INTARSIA_HIP_MXFP4_A8_FUSE=1" "anfuse|INTARSIA_TP_ARNORM=1 INTARSIA_HIP_MXFP4_A8_FUSE=1"; do
      n=${a%%|*}; e=${a#*|}
      gpu 300 env $e $BIN generate $COMMON --spec off --profile-every 0 --ignore-eos --max-tokens $NTOK -p "$PROMPT" > "$O/gen-$n-$r.out" 2> "$O/gen-$n-$r.err"
      echo "[hdc] gen $n r$r rc=$? $(grep -aiE 'tok/s|decode' "$O/gen-$n-$r.err" | tail -2 | tr '\n' ' ' | cut -c1-250)"
    done
  done
fi

if has prof; then # kernel trace of both ranks, MTP off, graphs as in serving; hipGraph kernels must show up (checked in analyze.py)
  rm -rf "$HOME/intarsiaml-hdc/hdc-prof"
  for a in "base|A=1" "arnorm|INTARSIA_TP_ARNORM=1" "fuse|INTARSIA_HIP_MXFP4_A8_FUSE=1" "anfuse|INTARSIA_TP_ARNORM=1 INTARSIA_HIP_MXFP4_A8_FUSE=1"; do
    n=${a%%|*}; e=${a#*|}
    gpup 600 env $e rocprofv3 --kernel-trace --stats -f csv -d /work/hdc-prof/$n -o k -- $BIN generate $COMMON --spec off --profile-every 0 --ignore-eos --max-tokens $NTOK -p "$PROMPT" > "$O/prof-$n.out" 2> "$O/prof-$n.err"
    echo "[hdc] prof $n rc=$? $(now): $(find "$HOME/intarsiaml-hdc/hdc-prof/$n" -name '*kernel_trace.csv' | head -2 | xargs -r ls -la | awk '{print $5, $9}' | tr '\n' ' ')"
  done
  mkdir -p "$O/prof"; cp -r "$HOME/intarsiaml-hdc/hdc-prof/." "$O/prof/" 2>/dev/null
fi
if has gendev; then  # dev CLI vs base CLI, decode rate without the profiler, 3 rounds interleaved
  for r in 1 2 3; do
    for a in "base|/work/hdc-base-cli" "dev|/work/hdc-dev-cli"; do
      n=${a%%|*}; b=${a#*|}
      gpu 300 $b generate $COMMON --spec off --profile-every 0 --ignore-eos --max-tokens $NTOK -p "$PROMPT" > "$O/gendev-$n-$r.out" 2> "$O/gendev-$n-$r.err"
      echo "[hdc] gendev $n r$r rc=$? $(grep -a 'decode ' "$O/gendev-$n-$r.err" | grep -o 'decode [0-9]* ms ([0-9.]* tok/s)')"
    done
  done
fi

if has profdev; then  # kernel trace of the dev CLI (fused-quant default), MTP off
  for a in "dev|/work/hdc-dev-cli"; do
    n=${a%%|*}; b=${a#*|}
    gpup 600 rocprofv3 --kernel-trace --stats -f csv -d /work/hdc-prof/$n -o k -- $b generate $COMMON --spec off --profile-every 0 --ignore-eos --max-tokens $NTOK -p "$PROMPT" > "$O/prof-$n.out" 2> "$O/prof-$n.err"
    echo "[hdc] profdev $n rc=$? $(now)"
  done
  mkdir -p "$O/prof"; cp -r "$HOME/intarsiaml-hdc/hdc-prof/." "$O/prof/" 2>/dev/null
fi
echo "[hdc] attr $W done $(now)"
