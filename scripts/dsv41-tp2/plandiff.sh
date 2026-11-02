#!/usr/bin/env bash
# Lane dsv41/tp2: `intarsia plan show` of the 35B NVFP4 and Qwen3-8B on CUDA, base (BASE_REF, default dsv41/plan) vs this
# branch's binary ($SPARK_DIR/bin/intarsia, from build.sh). Builds the base on spark2 in $SPARK_DIR/src-base.
# EXL3=1 also diffs main's EXL3 checkpoints (Qwen3-8B EXL3 4.0 bpw, plain and bf16plain, on spark1): plan show and 32 greedy
# tokens with each binary (run inside a --quiet gpu,cpu slot on spark1 as well: the job must hold spark1 and spark2).
#   SPARK_DIR=~/intarsia-mylane fleetq run --host spark2 --load cpu --mem 20 --expect 25 --label "dsv41 tp2 plan diff" -- scripts/dsv41-tp2/plandiff.sh
# SPARK_DIR is required, no default: this script deletes and rebuilds $SPARK_DIR/src-base and tests $SPARK_DIR/bin/intarsia, so
# running it without naming your own lane's directory would use (and clobber) another lane's.
if [[ -z ${SPARK_DIR:-} ]]; then echo "plandiff.sh: set SPARK_DIR to your lane's Spark directory (e.g. SPARK_DIR=~/intarsia-dsv41-draftdev); it deletes \$SPARK_DIR/src-base" >&2; exit 2; fi
source "$(dirname "${BASH_SOURCE[0]}")/spark-dir.sh"
# The one recursive delete in these scripts: $SPARK_DIR/src-base, a real directory under the home directory, nothing else.
remove_base_tree() {
  [[ $SPARK_DIR == '$HOME/'* && $SPARK_DIR != *..* ]] || { echo "refusing to remove a base tree: SPARK_DIR is '$SPARK_DIR'"; exit 3; }
  rsh "d=$SPARK_DIR/src-base; case \"\$d\" in \"\$HOME\"/?*/src-base) ;; *) echo \"refusing to remove \$d\"; exit 3;; esac; [ ! -L \"\$d\" ] || { echo \"refusing to remove \$d: a symlink\"; exit 3; }; rm -rf -- \"\$d\""
}
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE=${BASE_REF:-dsv41/plan}
OUT=$REPO/.scratch/plandiff; mkdir -p "$OUT"
RP='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH; export MAX_JOBS=4 CARGO_BUILD_JOBS=4;'
rsh() { ssh -n -o BatchMode=yes spark2 "$RP export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
# GREEDY_ONLY=1 reuses the base binary a previous run built and proved.
if [[ ${GREEDY_ONLY:-0} != 1 ]]; then
# Always a clean base: no copied target dir and no files left from an earlier base. git archive stamps files with the
# commit time, so against an existing target cargo calls them fresh and relinks whatever was built there, and the CUDA
# build script compiles every kernels/cuda/*.cu it finds, left-overs included.
remove_base_tree
rsh 'mkdir -p '"$SPARK_DIR"'/src-base '"$SPARK_DIR"'/logs'
git -C "$REPO" archive --format=tar "$BASE" | ssh -o BatchMode=yes spark2 'tar -C '"$SPARK_DIR"'/src-base -xf -'
BASE_SHA=$(git -C "$REPO" rev-parse --short "$BASE")
BASE_STAMP="plandiff-base-$BASE_SHA-$(date +%m%d%H%M%S)"
echo "base $BASE_SHA, stamp $BASE_STAMP"
rsh "cd $SPARK_DIR/src-base && INTARSIA_BUILD_STAMP=$BASE_STAMP cargo build --release --features cuda -p intarsia-cli > $SPARK_DIR/logs/plandiff-base.log 2>&1; rc=\$?; tail -2 $SPARK_DIR/logs/plandiff-base.log; echo \"compiled \$(grep -c Compiling $SPARK_DIR/logs/plandiff-base.log) crates\"; exit \$rc"
# The base binary must really be this base: it carries this run's stamp, the workspace crates were compiled here, it lacks
# BASE_MUST_LACK (strings only the branch's code has, space separated), and the branch binary has them (so the check bites).
BB="$SPARK_DIR/src-base/target/release/intarsia"
rsh "strings $BB | grep -q $BASE_STAMP" || { echo "base binary lacks stamp $BASE_STAMP"; exit 3; }
rsh "grep -q 'Compiling intarsia-core' $SPARK_DIR/logs/plandiff-base.log" || { echo "intarsia-core was not compiled for the base"; exit 3; }
for x in ${BASE_MUST_LACK:-}; do
  git -C "$REPO" grep -q "$x" "$BASE" -- crates kernels && { echo "$x is in the sources of $BASE: not a usable guard"; exit 3; }
  rsh "strings $BB | grep -q $x" && { echo "base binary contains $x: not built from $BASE"; exit 3; }
  rsh "strings $SPARK_DIR/bin/intarsia | grep -q $x" || { echo "branch binary lacks $x: the guard proves nothing"; exit 3; }
  echo "guard $x: absent from the base binary, present in the branch binary"
done
fi
echo "branch binary: $(rsh "cat $SPARK_DIR/bin/STAMP")"
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
M8=/home/nvidia/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
[[ ${GREEDY_ONLY:-0} == 1 ]] || for m in 35b:$M35 8b:$M8; do
  n=${m%%:*}; d=${m#*:}
  rsh "$SPARK_DIR/src-base/target/release/intarsia plan show -m $d --backend cuda 2>&1" > "$OUT/$n-base.txt" || true
  rsh "$SPARK_DIR/bin/intarsia plan show -m $d --backend cuda 2>&1" > "$OUT/$n-tp2.txt" || true
  if diff -q <(grep -v -iE 'built|stamp|took|loaded in|ms\b' "$OUT/$n-base.txt") <(grep -v -iE 'built|stamp|took|loaded in|ms\b' "$OUT/$n-tp2.txt") >/dev/null; then
    echo "$n: plan show unchanged ($(wc -l < "$OUT/$n-tp2.txt") lines)"
  else
    echo "$n: plan show DIFFERS"; { diff "$OUT/$n-base.txt" "$OUT/$n-tp2.txt" | head -40; true; }
  fi
done
# GREEDY=1: 64 greedy tokens of the 35B and Qwen3-8B with both binaries (plan show says nothing about shared runtime code)
if [[ ${GREEDY:-0} == 1 ]]; then
  for m in 35b:$M35 8b:$M8; do
    n=${m%%:*}; d=${m#*:}
    for b in base tp2; do
      bin="$SPARK_DIR/bin/intarsia"; [[ $b == base ]] && bin="$SPARK_DIR/src-base/target/release/intarsia"
      rsh "$bin generate -m $d --backend cuda -p 'The sky appears blue because' --max-tokens 64 2>/dev/null" > "$OUT/$n-$b-greedy.txt" || echo "$n $b: generate exit $?"
    done
    if [[ -s "$OUT/$n-tp2-greedy.txt" ]] && cmp -s "$OUT/$n-base-greedy.txt" "$OUT/$n-tp2-greedy.txt"; then echo "$n: 64 greedy tokens identical: $(tr '\n' ' ' < "$OUT/$n-tp2-greedy.txt" | head -c 100)"; else echo "$n: greedy DIFFERS or empty"; { diff "$OUT/$n-base-greedy.txt" "$OUT/$n-tp2-greedy.txt" | head -6; true; }; fi
  done
  [[ ${GREEDY_ONLY:-0} == 1 ]] && exit 0
fi
if [[ ${EXL3:-0} == 1 ]]; then
  r1() { ssh -n -o BatchMode=yes spark1 "export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
  rsh 'scp -q '"$SPARK_DIR"'/src-base/target/release/intarsia spark1:'"$SPARK_REL"'/bin/intarsia-base'
  for d in qwen3-8b-exl3-4.0bpw qwen3-8b-exl3-4.0bpw-bf16plain; do
    m=/home/nvidia/models-m2/$d
    for b in base tp2; do
      bin="$SPARK_DIR/bin/intarsia"; [[ $b == base ]] && bin="$SPARK_DIR/bin/intarsia-base"
      r1 "$bin plan show -m $m --backend cuda 2>&1" > "$OUT/$d-$b.txt" || true
      r1 "$bin generate -m $m --backend cuda -p 'The sky appears blue because' --max-tokens 32 2>&1 | grep -vE 'loaded|finish=|steps=|prefill chunk'" > "$OUT/$d-$b-gen.txt" || true
    done
    if diff -q <(grep -v -iE 'built|stamp|took|loaded in|ms\b' "$OUT/$d-base.txt") <(grep -v -iE 'built|stamp|took|loaded in|ms\b' "$OUT/$d-tp2.txt") >/dev/null; then
      echo "$d: plan show unchanged ($(wc -l < "$OUT/$d-tp2.txt") lines)"
    else
      echo "$d: plan show DIFFERS"; { diff "$OUT/$d-base.txt" "$OUT/$d-tp2.txt" | head -20; true; }
    fi
    # stdout (tokens) and stderr (notes) interleave differently from run to run: the capture-failure note moves around the tokens
    norm() { sed 's/decode graph capture failed[^;]*; decoding eagerly from here on//' "$1" | tr -d ' \n'; }
    if [[ "$(norm "$OUT/$d-base-gen.txt")" == "$(norm "$OUT/$d-tp2-gen.txt")" ]]; then echo "$d: greedy output identical (outcome: $(tail -1 "$OUT/$d-tp2-gen.txt" | head -c 120))"; else echo "$d: greedy DIFFERS"; { diff "$OUT/$d-base-gen.txt" "$OUT/$d-tp2-gen.txt" | head; true; }; fi
  done
fi
