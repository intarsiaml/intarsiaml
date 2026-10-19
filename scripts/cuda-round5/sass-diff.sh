#!/usr/bin/env bash
# SASS identity of the bf16 attention kernels across a refactor (spark1, inside a fleetq slot; no GPU work):
#   scripts/cuda-round5/sass-diff.sh <base tree> <dev tree> [object ...]      (trees under ~/intarsia-decfuse, built)
# Per object (default attention_splitkv.o attention_flash.o) and per kernel: md5 of the instruction stream (cuobjdump -sass,
# encodings included, addresses and function headers dropped). A template parameter added to a kernel changes its mangled
# name, so the kernels are matched by their hash: every base kernel's hash must occur in the dev object. Prints
# "SAME <base kernel>" / "CHANGED <base kernel>" and the dev-only kernels; exit 1 if any base kernel changed.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
BT=${1:?base tree}; DT=${2:?dev tree}; shift 2
OBJS=${*:-attention_splitkv.o attention_flash.o}
HOST=${HOST:-spark1}
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
rsh() { ssh -n -o BatchMode=yes "$HOST" "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
hashes() { # tree object -> "md5 name" per kernel
  rsh "cd ~/intarsia-decfuse/$1 && d=\$(ls -td target/acc/release/build/intarsia-backend-cuda-*/out | head -1) && cuobjdump -sass \$d/$2 | awk '
    /Function :/ { if (f != \"\") { print f \"\\t\" body } f = \$3; body = \"\"; next }
    /^[[:space:]]*\\/\\*[0-9a-f]{4,}\\*\\// { line = \$0; sub(/^[[:space:]]*\\/\\*[0-9a-f]+\\*\\//, \"\", line); body = body line \"|\"; next }
    /^[[:space:]]*\\/\\* 0x[0-9a-f]+ \\*\\// { body = body \$0 \"|\"; next }
    END { if (f != \"\") print f \"\\t\" body }' | while IFS=\$'\\t' read -r name body; do echo \"\$(printf '%s' \"\$body\" | md5sum | cut -c1-16) \$name\"; done"
}
bad=0; changed=""
for o in $OBJS; do
  echo "== $o"
  b=$(hashes "$BT" "$o"); d=$(hashes "$DT" "$o")
  [[ -n "$b" && -n "$d" ]] || { echo "no SASS for $o"; bad=1; continue; }
  while read -r h name; do
    if echo "$d" | grep -q "^$h "; then echo "SAME    $name  ($h = $(echo "$d" | grep "^$h " | head -1 | cut -d' ' -f2 | cut -c1-70))"
    else echo "CHANGED $name"; bad=1; changed="$changed $name"; fi
  done <<< "$b"
  # first changed kernel: its dev twin is the one whose name has the same leading template arguments; show a diff
  for name in $changed; do
    key=$(echo "$name" | grep -Eo 'kernelILi[0-9]+ELi[0-9]+E' | head -1)
    [[ -n "$key" ]] || continue
    twin=$(echo "$d" | awk '{print $2}' | grep -F "$key" | grep -v 'EhEE' | head -1)
    echo "-- SASS diff $key (base vs dev twin ${twin:0:60}), first 30 lines:"
    rsh "dump() { cd ~/intarsia-decfuse/\$1 && d=\$(ls -td target/acc/release/build/intarsia-backend-cuda-*/out | head -1) && cuobjdump -sass -fun '\$2' \$d/$o | grep -E '^ +/\\*[0-9a-f]{4,}\\*/' | sed -E 's|^ +/\\*[0-9a-f]+\\*/||; s|/\\* 0x[0-9a-f]+ \\*/||'; }; diff <(dump $BT '$name') <(dump $DT '$twin') | head -30"
    break
  done
  changed=""
  echo "-- dev-only kernels:"
  while read -r h name; do echo "$b" | grep -q "^$h " || echo "  NEW $name"; done <<< "$d"
done
exit $bad
