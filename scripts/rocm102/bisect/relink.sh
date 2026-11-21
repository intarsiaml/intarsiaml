#!/usr/bin/env bash
# Lane rocm102-bisect (ON tatooine, CPU only): relink.sh <name> <base arm> [tu ...]: a CLI whose kernel archive holds the base
# arm's objects except the named translation units (file stems of kernels/hip/*.hip), which come from the other arm
# (or stem@dir: from objs/<dir>/, an experiment object of tu.sh). Re-runs the
# LINK arm's captured link command (LINK=r10 default) with the archive's -L directory and the output swapped, in that arm's image.
# Out: $B/mix/<name>/intarsia (+ tus.txt). The Rust objects are the LINK arm's; only the kernel archive changes.
set -u
. "$(dirname "$0")/lib.sh"
name=$1 base=$2; shift 2
other=r10; [[ $base == r10 ]] && other=r7
LINK=${LINK:-r10}
dk "$LINK" bash -c '
  set -e; name=$1 base=$2 other=$3 link=$4; shift 4
  m=/work/.bisect/mix/$name; rm -rf "$m"; mkdir -p "$m/o"
  for f in /work/.bisect/objs/$base/*.o; do
    s=$(basename "$f" .o); src=$base
    for t in "$@"; do [[ $t == "$s" ]] && src=$other; [[ $t == "$s@"* ]] && src=${t#*@}; done
    cp /work/.bisect/objs/$src/$s.o "$m/o/$s.o"; echo "$s $src" >> "$m/tus.txt"
  done
  ar rcs "$m/libintarsia_hip_kernels.a" "$m"/o/*.o; rm -rf "$m/o"
  od=$(cat /work/.bisect/outdir-$link.txt)
  cmd=$(sed -e "s#\"$od\"#\"$m\"#g" -e "s#\"-o\" \"[^\"]*\"#\"-o\" \"$m/intarsia\"#" /work/.bisect/link-$link.txt)
  grep -q "\"$m\"" <<<"$cmd"
  eval "$cmd"
  ls -la "$m/intarsia"' _ "$name" "$base" "$other" "$LINK" "$@" < /dev/null
