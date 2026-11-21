#!/usr/bin/env bash
# Lane rocm102-bisect (ON tatooine, CPU only): experiment set 1 on gdn_chunk.hip (the TU that moves the logits; win2).
# c7 / c10 = recompiles with build.rs's flags (controls), c10dl7 = 10.2 clang with 7.2.4's device-library bitcode.
# Then the relinked CLIs x-<variant> (7.2.4 kernels, gdn_chunk from the variant).
set -u
. "$(dirname "$0")/lib.sh"
H="$(dirname "$0")"
dk r7 bash -c 'for d in /opt/rocm/amdgcn/bitcode /opt/rocm/lib/llvm/lib/clang/*/lib/amdgcn/bitcode; do [[ -f $d/ocml.bc ]] && { echo "r7 devlibs $d"; mkdir -p /work/.bisect/devlibs7; cp $d/*.bc /work/.bisect/devlibs7/; break; }; done; ls /work/.bisect/devlibs7 | wc -l' < /dev/null 2>&1 | tee -a "$O/steps.log"
dk r10 bash -c 'hipcc --version | head -3; for d in /opt/rocm/amdgcn/bitcode /opt/rocm/lib/llvm/lib/clang/*/lib/amdgcn/bitcode /opt/rocm/core-10.2/lib/llvm/lib/clang/*/lib/amdgcn/bitcode; do [[ -f $d/ocml.bc ]] && echo "r10 devlibs $d"; done; hipcc -v --offload-arch=gfx1201 -x hip -c /dev/null -o /dev/null 2>&1 | grep -o "\-mlink-builtin-bitcode [^ ]*" | head' < /dev/null 2>&1 | tee -a "$O/steps.log"
"$H/tu.sh" r7 c7 gdn_chunk > "$O/tu-c7.log" 2>&1; log "tu c7 exit $?"
"$H/tu.sh" r10 c10 gdn_chunk > "$O/tu-c10.log" 2>&1; log "tu c10 exit $?"
DEVLIBS=/work/.bisect/devlibs7 "$H/tu.sh" r10 c10dl7 gdn_chunk > "$O/tu-c10dl7.log" 2>&1; log "tu c10dl7 exit $? $(tail -2 "$O/tu-c10dl7.log" | tr '\n' ' ' | cut -c1-300)"
for v in c7 c10 c10dl7; do
  [[ -f $B/objs/$v/gdn_chunk.o ]] || continue
  "$H/relink.sh" "x-$v" r7 "gdn_chunk@$v" > "$O/relink-x-$v.log" 2>&1; log "relink x-$v exit $?"
done
for p in "c7 r7" "c10 r10"; do set -- $p; cmp -s "$B/objs/$1/gdn_chunk.o" "$B/objs/$2/gdn_chunk.o" && log "objs $1 == $2 (bitwise)" || log "objs $1 != $2"; done
for v in c7 c10 c10dl7; do
  f=$B/isa/$v/gdn_chunk.s; [[ -f $f ]] || continue
  log "isa $v: $(grep -cE '^\s+v_' "$f") v_ insts; $(grep -oE '^\s+v_(exp|log|rsq|sqrt|rcp|fma|fmac|mul|add|sub|ldexp|frexp|cvt|wmma|pk_)[a-z0-9_]*' "$f" | sed 's/^\s*//' | sort | uniq -c | sort -k2 | awk '{printf "%s=%s ", $2, $1}')"
done
