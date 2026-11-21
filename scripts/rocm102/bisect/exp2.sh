#!/usr/bin/env bash
# Lane rocm102-bisect (ON tatooine, CPU only): experiment set 2 on gdn_chunk.hip. nc = -ffp-contract=off; dl7 = 7.2.4's
# device-library bitcode (exp1 copied it to $B/devlibs7). Relinked CLIs x-<variant> (7.2.4 kernels, gdn_chunk from the variant).
set -u
. "$(dirname "$0")/lib.sh"
H="$(dirname "$0")"
"$H/tu.sh" r7 c7nc gdn_chunk -ffp-contract=off > "$O/tu-c7nc.log" 2>&1; log "tu c7nc exit $?"
"$H/tu.sh" r10 c10nc gdn_chunk -ffp-contract=off > "$O/tu-c10nc.log" 2>&1; log "tu c10nc exit $?"
DEVLIBS=/work/.bisect/devlibs7 "$H/tu.sh" r10 c10dl7nc gdn_chunk -ffp-contract=off > "$O/tu-c10dl7nc.log" 2>&1; log "tu c10dl7nc exit $?"
for v in c7nc c10nc c10dl7nc; do
  "$H/relink.sh" "x-$v" r7 "gdn_chunk@$v" > "$O/relink-x-$v.log" 2>&1; log "relink x-$v exit $?"
done
