#!/usr/bin/env bash
# Lane rocm102-bisect (ON tatooine, CPU only, after mkbins.sh): the relinked CLIs. none / all = every TU from 7.2.4 / 10.2
# (controls for the relink itself), s-<tu> = 7.2.4 kernels except <tu> from 10.2. SET="..." limits the swaps; REV=1 also makes
# r-<tu> = 10.2 kernels except <tu> from 7.2.4.
set -u
. "$(dirname "$0")/lib.sh"
"$(dirname "$0")/relink.sh" none r7 > "$O/relink-none.log" 2>&1; log "relink none exit $?"
"$(dirname "$0")/relink.sh" all r10 > "$O/relink-all.log" 2>&1; log "relink all exit $?"
for f in "$B"/objs/r7/*.o; do
  t=$(basename "$f" .o)
  [[ -n ${SET:-} && " $SET " != *" $t "* ]] && continue
  "$(dirname "$0")/relink.sh" "s-$t" r7 "$t" > "$O/relink-s-$t.log" 2>&1; log "relink s-$t exit $?"
  [[ ${REV:-0} == 1 ]] && { "$(dirname "$0")/relink.sh" "r-$t" r10 "$t" > "$O/relink-r-$t.log" 2>&1; log "relink r-$t exit $?"; }
done
for a in r7 r10; do for f in "$B"/objs/r7/*.o; do t=$(basename "$f"); cmp -s "$f" "$B/objs/r10/$t" && log "identical object $t"; done; break; done
du -sh "$B" | tee -a "$O/steps.log"
