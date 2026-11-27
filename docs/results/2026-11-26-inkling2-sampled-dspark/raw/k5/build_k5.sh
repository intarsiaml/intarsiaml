#!/usr/bin/env bash
# k5 refresh (lane ink-sampled-k5). ON tatooine, inside a `fleetq run --host tatooine-r9700 --load cpu` job: the branch CLI at the merged head
# f44766b2d as intarsia-br5, then main 85e14bbcb (.scratch/main-src) as intarsia-main5. CLI only (no test binaries).
set -uo pipefail
T=$HOME/intarsiaml-ink-sampled/.scratch/tat
echo "[build] branch $(date +%T)"
MTESTS="" bash $T/build.sh br5 || exit $?
echo "[build] main $(date +%T)"
MTESTS="" bash $T/build.sh main5 .scratch/main-src || exit $?
echo "[build] done $(date +%T)"
md5sum $HOME/intarsiaml-ink-sampled/.scratch/bin/intarsia-br5 $HOME/intarsiaml-ink-sampled/.scratch/bin/intarsia-main5
