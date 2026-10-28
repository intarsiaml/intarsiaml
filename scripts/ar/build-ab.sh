#!/usr/bin/env bash
# Lane ars (from the Mac), ONE fleetq cpu job: hipcc check + asm of the B kernels, then the A tree (~/intarsiaml-ars, binary ars-intarsia) and the B
# tree (~/intarsiaml-ars/.ars/btree, binary copied to ~/intarsiaml-ars/ars-intarsia-b, tests built):
#   fleetq run --host tatooine --load cpu --expect 40 --label "ars: build A and B" -- env STAMPA=<a> STAMPB=<b> scripts/ar/build-ab.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SSH="ssh -o BatchMode=yes tatooine"
"$HERE/hipcheck.sh"
[[ ${ONLY:-} == b ]] || $SSH "STAMP=${STAMPA:-a} bash -s" < "$HERE/build.sh"
$SSH "STAMP=${STAMPB:-b} TREE=\$HOME/intarsiaml-ars/.ars/btree TESTS=1 bash -s" < "$HERE/build.sh"
$SSH 'cp ~/intarsiaml-ars/.ars/btree/ars-intarsia ~/intarsiaml-ars/ars-intarsia-b && ls -la ~/intarsiaml-ars/ars-intarsia-b' < /dev/null
