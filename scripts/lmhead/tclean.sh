#!/usr/bin/env bash
# Lane lmhead-fp8: remove everything this lane made on tatooine (trees, bin dirs, cargo volumes, golden / gate scratch). Run inside a slot:
#   fleetq run --host tatooine --load none --mem 0 --expect 5 --label "lmhead-fp8: tatooine cleanup" -- scripts/lmhead/tclean.sh
set -uo pipefail
ssh -n -o BatchMode=yes tatooine 'for n in lmhead lmhbase; do rm -rf ~/intarsiaml-$n ~/$n-bin; docker volume rm -f intarsia-cargo-target-$n >/dev/null; done; rm -rf ~/lmh-golden ~/lmh-gates ~/lmh-deadman.pid
echo "left over:"; ls -d ~/intarsiaml-lmh* ~/lmh* 2>/dev/null; docker volume ls --format "{{.Name}}" | grep -E "lmh" ; docker ps -a --format "{{.Names}}" | grep -E "^lmh" ; echo "(end)"; systemctl --user is-active radiance; curl -s -m 5 http://tatooine:8080/v1/models | head -c 80'
