#!/usr/bin/env bash
# Lane PF27, FROM THE MAC (worktree root) inside one fleetq --load cpu job: build the base tree that is on tatooine now (BASE=1), then
# rsync this worktree over it and build it as dev. Logs on stdout.
#   fleetq run --host tatooine-r9700,tatooine-r9700-2 --load cpu --expect 40 --label "pf27 build" -- env BASE=1 scripts/pf27/buildboth.sh
set -uo pipefail
SSH="ssh -o BatchMode=yes tatooine"
if [[ "${BASE:-0}" == 1 ]]; then $SSH "CLI=${BASE_CLI:-1} ISA=0 bash ~/intarsiaml-pf27/scripts/pf27/build.sh base" < /dev/null; fi
bash scripts/pf27/sync.sh
$SSH "CLI=${DEV_CLI:-1} bash ~/intarsiaml-pf27/scripts/pf27/build.sh ${TAG:-dev}" < /dev/null
