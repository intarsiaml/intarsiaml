#!/usr/bin/env bash
# Lane mtp-snap-gpu (ON THE MAC, under `fleetq run --load cpu`): run build.sh on tatooine for <pr|main> and keep the log.
set -u
exec ssh -o BatchMode=yes tatooine "bash ~/intarsiaml-mtpsnap-gpu/scripts/mtp-snap-gpu/build.sh ${1:?pr|main}" < /dev/null
