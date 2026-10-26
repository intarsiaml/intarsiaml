#!/usr/bin/env bash
# Lane l5r7, LOCAL, ONE fleetq job (radiance-rank1 down, card 1): the gate window g2 on the merged tree (stages-g1.sh: moe_pairs + quantmoe, G-C, no-op, self-golden
# identity), then the served A/B (ab.sh). window.sh restores radiance at its end and ab.sh stops it again. usage:
#   NAME=z3 FLEETQ_OWNER=claude-f31ccf91-l5 fleetq run --host tatooine-r9700-2 --takes-down radiance-rank1 --quiet gpu,cpu --mem 4 --expect 75 --label "..." -- scripts/l5r7/g2ab.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}" "${NAME:?}"
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WIN=g2 STAGES=scripts/l5r7/stages-g1.sh CAP=3000 "$S/window.sh"; echo "g2 window exit $?"
"$S/ab.sh"; echo "ab exit $?"
