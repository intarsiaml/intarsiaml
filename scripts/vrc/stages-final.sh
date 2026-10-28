#!/usr/bin/env bash
# Lane vrc (ON tatooine, radiance down; PRE=final of cat-window.sh): the ids gate (stages-ids.sh), then the gates (stages-gates.sh).
O=${1:?out dir}
bash "$HOME/intarsiaml-vrc/.vrc/stages-ids.sh" "$O"
bash "$HOME/intarsiaml-vrc/.vrc/stages-gates.sh" "$O"
