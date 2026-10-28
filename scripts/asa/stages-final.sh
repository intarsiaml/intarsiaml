#!/usr/bin/env bash
# Lane asa (ON tatooine, radiance down; PRE=final of cat-window.sh): the ids gate (stages-ids.sh), then the gates (stages-gates.sh).
O=${1:?out dir}
bash "$HOME/intarsiaml-asa/.asa/stages-ids.sh" "$O"
bash "$HOME/intarsiaml-asa/.asa/stages-gates.sh" "$O"
