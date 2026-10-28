#!/usr/bin/env bash
# Lane vat (ON tatooine, radiance down; PRE=final of cat-window.sh): the ids gate (stages-ids.sh), then the gates (stages-gates.sh).
O=${1:?out dir}
bash "$HOME/intarsiaml-vat/.vat/stages-ids.sh" "$O"
bash "$HOME/intarsiaml-vat/.vat/stages-gates.sh" "$O"
