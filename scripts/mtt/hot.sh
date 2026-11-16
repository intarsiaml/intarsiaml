#!/bin/zsh
# heat the box, then the TTFT probe on alternating arms (lane mtt: is the slow state the box's or the MTP default's?)
D=${0:h}; $D/heat.sh ${HEAT:-150}; shift 0; $D/ttft.sh "$@"
