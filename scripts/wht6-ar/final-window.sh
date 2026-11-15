#!/usr/bin/env bash
# Lane wht6-ar: the last window (ON tatooine, inside window.sh): gate 1 again on the final build, the rate sweep, then the
# accuracy steps that were still open (the vLLM golden with its reference noise on / off; the long golden with a floor).
cd "$(dirname "$0")/../.."
O=$O STEPS="wht6 comm rate" scripts/wht6-ar/kernels-window.sh
O=${ACC_O:?the accuracy out dir holding accl-off.safetensors} STEPS="gref floorl" scripts/wht6-ar/accuracy-window.sh
