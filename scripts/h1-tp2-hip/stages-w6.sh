#!/usr/bin/env bash
# Lane H1 window 6 (ON tatooine, radiance down): window 5's fusion plan-diff run again, with the dev dump's provenance.
set -uo pipefail
export O=${1:?out dir}
source "$HOME/h1/lib.sh"
cp $HOME/h1/out/w1/acc-dev-35b-tp2g-short.safetensors "$O/"
E="INTARSIA_TP_FUSION=1" self dev 0,1 dev-tp2gf-vs-dev-tp2g-short acc-dev-35b-tp2g-short.safetensors $TP2 --pre-change 2ec085e \
  --self-max-kl 1e-9 --self-max-pos-kl 1e-6 --self-max-flip-pct 0 --self-max-tail 5e-8
rm -f "$O"/acc-dev-35b-tp2g-short.safetensors
log "done"
