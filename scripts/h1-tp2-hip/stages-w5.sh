#!/usr/bin/env bash
# Lane H1 window 5 (ON tatooine, radiance down, called by window.sh): the open items of the review.
#  1. prefix reuse at tp=2 with graphs (the new default), and with graphs + fusion (35b-long, --mode reuse);
#  2. graphs + fusion against the graphs dump with the floor budgets: the plan-diff line names the fused kernels that ran;
#  3. tp=1 with the scalar MoE GEMMs (an equally correct reordering) through the real change-vs-base gate.
set -uo pipefail
export O=${1:?out dir}
source "$HOME/h1/lib.sh"
PV="--pre-change 0294b77"; PD="--pre-change 2ec085e"
FLOOR="--self-max-kl 1e-9 --self-max-pos-kl 1e-6 --self-max-flip-pct 0 --self-max-tail 5e-8"
run dev 0,1 400 reuse-dev-tp2g accuracy --backend hip --model $M35 --golden $LG --mode reuse --mem-budget 29 --max-seq 17408 $TP2
E="INTARSIA_TP_FUSION=1" run dev 0,1 400 reuse-dev-tp2gf accuracy --backend hip --model $M35 --golden $LG --mode reuse --mem-budget 29 --max-seq 17408 $TP2
cp $HOME/h1/out/w1/acc-dev-35b-tp2g-short.safetensors $HOME/h1/out/w1/acc-base-35b-short.safetensors "$O/"
E="INTARSIA_TP_FUSION=1" self dev 0,1 dev-tp2gf-vs-dev-tp2g-short acc-dev-35b-tp2g-short.safetensors $TP2 $PD $FLOOR
E="INTARSIA_HIP_MOE_WMMA=0" self base 0 base-tp1-moewmma0-vs-base-tp1-short acc-base-35b-short.safetensors $PV --self-vs-ref-noise --ref-noise $NS
rm -f "$O"/acc-dev-35b-tp2g-short.safetensors "$O"/acc-base-35b-short.safetensors
log "done"
