#!/usr/bin/env bash
# Lane T2 window 4 (ON tatooine, radiance down): (a) the change-vs-base self-golden checks window 3 could not run (the dump's
# commit needs --pre-change), against window 3's dumps; (b) the speed A/B (stages-bench.sh), rounds 1-3.
set -uo pipefail
export O=${1:?out dir}
source "$HOME/t2/lib.sh"
cp "$HOME"/t2/out/w3/acc-tp1.safetensors "$HOME"/t2/out/w3/acc-tp2g.safetensors "$O/"
self head 0,1 tp2g-vs-tp1 acc-tp1.safetensors $TP2 --self-vs-ref-noise
E="INTARSIA_TP_FUSION=1" self head 0,1 tp2gf-vs-tp1 acc-tp1.safetensors $TP2 --self-vs-ref-noise
E="INTARSIA_TP_FUSION=1" self head 0,1 tp2gf-vs-tp2g acc-tp2g.safetensors $TP2
self head 0 tp1-vs-tp1 acc-tp1.safetensors
for f in "$O"/self-*.out; do echo "## $(basename "$f")"; grep -aE "SELF-STATS|self-golden|verdict|KL|flip|noise" "$f" | tail -14; done > "$O/self-summary.txt"
rm -f "$O"/*.safetensors
ROUNDS="1 2 3" bash "$HOME/t2/stages-bench.sh" "$O"
