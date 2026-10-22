#!/usr/bin/env bash
# Lane T2 window 3 (ON tatooine, radiance down): 27B accuracy against P1's W4A4 golden (vLLM-free: HF Quark reference) with
# its reference-noise sidecar, tp=1 vs tp=2, and tp=2 against tp=1 (change vs base, A2's --self-vs-ref-noise form).
set -uo pipefail
export O=${1:?out dir}
source "$HOME/t2/lib.sh"
acc head 0 tp1
acc head 0,1 tp2g $TP2
E="INTARSIA_TP_FUSION=1" acc head 0,1 tp2gf $TP2
# change vs base: tp=2 (graphs + fusion) against the tp=1 dump; and fusion against graphs at tp=2 (must be the floor)
E="INTARSIA_TP_FUSION=1" self head 0,1 tp2gf-vs-tp1 acc-tp1.safetensors $TP2 --self-vs-ref-noise
E="INTARSIA_TP_FUSION=1" self head 0,1 tp2gf-vs-tp2g acc-tp2g.safetensors $TP2
for f in "$O"/acc-*.out "$O"/self-*.out; do echo "## $(basename "$f")"; grep -aE "mode|KL|top-1|PASS|FAIL|noise|verdict|SELF-STATS|flip" "$f" | tail -25; done > "$O/acc-summary.txt"
log done
