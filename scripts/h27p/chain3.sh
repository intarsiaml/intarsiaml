#!/usr/bin/env bash
# Lane H27P window 8: (a) the kernel admission summary of the dev binary, (b) the bf16-partials option (INTARSIA_TP_PARTIAL=bf16): how many of
# the 4k / 16k gate prompts keep their ids (an option, not a default: it rounds each rank's partial to bf16 before the sum)
S="$(dirname "${BASH_SOURCE[0]}")"; source "$S/lib.sh"
echo "== kernel admission summary (dev binary)"
gpu 230 /work/h27p-intarsia kernels test --backend hip --quick 2>&1 | grep -av "^SKIPPED" | tail -12
d=$HOME/intarsiaml-h27p-logs/bf16; mkdir -p $d; cp $HOME/intarsiaml-h27p-logs/ids/gen-*-base.ids $d/
echo "== bf16 partials vs base ids (informational)"
DEVENV="env INTARSIA_TP_PARTIAL=bf16" REUSE_BASE=1 O=$d bash "$S/ids.sh" 4k-0 4k-1 4k-2 4k-3 4k-4 4k-5 16k-0 16k-1 16k-2 16k-3 16k-4 16k-5 2>&1 | grep -aE "^(PASS|FAIL)|fails="
