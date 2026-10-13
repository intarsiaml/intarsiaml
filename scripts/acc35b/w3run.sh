#!/usr/bin/env bash
# Window 3 with the section 2.2 budgets (scripts/acc35b/cpuref_gate.py: k = 2 x max over oldmoe / SIMT / HIP, larger mode).
CPUREF_BUDGET="--self-max-kl 2.0594e-02 --self-max-pos-kl 2.0801 --self-max-flip-pct 3.7500 --self-max-tail 4.9152e-03 --self-prompt-factor 7.6044" exec bash $HOME/intarsia-acc35b/scripts/w3.sh
