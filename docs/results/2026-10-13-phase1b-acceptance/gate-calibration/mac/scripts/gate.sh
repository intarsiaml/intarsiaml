#!/bin/bash
# usage: gate.sh <name> <envfile|none> [VAR=VALUE ...]   -> $S/gate-<name>.log
S=/private/tmp/claude-501/-opt-code-intarsiaml/f31ccf91-6032-412f-aeab-eea38852047d/scratchpad/mac
name=$1; env=$2; shift 2
E=(); [ "$env" != none ] && E=(--noise-envelope "$env")
env "$@" $S/acc.sh --baseline $S/base.json "${E[@]}" --allow-same-code --allow-dirty > $S/gate-$name.log 2>&1
echo "rc=$?"
grep -E "verdict|golden gate|failing per-prompt|per-prompt gate: |^  (top-1|mean KL|tail-bucket|top-5|max KL|mean \|tail|\|tail excess)" $S/gate-$name.log | grep -E "FAIL|ok|verdict|golden gate|failing" | cut -c1-230
