#!/usr/bin/env bash
# Lane H27P window 6: planted mutants + kernel admission, then the batched-serve context check
S="$(dirname "${BASH_SOURCE[0]}")"
bash "$S/mutants.sh"
bash "$S/ctx.sh"
