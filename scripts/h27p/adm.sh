#!/usr/bin/env bash
# Lane H27P: kernel admission, every row that is not ADMITTED (final binary)
S="$(dirname "${BASH_SOURCE[0]}")"; source "$S/lib.sh"
gpu 230 /work/h27p-intarsia kernels test --backend hip --quick > $HOME/intarsiaml-h27p-logs/adm.log 2>&1; echo "exit $?"
grep -av " ADMITTED" $HOME/intarsiaml-h27p-logs/adm.log | grep -av "^SKIPPED" | tail -20
echo "admitted rows: $(grep -ac " ADMITTED" $HOME/intarsiaml-h27p-logs/adm.log)"
