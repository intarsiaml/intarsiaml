#!/bin/zsh
# so lane, ONE non-exclusive fleetq slot (CPU only):
#   1. main (wt-so-base): run mask_digests_are_stable (pinned digests absent there, so it prints "digest NAME HEX" lines)
#      plus the CPU phase microbench with main's apply
#   2. write main's digests into the head tree's EXPECTED_DIGESTS
#   3. head (wt-so): structured tests + microbench, then the full workspace test
# fleetq run --host endor --mem 16 --expect 30 --label "so: cargo test (main digests, head structured + workspace, cpu)" -- zsh test.sh <out dir>
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
OUT=${1:?out dir}
mkdir -p $OUT
export INTARSIA_TEST_TOKENIZER=${INTARSIA_TEST_TOKENIZER:-/Users/josef/models/Qwen3-8B/tokenizer.json}
if [[ -z "${SKIP_BASE:-}" ]]; then
echo "== main digests + microbench $(date)"
(cd /opt/code/wt-so-base && CARGO_TARGET_DIR=/opt/code/wt-so-base/target-test cargo test --release -p intarsia-server structured -- --nocapture --test-threads 1) > $OUT/base-structured.log 2>&1
grep -E "^digest |phase |mean allowed|test result|FAILED|panicked" $OUT/base-structured.log | head -40
D=("${(@f)$(grep -oE 'digest [a-z]+ [0-9a-f]{16}' $OUT/base-structured.log)}")
if (( ${#D} == 4 )); then
  python3 - "${D[@]}" <<'PY'
import sys,re
vals=[l.split()[1:3] for l in sys.argv[1:]]
want=", ".join('"%s=%s"'%(n,h) for n,h in vals)
p="/opt/code/wt-so/crates/intarsia-server/src/structured.rs"
s=open(p).read()
s=re.sub(r'const EXPECTED_DIGESTS: \[&str; 4\] = \[[^\]]*\];','const EXPECTED_DIGESTS: [&str; 4] = [%s];'%want,s)
open(p,"w").write(s)
print("wrote",want)
PY
else
  echo "no digests from main (${#D}); head digest test will fail" >&2
fi
fi
echo "== head structured + microbench $(date)"
(cd /opt/code/wt-so && CARGO_TARGET_DIR=/opt/code/wt-so/target-test cargo test --release -p intarsia-server -p intarsia-core -- structured constrain --nocapture --test-threads 1) > $OUT/head-structured.log 2>&1
grep -E "^digest |phase |mean allowed|test result|FAILED|panicked" $OUT/head-structured.log | head -40
echo "== head workspace $(date)"
(cd /opt/code/wt-so && CARGO_TARGET_DIR=/opt/code/wt-so/target-test cargo test --release --workspace --no-fail-fast) > $OUT/head-workspace.log 2>&1
echo "workspace exit $?"
grep -E "test result|FAILED|failed" $OUT/head-workspace.log | sort | uniq -c | head -20
