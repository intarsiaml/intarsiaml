#!/usr/bin/env bash
# Window 1 (lane ink-hip-spec, go / no-go): tiny tests on hip, device top-k bf16, hc_pre admission, the real-model verify probe
# (rows 2..4 bitwise the one-row steps, verify cost against a decode step), one 1k prompt spec off vs DSpark k3.
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-spec
S=$REPO/.scratch
M=/models/Inkling-2-Mini
B=/work/.scratch/bin/intarsia-${BR:-spec1}
now() { date '+%F %T'; }
r() { bash "$S/run.sh" "$@"; }
echo "[stage] tiny inkling2 tests on hip $(now)"
r env INTARSIA_INKLING2_BACKEND=hip cargo test --release -p intarsia-models --test inkling2 --test inkling2_verify_softcap -- --test-threads 1 2>&1 | grep -aE "^test |test result|panicked|SKIPPED" | head -40
echo "[stage] tiny dspark tests on hip $(now)"
r env INTARSIA_INKLING2_BACKEND=hip cargo test --release -p intarsia-models --test inkling2_dspark --test inkling2_dspark_cut -- --test-threads 1 2>&1 | grep -aE "^test |test result|panicked|dspark:|SKIPPED" | head -30
echo "[stage] hip topk $(now)"
r env INTARSIA_REQUIRE_GPU=1 cargo test --release -p intarsia-backend-hip --test topk -- --test-threads 1 --nocapture 2>&1 | grep -aE "^test |test result|panicked|G-B" | head -20
echo "[stage] kernels test hc_pre $(now)"
r $B kernels test --backend hip --op hc_pre 2>&1 | grep -aE "ADMIT|REJECT|FAIL|NO CASES|Error|panicked" | head -6
echo "[stage] verify probe rows 4 $(now)"
r env INTARSIA_VERIFY_DIR=$M INTARSIA_BATCH_TEST_BACKEND=hip INTARSIA_VERIFY_ROWS=4 INTARSIA_VERIFY_LENS=17,1024,4000 cargo test --release -p intarsia-models --test verify_rows_real -- --nocapture 2>&1 | grep -aE "PROBE|DIFF|checked|panicked|test result|plan notes" | cut -c1-400 | head -30
echo "[stage] 1k spec off vs dspark k3 $(now)"
G="generate -m $M --backend hip --max-seq 8192 --profile-every 0 --show-ids --max-tokens 128 --prompt-ids /work/.scratch/prompts/len1k.json"
r $B $G --spec off > $S/w1-off.out 2> $S/w1-off.err; echo "off rc $?"
r $B $G --spec dflash --spec-draft $M --spec-k 3 > $S/w1-on.out 2> $S/w1-on.err; echo "on rc $?"
for a in off on; do grep -aE "decode\[|spec:|prefill overall|dspark:|device top-k|no row-exact|graph|Error|panicked" $S/w1-$a.err | cut -c1-300 | head -12; done
cmp <(grep -a '^ids\[' $S/w1-off.out $S/w1-off.err 2>/dev/null | sed 's/^[^:]*://' ) <(grep -a '^ids\[' $S/w1-on.out $S/w1-on.err 2>/dev/null | sed 's/^[^:]*://') && echo "IDS 1k: identical" || echo "IDS 1k: DIFFER"
echo "[stage] done $(now)"
