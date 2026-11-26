#!/usr/bin/env bash
# Window 2 (lane ink-hip-spec), both cards at once; full logs in logs/w2/. Binaries: base = main 87c9745c0 + #526 (da8501b51), spec2 =
# the branch (with #526 merged). Nothing set in the environment unless named.
#   card 0: the verify probe with the softcapped comparison (rows 2..4 at 17 / 1024 / 4000, verify graph default = on), its cost table
#           again with INTARSIA_VERIFY_GRAPH=0 (1024 only), the DSpark phase profile at 1k (INTARSIA_DSPARK_PROFILE=1, gate rule).
#   card 1: identity gate. (a) greedy ids, the 12 greedy golden prompts + the 2 prose texts x 128 tokens (--show-ids): base --spec off,
#           branch --spec off, branch --spec dflash k3 gate 0.8 (rule default off Metal), branch k3 cut 0.5 (TAU=0 CUT=0.5);
#           (b) short-0 / 4k-0 text prompts, 256 tokens --ignore-eos --show-ids: base off, branch off, gate, cut (md5 of ids[:64] and
#           ids[:256]). Card 0 then runs G2 short / long at spec off on the branch.
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-spec
S=$REPO/.scratch
O=$REPO/logs/w2
rm -rf "$O"; mkdir -p "$O"
M=/models/Inkling-2-Mini
BB=/work/.scratch/bin/intarsia-base
B=/work/.scratch/bin/intarsia-${BR:-spec2}
D="--spec dflash --spec-draft $M --spec-k 3"
CUT="INTARSIA_DSPARK_TAU=0 INTARSIA_DSPARK_CUT=0.5"
now() { date '+%T'; }
r0() { HIP_VISIBLE_DEVICES=0 bash "$S/run.sh" "$@"; }
r1() { HIP_VISIBLE_DEVICES=1 INK_CONTAINER=ink-hip-spec-run2 bash "$S/run.sh" "$@"; }
card0() {
  echo "[c0] verify probe rows 4, softcapped compare $(now)"
  r0 env INTARSIA_VERIFY_DIR=$M INTARSIA_BATCH_TEST_BACKEND=hip INTARSIA_VERIFY_ROWS=4 INTARSIA_VERIFY_LENS=17,1024,4000 cargo test --release -p intarsia-models --test verify_rows_real -- --nocapture > $O/probe.log 2>&1
  echo "probe rc $?"; grep -aE "PROBE_TIME|PROBE_RESULT|PROBE_GRAPHS|DIFF|panicked|test result" $O/probe.log | cut -c1-400 | head -24
  echo "[c0] verify probe, eager (INTARSIA_VERIFY_GRAPH=0), len 1024 $(now)"
  r0 env INTARSIA_VERIFY_GRAPH=0 INTARSIA_VERIFY_DIR=$M INTARSIA_BATCH_TEST_BACKEND=hip INTARSIA_VERIFY_ROWS=4 INTARSIA_VERIFY_LENS=1024 cargo test --release -p intarsia-models --test verify_rows_real -- --nocapture > $O/probe-eager.log 2>&1
  echo "probe eager rc $?"; grep -aE "PROBE_TIME|PROBE_RESULT|PROBE_GRAPHS|panicked" $O/probe-eager.log | cut -c1-400 | head -6
  echo "[c0] dspark profile 1k (gate) $(now)"
  r0 env INTARSIA_DSPARK_PROFILE=1 $B generate -m $M --backend hip --max-seq 8192 --profile-every 0 --max-tokens 128 --prompt-ids /work/.scratch/prompts/len1k.json $D > $O/prof.out 2> $O/prof.err
  echo "prof rc $?"
  grep -a 'dspark: ms' $O/prof.err | awk '{for(i=1;i<=NF;i++) if($i ~ /^(flush|block|head|step0|cycle)$/) {s[$i]+=$(i+1); n[$i]++}} END {for(k in s) printf "%s %.2f ms (mean of %d)\n", k, s[k]/n[k], n[k]}'
  grep -a 'dspark: ingest' $O/prof.err | awk '{s+=$(NF); n++} END {if (n) printf "ingest %.2f ms (mean of %d)\n", s/n, n}'
  grep -aE "^decode\[|^spec:" $O/prof.err | head -4
  for g in short long; do
    echo "G2 $g branch spec off [$(now)] $(r0 $B accuracy -m $M --backend hip --max-seq 4096 --mode prefill --golden golden/oracle/inkling2-mini-oracle-w4a16-$g-top20.safetensors 2>&1 | grep -aE "prefill floor|floor verdict|Error|panicked" | cut -c1-400)"
  done
  echo "[c0] done $(now)"
}
ids() {  # arm binary "env" extra file
  local f=$O/ids-$1-$(basename $5 .json)
  r1 env $3 $2 generate -m $M --backend hip --max-seq 8192 --profile-every 0 --show-ids --max-tokens 128 --prompt-ids /work/.scratch/prompts/$5 $4 > $f.out 2> $f.err
  echo "IDS $1 $5 rc $? md5 $(grep -a '^ids\[' $f.out $f.err | sed 's/^[^:]*://' | md5sum | cut -c1-12) | $(grep -aE '^decode overall' $f.err) | $(grep -aE '^spec: verify' $f.err) [$(now)]"
}
md5() {  # arm binary "env" extra prompt-name
  local f=$O/md5-$1-$5
  r1 env $3 $2 generate -m $M --backend hip --max-seq 8192 --profile-every 0 --show-ids --ignore-eos --max-tokens 256 $4 -p "$(cat $REPO/scripts/cmx/ids-prompts/$5.txt)" > $f.out 2> $f.err
  local rc=$?
  python3 - "$f.out" "$f.err" "$1" "$5" "$rc" <<'PY'
import hashlib, re, sys
ids = None
for p in sys.argv[1:3]:
    for line in open(p, errors="replace"):
        m = re.match(r"^ids(\[[^\]]*\])?: \[(.*)\]", line)
        if m:
            ids = [int(x) for x in m.group(2).split(",") if x.strip()]
h = lambda v: hashlib.md5(",".join(map(str, v)).encode()).hexdigest()[:12]
print(f"MD5 {sys.argv[3]} {sys.argv[4]} rc {sys.argv[5]} n {len(ids) if ids else 0} ids64 {h(ids[:64]) if ids else '-'} ids256 {h(ids[:256]) if ids else '-'}")
PY
  grep -aE '^decode overall|^spec: verify' $f.err | tr '\n' ' '; echo
}
card1() {
  for p in greedy.json prose.json; do
    ids base $BB "INK=base" "--spec off" $p
    ids off $B "INK=off" "--spec off" $p
    ids gate $B "INK=gate" "$D" $p
    ids cut $B "$CUT" "$D" $p
  done
  for p in short-0 4k-0; do
    md5 base $BB "INK=base" "--spec off" $p
    md5 off $B "INK=off" "--spec off" $p
    md5 gate $B "INK=gate" "$D" $p
    md5 cut $B "$CUT" "$D" $p
  done
  echo "[c1] done $(now)"
}
echo "[stage] start $(now)"
card0 > "$O/c0.log" 2>&1 &
card1 > "$O/c1.log" 2>&1 &
wait
echo "---- card 0"; cat "$O/c0.log"
echo "---- card 1"; cat "$O/c1.log"
echo "[stage] done $(now)"
