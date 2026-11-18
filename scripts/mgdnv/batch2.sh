#!/bin/zsh
# mgdnv: served check of the BATCHED verify branch (segs > 1 in gdn_block.rs): `serve --max-batch 2 --spec mtp --spec-off-at 9` with INTARSIA_BATCHED_MTP=2 INTARSIA_BATCHED_VERIFY=1 (both off by default on Metal), two concurrent cells, knob off then on;
# the reply shas must be equal. BIN defaults to the new binary; with the counter-patched debug binary (BIN=...) the stderr shows the fused calls.
#   fleetq run --host endor --quiet gpu,cpu --mem 40 --expect 10 --label "mgdnv: batch2" -- scripts/mgdnv/batch2.sh
source ${0:A:h}/lib.sh
BIN=${BIN:-$NEW_BIN}; O=$RAW/batch2; mkdir -p $O; C=$W/scripts/mcopy/cell.py
for knob in 0 1; do
  env INTARSIA_BATCHED_MTP=2 INTARSIA_BATCHED_VERIFY=1 INTARSIA_METAL_GDN_VERIFY_FUSE=$knob $BIN serve --backend metal --model $M --served-model-name m --max-batch 2 --max-seq 8192 --spec mtp --spec-k 2 --spec-off-at 9 --profile-every 0 --no-supervisor --no-ui \
    --host 127.0.0.1 --port $PORT > $O/serve-k$knob.out 2> $O/serve-k$knob.err < /dev/null &
  TP=$!; ok=0
  for i in {1..300}; do curl -sf -m 2 $URL/health >/dev/null 2>&1 && { ok=1; break; }; kill -0 $TP 2>/dev/null || break; sleep 1; done
  [ $ok = 1 ] || { echo "SERVER FAILED k$knob: $(tail -3 $O/serve-k$knob.err)"; kill $TP 2>/dev/null; exit 1; }
  python3 $C $PORT $CELLS_DIR/chat-short.json 1 warm > /dev/null 2>&1
  for rep in 1 2; do
    python3 $C $PORT $CELLS_DIR/code-gen.json 2 bA$rep >> $O/k$knob-A.jsonl 2>&1 &
    p1=$!
    python3 $C $PORT $CELLS_DIR/reason.json 2 bB$rep >> $O/k$knob-B.jsonl 2>&1 &
    p2=$!
    wait $p1 $p2
  done
  curl -s $URL/metrics | grep -E '^intarsia_(spec_|batch)' | sed 's/{[^}]*}//' | tr '\n' ' ' > $O/metrics-k$knob.txt
  kill -INT $TP; for i in {1..60}; do kill -0 $TP 2>/dev/null || break; sleep 1; done; kill -9 $TP 2>/dev/null; wait $TP 2>/dev/null
  echo "knob=$knob: $(cat $O/k$knob-A.jsonl $O/k$knob-B.jsonl | python3 -c 'import sys,json; print(sorted((json.loads(l)["cell"],json.loads(l)["tag"],json.loads(l)["rep"],json.loads(l)["sha"],json.loads(l)["dec_tps"]) for l in sys.stdin if l.startswith("{")))')"
  grep -a mgdnvdbg $O/serve-k$knob.err | tail -2
done
python3 - $O <<'PY'
import sys, json, glob
o = sys.argv[1]
def shas(k):
    r = {}
    for f in glob.glob(f"{o}/k{k}-*.jsonl"):
        for l in open(f):
            if l.startswith("{"):
                d = json.loads(l); r[(d["cell"], d["tag"], d["rep"])] = d["sha"]
    return r
a, b = shas(0), shas(1)
print("BATCH2", "PASS" if a == b and a else "FAIL", len(a), "replies compared")
PY
