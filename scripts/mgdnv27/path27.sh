#!/bin/zsh
# mgdnv27: proves the fused rows run on the 27B (and in batched 35B serving): counter-patched debug binary (scripts/mgdnv/counter-debug.patch.txt, never committed) built in target-mgdnv27-dbg.
source ${0:A:h}/lib.sh
B=$W/target-mgdnv27-dbg/release/intarsia
for knob in 1 0; do
  echo "== 27B generate default recipe knob=$knob"
  INTARSIA_METAL_GDN_VERIFY_FUSE=$knob $B generate -m $M27 --backend metal --max-seq 4096 --chat --no-think --temperature 0 --profile-every 0 -p "$(cat $CELLS_DIR/code-gen.txt)" --max-tokens 300 < /dev/null 2>&1 >/dev/null | grep -a "mgdnvdbg" | tail -1 | cut -c1-300
done
for cfg in 4:4 16:8; do
mb=${cfg%%:*}; n=${cfg#*:}
echo "== 35B serve mb$mb batched MTP N=$n knob=1, $n concurrent"
INTARSIA_BATCHED_MTP=$n INTARSIA_BATCHED_VERIFY=1 INTARSIA_METAL_GDN_VERIFY_FUSE=1 $B serve --backend metal --model $M35 --served-model-name m --max-batch $mb --max-seq 8192 --spec mtp --spec-k 2 --spec-off-at 9 --profile-every 0 --no-supervisor --no-ui --host 127.0.0.1 --port $PORT > $RAW/path-serve-mb$mb.out 2> $RAW/path-serve-mb$mb.err < /dev/null &
TP=$!; for i in {1..300}; do curl -sf -m 2 $URL/health >/dev/null 2>&1 && break; sleep 1; done
python3 $S/conc.py $PORT $CELLS_DIR $n p 200 | tail -1
kill -INT $TP; for i in {1..60}; do kill -0 $TP 2>/dev/null || break; sleep 1; done; kill -9 $TP 2>/dev/null
grep -a "mgdnvdbg" $RAW/path-serve-mb$mb.err | tail -1 | cut -c1-300
done
