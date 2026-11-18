#!/bin/zsh
# mrows16k profile: `generate` (copy3k prompt, greedy, 35B, Metal) with a GPU-timestamp trace, arms = copy length (len 7 = 8-row verify passes, len 15 = 16-row).
#   fleetq run --host endor --quiet gpu,cpu --mem 40 --expect 20 --label "mrows16k: profile" -- env MODE=kernel scripts/mrows16k/prof.sh
# MODE: kernel (INTARSIA_METAL_PROFILE=kernel:FILE, every launch its own command buffer) or ce1 (cb:FILE + INTARSIA_METAL_COMMIT_EVERY=1) or cb.
source ${0:A:h}/lib.sh
MODE=${MODE:-kernel}; CELL=${CELL:-copy3k}; NT=${NT:-700}; O=$RAW/prof; mkdir -p $O; BIN=${BIN:-$NEW_BIN}
for len in ${=LENS:-7 15}; do
  f=$O/$MODE-$CELL-L$len; E=(INTARSIA_SPEC_COPY=20,$len INTARSIA_SPEC_TRACE=1)
  case $MODE in kernel) E+=(INTARSIA_METAL_PROFILE=kernel:$f.jsonl) ;; ce1) E+=(INTARSIA_METAL_PROFILE=cb:$f.jsonl INTARSIA_METAL_COMMIT_EVERY=1) ;; cb) E+=(INTARSIA_METAL_PROFILE=cb:$f.jsonl) ;; none) ;; esac
  echo "-- $MODE $CELL len $len $(cool)"
  env $E $BIN generate -m $M --backend metal --max-seq 8192 --spec mtp --spec-k 2 --chat --no-think --temperature 0 --profile-every 0 --show-ids \
    -p "$(cat $CELLS_DIR/$CELL.txt)" --max-tokens $NT < /dev/null > $f.out 2> $f.err
  echo "exit $? thermal $(th)"; grep -a "finish=\|spec: " $f.err | grep -av "spec: copy token\|spec: token\|spec: mtp\|spec: cost" | cut -c1-220
  [ -f $f.jsonl ] && { wc -l $f.jsonl; gzip -f $f.jsonl; }
done
