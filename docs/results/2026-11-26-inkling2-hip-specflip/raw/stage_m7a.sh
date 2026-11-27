#!/usr/bin/env bash
# Lane ink-hip-specflip, main7 check (coordinator 02:08), window M1 = bitwise + serve, ON tatooine inside window.sh. Nothing timed.
#   card 0: `serve` main7 solo (no spec flags: the default), started first and alone (cold admission, up to 240 s); its 16 replies one at a
#           time (serve_bitwise.py solo, SB_FIT=1) once it answers.
#   card 1: greedy md5.json 256 tokens --ignore-eos: --spec off and the default; the default at 64 tokens (vs the first 64 ids of off);
#           100 seeds at T 1.0 (24 tokens; the seed harness of main7, --spec dflash with the checkpoint's DSpark draft, k5 = GA2's path):
#           INTARSIA_SPEC_SAMPLED_DFLASH=0 (off) vs unset (the recipe's sampled default). Then, once solo answers: `serve --max-batch 16`
#           main7 (cold admission), its 16 replies at once; b16 vs solo (serve_bitwise.py compare).
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-specflip
S=$REPO/.scratch
RAW=$REPO/docs/results/2026-11-26-inkling2-hip-specflip/raw
BR=${BR:-main7}; M=/models/Inkling-2-Mini
OUT=$REPO/logs/m7a-$BR; mkdir -p "$OUT" "$S/prompts" "$S/home"
cp "$RAW/md5.json" "$S/prompts/"
[ -x "$S/bin/intarsia-$BR" ] && [ -x "$S/bin/t-$BR-inkling2_sampled_real" ] || { echo "[m7a] guard: binaries missing"; exit 3; }
T0=$(date +%s); el() { echo $(( $(date +%s) - T0 )); }
rc() { local c=$1; shift; env CARD=$c CNAME=ink-hip-specflip-run$c bash "$RAW/run.sh" env HOME=/work/.scratch/home "$@"; }
B=/work/.scratch/bin/intarsia-$BR
echo "[m7a] start $(date +%T) $BR $(md5sum "$S/bin/intarsia-$BR" | cut -c1-32) $(cat "$S/bin/intarsia-$BR.commit" 2>/dev/null | head -1)"
start() {  # card name serve-args
  local c=$1 n=$2; shift 2
  docker rm -f ink-hip-specflip-run$c > /dev/null 2>&1
  ( rc $c bash -c "exec $B serve -m $M --backend hip --host 0.0.0.0 --port 8132 --model-id ink $*" > "$OUT/$n.log" 2>&1 ) &
}
up() {  # card name wait-s -> ip
  local c=$1 n=$2 w=$3 ip="" i
  for i in $(seq 1 "$w"); do
    ip=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' ink-hip-specflip-run$c 2>/dev/null)
    [ -n "$ip" ] && curl -sf -m 2 "http://$ip:8132/health" > /dev/null 2>&1 && { echo "[m7a c$c] $n up after ${i}s (t+$(el)s)" >&2; echo "$ip"; return 0; }
    sleep 1
  done
  echo "[m7a c$c] $n: no server after $w s (t+$(el)s)" >&2; return 1
}
client() {  # card name mode ip
  SB_FIT=1 timeout 90 python3 "$RAW/serve_bitwise.py" "http://$4:8132/v1" $3 "$OUT/$2.json" > "$OUT/$2.client" 2>&1
  echo "[m7a c$1] $2 $3 rc $? t+$(el)s | $(tail -n 1 "$OUT/$2.client")"
  docker rm -f ink-hip-specflip-run$1 > /dev/null 2>&1
  echo "[m7a c$1] $2 lines: $(grep -aE '^admission|^loaded|DSpark default|DFlash-family drafting|draft length per request|continuous batching|^--spec' "$OUT/$2.log" | tr '\n' ' ' | cut -c1-700)"
}
md5() {  # name tokens extra
  local f=$OUT/md5-$1
  rc 1 $B generate -m $M --backend hip --max-seq 8192 --profile-every 0 --show-ids --ignore-eos --max-tokens $2 --prompt-ids /work/.scratch/prompts/md5.json $3 > "$f.out" 2> "$f.err"
  echo "[m7a c1] md5 $1 rc $? t+$(el)s $(cat "$f.out" "$f.err" | grep -a '^ids\[' | md5sum | cut -c1-32) | $(grep -aE '^decode overall|^spec: verify|DSpark default' "$f.err" | tr '\n' ' ' | cut -c1-300)"
}
seeds() {  # arm off | def
  local f=$OUT/seeds-$1 e=""
  rm -f "$f.txt"
  [ $1 = off ] && e="INTARSIA_SPEC_SAMPLED_DFLASH=0"
  rc 1 env $e INTARSIA_TEST_REAL_MODEL=1 INTARSIA_INK_MODEL=$M INTARSIA_INKLING2_BACKEND=hip INTARSIA_SS_SEEDS=0..100 INTARSIA_SS_TEMPS=1.0 \
    INTARSIA_SS_TOKENS=24 INTARSIA_SS_K=5 INTARSIA_SS_BUDGET_S=75 INTARSIA_SS_OUT=/work/logs/m7a-$BR/seeds-$1.txt \
    /work/.scratch/bin/t-$BR-inkling2_sampled_real --nocapture > "$f.log" 2>&1
  echo "[m7a c1] seeds $1 rc $? t+$(el)s lines $(wc -l < "$f.txt" 2>/dev/null) | $(grep -aE 'DFlash-family drafting|draft length per request|requests in' "$f.log" | tr '\n' ' ' | cut -c1-300)"
}
start 0 solo
( md5 off 256 "--spec off"; md5 def 256 ""; md5 def64 64 ""; seeds off; seeds def ) > "$OUT/c1.log" 2>&1 &
C1=$!
if ip0=$(up 0 solo 240); then
  client 0 solo solo "$ip0" > "$OUT/solo-client.log" 2>&1 &
  SC=$!
  wait $C1
  start 1 b16 --max-batch 16
  left=$(( 278 - $(el) ))
  if ip1=$(up 1 b16 $left); then client 1 b16 conc "$ip1"; else docker rm -f ink-hip-specflip-run1 > /dev/null 2>&1; echo "[m7a c1] b16 lines: $(grep -aE '^admission|^loaded' "$OUT/b16.log" | tr '\n' ' ')"; fi
  wait $SC; cat "$OUT/solo-client.log"
else
  docker rm -f ink-hip-specflip-run0 > /dev/null 2>&1
  wait $C1
fi
cat "$OUT/c1.log"
python3 "$RAW/serve_bitwise.py" compare "$OUT/solo.json" "$OUT/b16.json" > "$OUT/serve-compare.txt" 2>&1
echo "[m7a] serve b16 (card 1) vs solo (card 0): $(tail -n 1 "$OUT/serve-compare.txt")"
python3 - "$OUT" <<'EOF'
import re, sys
from pathlib import Path
d = Path(sys.argv[1])
def ids(n):
    out = {}
    for f in (d / f"{n}.out", d / f"{n}.err"):
        if f.exists():
            for l in f.read_text(errors="replace").splitlines():
                m = re.match(r"^ids\[([^\]]*)\]: (.*)$", l)
                if m:
                    out[m.group(1)] = m.group(2).strip().strip("[]").replace(",", " ").split()  # generate prints a Debug list
    return out
off, de, d64 = ids("md5-off"), ids("md5-def"), ids("md5-def64")
print(f"[m7a] greedy 256: default = off: {'EQUAL' if off and de == off else 'DIFFER'} ({len(off)} prompts)")
ok64 = bool(d64) and all(len(v) == 64 and off.get(k, [])[:64] == v for k, v in d64.items())
print(f"[m7a] greedy 64: default's ids = the first 64 of off's: {'EQUAL' if ok64 else 'DIFFER'}")
a = sorted((d / "seeds-off.txt").read_text().splitlines()) if (d / "seeds-off.txt").exists() else []
b = sorted((d / "seeds-def.txt").read_text().splitlines()) if (d / "seeds-def.txt").exists() else []
print(f"[m7a] seeds T 1.0: off {len(a)} default {len(b)} identical {len(set(a) & set(b))}")
EOF
echo "[m7a] done $(date +%T) t+$(el)s"
