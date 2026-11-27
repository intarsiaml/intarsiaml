#!/usr/bin/env bash
# Lane ink-hip-specflip, (a) diagnostic window WA4 (only if the coordinator picks it after WA2's free-memory fail), ON tatooine inside
# window.sh, card 1 only: one process, 60 requests alternating greedy / seeded sampled (seeds 0..30 x T 0.0, 1.0; every request a draft
# switch on def), free memory after every request; then the same 60 requests with no speculation (INTARSIA_SPEC_DEFAULT=0). Prompt x1.
# Reads: the per-request free memory after the first few requests: flat = one-time first-use cost; falling per switch = creep.
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-specflip
RAW=$REPO/docs/results/2026-11-26-inkling2-hip-specflip/raw
BR=${BR:?set BR}; M=/models/Inkling-2-Mini
OUT=$REPO/logs/a4-$BR; mkdir -p "$OUT" "$REPO/.scratch/home"
rc() { local c=$1; shift; env CARD=$c CNAME=ink-hip-specflip-run$c bash "$RAW/run.sh" env HOME=/work/.scratch/home "$@"; }
echo "[a4] start $(date +%T) BR $BR $(md5sum "$REPO/.scratch/bin/intarsia-$BR" | cut -c1-32) $(cat "$REPO/.scratch/bin/intarsia-$BR.commit" 2>/dev/null)"
for arm in def off; do
  e="INTARSIA_SS_DEFAULT=1"; [ $arm = off ] && e="$e INTARSIA_SPEC_DEFAULT=0"
  rc 1 env $e INTARSIA_TEST_REAL_MODEL=1 INTARSIA_INK_MODEL=$M INTARSIA_INKLING2_BACKEND=hip INTARSIA_SS_SEEDS=0..30 INTARSIA_SS_TEMPS=0.0,1.0 \
    INTARSIA_SS_TOKENS=24 INTARSIA_SS_OUT=/work/logs/a4-$BR/mixed60-$arm.txt /work/.scratch/bin/t-$BR-inkling2_sampled_real --nocapture > "$OUT/mixed60-$arm.log" 2>&1
  echo "[a4] $arm rc $? $(date +%T) | $(grep -aE 'requests in|^free_mem|DFlash2 drafter .*loaded beside|panicked' "$OUT/mixed60-$arm.log" | tr '\n' ' ' | cut -c1-400)"
  grep -aE '^request [0-9]+ ' "$OUT/mixed60-$arm.log" | awk -v a=$arm '{n=$2; f=$(NF-1); if (n%6==0 || n>=57) printf "[a4]   %s request %s free %s MiB\n", a, n, f}'
done
cmp -s "$OUT/mixed60-def.txt" "$OUT/mixed60-off.txt" && echo "[a4] ids def = off (60)" || echo "[a4] ids DIFFER"
python3 - "$OUT" <<'PY'
import re, sys
from pathlib import Path
d = Path(sys.argv[1])
free = {}
for arm in ("def", "off"):
    v = {}
    for l in (d / f"mixed60-{arm}.log").read_text(errors="replace").splitlines():
        m = re.match(r"^request (\d+) .*free (\d+) MiB", l)
        if m:
            v[int(m.group(1)) + 1] = int(m.group(2))  # requests 1-60
    free[arm] = v
ok = all(len(free[a]) == 60 for a in free)
if not ok:
    print(f"[a4] VERDICT VOID: requests logged def {len(free['def'])} off {len(free['off'])} (need 60 each)")
    sys.exit()
dr = {a: free[a][10] - free[a][60] for a in free}
worst = max(free["def"][10] - free["def"][n] for n in range(11, 61))
print(f"[a4] free after request 10 / 60: def {free['def'][10]} / {free['def'][60]} MiB (drift {dr['def']}), off {free['off'][10]} / {free['off'][60]} (drift {dr['off']})")
print(f"[a4] def: largest drop below request 10 over requests 11-60: {worst} MiB")
p = dr["def"] <= dr["off"] + 8 and worst <= 16
print(f"[a4] VERDICT {'PASS' if p else 'FAIL'} (drift def {dr['def']} <= off {dr['off']} + 8: {dr['def'] <= dr['off'] + 8}; largest drop {worst} <= 16: {worst <= 16})")
PY
echo "[a4] done $(date +%T)"
