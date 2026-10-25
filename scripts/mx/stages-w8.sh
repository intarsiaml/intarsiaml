#!/usr/bin/env bash
# Lane mx window 8 (card 0): the 16-row peel. w7 showed peel 8 bitwise for 1..8-row tails but not for 9..16-row tails
# (split-KV attention at <= 16 rows on the 256 schedule, flash above); this window repeats w7's checks with
# INTARSIA_PREFILL_PEEL=16 on the PR-head build (e5f849b; its qwen3_5_vl.toml already resolves to 4096 + 16): tails dumps (vs w7's main chunk-256 dump), the 27B golden gate,
# 256 greedy ids through the engine at lengths whose last multi-row pass is 10..16 rows inside a 4096 pass (525, 529, 8459)
# or after it (4109), prefill speed.
set -uo pipefail
O=$1; S="$(dirname "${BASH_SOURCE[0]}")"; source "$S/lib.sh"
for t in base dev; do [[ $(T $t)/mx-intarsia -nt $(T $t)/Cargo.toml ]] || { log "NO FRESH BINARY in $(T $t); abort"; exit 3; }; done
log "w8 start; budget $(left) s; base $(cat $(T base)/.stamp) dev $(cat $(T dev)/.stamp)"
# head default with no flags or env: plan main vs dev (only prefill_chunk / _source / prefill_peel may differ; scratch and
# kv cap at the window's max-seq are printed), 256 ids short / 4k dev vs main, 35B acc-short with the golden dir
for t in base dev; do  # at serve's --max-seq 17408 (loc-window.sh), so a kv cap cut by the 4096-row workspace shows
  need 60 "plan" && run $t 200 "plan-$t-27" "$(mdir "$M27")" plan show --backend hip --model "$M27" --mem-budget 29 --max-seq 17408 --json
done
log "plan 27B main vs head default: $(python3 - "$O/plan-base-27.out" "$O/plan-dev-27.out" <<'PY'
import json, sys
def flat(d, p=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items(): out.update(flat(v, f"{p}.{k}" if p else k))
    elif isinstance(d, list):
        for i, v in enumerate(d): out.update(flat(v, f"{p}[{i}]"))
    else: out[p] = d
    return out
try:
    a, b = (flat(json.load(open(f))) for f in sys.argv[1:3])
except Exception as e:
    print(f"cannot parse: {e}"); sys.exit()
diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
print("; ".join(f"{k}: {a.get(k)} -> {b.get(k)}" for k in diff)[:900] or "identical")
PY
)"
for k in short 4k; do
  need 80 "ids" && ids dev hd-$k "$M27" $k
  need 80 "ids" && ids base hb-$k "$M27" $k
  same ids-hd-$k.ids ids-hb-$k.ids "27B $k ids head default (no flags) vs main"
done
need 300 "qual" && qual dev 35g "$M35" --golden-dir /work/golden
need 200 "acc" && GT=g27-tails2 E="INTARSIA_PREFILL_PEEL=16" acc dev dev-peel16 4096
need 100 "accg" && E="INTARSIA_PREFILL_PEEL=16" accg dev dev16 --prefill-chunk 4096
for k in 525 529 8459 4109; do
  need 80 "ids" && E="INTARSIA_PREFILL_PEEL=16" ids dev dev16-$k "$M27" $k --prefill-chunk 4096
  need 80 "ids" && ids base base-$k "$M27" $k
  same ids-dev16-$k.ids ids-base-$k.ids "27B $k ids dev (4096 + peel 16) vs main (256)"
done
for n in 1024 4096 16384; do
  need 60 "pf" && E="INTARSIA_PREFILL_PEEL=16" pf dev peel16-$n "$M27" $n --prefill-chunk 4096
done
# LDS / VALU counters on the v2 GEMM at M = 4096 (w7 passed the counters without the M argument)
need 120 "pmc" && pmcbench dev lds 4096 SQ_LDS_BANK_CONFLICT SQ_LDS_IDX_ACTIVE
need 120 "pmc" && pmcbench dev valu 4096 SQ_INSTS_VALU SQ_INSTS_LDS SQ_WAIT_INST_LDS
log "w8 done; $(left) s left"
