#!/usr/bin/env bash
# Lane mx fold-default, window G2 (card 0): the one-time re-record of the 27B W4A4 baseline + noise envelope from the fold
# build (approved by the user 2026-10-25 13:14, only if every G1 gate passed), its self-check, then 27B acc-short against it.
# Runs only if ~/mx/G1_OK holds this tree's stamp: the Mac writes it after reading G1 (hard stops: widened fold_rows,
# S4 batched == solo, tails sweep chunkcmp 0 rows) and nothing else writes it. The commands are #150's (scripts/ac/stages.sh).
set -uo pipefail
O=$1; S="$(dirname "${BASH_SOURCE[0]}")"; source "$S/lib.sh"
[[ $(T dev)/mx-intarsia -nt $(T dev)/Cargo.toml ]] || { log "NO FRESH BINARY in $(T dev); abort"; exit 3; }
stamp=$(cat "$(T dev)/.stamp")
if [[ "$(cat "$HOME/mx/G1_OK" 2>/dev/null)" != "$stamp" ]]; then log "G1_OK missing or not for $stamp: NO re-record, abort"; exit 4; fi
log "fd2 start; budget $(left) s; dev $stamp (G1_OK)"
# G1 completion: G1's golden-gate runs were refused (the binaries lacked a valid INTARSIA_BUILD_STAMP: fd-sync wrote no
# head=), so they run here first, each a condition of the re-record.
G16=/work/p1-golden/qwen3.8-27b-mxfp4-hfquark-w4a16-top20.safetensors
B16=/work/golden/baselines/qwen3.8-27b-mxfp4-hfquark-w4a16-top20.hip.baseline.json
E16=/work/golden/baselines/qwen3.8-27b-mxfp4-hfquark-w4a16-top20.hip.envelope.json
OB4=/work/golden/baselines/qwen3.8-27b-mxfp4-hfquark-top20.hip.baseline.json
OE4=/work/golden/baselines/qwen3.8-27b-mxfp4-hfquark-top20.hip.envelope.json
PROV="--pre-change bbc0cde --revlist /work/mx-out/revlist"
A16="--golden $G16 --mode both --decode-positions 16 --max-seq 4096 --mxfp4-act none"
ok=1
run base 300 a16-dump-base "$HUB" accuracy --backend hip --model $M27 --mem-budget 29 $A16 --write-self-golden /work/mx-out/a16-base.safetensors || ok=0
run dev 300 a16-gate "$HUB" accuracy --backend hip --model $M27 --mem-budget 29 $A16 --baseline $B16 --noise-envelope $E16 $PROV --write-self-golden /work/mx-out/a16-dev.safetensors || ok=0
cp "$(T base)/mx-out/a16-base.safetensors" "$(T dev)/mx-out/a16-dev.safetensors" "$O/" 2>/dev/null
python3 -I "$S/dumpeq.py" "$O/a16-base.safetensors" "$O/a16-dev.safetensors" > "$O/a16-dumpeq.txt" 2>&1 || ok=0
log "W4A16 dumps main vs branch: $(tail -1 "$O/a16-dumpeq.txt")"
run dev 300 a4-gate-old-baseline "$HUB" accuracy --backend hip --model $M27 --mem-budget 29 --golden $G27 --mode both --decode-positions 16 --max-seq 4096 --ref-noise $N27 --baseline $OB4 --noise-envelope $OE4 $PROV --write-self-golden /work/mx-out/a4-dev.safetensors
grep -aqE '^prefill floor .*=> PASS' "$O/a4-gate-old-baseline.out" || { log "W4A4 prefill floor not PASS"; ok=0; }
cp "$(T dev)/mx-out/a4-dev.safetensors" "$O/" 2>/dev/null
if ((ok == 0)); then log "a G1 golden step failed: NO re-record, abort"; exit 6; fi
log "G1 golden steps: W4A16 gate PASS + dumps bitwise main, W4A4 prefill PASS"
# HARD STOP first: the fold_rows test with the wide-span (rounding) regime added after G1 (its synthetic data summed exactly)
need 200 "hip tests" && hiptest dev fold2 --test mxfp4_fold -- --test-threads=1 --nocapture
if ! grep -aq "^test result: ok. 4 passed; 0 failed" "$O/test-fold2.txt"; then log "mxfp4_fold FAILED or did not run: NO re-record, abort"; exit 5; fi
PC=$(sed -E "s/^head=([0-9a-f]+).*/\1/" <<< "$stamp")
B4=/work/mx-out/qwen3.8-27b-mxfp4-hfquark-top20.hip.baseline.json
E4=/work/mx-out/qwen3.8-27b-mxfp4-hfquark-top20.hip.envelope.json
A4="--mxfp4-act dynamic --golden $G27 --mode both --decode-positions 16 --max-seq 4096"
acc4() { local to=$1 tag=$2; shift 2; run dev "$to" "$tag" "$HUB" accuracy --backend hip --model $M27 --mem-budget 29 "$@"; }
need 300 "record" && acc4 300 a4-record $A4 --ref-noise $N27 --write-baseline $B4 --write-self-golden /work/mx-out/a4-rec.safetensors
need 300 "envelope" && acc4 300 a4-envelope $A4 --noise-ref $B4 --noise-perturb chunk --write-noise-envelope $E4 --allow-same-code
need 300 "gate" && acc4 300 a4-gate-new $A4 --ref-noise $N27 --baseline $B4 --noise-envelope $E4 --pre-change "$PC" --allow-same-code
for f in baseline envelope; do cp "$(T dev)/mx-out/qwen3.8-27b-mxfp4-hfquark-top20.hip.$f.json" "$O/" 2>/dev/null; done
# acc-short reads golden/baselines of the tree: put the new pair there (this tree only; the repo gets them from $O)
if [[ -s $O/qwen3.8-27b-mxfp4-hfquark-top20.hip.baseline.json && -s $O/qwen3.8-27b-mxfp4-hfquark-top20.hip.envelope.json ]]; then
  cp "$O"/qwen3.8-27b-mxfp4-hfquark-top20.hip.{baseline,envelope}.json "$(T dev)/golden/baselines/"
  need 600 "qual" && qual dev 27 "$M27" --golden $G27
else log "no new baseline pair: acc-short not run"; fi
cp "$(T dev)/mx-out/a4-rec.safetensors" "$O/" 2>/dev/null
log "fd2 done; $(left) s left"
