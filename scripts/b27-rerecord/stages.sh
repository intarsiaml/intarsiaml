#!/usr/bin/env bash
# Lane b27-rerecord, the radiance-down window (STEPS selects; default s1 only; one GPU, tp=1; set O to share a log dir across windows):
#   s1   the gates with the CURRENT (stale) baselines: first INTARSIA_GDN_CHUNK=0 (expected PASS: old baselines + this setup match),
#        then the default (chunked GDN on): the 27B change-vs-base numbers
#   rec  record the baselines + envelopes from this build (chunk on, the recipe default chunk 2048): W4A4 through `qualify` (its own
#        plan; the stale W4A4 files are moved away first, acc-short records the baseline, the envelope stage the envelope), W4A16 with the
#        explicit accuracy command (`--mxfp4-act none`); everything lands in $O/new/ under the canonical file names
#   chk  gate with the new files: W4A4 `qualify` again (expect PASS), W4A16 the explicit gate; the absolute floors are in the same output
#   self the change-vs-base number of the GDN flip (method of scripts/gdn-chunk/accuracy-window.sh `self`): per variant, a base run with
#        INTARSIA_GDN_CHUNK=0 writes a self-golden, the default run is compared to it with --self-vs-ref-noise (W4A4: its hash-bound
#        sidecar; W4A16 has no sidecar, see the doc), plus a control (the default against itself)
# The accuracy flags are the ones recorded in the baselines' exec.cmdline (docs/results/2026-10-25-ac-27b-accuracy-gate.md section 7);
# Two flags the original runs did not need: W4A4 now passes --mxfp4-act dynamic (the Qwen3.8-27B recipe default became none, W4A16, on
# 2026-10-26), and the stale baselines were recorded at prefill chunk 256 (then the default; the recipe default is now 2048), so s1
# passes --prefill-chunk 256 to gate them like for like. rec / chk use no chunk flag, i.e. the recipe default, like qualify.
set -u
. "$(dirname "$0")/lib.sh"
cd "$TREE"
STEPS=${STEPS:-s1}
has() { [[ " $STEPS " == *" $1 "* ]]; }
W=/work/${O#"$TREE"/}
M27=/models/Qwen3.8-27B-Quark-AWQ-MXFP4
HEAD7=$(echo "$STAMP" | sed -n 's/.*head=\([0-9a-f]*\).*/\1/p')
BN=qwen3.8-27b-mxfp4-hfquark
COMMON="--backend hip --model $M27 --mem-budget 29 --mode both --decode-positions 16 --max-seq 4096"
# variant flags
V_w4a4="--mxfp4-act dynamic --golden /work/golden/$BN-top20.safetensors --ref-noise /work/golden/ref_noise/$BN-top20.safetensors.json"
V_w4a16="--mxfp4-act none --golden /work/golden/$BN-w4a16-top20.safetensors"
STEM_w4a4=$BN-top20.hip
STEM_w4a16=$BN-w4a16-top20.hip
mkdir -p "$O/new"
log "stages start; $STAMP; STEPS=$STEPS"
sha256sum golden/$BN-top20.safetensors golden/$BN-w4a16-top20.safetensors | tee -a "$O/steps.log"
summ() { grep -aE "floor verdict|baseline verdict|golden gate|^Error|mxfp4 activations|this build" "$O/$1.out" "$O/$1.err" | cut -c1-260 | tee -a "$O/steps.log"; }
if has s1; then
  for part in chunk0 chunkon; do
    e=""; [[ $part == chunk0 ]] && e="INTARSIA_GDN_CHUNK=0"
    for v in w4a16 w4a4; do
      sv="STEM_$v"; stem=${!sv}; vv="V_$v"
      pc=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['code']['head'])" "golden/baselines/$stem.baseline.json")
      # shellcheck disable=SC2086
      E="$e" run 0 900 "s1-$v-$part" $BIN accuracy $COMMON ${!vv} --baseline /work/golden/baselines/$stem.baseline.json \
        --noise-envelope /work/golden/baselines/$stem.envelope.json --pre-change "$pc" --allow-same-code --prefill-chunk 256
      summ "s1-$v-$part"
    done
  done
fi
if has s1 && { has rec || has chk; }; then
  # step 1a (chunk off, the stale baselines) must PASS before anything is re-recorded
  for v in w4a16 w4a4; do
    if ! grep -aq "baseline verdict.*: PASS" "$O/s1-$v-chunk0.out"; then log "STOP: 1a ($v, INTARSIA_GDN_CHUNK=0) did not PASS; not re-recording"; exit 3; fi
  done
fi
SK="--skip kernels --skip bench --skip acc-long --skip acc-4k --skip bench-long --skip ab --skip acc-format --skip acc-oracle"
QV="qualify --backend hip --golden /work/golden/$BN-top20.safetensors --model $M27 --mem-budget 29 $SK"   # the W4A4 golden; qualify plans chunk + act mode
qsum() { grep -aE "\] [a-z0-9-]+: (PASS|FAIL|SKIP|ERROR|WARN|MISSING-BASELINE)" "$O/$1.out" | cut -c1-300 | tee -a "$O/steps.log"; }
if has rec; then
  # W4A4: qualify. The stale files are moved out of the tree so qualify reports MISSING-BASELINE and records its own.
  mkdir -p "$O/old-baselines"; mv golden/baselines/$STEM_w4a4.baseline.json golden/baselines/$STEM_w4a4.envelope.json "$O/old-baselines/" 2>/dev/null
  run 0 1800 q-rec-w4a4 $BIN $QV --out $W/q-rec-w4a4
  qsum q-rec-w4a4
  cp "$O/q-rec-w4a4/baselines/acc-short.baseline.json" "$O/new/$STEM_w4a4.baseline.json"
  cp "$O/q-rec-w4a4/baselines/envelope.json" "$O/new/$STEM_w4a4.envelope.json"
  # W4A16: the explicit accuracy command
  v=w4a16; stem=$STEM_w4a16; vv="V_$v"
  # shellcheck disable=SC2086
  run 0 900 "rec-$v-baseline" $BIN accuracy $COMMON ${!vv} --write-baseline $W/new/$stem.baseline.json --write-self-golden $W/new/$v-self.safetensors
  summ "rec-$v-baseline"
  # shellcheck disable=SC2086
  run 0 900 "rec-$v-envelope" $BIN accuracy $COMMON ${!vv} --noise-ref $W/new/$stem.baseline.json --noise-perturb chunk --write-noise-envelope $W/new/$stem.envelope.json --allow-same-code
  summ "rec-$v-envelope"
  ls -la "$O/new" | tee -a "$O/steps.log"
fi
if has chk; then
  cp "$O/new/$STEM_w4a4.baseline.json" "$O/new/$STEM_w4a4.envelope.json" golden/baselines/
  run 0 1800 q-chk-w4a4 $BIN $QV --out $W/q-chk-w4a4 --skip envelope
  qsum q-chk-w4a4
  v=w4a16; stem=$STEM_w4a16; vv="V_$v"
  # shellcheck disable=SC2086
  run 0 900 "chk-$v" $BIN accuracy $COMMON ${!vv} --baseline $W/new/$stem.baseline.json --noise-envelope $W/new/$stem.envelope.json --pre-change "$HEAD7" --allow-same-code
  summ "chk-$v"
fi
if has self; then
  mkdir -p "$O/self"
  A_w4a4="--mxfp4-act dynamic"; A_w4a16="--mxfp4-act none"
  NS=/work/golden/ref_noise/$BN-top20.safetensors.json
  AB="--allow-same-code --allow-config-mismatch"
  for v in w4a4 w4a16; do
    av="A_$v"; vv="V_$v"
    # shellcheck disable=SC2086
    E="INTARSIA_GDN_CHUNK=0" run 0 900 "self-$v-base" $BIN accuracy $COMMON ${!vv} --write-self-golden $W/self/$v-base.safetensors
    # shellcheck disable=SC2086
    E="" run 0 900 "self-$v-change" $BIN accuracy $COMMON ${!av} --self-golden $W/self/$v-base.safetensors $AB --self-vs-ref-noise --ref-noise $NS
    # shellcheck disable=SC2086
    E="" run 0 900 "self-$v-plain" $BIN accuracy $COMMON ${!av} --self-golden $W/self/$v-base.safetensors $AB
    # shellcheck disable=SC2086
    E="INTARSIA_GDN_CHUNK=0" run 0 900 "self-$v-control" $BIN accuracy $COMMON ${!av} --self-golden $W/self/$v-base.safetensors $AB --self-vs-ref-noise --ref-noise $NS
    for t in change plain control; do echo "-- self-$v-$t" | tee -a "$O/steps.log"; grep -aE "threshold|mean KL|noise|FAIL|PASS" "$O/self-$v-$t.out" | cut -c1-260 | head -12 | tee -a "$O/steps.log"; done
  done
fi
log "stages done"
