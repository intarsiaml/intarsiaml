# Why the Qwen3.8-27B MXFP4 accuracy gate fails on HIP (lane ac, 2026-10-25)

User approval: 2026-10-25 11:05 (investigate; report before changing any golden in main; if the reference is at fault, prepare a
re-record as a draft PR and stop for review). No GPU time was used for the findings below: they come from the committed logs of
lane tp2 (`origin/lane-tp2/batch`, `docs/results/2026-10-25-tp2-hip/raw/`), the three engine dumps those runs wrote
(`tatooine:~/tp2/out/{27b,w12,w13}/`), the three goldens (`tatooine:~/intarsiaml-p1/p1-golden/`), and offline arithmetic with the
gate's own estimator. A GPU window that would add the missing main-build controls and a baseline + noise envelope is written and
not yet run (section 7).

## 0. Verdict

* **No evidence of an engine accuracy fault.** Measured builds only: main `5241377b` and `#144` were not run (the window in
  section 7 would). The 27B W4A4 output on this gate is bitwise unchanged from build `ba0fa9e` (w9) through `5bee812` (w12) to `020b28c` (w13, the merged tree): all 60 dump tensors
  are equal. Its numbers (prefill 0.1799, decode 1.392x noise) are the same to the printed digit as lane T2's on 2026-10-22 and
  lane M1's on 2026-10-21 (`#145` is bitwise against the previous main by its own gate).
* **The two failures have two different causes, and neither is "the golden is wrong".**
  1. **W4A16 golden (w12): a flag mismatch in the run.** The golden is W4A16 (activations none). The engine ran W4A4
     (`mxfp4 activations: dynamic (declared by the checkpoint)`) because `stages-w12.sh` did not pass `--mxfp4-act none`. A W4A4
     engine is 0.138 KL21 away from a W4A16 reference, exactly as far as the W4A4 HF reference itself is (0.139).
  2. **W4A4 golden (w9): prefill passes; decode has failed since the gate was written** (1.392x the reference noise on
     2026-10-21, 2026-10-22 and 2026-10-25). The reference-noise record the gate uses for decode was measured on the prefill rows
     (all 1441, 0.197), not on the decode window (the last 16 rows of each prompt, 118 scored rows). On the window the same two HF
     runs are 0.2527 apart. The engine's 0.3397 is 1.34x that. This is the follow-up the user deferred at LOG 2026-10-21 13:23
     ("3 later").
* **Correcting the reference record does not make the W4A4 decode verdict pass.** At factor 1 it is still FAIL (1.34x). Making
  it pass needs a factor of at least 1.35, or a different judge for this class. That is the user's call; none was chosen here.

## 1. What the three goldens are

All three are HF `transformers` 5.19.0 + `torch` 2.14.1 with Quark MXFP4 emulation (weights dequantised to bf16), layer-streamed
on endor's CPU, one full-sequence forward per prompt, 10 prompts (`golden.jsonl`, `golden_long.jsonl`), 1441 positions, top-20.
None is vLLM: no vLLM golden exists for this model (T2, `2026-10-22-t2-27b-tp2.md` section 2). Model
`amd/Qwen3.8-27B-Quark-AWQ-MXFP4` (snapshot `5233554c`). Built by lane M1 with `tools/golden_quark_mxfp4.py`
(`2026-10-21-m1-qwen38-drill.md` section 5). Provenance files are added next to them in `golden/*.provenance.json` (the
safetensors are gitignored and only exist on tatooine/endor).

| golden | activations | KV | built (UTC) | sha256 (first 12) | gate role |
|---|---|---|---|---|---|
| `qwen3.8-27b-mxfp4-hfquark-top20` | dynamic fp4 (W4A4, 496 QDQ'd linears) | bf16 | 2026-10-21 14:05 | `1b32357eaa08` | the gate golden, ref-noise sidecar |
| `qwen3.8-27b-mxfp4-hfquark-f32-top20` | dynamic fp4 (W4A4) | f32 | 2026-10-21 14:42 | `d5186d875800` | noise only (bf16 vs f32 of the same HF run) |
| `qwen3.8-27b-mxfp4-hfquark-w4a16-top20` | none (W4A16, 0 QDQ'd linears) | bf16 | 2026-10-21 14:42 | `32e2b96d84cc` | gate golden for `--mxfp4-act none` only |

## 2. W4A16 golden: the run did not match the golden

* `scripts/tp2-hip/stages-w12.sh` (lane tp2, branch `lane-tp2/batch`): `run $D 0 600 acc-27b-tp1 accuracy ... --golden $G27 --mode
  both ...` with `G27=.../qwen3.8-27b-mxfp4-hfquark-w4a16-top20.safetensors` and no `--mxfp4-act none`.
* The run's own stderr (`raw/w12/acc-27b-tp1.err`) prints both sides: golden reference `activations none (Quark even_round), 0
  QDQ'd linears`, engine `mxfp4 activations: dynamic (declared by the checkpoint)`. The gate does not compare the two.
* Result (`raw/w12/acc-27b-tp1.out`): prefill KL21 0.1387 and decode 0.1897 against the W4A16 static floor 0.03 (no sidecar for
  this golden), FAIL.
* That 0.138 is the activation-format gap, not an engine error. Same pair of numbers, computed here with the gate's estimator
  (`scripts/ac/w4a16.py`), scored rows, reference = the W4A16 golden:

  | candidate | KL21 |
  |---|---:|
  | engine, W4A4 (the w12 dump) | 0.1380 |
  | HF W4A4 bf16 golden | 0.1394 |
  | HF W4A4 f32 golden | 0.1354 |

  And M1 measured the engine at W4A16 against this golden on 2026-10-21: prefill 1.4e-3, decode 8.3e-4, PASS
  (`2026-10-21-m1-qwen38-drill.md` section 5). T2 on 2026-10-22 (w6, `--mxfp4-act none`): 1.484e-3 (tp1) and 6.003e-4 (tp2),
  PASS. Neither has been re-run on current main (section 7).
* Why it looks like a property of the checkpoint: it is a property of the flag. The same engine, same build, passes this golden
  as W4A16 and fails it as W4A4.

## 3. W4A4 golden: what the gate run used

`raw/27b/acc-tp1-27b.out` (w9; w12 and w13 dumps are bitwise the same): golden W4A4 bf16, engine default flags (dynamic
activations = the golden's mode, so this one is matched), prefill chunk 256, `--ref-noise` sidecar
`qwen3.8-27b-mxfp4-hfquark-top20.safetensors.json` (precision_spread, all rows 0.197, no scored value).

| mode | engine KL21 (scored rows) | limit (ref noise x 1) | ratio | verdict |
|---|---:|---:|---:|---|
| prefill (n=1279) | 0.1799 | 0.1970 | 0.91 | PASS |
| decode (n=118) | 0.3397 | 0.1970 | 1.72 (1.392 on all rows) | FAIL |

The sidecar was back-filled on 2026-10-21 from M1's table; its golden was "not available when this record was filled in" and was
matched by name and position count only (`2026-10-21-a2-ref-noise-gate.md`). The goldens are available now. Re-measured with
`intarsia ref-noise` (this PR's `golden/ref_noise/...json`, bound by FNV-1a `725eb3ea093c0264`): all rows 0.19692 (n=1441), scored
rows 0.19913 (n=1279). The old 0.197 is reproduced. All 12 copies of the W4A4 golden on tatooine (every lane's `p1-golden`, incl.
`~/intarsiaml-p1/`) have the same sha256 `1b32357eaa08...`, so the binding does not orphan any lane's `--ref-noise` run.

## 4. Decode: the reference noise on the decode window

The decode mode feeds the last 16 positions of each prompt through the single-token path (`dec_pos` in the dumps equals `arange(n-16, n)` for all 10
prompts, checked directly; 118 of the 160 rows are scored). The same two HF runs, restricted to those rows (`scripts/ac/win.py`; the all-rows
and scored numbers reproduce the Rust tool to 4 digits):

| HF W4A4 bf16 vs f32 | all rows | scored rows |
|---|---:|---:|
| prefill rows (1441 / 1279) | 0.1969 | 0.1991 |
| decode window (160 / 118) | 0.2227 | **0.2527** |

Engine decode 0.3397 / 0.2527 = **1.34**. Not a pass at factor 1. The statistic is a mean over 118 rows with a heavy tail:

* The 10 largest rows carry 89 % of the engine's decode sum (3 rows: 56 %). In the HF noise they carry 81 % (3 rows: 49 %).
* The largest row is `chat_multiturn` position 55: engine decode 10.05, engine prefill 9.84, HF bf16-vs-f32 7.51. Its bf16
  reference top-1 is token 326, the f32 and W4A16 references' top-1 is 248045. The reference itself flips there.
* Per row, engine decode KL against the bf16 golden and the HF bf16-vs-f32 spread have Spearman rank correlation 0.91: the same
  rows are hard for both.

## 5. Is the engine's decode path wrong?

All offline, on the three engine dumps. The dumps hold the engine's top-64 only, so this estimator reads lower than the gate's
(decode scored 0.2915 here against 0.3397 in the gate); every comparison below uses the same estimator on both sides.

| quantity (118 scored decode rows) | mean | bootstrap 95 % |
|---|---:|---|
| engine decode vs bf16 golden | 0.2915 | [0.113, 0.534] |
| engine prefill vs bf16 golden, same rows | 0.2192 | [0.093, 0.416] |
| HF bf16 vs f32, same rows | 0.2527 | [0.120, 0.430] |
| engine decode vs engine prefill, same rows | 0.1619 (median 0.007) | [0.082, 0.263] |
| paired: engine decode minus HF noise | +0.039 | [-0.059, +0.166] |
| paired: engine decode minus engine prefill | +0.072 | [-0.012, +0.178] |

* **Not distinguishable from the reference's own spread at this sample size.** Dropping `chat_multiturn` (5 rows) gives 0.2155
  against noise 0.1974 (1.09x). Dropping it and `long_prose_700` gives 0.1135 against 0.0681 (1.67x) on 97 rows of small absolute
  size; engine prefill on those rows is 0.0518. So the engine's decode rows sit above its prefill rows and above one HF spread
  draw on the ordinary rows too. One draw of a heavy-tailed spread cannot say more than that.
* **The decode path itself is right in W4A16.** Same engine, same GDN recurrence, attention, KV cache and state: decode KL
  8.3e-4 against the W4A16 golden (M1) and 6.0e-4 (T2). Only the W4A4 activation rounding is added.
* **Why decode and prefill differ in W4A4:** under W4A4 every MXFP4 product is exact (T2 section 3), so the engine's two paths differ only in
  how the non-MXFP4 ops (norms, attention, GDN step) round to bf16. Rounding activations to fp4 turns a one-ulp difference at an
  e2m1 near-tie into a whole step (M1 section 5). Engine decode against engine prefill is 0.16 mean with a median of 0.007: a few
  rows flip, the rest agree. That is the same mechanism, at the same size, as HF bf16 against HF f32.
* The decisive control, a W4A4 engine noise envelope on the decode rows (`--noise-perturb chunk`, section 7), is not run.

## 6. What to do (for the user to choose)

1. **W4A16 golden gate**: run it with `--mxfp4-act none`. That is a one-flag fix to lane tp2's `stages-w12.sh` / `w13` (their
   branch; this lane did not touch it). A guard in `accuracy` (refuse, or warn, when the golden's `engine_version` says
   `activations none` and the run is dynamic, or the reverse) would have caught it; goldens carry no structured field for it
   today, so that is a code change and was not made.
2. **W4A4 golden gate, decode.** Options, none applied:
   * (a) keep the verdict honest and leave it FAIL until the gate carries a decode-window noise (per-mode noise in
     `refnoise.rs`; the window value is 0.2527) *and* the user picks a factor; at the measured 1.34 a factor of 1.5 passes.
     `2026-10-21-a2-ref-noise-gate.md` (follow-up 3) records that no factor was chosen to make a verdict pass.
   * (b) give the 27B the regression gate the 35B has (`golden/baselines/*.baseline.json` + `*.envelope.json`, as `5c769c8a3`
     did for CUDA 35B): any change in what the engine computes is caught, and the absolute decode floor becomes report-only for
     this golden. Needs the GPU window of section 7.
   * (c) use the W4A16 golden (item 1) as the engine-correctness gate for the 27B (weights, loader, GDN, attention, KV, decode
     path to 1e-3, no near-tie noise) and gate W4A4 on prefill only. The checkpoint's declared mode is W4A4, so (c) alone leaves the
     activation quantiser ungated; combine with (b).
   Recommendation: (c) + (b). It needs no factor, and it would have failed on a real regression.
3. **Keep the goldens in git** (735 KB; `.gitignore` allow-lists only the gemma and 35B ones). They exist on tatooine and endor only.
   Not done here; this PR adds provenance files and no safetensors.

## 7. What this PR contains, and what it does not

Contains: this note; `golden/ref_noise/qwen3.8-27b-mxfp4-hfquark-top20.safetensors.json` re-measured and hash-bound (adds the
scored-row value 0.1991, FNV-1a and size; **no per-prompt values**, because those would tighten `--self-vs-ref-noise`, which now judges
every prompt against the golden-wide mean; the prefill verdict is unchanged, 0.1799 against 0.1991 = 0.90x; decode
is still FAIL; the golden-bound test in `refnoise.rs` is updated and `cargo test -p intarsia-cli` passes, 171 tests);
`golden/*.provenance.json` for the three goldens; `scripts/ac/` (the offline analysis, reproducible with `DUMPDIR=<dir with the
three dumps> uv run --with numpy --with tokenizers python scripts/ac/pair.py`, and the GPU window); a LOG row (status proposed).

Does not contain, and is not run: the GPU window `scripts/ac/window.sh` / `stages.sh` (tatooine, one card, radiance down ~10
min, restored in an EXIT trap). It runs, on a freshly built main `5241377b` (`~/intarsiaml-ac/ac-intarsia`, built under fleetq
#5318), (0) the W4A16 golden with the default flags (control, expect FAIL as w12), (1) the same with `--mxfp4-act none` (expect
PASS), then for each golden a baseline, a `--noise-perturb chunk` envelope, and the baseline + envelope gate. Its outputs would
be the two engine baselines and envelopes (`golden/baselines/qwen3.8-27b-mxfp4-hfquark-{,w4a16-}top20.hip.{baseline,envelope}.json`)
stamped `head 5241377 code-tree 326167668388`. tatooine's queue put it at about 15:00 (3 h 40 min); no job is queued. Until it runs, the
W4A16 PASS on main is inferred from the 10-21 and 10-22 runs and the bitwise-unchanged W4A4 output, not measured.
