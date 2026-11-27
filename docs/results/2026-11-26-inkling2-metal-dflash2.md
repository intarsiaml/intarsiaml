# Inkling-2-Mini + DFlash2 drafter on Metal (lane ink-metal-dflash2, 2026-11-26 / 27)

Status: **GO by the pre-registered rule at k5 (ABBA 4 + 4); `--spec dflash` stays opt-in, no default changed (the user decides).** Branch
`ink/metal-dflash2` from origin/main `3fb1b5f8d` (main `1bf40e29f`, #544, merged in; a merge, not a rebase). The Metal refusal of an Inkling
DFlash2 draft is lifted; everything else is the HIP / cpu path of #541 running on Metal's existing DFlash2 ops (lane dm). Greedy ids equal
`--spec off` at every k tried (3, 4, 5, 7).

Turn it on: `intarsia generate|serve -m <Inkling-2-Mini> --backend metal --spec dflash --spec-draft <Inkling-2-Mini-DFlash2 dir> --spec-k 5`.
**Pass `--spec-k 5`**: without it the DFlash2 draft's k is its block - 1 = 7 (the HIP default), which on Metal loses 1k (0.967) and 16k (0.920)
against the default in the k sweep. The default (no `--spec`) stays DSpark k3 (recipe `mtp_default_k`).

## Speed: DFlash2 k5 / the Metal default (DSpark k3), ABBA 4 + 4 on endor (binary `br3` = the PR head's code)

Decode tok/s ratio per cell, geometric mean over the 4 rep pairs, 95 % CI = t-interval on the log of the reps' cell values (`raw/rule.py`).
One `generate` process per arm and rep over `speed.json` (19 prompts, 256 tokens, EOS honoured, greedy; #541's prompts and cells), Metal,
`--max-seq 20480`, 45 s GPU idle before every process (`raw/slot.sh`, macmon logged), exclusive endor slots #12495 (r1 r2: dspark dfl5 dfl5
dspark) and #12502 (r3 r4: dfl5 dspark dspark dfl5).

| DFlash2 k5 / DSpark k3 | short | 1k | prose | chat | 4k | 16k | **mean of the 6 cells (geo / arith)** | code (extra) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ratio [95 % CI] | **1.171** [1.163-1.178] | **1.031** [1.002-1.062] | **1.051** [1.019-1.083] | **1.159** [1.149-1.169] | **1.017** [0.983-1.052] | **1.014** [0.965-1.066] | **1.072 / 1.074** | 1.352 [1.318-1.386] |

Mean of the six cells per rep: 1.074, 1.077, 1.049, 1.088 -> 1.072 [1.046-1.098]. Prefill (the same processes, one prompt each, so noisy):
**4k 0.894 [0.784-1.020]** (dspark 3093, dfl5 2731 tok/s), **16k 0.943 [0.827-1.075]** (2785, 2756 tok/s); both CIs include 1.0, the point
estimates say the draft's context ingest costs prefill time (about 0.2 s on a 4k prompt). Prefill is not in the rule.

| tok/s, median [range] over 4 processes | short | 1k | prose | chat | 4k | 16k | code |
|---|---:|---:|---:|---:|---:|---:|---:|
| Metal default (DSpark k3) | 97.93 [97.78-98.14] | 84.01 [83.63-84.37] | 103.09 [102.59-103.46] | 96.06 [95.85-96.41] | 105.79 [105.48-106.14] | 77.79 [77.53-78.16] | 116.31 [114.85-116.51] |
| DFlash2 k5 | 114.57 [114.35-115.20] | 87.15 [84.43-87.88] | 108.88 [104.99-110.40] | 111.34 [110.86-111.92] | 108.69 [104.01-109.15] | 79.89 [75.19-80.82] | 157.87 [151.90-159.58] |
| tokens per verify pass: DSpark k3 / DFlash2 k5 | 2.33 / 3.44 | 2.21 / 2.95 | 2.61 / 3.61 | 2.30 / 3.41 | 2.99 / 3.79 | 2.29 / 3.17 | 2.83 / 5.00 |

**Verdict by the pre-registered rule (below): GO.** Mean of the six cells 1.072 >= 1.03; no cell has a CI upper bound under 0.99 (the
lowest, 16k, has 1.066); ids identical across all 8 processes and both arms. Caveats the rule does not see: the 4k, 16k and 1k cells'
CIs include 1.0 (their point estimates are +1.4 % to +3.1 %), and the r3 pair (the first process of its slot, 45 s after a 3-min cargo test
suite, DFlash2 mean GPU 39.9 W against 45-50 W in the other DFlash2 runs) is the lowest of the four reps (per-rep mean 1.049; without
it 1.080). It stays in the numbers. The gain is at short / chat / code (acceptance 3.4-5.0 tokens a pass against 2.3-2.8); at 1k / 4k /
16k the cell is within a few percent of the default.

### k sweep (slot 2, #12478; one process per arm, not paired, no CI; decided before the ABBA)

DFlash2 k / DSpark k3, the same cells (`raw/s2-sweep/`; ids identical in all five processes):

| k | short | 1k | prose | chat | 4k | 16k | mean of 6 (geo) | code |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 1.142 | 1.083 | 1.007 | 1.105 | 0.982 | 1.100 | 1.070 | 1.174 |
| 4 | 1.179 | 1.064 | 1.006 | 1.141 | 0.982 | 1.054 | 1.071 | 1.259 |
| **5** | 1.196 | 1.023 | 1.042 | 1.144 | 1.016 | 1.037 | **1.076** | 1.369 |
| 7 (the block - 1 default) | 1.167 | 0.967 | 0.994 | 1.114 | 1.020 | 0.920 | 1.029 | 1.464 |

K3 / k4 / k5 are within 0.006 of each other (inside this sweep's noise); k5 had the highest mean and was the ABBA's k as registered.
(k5 in the ABBA, 4 pairs: 1.072; the sweep's single pair said 1.076.) k7 is the only k that loses on 1k and 16k and wins most on code.

### Secondary: k4, ABBA 2 + 2 (slot 5, #12514, `raw/k4/`, same binary and protocol; reported, not the verdict)

DFlash2 k4 / DSpark k3: short 1.136 [1.077-1.198], 1k 1.076 [1.033-1.120], prose 1.072 [0.831-1.383], chat 1.135 [1.067-1.208], 4k 0.996
[0.910-1.091], 16k 1.071 [0.883-1.299]; mean of the six 1.080 (per rep 1.083, 1.077); ids identical; prefill 4k 0.939 [0.656-1.346], 16k
1.076 [0.560-2.067]. By the rule's numbers it is GO too, with n = 2 (t = 12.7 intervals), so it cannot separate k4 from k5; k4 is a bit
better at 1k / 16k and worse at short / chat / 4k. The first attempt of this slot (#12510) ran no process: `slot.sh`'s guard matched another
lane's shell command line (no Metal time used).

## Bitwise results (greedy ids == `--spec off`)

- `raw/s1-md5/` (slot 1, #12474, binary `br1`) and `raw/s3-md5/` (slot 3, #12495, binary `br3`): `md5.json` = the golden `code` prompt and the
  4108-id prompt, `--ignore-eos`, 256 tokens: **off = DSpark default = DFlash2 k3 = k4 = k5 = k7** on both prompts (`br3`: off, default, k5).
  Whole-ids md5 (`[a, b, ...]` as printed) code `c69d1438ab3e715feb93a0614ebddcfe`, 4k `9e85ba1117e4a6a6cc12c5c917532e7f`; 64-token prefixes
  `3005dc6666cd9d7c81649e6b0eb4128b` / `c71814c5d2fd380c5b7d9e0486942ff6`.
- `raw/s2-md5-64/` (slot 2): the same prompts at `--max-tokens 64` (its own runs, not prefixes): off = default = k4 = k7
  (`ids` md5 `f91d92b86b45f7b6893d2a142027aa3c` over both prompts' lines).
- All 8 ABBA processes and the 5 sweep processes: the ids of all 19 speed.json prompts are identical across processes (`speed.py`, `rule.py`).
- Tests on Metal (`INTARSIA_INKLING2_BACKEND=metal`): the Inkling-2 tiny suites `inkling2` 17, `inkling2_dspark` 4, `inkling2_verify_softcap` 1,
  and `inkling2_dflash2` 8 (slot 1, `br1`) then 9 of 9 on the merged head (slot 4, `raw/tiny/full-suite-metal.txt`), incl.
  `greedy_ids_equal_spec_off` (k 1 / 3 / 7, prompts past the chunk and the draft window), `partial_acceptance_keeps_ids_identical`,
  `runs_with_admitted_kernels_only`, and the new `draft_pass_in_bf16_on_the_backend_matches_host_reference` (the real Metal draft pass at the
  tiny shapes against the f64 reference, rel l2 <= 5e-2; cpu bf16 measures 7e-3 to 2.5e-2). `intarsia-backend-metal --test dflash2` (dyn_conv
  at the Inkling draft's H 2304 / K 2 / G 144 added, window attention): 3 of 3.
- cpu (endor, `--load cpu`; `raw/cpu-tests-1.log`, `raw/cpu-tests-2.log`): `inkling2_dflash2` 8 then 9, `dflash2_cpu` 13, `dflash_cpu` 7,
  `inkling2_dspark` 4, `inkling2_dspark_cut` 1, `recipes_doc` 3, `intarsia-core --lib` 281 then 283, `docs_env_vars` 2; all ok on the merged head.
- Why ids stay exact: the draft only proposes; every emitted token is a verify row's argmax, and the Metal verify rows (<= 8) are the target's
  own row-exact path, already what DSpark k3 / k5 use. What this lane had to show is that nothing in the draft's pass breaks the target's state:
  the tests above and the md5 runs show it does not (k7 is an 8-row verify).

## What changed

- `crates/intarsia-core/src/model.rs`: an Inkling-2 DFlash2 draft is accepted on `metal` as on `hip` and `cpu` (the refusal said "hip and cpu only").
- `crates/intarsia-core/src/model/dflash.rs`: `INTARSIA_DFLASH2_HEAD=mxfp4|fp8` on Metal is refused with an Unsupported error (no gate has run a
  quantised draft head's kernels there). **The Metal draft's logits use the target's own bf16 head** (no copy: `dflash2_head_copy` returns
  `None` where it is not HIP, as before); draft numerics only, so ids are unchanged. The HIP default (mxfp4 copy, a speed choice) is not
  carried over.
- Nothing else was needed: the DFlash2 ops the draft runs (Matmul, Logits, Rope, QkNorm, KvAppend, windowed AttnDecode, SiluGlu, DynConv,
  RmsNorm, Embedding, Add) were all admitted on Metal by lane dm and the DSpark draft; the taps, block rows `rms(embed[id], embed_norm)`,
  padded vocab, softcapped selector and `--spec-k` default of #541 are backend-independent. The selector's top-k runs on the host
  (`dflash2_candidates` downloads the draft's `7 x 200058` f32 logits each pass); HIP's device top-k does not exist on Metal.
- Tests: `crates/intarsia-models/tests/inkling2_dflash2.rs` (backend `metal`, a bf16 draft-pass test), `crates/intarsia-backend-metal/tests/dflash2.rs`
  (H 2304 shape). Scripts and logs: `raw/` (`slot.sh` the one timing script, after #535's `slotB.sh`; `slot1.sh` .. `slot3.sh`; `rule.py`; `speed.py`; `prefill.py`).

## Not verified

- The secondary k4 ABBA is n = 2 pairs (wide CIs); k5's ABBA, the verdict's, is n = 4.
- Sampled requests (`temperature > 0`) with a DFlash2 draft on Metal: not run (the engine's greedy-only speculation gate is unchanged). No md5 at 16k.
- The mxfp4 / fp8 draft head on Metal (refused, not tried); the host top-k as a speed cost on Metal (not profiled; a Metal device top-k is a
  possible lever); whether a lower draft cost would move k.
- One machine (endor, a 140 W MacBook Pro), one session; the ABBA reps are deterministic to ~0.5 % process to process, but a thermal / power
  state shared by all reps would not show in the CI (the r3 process shows such a state can move a rep by ~4 %). The 4 reps cover two slots.
- `--max-batch > 1` and `serve` with this draft: not run on Metal (the loader refuses `--spec dflash` with batching, unchanged).
- Prefill is measured on one prompt per length, 4 pairs; the 4k / 16k prefill ratios are noisy and their CIs include 1.0.

## Metal time and fleetq

Exclusive Metal slots: #12474 (slot 1: tiny suites + md5, 23:45:17-23:52:23, 7.1 min), #12478 (slot 2: Metal kernel tests, k sweep, md5 at 64,
23:53:38-00:02:41, 9.1 min), #12495 (slot 3: bf16 draft pass on Metal, md5 on `br3`, ABBA half A, 00:25:17-00:33:06, 7.8 min), #12502 (slot 4:
full Metal suite, ABBA half B, 00:39:13-00:49:05, 9.9 min). #12514 (slot 5: the secondary k4 ABBA, 00:53:20-01:00:17, 7.0 min). Total 40.9 min. cpu / net jobs: #12464 (drafter copy, md5 `49860050629751794f4168a03620a5ee`,
sha256 `217980ab00d116693fa5946df7bbfd967caf5fac1f529afd541fbd470ddb54ee` = the HF LFS sha256 of #541's download), #12466 and #12488 (builds),
#12468, #12479, #12490 (cpu tests), #12512 (push). #12465 and #12487 and #12489 were released while still queued (never ran).
Drafter: `~/.cache/huggingface/hub/models--incoai--Inkling-2-Mini-DFlash2/snapshots/4f9088ea66c21f2b99ed657d1941c306b03e84d2/` on endor.

## Brief (coordinator, 2026-11-26 23:36 EST; the user is asleep and said "keep working")

Goal: make the published DFlash2 drafter (`incoai/Inkling-2-Mini-DFlash2`) work as **opt-in** `--spec dflash --spec-draft <dir>` for
Inkling-2-Mini on **Metal**, then measure it on endor against Metal's current default speculative path. Ship a PR. **No default changes.**
Whether this ever becomes a default is the user's call in the morning.

What exists: HIP and cpu DFlash2 for Inkling (#541, `docs/results/2026-11-26-inkling2-dflash2.md`; taps are target visits, block rows
`rms(embed[id], embed_norm)`, the draft's logits use its own quantised copy of the unembed's real rows, vocab padded, `--spec-k`
defaults to block - 1; on HIP k4 beat DSpark k5: mean 1.112, short 1.19, 4k 0.991, 16k 1.10). The Metal refusal is in `model.rs`
("hip and cpu only"). Metal DFlash2 ops exist from lane dm (Qwen3.8-27B): `crates/intarsia-backend-metal/src/dflash.rs`,
`kernels/metal/dflash.metal` (`DynConv`), `crates/intarsia-backend-metal/tests/dflash2.rs`, `docs/results/2026-11-06-metal-dflash2.md`.
Device top-k is HIP only. Baseline: main's default Inkling spec on Metal (DSpark; k from the code and startup line). Metal slot harness
of #535: `docs/results/2026-11-26-inkling2-sampled-dspark/raw/metal/` (reuse, no new timing scripts).

Where: worktree `/opt/code/intarsiaml/.worktrees/ink-metal-dflash2`, scratch under `.scratch/`. #544 (int8 HIP, opt-in) is about to
merge; once MERGED merge `origin/main` into the branch (a merge, never a rebase). Drafter weights on tatooine
`/data/models/intarsia/Inkling-2-Mini-DFlash2` (945 MB): copy next to endor's Inkling-2-Mini via fleetq, check md5/size. No other new folders.

Steps: (1) exact on cpu first (Inkling DFlash2 cpu tests on endor, `--load cpu`). (2) Lift the Metal refusal only for what Metal can run
exactly; fill missing Metal pieces (draft head / unembed copy bitwise to the HIP lane's reference rules; anything else the draft needs);
Metal tests in the style of `crates/intarsia-backend-metal/tests/dflash2.rs`. (3) Metal exactness (Metal slot): greedy ids `--spec dflash`
== `--spec off` on the golden md5 prompts, short and 4k, 64 and 256 tokens; plus the Metal Inkling tiny suite. (4) Speed: one-process k
sweep (k 3/4/5/7), then ABBA 4 + 4 at the best k against main's Metal default, the same six cells as #541 (short, 1k, prose, chat, 4k,
16k), plus prefill 4k and 16k. endor is a 140 W MacBook Pro whose GPU speed depends on recent power draw: 45 s idle gaps between runs,
arms grouped as `slot.sh` does with macmon. (5) The pre-registered rule (below) is written before any speed number and not relaxed.
(6) PR to `main` via `fleetq run --host endor --load cpu` push; do not merge; body: what changed, speed table first, bitwise results,
what is not verified, how to turn it on; ends with the Claude Code line. (7) This doc, a LOG.md row, a unanimis capture (`unim_capture_new`,
no `project`/`scope`, `intarsiaml` in `source`).

Budget and fleet: every Metal run takes an exclusive endor slot (`fleetq run --host endor --exclusive --expect N --label ...`), each slot
<= 20 min, <= 60 min in total, none after 07:00 EST; another lane needs one <= 15-min exclusive Metal slot tonight (fleetq orders them);
queue one slot at a time and do not hold the box idle. Other fleet commands through fleetq with an honest `--expect` (builds / tests
`--load cpu`); no tatooine GPU windows (radiance is not ours tonight). Never `fleetq release` a running job (kill its process tree).
No single tool call or wait over 4 min.

Stop rules: if a Metal piece cannot be made bitwise with a small change, stop and report; do not ship a non-exact path. If the k sweep
shows no k faster than the Metal default on any cell, skip the ABBA, record NO-GO, and ship the opt-in PR only if the code is clean and
tested, else leave the branch unpushed and report.

Final report (<= 15 lines): speed table first (per-cell ratio vs the Metal default with 95 % CI, mean, verdict, best k, prefill);
bitwise results; PR number and head sha; Metal minutes used; what is not verified; fleetq ids.

## Pre-registered rule (written 2026-11-26 23:37 EST, before any speed number; not to be relaxed)

Cells: short, 1k, prose, chat, 4k, 16k (the #541 definitions, `raw/speed.json`), plus prefill 4k / 16k. Ratio = DFlash2 tok/s / Metal
default tok/s per (prompt, rep), geometric mean per cell, 95 % CI = t-interval over the rep pairs.

- **GO** (for a later default-on proposal, the user's call): mean of the six cells >= 1.03, no cell with a 95 % CI upper bound < 0.99,
  and greedy ids identical to `--spec off` (and across arms).
- **OPT-IN**: faster on some cells, not GO.
- **NO-GO**: faster on none.
- Whatever the verdict, `--spec dflash` stays opt-in.
- The k sweep (k 3 / 4 / 5 / 7, one process each) only picks the ABBA's k; no k faster than the default on any cell = skip the ABBA, NO-GO.

## Log

- 2026-11-27 00:05 EST, **k sweep done (slot 2, #12478, one process per arm, 45 s idle before each, not paired), before any ABBA number**.
  Ratio DFlash2 k / DSpark k3 (the Metal default) per cell, single process each (no CI): mean of the six cells k3 1.070, k4 1.071,
  **k5 1.076**, k7 1.029 (k7 loses 1k 0.967, 16k 0.920). Every k is faster than the default on some cells, so the stop rule (no k faster
  anywhere) does not fire. The three lowest k are within 0.006 of each other, which is inside this sweep's noise. **ABBA k = k5, the k with
  the highest six-cell mean in the sweep (decided here, before the ABBA).** k4 gets a smaller ABBA afterwards only if the Metal budget
  allows it; it is reported as a secondary result and does not change the verdict.
