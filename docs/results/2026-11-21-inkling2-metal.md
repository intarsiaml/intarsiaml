# Inkling-2-Mini end to end on Metal (lane ink-metal, 2026-11-21)

Status: Inkling-2-Mini runs end to end on Metal (endor, M5 Max): G1, G2 and G3 short / long pass (last full check #10901 on
de2b5c9bc; the m-mix final + ink/batch merge gate is #11048, see the end), `intarsia serve` loads and serves chat / reasoning /
tools / json_schema. Decode 63-64 tok/s at short context (`--profile-every 0`, localeval 63.4), 56.7 at 16k; prefill 1.4-1.7k
tok/s (chunk 256), 2.3k at chunk 2048. Open: batched decode now merged (ink/batch) but not yet gated here, TP / MTP / prefix store
refused, decode graph replay blocked (Changes item 8). Branch `ink/metal` (not pushed, not merged). Op contracts unchanged.

## Briefs received

- 2026-11-21 (coordinator, SendMessage): ink/skel is done at 46349d879 (G1, G2, G3 short / long / 4k pass on cpu; new: ring-safe prompt reuse, faster cpu experts bitwise the same, the accuracy turn format, recipe coverage, results doc; op contracts unchanged). Merge ink/skel at the next clean point, rerun the Metal gates on the merged tip, and message main with the first endor decode tok/s and the per-op profile.

- 2026-11-21 (coordinator, SendMessage): scope from here: (1) finish the per-dispatch trace (#10756) and the localeval run
  (speed-plain, conc-plain-1k vs oMLX 20261120-222536); write the trace path into this doc as soon as it lands (the Fable
  reviewer reads it here). (2) Do not start controller / router fusion or host-gap / command-buffer work (decode graphs
  included): lanes ink-dfuse (`.worktrees/ink-dfuse`: controllers, router fusion, <= 350 dispatches) and ink-hostgap
  (`.worktrees/ink-hostgap`: cb pipelining, GPU sampling, encode-ahead) own them, both branched from a88db4620. (3) This
  lane stays the integration owner: merge lane branches when they report done with passing gates and rerun G1-G3 after each
  merge; next in line are m-mix's final tip and ink/serve aaad4c9b6 (chat / reasoning / tools).

- 2026-11-21 (coordinator): merge ink/serve (done at b447e3da4, which superseded the aaad4c9b6 the scope brief named);
  append ink/serve's Metal smoke, the `--profile-every 0` decode runs and ink-hostgap's measurement A to the serve job and
  release #10812; move the `--no-fail-fast` core / cpu / cli rerun to spark2 (#10868) and keep only the metal crate on endor.
- 2026-11-21 (coordinator): merge ink/m-mix at 47570539e (InkInProj GEMV up to 8 rows, blend gate knobs, docs / raw) before
  #10867 starts, record it here, and take no default flips from m-mix (those come later as their own commit). ink-batch
  merges this lane's 761a1561b (serve admission fix).

- 2026-11-21 (coordinator): merge ink/d1row-doc 5b0b0bd2f (docs only: docs/results/2026-11-21-inkling2-d1row/ and its results
  doc; the d1row kernels lost and stay unmerged on ink/d1row); check the diff touches only docs/results/; ops 84-87 are free
  again (mark them free in an op reservation note if ink/metal has one: it has none). Done: merge 2343bb720, 27 files, all under
  docs/results/.
- 2026-11-21 (coordinator): guard the two `intarsia_backend_metal` uses in tests/inkling2.rs with `cfg(target_os = "macos")`
  (Linux test builds of intarsia-models failed), check on spark2, commit separately. Done: e3f92beff; `cargo check --tests -p
  intarsia-models` EXIT 0 on spark2 and clean on endor.

- 2026-11-21 (coordinator): #10890 is already running (started ~23:14); #10890 and #10901 build from this working tree, so no
  code changes here until #10901 has finished its build (docs-only commits are fine); when #10890 reports, record which
  commit it built. Recorded: its `cargo build` ran at 23:14:12 and finished in 0.27 s (nothing to rebuild), so the binary is the
  one built at 22:32 (#10898) from f55bfe8b7's code; the tree at 23:14 was e3f92beff, which differs from f55bfe8b7 only in
  `tests/inkling2.rs` and docs. The 2343bb720 merge (23:32) is docs only, so #10890's binary equals the tip's code.

- 2026-11-21 (coordinator, rule for all lanes): this session and every worktree run on endor, so no local cargo build / check /
  test or shader compile on endor outside a fleetq slot. macOS compile checks: bespin (`fleetq run --host bespin --load cpu
  --mem 6 --expect <min>`, a `git archive` copy in ~/intarsiaml-<lane>, toolchain in ~/.cargo/bin; no GPU timing or real model
  there). CPU tests: spark2. GPU work: endor fleetq jobs. Reading files, git and editing locally are fine. For #10890: it ran
  while other lanes built and tested on endor (ink-pack12 around 23:35); compare its localeval decode numbers with the baseline
  62.5 / 62.0 / 60.6 / 57.5 tok/s at 142 / 1k / 4k / 16k, and if any length is more than 1 % lower mark the run possibly
  contaminated and requeue it.

- 2026-11-22 (coordinator): (1) cancel #11023 (done): the coordinator overrides the > 1 % rule; three spec-off runs at 16k are
  57.5 / 57.4 (mtp 17c67dacf) / 56.7 (#10890) and at 142 62.5 / 63.8 / 63.4, a 1.3 % spread within run-to-run noise; mtp's
  #10991 off arm gives a fresh baseline. (2) Investigate the `intarsia bench speed` calibration failure now (ink-batch's #10946
  runs the engine bench at 1-16 streams): compare the calibration call with chat / localeval, repro on a tiny-model serve on
  spark2 if needed (not endor), fix on ink/metal with a test, send the hash.

- 2026-11-22 (coordinator): m-mix is done (quiet #10909: ConvMix at 2048 rows 15 % faster, blend at 16 rows 13-14 % faster,
  +0.8 us at 1 row, which dfuse's T=1 split replaces). After #10901, merge ink/m-mix 52b337aa9 (with 25ff7864a, the defaults
  change: NE 4 row tiling, gate rows) and gate the merged tip (rows-exact kernels test, G2 / G3, greedy md5); #10901 does not
  cover it. Send the hash; brief here first.
- 2026-11-22 (coordinator, supersedes the separate m-mix gate): ink/batch passed its criterion (#10914: solo decode -0.62 %,
  identical output). After #10901 merge m-mix 52b337aa9 first, then ink/batch bbc0bec00 (177cfd2f5 plus docs). Both touch the
  mix shader / inkmix.rs (m-mix: NE 4 tiling, gate-rows default; batch: the MX_ROWTAB row-table branch, longer InkP / InkMixP
  params): keep both, a kernel-body conflict is a semantic merge. ONE gate on the merged tip: shader compile, rows-exact
  kernels test, G2 / G3, greedy md5 against the current tip, ink-batch's batched == solo at B = 2 / 5 / 16. Send the hash to
  the coordinator and ink-batch (a716d7e057a6fbfc3).

- 2026-11-22 (coordinator): after #11048 passes, merge three more branches in this order, each when the coordinator or its
  lane says it is ready: (1) ink/hostgap 9a38f0224, once its spark2 #11072 and bespin #11073 pass (greedy +1.5 %, sampled
  +2.7-3.6 %, bitwise identical; `generate` defaults to `--profile-every 0` on Metal); (2) ink/fp8, after its #10974 (all
  opt-in, default path identical, #10941); (3) ink/pack12, after its flip gate #11066 (contains ink/fp8; P12 default-on on
  Metal, `INTARSIA_INK_PACK12=0` reverts; lossless, +8.5-9.7 % decode). Then ONE combined gate on the tip after all three (or
  whichever are in): rows-exact, G2 / G3, greedy md5 with P12 on vs explicit off, batched == solo at B 2 / 5 / 16, and a quiet
  localeval speed-plain at 142 / 1k / 4k / 16k, which becomes the new Metal baseline. Send the hash.

- 2026-11-22 (coordinator): ink/hostgap 9a38f0224 is ready (spark2 #11072: 45 test binaries, 0 failed; bespin #11073 compiles
  clean on macOS). Merge it after #11048 passes, before fp8 and pack12; leave its later fb38fe4a5 (opt-in pre-wake / QoS /
  vsort, untimed) out.

- 2026-11-22 (coordinator): add ink/prefill to the merge queue, after hostgap, fp8 and pack12 and before the ONE combined gate
  (which includes it: rows-exact, G2 / G3, md5, batched == solo B 2 / 5 / 16, then the localeval baseline). Its gate: localeval
  speed-plain #10967, base e3f92beff vs 378dc214e: prefill 1k +11.5 / +12.0 %, 4k +21.5 %, 16k +9.4 / +7.0 %, decode within
  1 % everywhere. ink-prefill sends the exact hash: 378dc214e, or a recipe commit on top (`[tuning.metal] prefill_chunk = 2048`
  in recipes/inkling2.toml) if its #11109 passes. With the recipe commit in, the combined gate and the baseline run at chunk
  2048 (say so: earlier Inkling Metal localeval numbers were at chunk 256). Report the merged tip hash.

- 2026-11-22 (coordinator): queue item 3 changed. fp8's #10974 screen md5 gate was vacuous (the opt-in head screen never ran
  in the "on" runs); fixed on ink/fp8 a810fbb5c (opt-in screen code only; bespin #11112 compiles), rerun queued on endor. The
  default path is proven (#10941 bit-identity, #11055 admission) and the screen is off by default, so do not wait for the
  rerun: merge ink/fp8 a810fbb5c (or a newer tip touching only docs or the screen) when fp8 is next; it unblocks pack12. Add
  one LOG line (below). A failing rerun is fixed by the fp8 lane in a follow-up and does not block the combined gate, which
  runs with every fp8 env var unset (the default path).

- 2026-11-22 (coordinator): the LOG line belongs in this doc's LOG section; leave docs/decisions/LOG.md alone. Its rows (P12
  default-on, MTP flip, argmax v4t, profile-every 0, chunk 2048 if it lands, head screen pending) go in once at merge-to-main
  time, from the LOG section here. The env reset in the combined gate is right. Carry on with the queue.

- 2026-11-22 (coordinator, before the merge queue): likely accuracy regression. ink-dfuse job E #11003 (ink/dfuse ac2371c81,
  containing ink/metal 0ee1bec29): G2 teacher-forced DECODE rows 1.109e-2 short / 9.205e-3 long (before the ink/metal merge,
  #10952: 2.589e-3 / 2.474e-3; limits 4e-3 / 6e-3; the tool exits 1), carried by one chat_reasoning row (0.43 nats) and one
  chat_system row (0.30); prefill G2 2.380e-3 -> 2.719e-3. Suspects: the m-mix and / or batch merges. Do: (1) gate the decode-mode
  G2 rows in #11048 and every later gate, same limits; (2) if #11048 shows it on 8546d143f, hold every queued merge (hostgap,
  fp8, pack12, prefill) and bisect G2 short decode mode on f55bfe8b7, 331238a4e (m-mix), 45887198b (batch) in one default-load
  endor job; (3) on the culprit, run the two outlier rows against the fp32 oracle (tools/inkling2_ref.py, the replay / capture
  setup on ink/spec): farther from the oracle = a bug to fix or revert, equally close = chaotic amplification through the conv /
  ring state (report; a gate-design question); (4) check whether `intarsia accuracy`'s teacher-forced decode path runs the
  row-table (batched) path at B = 1 (it must take the solo path). Report each step; cc the m-mix lane (ae322a37a246f2443) or
  ink-batch (a716d7e057a6fbfc3) if theirs.

- 2026-11-22 (coordinator): agrees it looks like bad state from the 1-row decode path. Whatever the bisect names, the fix is
  not done without a test that would have caught it: Metal 1-row decode at real D (2304, so NE 4 runs) against the CPU
  reference (not against the rows-exact kernels), ~20 decode steps, comparing the conv-ring / blend / KV state each kernel
  writes and the logits, within the tiny CPU-vs-Metal tolerances; a synthetic D 2304 layer with random weights in the Metal
  test suite is fine; the same test for NE 3. If the bisect points at NE 4: fix the kernel or make NE 3 the default
  (`INTARSIA_INK_MIX_NE`) until fixed, whichever is faster; then rerun decode-mode G2 (must be back near 2.6e-3) and release
  the merge queue. Send the bisect table when #11142 lands.

- 2026-11-22 (lane): test for the brief, step 1: `crates/intarsia-models/tests/inkling2_mix_decode_cpu.rs` (02d157d8b), the
  m-mix kernels at D 2304 against the cpu reference: `ConvMix` as a 20-step 1-row recurrence (ring and output carried per
  backend, all four outputs compared each step), `AttnResMix` and `LoopBlend` at 1, 2, 5, 16 rows; per-tensor rel L2 <= 1e-2;
  NE 3 runs in a child process (`INTARSIA_INK_MIX_NE=3`). It covers the mix kernels only, not the KV state or the logits; a
  whole synthetic D 2304 layer follows if this one does not catch the regression. Compiles on macOS (bespin #11159, which has
  no Metal toolchain); its first GPU run is in #11142 (tip, 331238a4e, 331238a4e with `INTARSIA_INK_BLEND_RB=16`), before the
  G2 bisect arms.

- 2026-11-22 (coordinator): build hazard 1: a target/ cloned from another tree (cp -c) has newer mtimes than the checked-out
  sources, so cargo keeps the other tree's crates and can build the wrong commit silently. Check the bisect's builds.
  Lane: the #11142 trees had no target yet (not run); bisect.sh now touches every tracked file after the clone and logs
  HEAD plus the Compiling count for core / backend-metal / models / cli per tree. #11048 built in this worktree's own target
  (no clone): not affected.
- 2026-11-22 (coordinator): build hazard 2: `cargo test -p intarsia-cli --test <x>` without the backend feature rebuilds
  target/<profile>/intarsia without Metal. Build the backend binary last or give tests their own target dir; count md5 /
  self-golden matches only with exit 0 and non-empty output. Lane: mergegate.sh and combgate.sh run docs_env_vars with
  `--features metal` (binary keeps Metal; #11048's G2 logs show real KL values, so not affected); bisect.sh runs only
  intarsia-models tests in the debug profile, which do not rebuild target/release/intarsia.
- 2026-11-22 (coordinator): bisect moved up. The coordinator's waiter runs `fleetq wait 11104`, releases #11142 and runs
  `.scratch/bisect.sh` with FLEETQ_JOB=11104 (batch correctness slot, default load, 20 GB), expected start ~03:47 when #11093
  ends; same `.scratch/bisect_job.log` and `.scratch/bis/out/`. The #11142 client exits 4 on release: expected, do not
  requeue. Coordinator messages the lane when it lands.
- 2026-11-22 (ink-prefill lane): merge ink/prefill-merge at 96740cbe2 instead of 378dc214e (coordinator brief 12):
  378dc214e plus the cherry-picked brief 7 fix ed27f4462 (`ink_in_proj_bf16_gemv_chunks`, NoKernel at m 31-63 without
  metal4); endor #11105 tiny Metal 8/8 and inkmix rows-exact 3/3 with metal4 off and on, gemv_chunks ADMITTED on 11 shapes;
  bespin #11173 check clean. If #11109 passes at chunk 2048, one more commit (`prefill_chunk = 2048`) replaces this hash.
- 2026-11-22 (coordinator): bisect landed; culprit NE 4 from D 2048 (m-mix 25ff7864a); BLEND_RB, InkMixP and ink/batch
  cleared. Fix brief: (1) find the NE 4 bug (30 min timebox), else NE 3 back at D 2304 with a TODO; (2) a test that fails
  on the unfixed tree (show it on 331238a4e) and passes on the fix; (3) gate: G2 short both modes (decode ~2.6e-3), greedy
  md5 non-empty, tok/s 1-row short vs the tip, what the fix costs; (4) then the merge queue: hostgap, fp8, pack12 after
  #11161, prefill (new hash), dfuse, then the combined gate. Message the gate job id; the coordinator nests it.

## Bisect #11142 (G2 short, decode mode, KL21 scored)

| Tree | KL21 scored | max | chat_system | chat_reasoning |
|---|---|---|---|---|
| f55bfe8b7 (pre-merge) | 2.589e-3 | 5.308e-2 | 2.395e-3 | 1.009e-2 |
| 331238a4e (m-mix merge) | 1.109e-2 | 4.334e-1 | 2.376e-2 | 5.808e-2 |
| 45887198b (+ ink/batch) | 1.109e-2 | 4.334e-1 | 2.376e-2 | 5.808e-2 |
| 331238a4e, `INTARSIA_INK_MIX_NE=3` | 2.589e-3 (identical to base) | 5.308e-2 | 2.395e-3 | 1.009e-2 |
| 331238a4e, NE 3 + `INTARSIA_INK_BLEND_RB=16` | 2.589e-3 | 5.308e-2 | 2.395e-3 | 1.009e-2 |

All three trees recompiled core / backend-metal / models / cli (4/4). The synthetic D 2304 mix-vs-cpu test
(inkling2_mix_decode_cpu.rs) passed on the tip, on 331238a4e and with RB 16 (NE 4 rel L2 <= 1.7e-3, same as NE 3), so it
does not catch the fault.

Fix (7eca1b012): the bug was not found in the 30 min timebox (the NE 4 kernel reads and writes only its own channels and
covers D 2304 exactly; reductions are nsg-agnostic; the NS 8 / QB 4 instance passed the synthetic test, so a pipeline
thread limit is unlikely). `mix_geom` is back to the fewest NE that fits (NE 3 at D 2304) with a TODO naming the suspects:
a real call layout the synthetic test lacks (views with element skips, outputs aliasing stored sources, device position).
New test `inkling2_real_decode_ne.rs` (1dfd554e6): real checkpoint, 48-token prefill of the golden's chat_system prompt,
24 teacher-forced decode steps, per-step KL(NE 3 || default) <= 2e-3; NE 4 forced is reported with the dumped decode-step
tensors furthest from NE 3 (to localize the bug later). Gate job: #11179 (`.scratch/fixgate.sh`).

## Integrated

| Branch | Head | What | Conflicts |
|---|---|---|---|
| `ink/skel` | b5e283927 | base: cpu port, model block, loader, recipe, tiny tests | - |
| `ink/m-attn` | 9410a9b67 | InkKvPrologue (70), RelSinkAttn (71, split-KV + NAX prefill), InkAttnDecode (75) | none |
| `ink/m-moe` | 98317825d | MoeRoute, NVFP4 SiTU experts, MXFP8 experts (decode + NAX prefill) | `build.rs` (metal4 sources, launcher sources), `kernels.rs` (metal4 list, TABLE, the count test): kept both |
| `ink/batch` | bbc0bec00 (177cfd2f5 + docs), merged at 45887198b after m-mix | batched decode: row-table `conv_mix_bf16_rows`, `MX_ROWTAB` in the mix shader, row-table attention / GEMV rows | no textual conflicts; inkmix.rs and the mix shader changed in disjoint places (m-mix: NE 4 from D 2048 and `BLEND_RB` 4; batch: the row-table kernel, `InkMixP` 112 -> 144 bytes); gated together (merge gate below) |
| `ink/d1row-doc` | 5b0b0bd2f | d1row results (docs only; kernels not merged) | none; diff only docs/results/ |
| `ink/m-mix` | 52b337aa9 final, with the defaults change 25ff7864a (merged at 331238a4e; earlier 47570539e at 606fd36e4, 3e14f6aa3, d99d2c0fc) | InkInProj (73), ConvMix (74), AttnResMix (67), LoopBlend (68), SconvRes (69) | `build.rs`, `kernels.rs`, `lib.rs` (module, launch dispatch): kept all three lanes' entries |

`intarsia kernels test --backend metal --op <op>`, every Inkling op (fleetq #10681): all ADMITTED.

| op | kernel | cases | fail | result |
|---|---|---:|---:|---|
| ink_kv_prologue | ink_kv_prologue_bf16 | 117 | 0 | ADMITTED |
| rel_sink_attn | rel_sink_attn_bf16 / _nax | 63 / 7 | 0 | ADMITTED |
| ink_attn_decode | ink_attn_decode_bf16 | 70 | 0 | ADMITTED |
| moe_route | moe_route_f32 | 19 | 0 | ADMITTED |
| moe_experts | moe_experts_situ_bf16 / _rows | 10 / 7 | 0 | ADMITTED |
| moe_experts_mxfp8 | moe_experts_mxfp8_bf16 / _rows | 19 / 14 | 0 | ADMITTED |
| ink_in_proj | _gemv / _gemv_rows / _nax | 6 / 24 / 23 | 0 | ADMITTED |
| conv_mix | conv_mix_bf16 | 532 | 0 | ADMITTED |
| attn_res_mix | attn_res_mix_bf16 | 97 | 0 | ADMITTED |
| sconv_res | sconv_res_bf16 | 64 | 0 | ADMITTED |
| loop_blend | loop_blend_bf16 | 27 | 0 | ADMITTED |

## Changes in this lane

1. **State plumbing (step 2): nothing Metal-specific was missing.** `SlotSpec::Ring` slots are allocated and zeroed through
   `Backend::alloc` / `Backend::zero` in `SequenceState::new` (backend-generic, `crates/intarsia-core/src/state.rs`); rollback by
   length within the margin (`check_ring_cut`), the ring reset on a cut to 0, the failed-pass reset (`model.rs`, a failed
   Inkling-2 pass starts the sequence over), the `reuse_len` margin rule and the prefix-store refusal of ring slots are all
   backend-generic too. The Metal state tests below (rollback, interleaved, failed pass) exercise them on Metal.
2. **Op 75 selection** (`model/inkling2.rs`, `ink_attn_fused_rule(rows, local, ctx, window)`): per visit, on a pass of up to 16
   rows, the fused `InkAttnDecode` where the rule says so and the backend admits the call (`Registry::select`), else the
   `InkKvPrologue` + `RelSinkAttn` pair. Rule (Metal; the pair elsewhere), keyed on the keys a row reads (min(ctx, window) on
   SWA visits): 1 row fused up to 4096 keys (so every SWA visit and full visits to 4k; 8k / 16k unmeasured, the pair there);
   2..=4 rows up to 512 keys; more rows the pair. Override `INTARSIA_INK_ATTN_FUSED=0|1` (docs/tuning.md). A dump
   (`ink_dump`, the G1 tests) always takes the pair (it needs q / k / v / rel). Both paths give the same bits.
3. **Router on Metal**: Metal has no `Matmul` with an f32 output; where the registry has no kernel for it the router runs as
   `Logits` (same math, `ops.rs`: "same math as `Matmul`, f32 output"). cpu keeps `Matmul`.
4. **Rows-exact small passes on Metal** (`ink_rows_exact`): an Inkling-2 pass of 2..=16 rows, and a head group of 2..=16 rows,
   runs under `exact_rows` (the row-exact kernels: `matmul_bf16_gemv_rows`, `logits_bf16_gemv_rows`,
   `ink_in_proj_bf16_gemv_rows`, `moe_experts_*_rows`), so token-by-token decode, small chunks and a rolled-back
   verify-sized pass leave the same state and the same logits, bitwise. Larger passes (prefill chunks) take the fast kernels
   (NAX). `INTARSIA_INK_ROWS_EXACT=0` turns it off. The head marking is guarded by `self.ink.is_some()`: no other model sees it.
5. **Host `pos0` for prefill**: already the case (Inkling-2 refuses device-position passes), so the NAX prefill attention is
   taken for every chunk above 16 rows (`rel_sink_attn_bf16_nax` in the plan; no fallback to split-KV).
6. **NoKernel refusals**: Metal never had an explicit Inkling refusal; the load-time plan check now resolves every op (plan
   below). The tiny-test assertion "Metal registers none of the Inkling ops" is flipped to "registers every one", and the cpu
   NoKernel test no longer expects a refusal for `InkAttnDecode` (optional: the pair covers it).
7. Kept refused: batched / packed decode, TP, MTP / DSpark, the prefix store, device-position passes.
8. **Decode graphs / command-buffer replay**: not available on Metal for any model. `MetalBackend` does not override
   `Backend::supports_graphs` (false), so `graph_step.rs` never records on Metal; `crates/intarsia-backend-metal/src/decode.rs`
   holds an indirect-command-buffer spike over synthetic Qwen3-8B GEMVs (examples `icb_check`, `gemv_bw`), not wired to the
   model. `generate` reports `graph replays=0`. Metal decode is eager: launches are encoded into command buffers committed
   every `INTARSIA_METAL_COMMIT_EVERY` (default 128) dispatches. Enabling replay for Inkling needs (a) a Metal graph backend
   (ICB recording of arbitrary compute pipelines, or encoder reuse) and (b) device-position variants of every
   position-dependent Inkling op (InkKvPrologue / RelSinkAttn / InkAttnDecode take `OpCall::pos` already; ConvMix and the
   conv rings read `pos0`), plus removing the Inkling refusal of device-position passes. That is not a small step, so it is
   left to a later lane; the commit cadence A/B is below.
9. Tests: `crates/intarsia-models/tests/inkling2.rs` takes `INTARSIA_INKLING2_BACKEND=metal`; the test wrapper backend now
   forwards the Metal scheduling hints and `wrap_host` (without it the uploaded weights took another GEMV variant than the
   zero-copy ones, a 5e-8 logits difference). docs/tuning.md rows for the new variables and for the m-* lanes' knobs that
   `docs_env_vars` flagged (INTARSIA_INK_MOE, _MOE_NAX, _SPLIT_KEYS, _NAX_SG, _MIX_NE, _PROJ_CFG / _P / _RB / _GRID).

`plan show --model ~/models/Qwen3-8B --backend metal`: byte-identical stdout and stderr, base (b5e283927, built with
`--features metal`) vs this branch.

## Tests

| Suite | Where / job | Tree | Result |
|---|---|---|---|
| `cargo test --release --no-fail-fast -p intarsia-core -p intarsia-backend-cpu -p intarsia-cli` | spark2 #10868 (`git archive` of 41c93f814) | 41c93f814 | 40 binaries, 0 failed |
| the same plus `-p intarsia-bench`, rerun on the merged tip | spark2 #11053 | 8546d143f (m-mix 52b337aa9 + ink/batch) | 637 passed, 1 failed: `docs_env_vars` (ink/batch's `INTARSIA_METAL_ROWS_SINGLE` had no row); row added at 9782778c1, `docs_env_vars` then 2 / 2 |
| `cargo test --release --no-fail-fast -p intarsia-backend-metal` | endor #10867 | f55bfe8b7 (m-mix 47570539e merged) | 46 binaries, 213 passed, 0 failed (incl. `every_metal_kernel_has_a_quick_case`, `rows_exact_registrations_are_pinned`) |
| `cargo test -p intarsia-models --test inkling2`, cpu and `INTARSIA_INKLING2_BACKEND=metal` | endor #10715 (both); Metal again in #10890 | #10890: e3f92beff (m-mix 47570539e, ink/serve merged) | 8 / 8 each |
| intarsia-models Metal tests: batch_metal, batch_metal_real, gdn_verify_metal, mtp_store_metal, prefix_store_metal, hybrid_reuse_real, prefix_reuse_real, batch_recurrent | endor #10901 | de2b5c9bc | 1, 3, 7, 2, 1, 2, 6, 15 passed, 0 failed |
| `cargo check --tests -p intarsia-models` on Linux (cfg guard e3f92beff) | spark2 | e3f92beff | 0 errors |
| intarsia-bench (calibration probe fix b9dd0d6b4) | spark2 | b9dd0d6b4 | 25 / 25; the new test fails without the fix |
| `docs_env_vars` | endor #10898 | f55bfe8b7 | 2 / 2 |
| `admission::plan_case_tests` (new SiTU case test) | endor | 761a1561b | 3 / 3 |

Two failures found and fixed on the way: `quick_coverage` (`rel_sink_attn_bf16_nax` had no quick case; a quick-only copy of
the first mid-size NAX case, full sweep unchanged) and `rows_exact_registrations_are_pinned` (m-mix's
`ink_in_proj_bf16_gemv_rows` added to the pin list; its G-B test is `tests/inkmix.rs::in_proj_rows_exact`).

## Gates

Real-model runs on endor through fleetq (#10690, rerun #10721, and rerun on the tip de2b5c9bc in #10901 after the ink/serve, SiTU-admission and m-mix 47570539e merges: the same numbers to the last digit each time), checkpoint
b85c3cfa, W4A16, bf16, `intarsia accuracy --backend metal --max-seq 4096 --golden ... --write-self-golden`; KL21 = the tool's
"KL21 scored" column; confident-row misses from the skel lane's side script (`p1 - p2 >= 0.5` on the golden) over the
self-golden's prefill rows.

| Gate | Golden | KL21 (limit) | confident misses (allowance) | top-1 | cpu skel |
|---|---|---|---|---|---|
| G1(a) | tiny f64 | cpu only (f32 activations; Metal kernels are bf16) | | | pass |
| G1(b) | tiny `ref_dump_bf16_tf`, `tf_check(factor=3)` | 0 failures, all 16 visits; every tensor at about 1/3 of its budget | | | 0 failures |
| G1 state | tiny | chunks 7 / 13 == 16-row passes == prefill 9 + decode, bitwise; rollback by length, reuse at 30, refusal past the margin, reset to 0; interleaved == solo; failed pass resets | | | pass |
| G2 short | oracle W4A16 short | **2.380e-3** (4e-3) | 0 / 363 (1) | 98.95 % | 2.994e-3, 0 |
| G2 long | oracle W4A16 long | **3.418e-3** (6e-3) | 1 / 1206 (2) | 99.17 % | 3.879e-3, 1 |
| G3 short | vLLM short | **8.391e-3** (1.05e-2) | 0 / 367 (1) | 97.89 % | 9.218e-3, 0 |
| G3 long | vLLM long | **1.248e-2** (1.65e-2) | 0 / 1211 (2) | 97.64 % | 1.416e-2, 0 |

Decode mode (teacher-forced decode rows, which exercise the 1-row kernels and the fused attention; reported, not gated):
G2 short 2.589e-3 over 128 rows (cpu: 9.275e-3, one chat_reasoning row at 0.61 nats; Metal's worst row 5.3e-2), G2 long
2.474e-3 over 160 rows. The 4k G3 set was not run on Metal (out of this lane's gate list; cpu passed it).

On the tiny fixture the 40-row single prefill (NAX / prefill kernels) against the 16-row passes free-running is rel l2 0.67,
top-1 12/40: the tiny random net is chaotic under bf16 (spec section i), so that comparison is printed, not gated; the prefill
kernels are gated teacher-forced (G1(b) is one 40-row pass).

Bitwise on real weights (19-token raw prompt, 64 greedy tokens, #10721): `INTARSIA_INK_ATTN_FUSED` unset / `0` / `1` and
`INTARSIA_INK_ROWS_EXACT=0` give the same text (md5 identical). Peak memory footprint of `generate` at `--max-seq 4096`:
16.0 GB (`/usr/bin/time -l`; RSS 27.3 GB counts the mmapped checkpoint).

Greedy text on Metal (W4A16): raw "The capital of France is" -> " Paris.\n\nA. Noun\n\nB. Pronoun\n\nC. Adjective ..." (the
oracle's continuation shape); chat "What is the capital of France?" -> thinking "The user asks a simple factual question:
capital of France. Need answer accurately. France's capital is Paris. ..." then "Paris." (the cpu / oracle text); the
lighthouse prompt continues into a coherent story.

## Speed

**Correction (Fable review, `.worktrees/ink-fable/decode-profile-review.md` section 0):** every `generate` number in the table
below ran with `generate`'s default `--profile-every 16` on Metal (`crates/intarsia-cli/src/common.rs:357`): every 16th
decode step runs as ~419 waited command buffers (77-82 ms). So the 19.8 ms / 50-52 tok/s decode figures, and the "about 20 %
host gap" in my first message, are **profiler-inflated**. Normal steps in the same cb trace are 15.76 ms (about 63.4 tok/s) with
1 us median GPU idle: no real host gap. The kernel-mode per-op shares are inflated 2.8x and unevenly (small kernels 4-7x);
the corrected split is the review's section 0.2 and the table under "Per-op profile" below. Decode numbers from here on use
`--profile-every 0` (serve, and so localeval, already does).

### Engine timing (`intarsia generate`, greedy, fleetq #10730 `--quiet gpu,cpu`, `--profile-every 16`: inflated, see above)

19-token raw prompt, 256 tokens; decode = (tokens - 1) / decode wall time, prefill includes the first-call warm-up.

| Run | prefill | decode tok/s |
|---|---|---:|
| default (`INTARSIA_METAL_COMMIT_EVERY` 128), two runs | 467 / 797 ms | 49.9 / 45.5 |
| `COMMIT_EVERY=512`, two runs | 408 / 416 ms | 48.8 / 51.8 |
| `COMMIT_EVERY=1024`, two runs | 806 / 432 ms | 39.9 / 51.7 |
| `INTARSIA_INK_ATTN_FUSED=0` (pair everywhere) | 405 ms | 50.0 |
| `INTARSIA_INK_ATTN_FUSED=1` | 403 ms | 51.8 |
| 5051-token prompt, chunk 256 (default), 128 tokens | 4144 ms = **1219 tok/s** | **48.3** |
| 5051-token prompt, `--prefill-chunk 2048` | 2274 ms = **2221 tok/s** | 48.7 |

Run-to-run spread is about +-5 % (the two default runs: 49.9 and 45.5), so the commit cadence A/B (128 / 512 / 1024) and the
fused-vs-pair choice are inside the noise at short context; the fused rule saves ~0.35 ms / token by lane ink-m-attn's
harness, which is ~2 %. A larger prefill chunk is a clear prefill win (1.8x at 5k): the recipe default 256 is a candidate
for a later lane (memory: the workspace grows with the chunk; not changed here).

### Traces (raw, `docs/results/2026-11-21-inkling2-metal/`)

| File | Run | What it shows |
|---|---|---|
| `prof_kernel.jsonl` | #10730, `INTARSIA_METAL_PROFILE=kernel`, 33 tokens | op order and labels of a decode step (613 dispatches); durations inflated (one waited command buffer per launch) |
| `prof_cb.jsonl` | #10730, `cb`, 65 tokens, `--profile-every 16` | normal steps = 2 command buffers (16 + 597 dispatches), 15.8 ms; profiled steps 419 buffers (Fable review section 0.1) |
| `prof_cb1.jsonl` | #10756, `cb`, `INTARSIA_METAL_COMMIT_EVERY=1`, 33 tokens | **not per dispatch for normal steps**: with token feedback the step is held and committed as 16 + 597 dispatches regardless of the commit cadence; 29 normal steps, median wall 15.86 ms, GPU busy 15.82 ms (the 3034 one-dispatch buffers are the prefill and the profiled steps) |
| `prof_cb2.jsonl` | #10857, `cb`, `INTARSIA_METAL_COMMIT_EVERY=1 INTARSIA_TOKEN_FEEDBACK=0 --profile-every 0`, 33 tokens | **per-dispatch timeline of unprofiled steps**: every dispatch its own buffer, labelled by the backend; 30 steps x 613 dispatches; GPU busy 16.74 ms / token, span 16.96 ms (the one-dispatch buffers add ~1.5 us each over the normal 15.8 ms step) |

Analysis scripts: `prof.py` (kernel trace per op), `prof_cb1.py` (per-dispatch labels by position).

### Per-op profile of one decode token (short context)

Launches: **417 op calls and 613 Metal dispatches per decode token** (the plan's calls/step: 96 ConvMix, 49 Logits = 48 router
+ unembed, 48 each of InkInProj / wo GEMV / InkAttnDecode / MoeRoute, 44 + 4 experts calls of 2 dispatches each, 26 AttnResMix,
2 LoopBlend + 2 RmsNorm, embedding, argmax). A normal (unprofiled) step is **2 command buffers** (16 + 597 dispatches), wall
15.86 ms, GPU busy 15.82 ms (#10756, 29 steps): no measurable host gap at batch 1 short context. That is **63 tok/s**
of GPU time per token; the 50-52 tok/s `generate` figures above include the `--profile-every 16` steps.

Per-op times: kernel mode (`prof_kernel.jsonl`, one waited buffer per launch) sums to 43.7 ms / token, 2.8x the real step and
not uniformly (small kernels 4-7x). The table below is the Fable review's correction (`.worktrees/ink-fable/decode-profile-review.md`
section 0.2) of the same runs: per-op GPU times of the profiled steps with ~1.8 us of per-buffer cost removed, scaled to the
normal 15.76 ms step. Floor = the term's bytes at 560 GB/s (perf-strategy section 1).

| Component | ms / token | share | floor ms | excess ms | launches / token |
|---|---:|---:|---:|---:|---:|
| Attention projections: InkInProj (48 x 73 us) + wo GEMV (48 x 27 us) | 4.78 | 30 % | 4.63 (2.58 GB) | 0.15 | 96 |
| Experts: 44 NVFP4 SiTU (89 us) + 4 MXFP8 (135 us) | 4.50 | 29 % | 3.28 (1.84 GB) | 1.22 | 96 dispatches |
| Controllers: ConvMix (96), MoeRoute (48), router GEMV (48), AttnResMix (26), LoopBlend (2) | 3.19 | 20 % | 0.25 (0.14 GB) | **2.94** | 220 calls, ~350 dispatches |
| Unembed (922 MB in 1.58 ms = 580 GB/s) | 1.58 | 10 % | 1.65 (0.92 GB) | 0 | 1 |
| Fused attention InkAttnDecode (48 x 32 us) | 1.52 | 10 % | 0.08 | 1.44 | 48 |
| argmax | 0.17 | 1 % | 0 | 0.17 | 1 |
| **Total** | **15.76** | | **9.9** | **5.9** | 613 dispatches |

Measured per dispatch (#10857, `prof_cb2.jsonl`, the backend's own labels, mean of 30 unprofiled steps; the sum is 16.74 ms,
about 1 ms above a normal step from the one-dispatch buffers):

| Kernel | us / token | share | dispatches / token | us / dispatch |
|---|---:|---:|---:|---:|
| moe_experts_situ_bf16 | 4057 | 24.2 % | 88 | 46.1 |
| ink_in_proj_bf16_gemv | 3579 | 21.4 % | 48 | 74.6 |
| conv_mix_bf16 | 2715 | 16.2 % | 234 | 11.6 |
| ink_attn_decode_bf16 | 1669 | 10.0 % | 48 | 34.8 |
| logits_bf16_gemv (unembed) | 1594 | 9.5 % | 1 | 1594 |
| matmul_bf16_gemv (wo) | 1354 | 8.1 % | 48 | 28.2 |
| moe_experts_mxfp8_bf16 | 549 | 3.3 % | 8 | 68.6 |
| moe_route_f32 | 501 | 3.0 % | 48 | 10.4 |
| attn_res_mix_bf16 | 231 | 1.4 % | 26 | 8.9 |
| logits_bf16_gemv (router) | 226 | 1.3 % | 48 | 4.7 |
| argmax_f32 | 174 | 1.0 % | 1 | 174 |
| loop_blend, rmsnorm, embedding, other | 87 | 0.5 % | 15 | |

It confirms the corrected split above: projections 4.93 ms (floor 4.63), experts 4.61 ms (floor 3.28), controllers
(ConvMix + MoeRoute + router + AttnResMix + LoopBlend) 3.74 ms (floor 0.25, 364 of the 613 dispatches), attention 1.67 ms,
unembed 1.59 ms (floor 1.65).

### Ranked costs (what to attack, by ms over the floor)

1. **Controllers, 2.9 ms over a 0.25 ms floor.** 96 ConvMix (46 blended at ~38 us, 4 dispatches each; 50 plain at ~16 us),
   48 MoeRoute (~10 us), 48 router GEMVs (~4 us), 26 AttnResMix. Latency-bound launches with the bus idle: fusion (lane ink-dfuse).
2. **Fused attention, 1.4 ms over floor** at short context: 48 x 32 us for a few KB of KV each. Fewer, wider threadgroups at
   short ctx, or folding attention into the controller fusion.
3. **Experts, 1.2 ms over floor**: per-call fixed cost (2 dispatches, union grouping) at 89 us vs ~65 us of bytes.
4. **Attention projections, at the floor** (0.15 ms over); the lever there is fewer bytes (the review's section 0.3: lossless
   12-bit bf16 packing), not kernel speed.
5. **Unembed, at the floor** (580 GB/s); argmax 0.17 ms in one launch (a single-threadgroup tail; small).

Host-side: no gap at batch 1 in normal steps; the profiler steps (every 16th, `generate` only) and the eager encode are lane
ink-hostgap's (graph replay / ICB is blocked as described in Changes item 8).

### Against the other engines (localeval speed-plain, decode tok/s / marginal prefill tok/s)

| prompt tok | **intarsia Metal bf16 W4A16** (endor, #10890) | oMLX oQ4e + DSpark (endor) | llama.cpp Q8_0 Metal (endor) | vLLM (spark2, GB10) |
|---:|---:|---:|---:|---:|
| ~145 | **63.4** / - | 50.7 / - | 33.0 / - | 36.8 / - |
| ~1020 | **61.5** / 1689 | 47.7 / - | 32.9 / 122 | 38.4 / 4018 |
| ~4110 | **60.0** / 1573 | 41.1 / 1368 | 32.8 / 126 | 42.8 / 2772 |
| ~16455 | **56.7** / 1406 | 44.2 / 1270 | 31.9 / 122 | 38.7 / 2902 |

Runs: ours 20261121-231510-adhoc-endor-intarsia-metal-ink-metal; oMLX 20261120-222536-endor-inkling-omlx-none-endor-omlx-oq4e-dspark
(its decode includes DSpark speculation; ours is spec-off); llama.cpp 20261120-232156-endor-inkling-llama-none-endor-llama-metal-q8;
vLLM 20261120-192706-spark2-inkling-none-quick-spec-effnone (another machine, for reference). On endor this is 1.25-1.46x
oMLX's decode and 1.1x its prefill at 4k-16k, and 1.8-1.9x llama.cpp's decode; vLLM on the GB10 prefills 1.8-2x faster.
Against the byte floor (102 tok/s at 560 GB/s), 63 tok/s is 62 %.

### localeval (`--tier adhoc --cells speed-plain,conc-plain-1k`, #10890)

`intarsia serve --backend metal --max-batch 1 --max-seq 20480`, run 20261121-231510-adhoc-endor-intarsia-metal-ink-metal in
/opt/code/deepeval/results (unanimis d9825440). Binary: f55bfe8b7's code (see Briefs). localeval ran inside #10890's slot: it
detects the inherited `FLEETQ_JOB` and does not queue a second one (`localeval/cli.py:296`).

| prompt tok | TTFT s | marginal prefill tok/s | decode tok/s | vs baseline 62.5 / 62.0 / 60.6 / 57.5 |
|---:|---:|---:|---:|---:|
| 146 | 0.15 | - | 63.4 | +1.4 % |
| 1019 | 0.67 | 1689 | 61.5 | -0.8 % |
| 4111 | 2.63 | 1573 | 60.0 | -1.0 % |
| 16455 | 11.41 | 1406 | 56.7 | **-1.4 %** |

The 16k row is 1.4 % under the baseline (other lanes built on endor during the run); the coordinator ruled it within
run-to-run noise (16k: 57.5 / 57.4 / 56.7 over three spec-off runs) and cancelled the rerun #11023. These numbers stand.

conc-plain-1k: **not supported** (batched decode is refused for Inkling-2; at `--max-batch 1` the server serializes the
streams): aggregate 50.8 / 51.6 / 50.9 tok/s at 1 / 4 / 16 streams, per-stream 50.8 / 12.9 / 3.2, TTFT 0.86 / 15.5 / 75.9 s.

`intarsia bench speed` against the same server failed at calibration ("no tokens received", `crates/intarsia-bench/src/speed.rs:324`);
the plain chat request and localeval's streaming requests work, so it is the bench's calibration call, not the server.
Cause and fix (b9dd0d6b4): the bench sizes prompts with 1-token probes; Inkling-2's first generated token is a block
marker (`<|content_thinking|>`) the server's text stream drops, so a 1-token request streams only usage, and the bench
discarded that usage. A probe with usage and no text now counts (measured calls still need text). Test
`speed::tests::calibration_takes_a_probe_whose_first_token_releases_no_text` (spark2: intarsia-bench 25 / 25; fails without
the fix). Not rerun against the real server here.

Repeated prompt (prefix reuse path): two identical greedy chat requests gave identical choices.

### `generate` at `--profile-every 0` (#10890, after localeval; the honest engine decode number)

| Run | prefill | decode tok/s |
|---|---|---:|
| 19-token raw prompt, 256 tokens, three runs | 283-286 ms | **64.3 / 64.0 / 62.5** |
| 5051-token prompt, chunk 256 (default), 128 tokens | 3655 ms = **1382 tok/s** | 59.7 |
| 5051-token prompt, `--prefill-chunk 2048` | 2209 ms = **2286 tok/s** | 60.1 |

ink-hostgap's measurement A in the same slot (`.worktrees/ink-hostgap/.scratch/mA/summary.txt`, copy in `.scratch/`): greedy
63.9 / 64.1 tok/s (md5 identical), `INTARSIA_TOKEN_FEEDBACK=0` 60.9 (same md5), temperature 0.7 55.8; cb trace of 125 greedy
steps: 2 buffers / 613 dispatches per step, span 15.55 ms median, GPU idle 1 us per step.

ink/serve's smoke on Metal (`scripts/inkling2-serve-smoke.sh` with `--backend metal`, #10890): every chat / reasoning / tool /
json_schema case answered correctly (effort none / high, multi-turn system, tool_call, tool result follow-up,
tool_choice required, json_schema and streamed json_schema); localeval quality smoke gsm8k 2 / 2; ifeval 2 replies truncated
at the smoke server's default 4096 context (4018 output tokens of thinking), not a correctness signal. G1 tiny on Metal
rerun in the same slot: 8 / 8.

### Tip check (#10901, tree de2b5c9bc)

G2 / G3 short and long identical to #10721 (above); `kernels test --backend metal` for all 11 Inkling ops: every kernel
ADMITTED (18 kernels, including m-mix's widened `ink_in_proj_bf16_gemv` at 18 cases); the intarsia-models Metal tests
(batch_metal 1, batch_metal_real 3, gdn_verify_metal 7, mtp_store_metal 2, prefix_store_metal 1, hybrid_reuse_real 2,
prefix_reuse_real 6, batch_recurrent 15): all pass; Qwen3-8B `plan show --backend metal`: stdout and stderr byte-identical
to the base b5e283927.

## LOG

Rows for docs/decisions/LOG.md at merge-to-main time (coordinator): P12 default-on, MTP flip, argmax v4t, `--profile-every 0`
default on Metal, chunk 2048 (if it lands), and:

- INTARSIA_INK_HEAD_SCREEN opt-in: md5 gate pending (rerun of #10974)

## Merge gate #11048 (tree 45887198b + docs: m-mix 52b337aa9, ink/batch bbc0bec00)

**FAIL: G2 decode mode** (now gated, coordinator brief). G2 short decode KL21 1.109e-2 (limit 4e-3), long 9.205e-3 (6e-3),
the tool exits 1 on both; carried by chat_reasoning (16 decode rows, 5.808e-2, max 0.433) and chat_system (16 rows, 2.376e-2,
max 0.303), top-1 100 % on both. Before the merges (f55bfe8b7 era, #10721 / #10901): 2.589e-3 / 2.474e-3. G2 prefill 2.719e-3
short / 3.544e-3 long (pass; was 2.380e-3 / 3.418e-3). G3 short 8.069e-3, long 1.227e-2 (pass). Confident misses 0 / 0 / 1 / 0.

Passed: rows_exact 11, inkmix 3, rows_single 2, quick_coverage 1, docs_env_vars 2; kernels test 18 Inkling kernels ADMITTED;
tiny inkling2 8 / 8 and inkling2_batch 6 / 6 on cpu and Metal; real-model `batch_step --ctx 600 --ctx-spread 150`, B 2 / 5 / 16:
batched == solo (asserted), 1.67x / 2.95x / 4.41x. Greedy md5 changed (64 tokens 65cde8f3, was 15d62682; 256 tokens 29ec9e50,
was d64363ea; 63.7 tok/s). Every queued merge is held; bisect #11142.

At B = 1 the teacher-forced decode path (`forward_logits_decode`, engine.rs:984) is `forward_prefill` then one-token
`model.forward`, never `forward_batch`, so no call has `rows`; ink/batch's inkling2.rs changes are no-ops there (batch None:
`pair_rows` true, the same fused rule). It did change solo-path kernels (inkling2_attn.metal, gemv_multirow.metal,
multirow.rs), which the bisect covers.

Per-row decode KL21 against the golden, previous tip (#10901 self-golden) vs 45887198b (#11048), G2 short
(`docs/results/2026-11-21-inkling2-metal/decrows.py`): six of eight prompts are unchanged within noise. chat_system: one
row (pos 57, golden margin 0.77) 0.011 -> 0.303. chat_reasoning: the error **grows across the consecutive teacher-forced
decode rows**: pos 92-104 old 0.007-0.018, new 0.036, 0.023, 0.123, 0.029, 0.131, **0.431** (pos 103, margin 0.06),
0.119. Mean 1.0e-2 -> 5.8e-2. Prefill rows of the same prompts barely move. Growth over successive decode steps points at
state written by the 1-row decode path (conv rings, KV), not at a single near-tie row.
