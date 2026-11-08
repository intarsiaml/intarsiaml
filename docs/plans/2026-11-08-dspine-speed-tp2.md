# DSpine-b16 on HIP: speed levers, tp=2, and work order (2026-11-08)

This document answers two questions about the DSpine draft: can it be made faster, and can it run at tp=2. Both answers are yes.

**Inputs.**
- Three research passes: a cost model, a reading of Tencent's reference, and a tp=2 design. This document reconciles them and does not paste them together.
- I re-read the code at `main 8532f5701`, in the worktree `/opt/code/intarsiaml/.worktrees/dspine-sweep`. All paths below are relative to it.
- Tencent/DSpine @6450af7 is cloned at `/opt/code/intarsiaml/.worktrees/dspine-research/repo`. The owning session should remove that clone when this is done.

**What I did not do.** No GPU job, no commit, and nothing written outside `.worktrees/dspine-research/`.

**Evidence tags used below.**
- **[M]** measured, in a committed doc.
- **[D]** derived by arithmetic from measured numbers.
- **[E]** estimated from bandwidth, launch or op counts.
- **[G]** a guess with no direct evidence, such as an acceptance change.

## 0. Answers

**Can its speed be improved? Yes, mostly by choosing the right arm and the right k. Kernel and host work adds less.**

*Short prompts.*
- Nothing on the draft side can make DSpine win short prompts at tp=1.
- A free draft at k3 would still reach only about 63 tok/s [D]. MTP k2 runs at 60.3 tok/s [M].
- The fix is routing: `--spec auto`, which picks MTP or DSpine per request. Expect about 59-60 tok/s, against 48.7 for DSpine k3 today (+22%).
- This needs no code, but it does need an ids gate, because auto has never run with a DSpine draft (section 2, lever 1).

*4k prompts.*
- **The k3 default leaves 8% unused.** DSpine k7 measures 73.5 tok/s against 68.0 for k3 [M].
- **Host-side candidates, a parallel walk, the verify graph and multi-row draft attention:** together about +7%, reaching about 78-79 tok/s [E].
- **fp8 draft weights:** about +5% more if acceptance holds [E/G].

*Where the time goes.* The verify pass is about 75% of a 4k k7 pass. The row-exact rule ("keep bitwise", user decision 2026-11-06, `docs/results/2026-11-06-dflash-win-split.md:12`) holds that part fixed.

**Can it be adapted for tp=2? Yes. Today it is refused, not missing.**

*Two steps.*
- **Option A, in about half a day:** keep the whole draft on rank 0. Predicted about 104 tok/s at 4k k7 [E].
- **Option B, in 1.5-2 days more:** split the draft over both ranks, reusing DFlash2's split. Predicted about 119 tok/s at 4k k7 [E, ±10%].
- MTP k2 at tp=2 runs at about 95 tok/s [M].

*Short prompts at tp=2.* DSpine stays below MTP on short prompts at tp=2 as well: about 77 against 95.6 tok/s. So auto routing matters there too.

## 1. Where a pass goes (tp=1, HIP, R9700, Quark 27B MXFP4)

### 1.1 Measured arms

Pass time is tokens per pass divided by tok/s. Source: `docs/results/2026-11-07-dspine.md:26-31`. The prompts are 24 tokens (short) and 4420 tokens (4k), with 512 tokens generated and one run each.

| arm | 4k tok/s | 4k t/p | 4k pass ms | short tok/s | short t/p | short pass ms |
|---|---:|---:|---:|---:|---:|---:|
| off | 35.07 | | 28.5 | 35.96 | | 27.8 |
| MTP k2 | 69.06 | 2.54 | 36.8 | **60.33** | 2.12 | 35.1 |
| DFlash2 k7 | 63.80 | 3.57 | 56.0 | 49.08 | 2.48 | 50.5 |
| DSpine k3 (default) | 67.95 | 3.07 | 45.2 | 48.72 | 2.07 | 42.5 |
| DSpine k7 | **73.49** | 4.19 | 57.0 | 41.61 | 2.14 | 51.4 |

**k15 is excluded.** It ran only in window w2, whose first pass carried the table build of about 1 s (`2026-11-07-dspine.md:34-35`).
- Corrected for that: 256 tokens in 6.29 s minus about 0.8-1.0 s, which is about 47-48 tok/s and about 96-100 ms per pass [D].
- Rows 9-16 cost about 5 ms each at tp=1, against about 3 ms for rows 4-8 (section 1.3).
- Do not use k > 7 at tp=1.

### 1.2 The draft/verify split, and the flush double-count corrected

**No HIP trace of the DSpine draft exists.**
- The trace line is at `dspine.rs:786-793`. It prints block, head, logits+cand and stash+walk in ms.
- `INTARSIA_SPEC_TRACE` was never set in the DSpine windows.
- The split below is therefore derived.

**Where the flush sits.**
- The draft's context ingest (the "flush": 5 fc matmuls, plus per-layer wk/wv, QkNorm, Rope and KvAppend) runs inside the verify call, not the draft call.
- `forward_verify` calls `dflash_after_forward` (`model.rs:2027`, `model.rs:2223`). For a verify pass (t > 1) that call ingests right away, because `DflashState.lazy` is `false` (`dflash.rs:387`, `dflash.rs:448-458`).
- The trace timers wrap `self.verify(...)` as "verify + rollback" (`engine.rs:2243-2257`).
- So the measured verify8 of 41.73 ms at 4k and 37.90 ms short (`docs/results/2026-11-06-27b-dflash-hip-tp1.md:39`, trace columns) **already contains the flush**, about 1.4 ms.
- The cost-model input counted the flush in the draft too. Its "draft about 13-15 ms including flush" therefore double-counts about 1.4 ms.

**Cleanest derivation.** Compare DSpine k7 with DFlash2 k7 on the same prompts. Both run 8 verify rows.
- At 4k: 57.0 − 56.0 = +1.0 ms [D]. Short: 51.4 − 50.5 = +0.9 ms [D].
- DFlash2's traced draft is 12.41 ms (4k) and 11.82 ms (short) (`2026-11-06-27b-dflash-hip-tp1.md:38`).
- DFlash2's window is 2048 keys, so its draft barely grows with context. The 4k trace is at 3135 tokens, not 4420.

| k7, tp=1 | 4k (4420 tok) | short |
|---|---:|---:|
| DSpine draft (block + head + candidates + walk), excluding flush [D] | about 13.4 | about 12.7 |
| verify8 + flush + rollback [D: 41.7 at 3135 tok, + about 1.6 ms for 1285 more keys over 8 rows] | about 43.3 | about 37.9 |
| host gaps outside both timers [D: DFlash2 trace vs served] | about 0.3-1.0 | about 0.8 |
| **pass** [M] | **57.0** | **51.4** |

**What this means.**
- The draft is about 23-25% of the pass, or about 26% if the flush is counted with it.
- A free draft would give about 4.19 / 43.6 ms = 96 tok/s at 4k k7 [D].
- A draft at its weight floor (4.94 GB / 640 GB/s = 7.7 ms; the practical floor is about 9 ms at the 505 GB/s the target's plain step reaches) would give about 79-80 tok/s [E].

**Uncertainty.** The subtraction is accurate to about ±1.5 ms.

### 1.3 Marginal cost of a row (tp=1)

**Verify.**
- 4k: (41.73 − 28.3) / 7 = 1.9 ms per row. Short: (37.90 − 27.8) / 7 = 1.45 ms per row [D, from `2026-11-06-27b-dflash-hip-tp1.md:36-39`].
- The probe-based figure is 2.1 ms per row (43.35 vs 28.39 at 3000 tokens). Use the trace values instead, since they were taken on served passes.

**Draft.**
- Total pass change from k3 to k7 [M]: 11.8 ms (4k) and 8.9 ms (short).
- Subtracting the verify rows leaves about 1.0 ms per draft row at 4k and about 0.8 ms at short [D].
- That draft-row cost comes from four places:
  - the `_rows` kernels the draft runs on (`dflash.rs:360`), which are row-exact and cost per row;
  - the head over t−1 rows (`dspine.rs:742`);
  - one `AttnDecode` launch per row per layer (`dspine.rs:648-664`);
  - the host candidate stage, which is per row (`dspine.rs:752-772`).

**Why k7 wins at 4k and k3 wins at short.**
- 4k: going from k3 to k7 buys 1.12 tokens per pass for 11.8 ms. That is 0.095 tok/ms at the margin, above the 0.068 average, so k7 wins.
- Short: the same step buys 0.07 tokens per pass for 8.9 ms, so k3 wins.

**Survival curve.** This is an illustrative fit to the two measured sums, not a measurement:
- 4k: s ≈ .86 .68 .53 .40 .31 .23 .18
- short: s ≈ .60 .31 .16 .04 …

The fit predicts that k5-k7 are flat at 4k, about 72-74 tok/s [G]. A k = 4, 5, 6 sweep settles this in one window.

### 1.4 The draft pass, op by op (k7, t = min(16, k+1) = 8 rows, `dflash.rs:635`)

**Flush (inside verify, see 1.2).**
- `dflash_ingest` (`dflash.rs:500-529`): about 40 launches, 0.37 GB.
- Measured 1.38-1.43 ms for DFlash2, which uses the same path [M].

**Block (`dspine_block`, `dspine.rs:603-698`).** Five Qwen3 layers in bf16, 639.6 MB of weights per layer, 3.20 GB in total.
- Per layer: q/k/v (`:638-640`), QkNorm+Rope and KvAppend, 8 `AttnDecode` calls (one per row, `:648-664`), o through `d2_proj_out` (`:665`), the MLP with down through `d2_proj_out` (`:670-673`), and the code bridge (`:676-685`, about 19 MB per layer).
- About 160 launches in total, roughly 66 of them matmul-class.
- No sync happens inside the block unless the trace is on (`dspine.rs:713-718`).

**Head (`dspine.rs:742`).**
- fp8 over the full vocabulary into f32 `[t−1, 248320]`.
- The DFlash2 equivalent measures 3.22 ms at 7 rows, 1.27 GB at 394 GB/s [M, `2026-11-06-27b-dflash-hip-tp1.md:43`].

**Host tail.**
1. *Candidates (`dspine.rs:752-772`).*
   - A fresh pageable 6.95 MB `Vec`, then a synchronous download.
   - A second f32 `Vec` filled through `par_fill`, which spawns threads each pass (`:170`).
   - Then one spawned thread per row running `candidates` (`:300-307`). Each allocates a bf16-rounded 1 MB key copy and runs `row_topk` over 248320 values.
   - [E] about 1.2-1.6 ms at k7.
   - Anchors: DFlash2's host download plus selection measured 1.18-1.47 ms, and its device top-k path 0.24 ms (lane dh).
2. *Stash (`:774-776`).* Three more synchronous downloads, about 0.1 ms.
3. *Walk (`walk` `:349`, `best_lane` `:312`).*
   - It is serial over rows: per row, 16 candidate `head.dot`s over 5120 fp8-LUT columns plus 512-wide TG dots.
   - [E] about 85 µs per row, so about 0.6 ms at k7.

**Exactness.** Every emitted token is a verify-row argmax (`dspine.rs:22`). Draft-side changes move only acceptance, never output.

## 2. Speed levers, ranked

**How to read the table.**
- **Baselines** are 4k k7 at 57.0 ms (73.5 tok/s) and short k3 at 42.5 ms (48.7 tok/s), the best fixed arm for each prompt length.
- **Gains** are per pass and refer to those baselines unless the row says "vs default k3".
- **Exactness risk** means risk to greedy ids. Every draft-side lever has none, because verify decides each token.
- **Effort** is in hours of coding. "Window" means a GPU measurement slot.

| rank | lever | 4k gain | short gain | effort | exactness risk | evidence |
|---:|---|---|---|---|---|---|
| 1 | Route per request: `--spec auto` with the DSpine draft | about −1 to −3% vs DSpine k7 (+5% vs MTP); +6% vs default k3 | **+22% vs DSpine k3** (about 59-60 tok/s) | 0 h to try; ids gate + cats window; 2-4 h for DSpine priors | none (verify-exact), but **no ids gate yet** for auto + DSpine | [M] au1/ag1 auto reaches 0.965-0.99 of the best fixed arm (DFlash2); [D] for DSpine |
| 2 | k = 7 for long prompts under `--spec dflash` (prompt-length default; later a spine-only depth controller) | **+8.2% vs default k3**; 0 vs best fixed k | 0 (k3 already best) | 2-4 h (length default); 6-8 h (controller) | none | [M] the arms in 1.1; threshold [G] until a category sweep |
| 3 | Device candidates, exact bf16 keys, plus a parallel walk (supersedes host-stage tuning) | +2.5-3% (−1.4 to −1.7 ms) | +1.2-1.5% | 6-8 h | none for ids; draft parity checkable with a `check` mode | [M] DFlash2 host 1.18-1.47 ms vs device 0.24 ms; [E] the rest |
| 4 | `INTARSIA_VERIFY_GRAPH=1` (exists, default off) | +0.2-2% | +0.2-2% | 0 h, one window arm | none: bitwise the eager pass (`docs/tuning.md:244`) | [M] on DFlash2 verify, lane dfh window 3 |
| 5 | fp8 e4m3 draft linears (fc, q/k/v/o, gate/up/down; bridge stays bf16) | +4.5-5.5% pass at flat t/p | +6-7% pass at flat t/p | 8-12 h + A/B window | none for ids; **acceptance [G]** | [E] bandwidth; break-even t/p loss equals the pass gain |
| 6 | Multi-row causal draft attention (5 launches instead of 40 at k7, one KV read per layer) | +1.5-3%; grows toward 8k context | +0.3-0.5% | 8-14 h | none (draft numerics may change bits, per the win-split precedent) | [M] DFlash2's 40 windowed calls cost 1.67 ms at 2048 keys; [E] at 4420 keys |
| 7 | Acceptance investigation (f32 or single-GEMM fc partial sums; real-weight parity against MLX) | unknown, 0 to +10% t/p [G] | unknown [G] | 6-10 h, mostly offline | none | [M] rel-L2 2e-2 vs the 2e-3 design target; real-weight parity not run (`2026-11-07-dspine.md:21-22`) |
| 8 | DSpark-style confidence-truncated verify (draft at k_max, verify up to the first low-confidence row) | 0 to +4% vs best fixed k [G]; oracle bound about +14% | 0 to +2% [G] | 12-20 h + calibration | none (capped accept) | Tencent uses it only at batch 8 (`repo/sglang_dspine/verification/ragged.py:121-128`); no measured gain in their repo |
| 9 | Cheaper recurrent row saves in verify (ReplaySSM-style), for all spec modes | 2-4 ms per verify8 [G] (+4-7%) | same | high (new GDN kernel) | **medium**: must stay bitwise | Estimate only (`docs/results/2026-10-26-dfh-hip-dflash.md:166`) |

**Rejected or deferred.**
- **A device tail** (TG/TSV on the device plus a walk kernel): only about 0.3-0.5 ms beyond lever 3, for 20-30 h and +0.76 GB VRAM. Its value is as a precondition for one sync per round.
- **HIP-graphing the draft:** at most about 0.5 ms. The tp=2 MTP capture measured 0.98x, no gain (`docs/results/2026-11-07-spec-graph-tp2.md`).
- **Bridge fusion:** under 0.25 ms. MLX measured no end-to-end gain from the same idea (`repo/dspine_mlx/dspine_mlx/sglang/kernels/config.py:21`).
- **MXFP4 draft weights:** they reach the ALU-bound W4A8 path at M=8, and fp8 activations would feed the bridge, the most rounding-sensitive part.
- **A reduced-vocab draft head:** saves about 2.5 ms but costs rare-token drafts, so the net is about zero [G].
- **Tencent's FP8 coarse-retrieve head:** our head is already fp8.
- **Their sparse accept:** sampled only, H100 only.

### 2.1 Lever details

**Lever 1: auto routing.**
- *How auto treats a DSpine draft.*
  - `--spec auto` loads the DFlash-family draft, DSpine included (`crates/intarsia-models/src/load.rs:791`).
  - The k3 DSpine default applies only to `--spec dflash` (`load.rs:814`). Under auto the DFlash passes run k7 (`AUTO_DFLASH_K_DEFAULT`, `spec.rs:48`, applied at `crates/intarsia-cli/src/common.rs:365-366`). That is the right k at 4k.
- *Entry.*
  - Entry needs MTP's tokens per pass at least `theta = 2.5` (`spec.rs:1046`, rule at `spec.rs:1230`).
  - Short prompts: MTP measures 2.12, so auto stays on MTP. That is where the +22% comes from.
  - 4k: MTP measures 2.54, so entry is marginal. If auto does not enter DSpine at 4k, it gives 69 rather than 73.5 tok/s.
  - A DSpine-specific theta or prior is the 2-4 h follow-up.
- *Priors.* The tp=1 priors are DFlash2's, `(2.2, 35.4), (3.6, 50.2)` (`spec.rs:1069`). DSpine k7 measured `(4.19, 57.0)` at 4k and `(2.14, 51.4)` short. The rate ranking is the same, so the priors are usable but should be refreshed.
- *Gate.* The port plan deliberately left auto untouched (`docs/plans/2026-11-07-dspine-port.md:197`). The ag1 ids gate covered auto with DFlash2 only. **Auto plus DSpine needs its own ids gate before anyone relies on it.**

**Lever 2: k by prompt length, then a spine-only depth controller.**
- *Why the existing `Depth` does not fit.*
  - `spec::Depth` (`spec.rs:312`) is wired for MTP only: `engine.rs:1449` and `engine.rs:2036` gate on `spec_mtp && !spec_dflash`.
  - Its climb rule needs a streak of full accepts. P(all 3 accepted at 4k) ≈ 0.53 [G, from the fit], so it would seldom climb from 3 to 7.
- *What to build.*
  - Cheapest: set the default `--spec-k` for `--spec dflash` from the prompt length at `load.rs:814-821`, for example k7 at 1-2k tokens and up. The threshold comes from a category sweep. Per-request granularity needs the choice to move to request time.
  - Better: a controller fed by an EWMA of per-row survival. It adds row i while ŝ_i > (tok/ms) x (row cost) and drops it otherwise.
- *Why it is safe.*
  - The draft rows follow k for free (`dflash.rs:635`), and the dry run admits every row count (`dflash.rs:1018-1024`).
  - A shorter k is a prefix of a longer one (causal block, `dspine.rs:23`), so ids are unaffected.

**Lever 3: device candidates with exact bf16 keys.**
- *Today.* The device top-k is off for a spine (`dflash2.rs:402`) because the tail ranks candidates by bf16-rounded logits, with the lower id winning ties (`dspine.rs:298-307`). The HIP kernel ranks by f32 (`kernels/hip/topk.hip:1-10`; ties by ascending index, with −0.0 and +0.0 tied).
- *An exact scheme with no kernel change.*
  1. Ask the device for the f32 top-K' with K' = 32. `cand <= 64` is already admitted (`dflash2.rs:402`).
  2. Re-rank those 32 on the host by bf16 key, lower id first.
  3. bf16 rounding is monotone, so any element outside the top-K' has a bf16 key no greater than bf(v_K').
  4. If bf(v_K') is below the 16th re-ranked key, the set is exactly the host's.
  5. If they are equal, a tie group may straddle the cut. Download that one row (about 1 MB) and run the host `candidates`.
  6. NaN-flagged rows take the same fallback.
- *Fallback rate.* Unknown [G], expected rare. A `check` mode like DFlash2's (`dflash.rs:888-897`) measures it and proves parity.
- *Parallel walk.* Compute the 7x16 `base` dots in parallel. Only the m chain stays serial (`best_lane` `:312-344`). `HeadRows::dot`'s summation order is unchanged, so the tail's bits are unchanged.
- *What it supersedes.* This lever supersedes tuning the host stage alone (reusing the buffer, dropping per-row spawns), which would be worth only about 0.7-1.2 ms.
- *At tp=2.* The same scheme composes with a shard merge (section 3.3).

**Lever 5: fp8 draft linears.**

The inputs disagreed on this one: "break-even", rank 2 and rank 3. Here is the reconciliation.
- *Bytes saved.* The layers plus fc are about 3.46 GB in bf16 and halve to about 1.73 GB.
- *Time saved.* At about 500 GB/s effective in bf16 and about 400 GB/s in fp8 (the fp8 head runs at 394 GB/s [M]), that is 6.9 − 4.3 ≈ 2.6 ms, or about 2.5-3 ms [E].
- *Gain.* That is +4.5-5.5% at 4k k7 and +6-7% at short k3, if tokens per pass stay flat.
- *Break-even.* A tokens-per-pass loss of the same size cancels it.
- *Acceptance risk.* Ours would be per-tensor e4m3 (`quant/head.rs:101-104` makes one whole-tensor scale). That is coarser than Tencent's affine int8 with group size 64 (`repo/dspine_mlx/dspine_mlx/model.py:268-292`). Tencent publishes no acceptance number for its quantised draft.
- *Ranking.* Expected value is about +2-3% (gain x P(flat), [G]). That puts it below levers 1-3, which are surer and cheaper, and above lever 6.
- *Gate.* An acceptance A/B over the category files. Proceed only if the tokens-per-pass loss is under 2%.
- *Implementation.* `dflash::lin` demands bf16 today (`dflash.rs:148-155`). The fp8 rows kernels and `quant::head::quantize_host` exist.

### 2.2 Overlaps and conflicts

- **Levers 1 and 2.**
  - Auto picks the drafter. Depth picks k inside DSpine.
  - Auto's EWMA would see a mix of k values once a depth controller runs, and its entry and exit thresholds were tuned for fixed k7.
  - So refresh auto's DSpine prior first, and gate auto plus the controller together.
  - Under auto the k3-vs-k7 problem mostly disappears already, because auto uses k7 and sends short prompts to MTP.
- **Levers 2 and 8 compete for the same rows.**
  - Lever 2 shrinks draft and verify rows together: about 1 ms plus 1.9 ms per row saved at 4k.
  - Lever 8 keeps the draft at k_max, paying about 1 ms per draft row, and cuts only verify rows per pass.
  - After lever 2, lever 8 earns only the within-prompt variance of accepted length.
  - Build lever 8 only if the lever-0 trace shows that variance is large. Its confidence term is cheap to compute in `best_lane` (Tencent's `conf = 1 / Σ exp((s_b − best) / max(rms, 1e-6))`, `repo/sglang_dspine/kernels/dspine.py:280-283`). The calibration table must be re-collected on our data, because theirs sits about 73% in the top bin (`repo/sglang_dspine/verification/calibration.json`).
- **Lever 3 and the device tail.** The device tail subsumes lever 3. Lever 3 first: it is cheaper and gets most of the value.
- **Levers 3 and 6.** These are independent and touch different parts of the draft. Lever 3 is the host tail. Lever 6 is the attention kernel.
- **Lever 5 and tp=2 option B.** Both change `load2`'s weight loading (`lin_part`, `dflash2.rs:229-265`). Sequence them. fp8 shards need either a whole-tensor scale (as `quantize_host` already does) or a per-shard scale. A per-shard scale is fine for a draft, which need not be bitwise.
- **Lever 4 and tp=2.** `INTARSIA_VERIFY_GRAPH` is one-rank only (`spec_graph.rs:570`), so it gives nothing at tp=2.
- **Combined tp=1 estimate** at 4k k7, levers 3 + 4 + 6:
  - 57.0 − 1.5 − 0.5 − 1.5 ≈ 53.5 ms, which is **78-79 tok/s (+7%)** [E].
  - Adding lever 5 at flat tokens per pass: about 50.8 ms, **about 82 tok/s (+12%)** [E/G].
  - Short prompts: **about 60 tok/s via lever 1**.

### 2.3 Metal note

- The DSpine draft pass on Metal is about 21-27 ms at 8 rows (block 13-15 ms, head 6-7.5 ms). The host tail is about 1 ms.
- Metal has no top-k kernel, and `supports_topk` is HIP only (`crates/intarsia-backend-hip/src/device.rs:1559`). So lever 3 needs a new Metal kernel and is worth at most about 1 ms there.
- On Metal, levers 5 and 6 matter more than on HIP. They are not in scope for this HIP plan.

## 3. tp=2 recommendation

### 3.1 What blocks it

There is no missing kernel. There are three guards:
- `crates/intarsia-core/src/model.rs:1484-1486` refuses `--spec dflash` with a DSpine draft when `tp.world > 1`.
- `model.rs:1471` gates the split-draft decision `dflash_tp` on `d.config.spine.is_none()`.
- `model/dflash2.rs:282-284`: `load2` refuses a spine when `world > 1 || shard.is_some()`.
- `load.rs` has no DSpine tp guard of its own. It only requires the draft on rank 0 (`load.rs:905-906`).

Tencent's reference is tp=1 only (`repo/sglang_dspine/spec.py:68-69`, `repo/sglang_dspine/runtime/worker.py:502-503`), so there is nothing upstream to copy.

### 3.2 Option A: the whole draft on rank 0 (first milestone and permanent fallback)

**Changes.**
1. Delete the refusal at `model.rs:1484-1486`.
   - `dflash_tp` is already false for a spine (`model.rs:1471`), so peers get no draft (`model.rs:1487`).
   - Under a vocab-parallel head, rank 0 gets a whole `full_head` copy (`model.rs:1515-1517`).
2. Add `fn dspine_head(&self) -> Result<&QTensor>` in `dspine.rs`. It returns `self.dflash.full_head` when present and `self.head()` otherwise.
   - Use it at `dspine.rs:489` (table build). Today the `[v, h]` shape check at `:493-494` rejects a vocab shard.
   - Use it at `dspine.rs:742` (pass-1 head). Today it would produce only half the logits.
   - Use it in `dspine_dry_run_tables` (`dspine.rs:592`).
   - DFlash2 already does the same thing at `dflash.rs:695-698`.
3. `dflash2.rs:282` is unchanged, since rank 0 loads the draft unsharded.
4. Tests: a tp=2 CPU case in `crates/intarsia-models/tests/dspine_cpu.rs`, modelled on `dflash2_cpu.rs:780` (split drafts equal the whole draft) and `:816` (greedy ids equal `--spec off`). Check that drafts at tp=2 equal tp=1 and that greedy ids equal tp=2 `--spec off`.
5. `docs/tuning.md`: note that `INTARSIA_DFLASH_TP` leaves a DSpine draft on rank 0 until option B lands.

**Memory on rank 0.**
- About 3.3 GiB of draft (`docs/plans/2026-11-07-dspine-port.md`, §3).
- +1.27 GB device for the fp8 `full_head`.
- About 1.9 GiB of host tables and head rows, built once by `dspine_prepare` (`load.rs:973-976`).
- KV capacity follows the smallest rank. Check it with `plan show`.

**Effort.** 4-6 h, plus one 2-card window.

### 3.3 Option B: split draft (the target)

**Changes.**
1. **Shard the layers as DFlash2 does.**
   - Drop the spine guards at `dflash2.rs:282-284` and `model.rs:1471`. The divisibility test at `model.rs:1475-1478` then decides: 8 kv heads and ffn 17408 both divide by 2. Option A stays as the fallback.
   - `lin_part` shards q/k/v/gate/up by rows and o/down by columns (`dflash2.rs:229-265`).
   - `dspine_block` already reaches the two all-reduces through `d2_proj_out` (`dspine.rs:665`, `:673`; helper at `dflash2.rs:492-507`). That makes 10 per pass, as for DFlash2.
2. **Replicate the bridge.**
   - The residual is whole after each all-reduce, and the bridge reads about 19 MB per layer. So it runs redundantly on each rank with no new collective.
   - `load_spine` (`dspine.rs:191`) runs per rank.
   - Guard peers out of `dspine_tabs` (`dspine.rs:479`) and `dspine_prepare` (`dspine.rs:466`). Otherwise each peer silently builds 2 GiB of host tables, taking about 1 s.
3. **Anchor row `s0`.** It is uploaded from rank 0's host tables (`dspine.rs:720-725`). Upload the 2 KB row into each peer's `sp.bufs.s0` before the peer threads start in `dflash_draft_tp` (`dflash.rs:542`).
4. **Restructure the draft path.**
   - The spine branch returns early at `dflash.rs:654-668`, before the `link`/`shard` logic at `dflash.rs:679-688`.
   - Make it honour `DraftLink2` (`dflash.rs:366`):
     - a Follow rank sends its shard's candidates and returns;
     - the Lead rank merges, downloads its own stash (`dspine.rs:774-776`) and runs `walk` unchanged.
5. **Head and candidates.**
   - Each rank projects through its own vocab shard, so there is no `full_head`; this is the existing rule at `model.rs:1515`.
   - Each rank produces its shard's candidates, either with the K'=32 device scheme of lever 3 or on the host, offset by `rank x V/2`.
   - `merge_topk` (`dflash2.rs:107-113`) needs no change when fed `(bf16 key, id)` pairs. It is a stable descending sort, and shard order is id order, so the lower id wins ties.
   - Two additions:
     - canonicalise −0.0 to +0.0 first, because `total_cmp` orders them while `row_topk` (`dflash2.rs:88-100`) and `topk.hip` tie them;
     - sort the final 16 ids ascending for lane order (`dspine.rs:304-305`).
   - With the device scheme, the boundary check from lever 3 is applied after the global merge.
6. **Head rows for the walk.**
   - Rank 0 concatenates both shards into one host `HeadRows` at table-build time.
   - The fp8 scale is the whole tensor's amax (`quant/head.rs:101-103`), so one LUT is valid for both halves and no device copy is needed.
7. **Dry run.** Make the table-build recording (`dspine_dry_run_tables`, `dspine.rs:592`) rank 0 only.

**Memory per rank.** About 1.9 GiB of draft per rank, against about 4.6 GiB on rank 0 for option A. Host tables stay on rank 0 only.

**Effort.** About 300-350 lines across `dflash2.rs`, `dspine.rs`, `dflash.rs` and `model.rs`. 1.5-2 days, plus one 2-card window.

### 3.4 Predicted tp=2 speed [E, ±10%]

**Method.**
- Start from DFlash2's measured tp=2 k7 passes: rank-0-only with win-split, 36.64 / 39.34 ms; split with win-split, 31.48 / 33.94 ms (short / 4k) [M, `docs/results/2026-11-06-dflash-win-split.md:73-82`].
- Add DSpine's +0.9 / +1.0 ms over DFlash2 (section 1.2), plus 0.2 ms of split overhead for B.
- For k3, take off 4 verify rows at 1.04 ms each [D: (25.2 − 17.9) / 7], plus 4 draft rows: about 1.0 ms per row for A (the whole tp=1 draft) and about 0.55 ms per row for B (split).
- Tokens per pass use the tp=1 values. The DFlash2 split moved them by up to 0.06.

**This corrects the tp=2 input's k3 rows.** That input subtracted only 0.3 ms of draft, which contradicts the tp=1 arms (about 1 ms per draft row, section 1.3).

| tp=2 | A pass ms | A tok/s | B pass ms | B tok/s |
|---|---:|---:|---:|---:|
| 4k k7 | 40.3 | **104** | 35.1 | **119** |
| 4k k3 | 32.1 | 96 | 28.7 | 107 |
| short k7 | 37.5 | 57 | 32.6 | 66 |
| short k3 | 30.3 | 68 | 26.8 | 77 |

**References at tp=2.**
- MTP k2: 95.6 tok/s short and 92.8-95.2 tok/s at 4k [M, `2026-11-06-27b-dflash-hip.md:17-25`, `2026-11-07-spec-graph-tp2.md`].
- DFlash2 split k7 at 4k: about 105 tok/s on DSpine's prompt (3.57 t/p at 33.94 ms) [D].
- Plain step: 17.9 ms [M].

**Reading the table.**
- At 4k, B at k7 is about +25% over MTP and +8-13% over DFlash2. A at k7 is about +9% over MTP.
- **On short prompts, every DSpine arm loses to MTP.**
- k3 at 4k is roughly on par with MTP under A and only +12% under B.
- So at tp=2, run DSpine at k7 behind auto. Never use the k3 default there.
- In auto's tp=2 priors (`spec.rs:1054`, `(2.2, 25.5), (3.6, 38.0)`), the second pair (the DFlash side) undervalues DSpine B (4.19 at about 35 ms). Refresh it.

**Measure at tp=2 before trusting these.**
- **The 8-row verify.** `spec.rs:36-37` records an 8-row GEMV shard-alignment cliff for MTP. DFlash2 k7 at tp=2 ran 8 rows at 25.2 ms without it, but include a k6 arm.
- **Tokens per pass.** Whether tp=2 matches tp=1 is unmeasured. The split o/down change the draft's rounding.
- **The draft share.** At tp=2 the draft is a larger share of the pass (about 9-10 of 35 ms under B), so lever 3 is worth about 3-4% there.

### 3.5 Recommendation

1. Ship A first. It is the only tp=2 path for a draft that does not divide, and it is half a day.
2. Then build B as the default under `INTARSIA_DFLASH_TP`, as for DFlash2.

**Pre-registered gates.**
- CPU split drafts equal the whole draft, within the `dflash2_cpu.rs:780` tolerance.
- Greedy ids at tp=2 equal `--spec off`, for A and B, k3 and k7, short and 4k, at 256 tokens.
- `plan show` is unchanged for existing configs with the knob unset.
- Speed: B ≥ 1.08x A at 4k k7, and A ≥ MTP k2 at 4k k7.
- Log rank 0's minimum free VRAM for A.

## 4. Order of work (PR-sized lanes)

Effort is coding hours. Every GPU window goes through fleetq on tatooine, with `--takes-down radiance` for the 2-card windows.

| lane | content | files touched | effort | depends on | gate |
|---|---|---|---|---|---|
| **L0 measure** (no code) | Traced `generate` (`INTARSIA_SPEC_TRACE=1`) for DSpine k3/k7, short/4k, plus a `INTARSIA_VERIFY_GRAPH=1` arm, a k4/5/6 sweep at 4k, `--spec auto --spec-draft <DSpine>` over `scripts/dfh/cats/` with an ids check against `--spec off` | none | 0 h; one window of about 40 min (tp=1, card 0) | | Turns sections 1.2-1.3 into [M]; settles levers 1, 2, 4 and how much lever 8 could earn |
| **L1 tp2-A** | Section 3.2 | `model.rs` (1484-1486), `model/dspine.rs` (`dspine_head`, :489, :592, :742), `tests/dspine_cpu.rs`, `docs/tuning.md` | 4-6 h | | CPU tp=2 test; 2-card ids + speed window |
| **L2 device candidates** | Lever 3: K'=32 device top-k for a spine, bf16 re-rank and boundary fallback, `check` mode, parallel `base` dots | `model/dflash2.rs` (398-402, topk buffers), `model/dspine.rs` (750-772, 312-363), `tests/dspine_cpu.rs` | 6-8 h | L0 (sizes it) | `check` mode: 0 differing rows over 512 tokens; ids; at least 1 ms off logits+cand in trace |
| **L3 auto + k** | DSpine prior/theta in `AutoParams`; prompt-length default k for `--spec dflash`; then optionally a spine-only depth controller | `spec.rs` (1046-1069, `Depth` 312), `engine.rs` (1449, 2036), `crates/intarsia-models/src/load.rs` (814-821) | 4-8 h | L0 | Auto + DSpine ids gate; cats ≥ 0.97x of the best fixed arm per category |
| **L4 tp2-B** | Section 3.3 | `model/dflash2.rs` (282, `merge_topk` callers), `model.rs` (1471), `model/dflash.rs` (542-593, 654-688), `model/dspine.rs` (s0, tabs guard, `HeadRows` concat) | 1.5-2 days | L1; after L2 (both rewrite the candidate path in `dspine.rs:750-772`) | Section 3.5 gates |
| **L5 fp8 draft** | Lever 5, behind a knob (default off) | `model/dflash.rs` (`lin`, 148-155), `model/dspine.rs` (`load_spine`), `model/dflash2.rs` (`load2`/`lin_part`) | 8-12 h + A/B window | After L4 (both touch `load2`) | t/p loss under 2% on cats, then speed |
| **L6 multi-row draft attention** | Lever 6 | `kernels/hip/` (new rows-causal attention), `model/dspine.rs` (648-664) | 8-14 h | L0 shows attention at 1.5 ms or more at 4k | Kernel oracle test; ids; trace block ms |
| **L7 acceptance research** | Lever 7: f32 fc partial sums, real-weight parity against MLX, and why short-prompt DSpine (2.14 t/p) trails DFlash2 (2.48) here, although Tencent reports the opposite on Qwen3-8B | `model/dflash.rs` (`dflash_ingest` 500-529), `tools/dspine/` | 6-10 h, mostly offline; any MLX run on endor is a fleetq job | | t/p on cats |

**Deferred.**
- DSpark confidence-truncated verify (lever 8): only if L0 and L3 show large within-prompt variance in accepted length.
- The device tail: only as groundwork for a captured draft.
- ReplaySSM-style row saves (lever 9): a verify-side project for every spec mode, outside DSpine.
- Draft graph capture.

**Parallelism.**
- L1, L2 and L3 touch disjoint code and can run at once after L0. L1 does not need L0.
- L4 waits for L1 and L2, and L5 for L4.
- L6 and L7 are independent of the rest.

## 5. Not verified

- **The DSpine phase split on HIP.** No trace exists. Every DSpine draft number is [D] (±1.5 ms) or [E].
- **The verify8 context correction.** The +1.6 ms for 4420 vs 3135 keys is linear extrapolation from short vs 4k.
- **The survival curve, and therefore the k5-k7 plateau.** Fitted to two sums [G].
- **The K'=32 boundary fallback rate.** Unknown. L2's `check` mode measures it.
- **The fp8 acceptance loss.** Unknown [G]. Tencent publishes none.
- **tp=2 tokens per pass, rank 0 VRAM headroom for A, and the 8-row shard-alignment behaviour for a DSpine verify.** All unmeasured. The tp=2 table combines tp=2 DFlash2 pass costs with tp=1 DSpine acceptance.
- **Auto with a DSpine draft.** It loads by the code path at `load.rs:791`, but no gate has run it.
- **Inherited claims.** From the inputs and not re-opened here: Tencent repo line numbers, the DFlash2 dry-run lines `dflash.rs:1018-1024`, and the plan-doc memory figures. The code citations in sections 1.2, 1.4, 2.1 and 3 were re-read at 8532f5701.

## 6. Corrections after verification and the category sweep (added 2026-11-08)

Two adversarial checks of this plan against the code (6/12 speed claims held, 21/31 tp=2 claims held) and the category sweep
(`docs/results/2026-11-08-dspine-cats.md`) change it as follows.

* **On short prompts the sweep overturns the ranking's premise; at 4k it does not test it.** On the 7 category files (39-47 token prompts only)
  DFlash2 k7 leads (geomean 72.1 tok/s) over MTP k2 (68.3), DSpine k7 (65.8) and DSpine k3 (63.9), and DSpine accepts less than DFlash2 at the same
  k on chat, prose and recall. At 4k the only data is still one prompt, where DSpine accepts more (4.19 vs 3.57 t/p). For short prompts, lever 7
  (acceptance) moves to the top and the pass-cost levers 3-6 matter only once acceptance reaches DFlash2's; the 4k ranking stands until a
  long-prompt sweep.
* **Lever 7 is a hypothesis with a test, not a finding.** Tencent's DSpine-over-DFlash2 numbers are b16 vs b8, bf16 target, H-class GPU; ours is k7
  truncation on an MXFP4 target. Test: dump our engine's real taps on a few sweep prompts, feed the same taps to the MLX reference, compare per-row
  draft argmax. Agreement means the gap is the model or the quantized target; disagreement means a port bug, and that is the lever.
* **Lever 1 loses DFlash2's short-prompt wins.** `--spec auto --spec-draft <DSpine>` puts DSpine in DFlash2's slot; on short prompts DFlash2 k7
  beats MTP on code (+24-33 %) and recall (+12-16 %), and DSpine k7 matches it only on code. Auto with DFlash2 stays the better short-prompt pairing.
* Lever 2: a prompt-length k cannot be a load-time default (`load.rs:814-821`). Load at k7 so the dry run admits 8 rows, lower k per request at
  `engine.rs:1424`. `spec::Depth` runs under `--spec mtp` and `--spec auto` (and acts on DSpine passes under auto), not under `--spec dflash`.
* Lever 4 (`INTARSIA_VERIFY_GRAPH=1`): about 0 at 4k k7, about +1 % at short k3. The combined tp=1 estimate is 54.0 ms / 77.6 tok/s at 4k k7
  (+5.6 %), 81.5 with fp8 draft weights at flat t/p. All [E].
* Lever 3 and lever 5 gains are unanchored: no DSpine trace exists, and the 394 GB/s fp8 rate is the head's shape, not the draft's small-N GEMVs.
* tp=2 option A: the failure today is an out-of-bounds slice in `dspine_prepare` at load (`load.rs:973-976`, `dspine.rs:546-554`), not a shape
  check; `dspine_dry_run_tables` needs no change. Option B's gate cannot be exact split-vs-whole draft equality (o/down summation order and
  the bridge's bf16 rounding change); pre-register draft-id agreement or stash rel-L2.
* tp=2 numbers with the context correction: A 4k k7 about 41.2 ms / 102 tok/s, B about 36.0 ms / 116 tok/s; k3 rows about 0.7 ms longer than
  §3.4. DFlash2 split k7 on DSpine's prompt is about 103 tok/s. Short-prompt tp=2 rows carry about ±7 % from acceptance drift (measured up to
  −0.27 t/p tp=1 → tp=2). The 8-row shard cliff is already measured absent on this target.
* Order of work: the tap-replay test (lever 7) or a long-prompt served sweep gates everything. L1 (tp2-A) and L4 (tp2-B) inherit the single 4k
  data point (102 / 116 tok/s [E]) and are only worth building if served traffic is long-context and DSpine beats DFlash2 there.
