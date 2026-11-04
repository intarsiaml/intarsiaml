# m27 row-table head split: the bitwise head split (HSB) for batched decode of the Qwen3.8-27B on Metal (2026-11-03)

Question: PR #292 made `INTARSIA_METAL_ATTN_HSB=3` the default for the Qwen3.8-27B single-stream decode attention (group 6, head_dim 256), but the row-table launcher of batched serving (`--max-batch > 1`) still called `head_split` directly and so ran the unsplit launch. Does the batched step have the same cost, and can the bitwise split be extended to it without changing a bit?

**Answer: yes to both.** `launch_attn_rowtab` now takes the same `head_plan` as the 1-row and verify launches. Batched decode output bits are unchanged (kernel test over 675 row-table windows plus a 4301-position sweep, and 12 concurrent greedy requests x several rounds at `--max-batch` 2, 4 and 8 with token ids compared; the only differences seen are a pre-existing rare near-tie flip that the baseline binary shows as often, section 4). Aggregate throughput of a concurrent burst improves by **+5.5 % to +9.2 %** (A/B/A/B, run-to-run spread under 2 %), so the stop rule (under 5 % at 16 streams and 16 K) is cleared. **Deviation: `--max-batch 16` cannot be loaded for the 27B on Metal** (section 2); 16 streams were run against `--max-batch 8` (8 slots, 8 streams queued).

Branch `m27/hs-rowtable` (local, not pushed; `origin/main` merged in after #292 landed). Commits: `fc6982dd1` (the change and the test), `65ab61c48` (opt-in sweep in the test), the merge `1daaf15b6`, and the final docs commit (hash in the report). Baseline binary `.scratch/intarsia.def` (sha256 in `raw/binaries.sha256`, = `m27/hs-bitwise` HEAD with HSB default 3), this lane's binary `.scratch/intarsia.rt` (built at `fc6982dd1`; `intarsia.rtm` is the post-merge rebuild used for the gates).

## 1. Spill analysis: does the row-table launch have the same padded-register problem?

Yes, by construction. `attn_decode_rowtab_t<GP>` (`kernels/metal/attention_splitkv.metal`) is the same template body as the 1-row kernel (same `ask_step`, same per-lane `qv[GP][8]`, `acc[GP][8]`, `m[GP]`, `l[GP]`). `launch_attn_rowtab` (`crates/intarsia-backend-metal/src/rowtab.rs`) picked `gp = (grp / hs).next_power_of_two()` with `hs = head_split(grp, d)`, which is 1 for group 6 at head_dim 256: GP = 8, i.e. 128 floats of q and accumulator per lane for 6 real heads (2 zero-padded), the footprint the 1-row launch had before #292. The 35B (group 8) already ran its row-table kernel at HS=4 (GP 2). The register spill itself was not measured (no GPU counters read), as in the previous lane; the evidence is the A/B below.

How much of a batched step this is cannot be read from a per-op timer in serve mode, so the quick A/B used the existing non-bitwise `INTARSIA_METAL_ATTN_HS=3` (the row-table path honours it) at `--max-batch 4`, MTP off, binary `.scratch/intarsia.def`, `localeval speed --concurrency 4 --gen 512 --reps 1`, one run per arm (jobs #8239 A, #8241 HS=3; raw in `raw/ab-hs3/`):

| prompt | A: default (aggregate tok/s, per-stream median) | C: HS=3 | C / A |
|---|---|---|---|
| 4 K, 4 streams | 28.4, 9.2 | 30.7, 10.1 | x1.081 |
| 16 K, 4 streams | 11.4, 5.2 | 13.1, 6.1 | x1.149 |

Above the 5 % bar at 4 K and 16 K (single runs, so indicative only), so the lane went ahead. The aggregate is completion tokens over the wall of the burst, and the burst's prefill (about 4.6 s per 4 K and 22 s per 16 K prompt, serialised) is the same in both arms, so the decode-phase gain is larger than the aggregate figure; for 16 K x 16 streams below, wall 718 s -> 655 s, of which roughly 355 s is prefill, i.e. about 17 % of the decode phase (derived, rough).

## 2. `--max-batch 16` is refused for the 27B on Metal

The step-1 jobs at `--max-batch 16` (#8242, #8243) failed at load, both times for the same reason (`raw/ab-hs3/{A,C}-mb16-server.err`): `a 9-row batched step runs matmul_mxfp4 on matmul_mxfp4_nax, which is not a row-exact (rows_exact) kernel and not the solo step's (matmul_mxfp4_gemv)` (`crates/intarsia-core/src/plan.rs`, `check_batch_ops`): the 27B's MXFP4 matmul has no row-exact 9..16-row kernel. This is the baseline's behaviour and not touched here. So the batch caps at 8 for this model (matches `docs/recipes.md`, "up to 8"). The brief's "16 streams" cells were run as 16 concurrent requests against `--max-batch 8` (8 sequences decode together, the other 8 queue and start as slots free up); "max-batch 16" itself is **not checked**.

## 3. The change (`crates/intarsia-backend-metal/src/rowtab.rs`, `kernels/metal/attention_splitkv.metal`, `docs/tuning.md`)

* `launch_attn_rowtab` uses `head_plan(grp, d)` (the 1-row and verify launchers' plan, unchanged: same opt-out `INTARSIA_METAL_ATTN_HSB=0`, same explicit-`INTARSIA_METAL_ATTN_HS`-wins rule, only group 6 at head_dim 256 changes): grid `nkv * hs` threadgroups wide, `max_split = decode_splits(cap, hp.policy_kv(nkv), MAX_SPLIT)` (the policy sees `nkv`), the launch's simdgroup count from `gp.max(hp.gpref)`, and `OpP::s1 = hp.s1()` (= hs | gpref << 8).
* Shader: the row-table stage 1 and combine kernels read `s1` through the helpers the 1-row kernels already use (`ask_hs`, `ask_pol`, `ask_gpn`): with no high bits they reduce to the old `max(p.s1, 1)` expressions, so every other shape (the 35B's HS=4, the 8B, group 6 at head_dim 128) keeps its exact launch.
* Why it stays bitwise, from the 1-row analysis of `2026-11-03-m27-hs-bitwise.md` section 1: the key partition is fixed by the split count (policy fed `nkv`, not `nkv * hs`) and the simdgroup count (derived from the unsplit GP = 8), heads do not interact, and the merge order is unchanged.
* No other source file changed (footprint grep against `origin/main` is empty, section 6).

## 4. Bit identity (all against the baseline `.scratch/intarsia.def`)

**Kernel test** `crates/intarsia-backend-metal/tests/hs_bitwise.rs`, extended to the row-table launch (`attn_decode_bf16_splitkv_rows`): R in {1, 2, 3, 4, 16} rows, row r on cache r mod 3 (three different random K/V caches) at positions taken from the same set that straddles every split-count change (127..129, 1023..1025, 3712..3968, 16000, the cache end), rotated per window so rows of one launch differ in length and split count. For each window the unsplit row-table launch (override 0) is first checked equal to each row's solo 1-row unsplit launch (the rows_exact contract), then HSB = 2, 3, 6 are compared with it: **675 row-table windows bit-identical** (plus the existing 1080 one-row and verify windows, `raw/hs_bitwise-test.log`, job #8250). The 8B, 35B and group-6-d128 shapes are also checked unchanged with HSB=3 on the row-table launch. An opt-in `HSB_SWEEP=<reps>` section sweeps every position 0..=4300 as a 2-row table (p, 4300 - p) and repeats an HSB=3 launch 300 times for four position pairs: all identical (`raw/hs_bitwise-sweep.log`, #8258).

**Mutation checks** (the test must fail on a partition change; both reverted before the commit): (1) dropping the GPREF bits from `OpP::s1` in `rowtab.rs` fails at `rowtab R=1 pos [0] HSB=2: differs from the unsplit row-table launch` (`raw/hs_bitwise-mutation.log`, #8247; an earlier try #8246 used `sed`, did not apply and the test passed, as in the previous lane); (2) the row-table stage-1 shader policy `nkv * ask_pol(s1)` -> `nkv * ask_hs(s1)` fails at `rowtab R=1 pos [1000] HSB=6` (`raw/hs_bitwise-mutation2.log`, #8248).

**Greedy ids under batching** (`raw/scripts/ident-rt.sh`, `chain-ident.sh`, `chain-flake.sh`, `cmp-ids.py`, `cmp-flake.py`; data `raw/ident/`): `intarsia serve --backend metal --max-batch MB --spec off --max-seq 12000`, 12 concurrent greedy `/v1/completions` requests per round (temperature 0, `ignore_eos`, `return_token_ids`, 200 tokens) whose prompts are the first 150, 400, 900, 1800, 3000, 5000, 8000, 11000, 15700, 22000, 31000 and 40000 characters of `prompt-16384.txt` (about 40 to 10600 tokens), each with a unique first line; ids per request saved.

| run | job | requests | identical to the baseline's ids (def-mb2 reference) | steps with 2 or more rows |
|---|---|---|---|---|
| def MB=2 / 4 / 8, 2 rounds each | #8251, #8253, #8257 | 24 + 24 + 24 | reference; def-mb4 and def-mb8 24/24 equal def-mb2 | 2218 of 2558 / 1104 of 1430 ... see `raw/ident/*-batch-counters.txt` |
| **rt MB=2**, 2 rounds | #8252 | 24 | 23 | 2219 of 2557 |
| **rt MB=4**, 2 rounds | #8256 | 24 | 24 (also 24/24 vs def-mb4) | 1296 of 1430 |
| **rt MB=8**, 2 rounds | #8259 | 24 | 24 (also 24/24 vs def-mb8) | 926 of 1024 |
| rt MB=2, 6 rounds (flake hunt) x2 | #8260, #8283 | 72 + 72 | 72, 71 | 6650 of 7678, 6653 of 7675 |
| def MB=2, 6 rounds x2 | #8267, #8268 | 72 + 72 | 71, 72 | (as above) |

The batch counter (`intarsia_batch_decode_rows_total{rows=N}` from `/metrics`, saved per run) shows that most decode steps were multi-row steps, so the row-table launch ran thousands of times in each run (for example rt MB=8: 330 eight-row steps, 926 steps with two or more rows). In the speed runs the counters of both arms are identical (for example MB=8, 16 K: 2259 multi-row steps of 3034 in each arm).

**The one difference class, and why it is not this change.** 3 requests in 408 differ in ids from the reference, always request q4 (the 3000-character prompt, about 800 tokens), always at generated token 184 with the same alternative (id 557 3799 instead of 14663 974: "The text was cut off" instead of "The text cuts off"): rt-mb2 round 2, rt flake4 round 6, **and the baseline binary** flake2 round 4 (`raw/ident/flake2-def-mb2-ids.json`). The baseline flips at the same token with the same alternative, 1 in 192 baseline requests against 2 in 216 for rt, which is not distinguishable. It is a pre-existing near-tie in the batched engine that resolves differently from run to run (what varies was not found: not the attention kernel, since the 4301-position sweep and 300 repeats are bit-stable). It is not caused by this change; a follow-up task was flagged for it. By the strict reading ("ids identical per request"), though, a rare 1-in-100 flip exists with both binaries, and the claim here is: **no difference attributable to the change was found, and the baseline has the same flip.**

## 5. Speed (A/B/A/B against the baseline binary, same window)

`serve --backend metal --max-seq 18000 --max-batch MB --spec off --profile-every 0`, `localeval speed --sizes S --gen 512 --reps 1 --thinking off --concurrency C` per size on one server, `raw/scripts/conc-arm.sh`, all in one fleetq slot (#8299, `--quiet gpu,cpu`, `--mem 45`) by `raw/scripts/speed-driver.sh`; A = `.scratch/intarsia.def`, B = `.scratch/intarsia.rt`, order A B A B (arms 1..4 for MB=8 at 16 K, 5..8 at 4 K; 9..12 for MB=4). Cell = one burst (aggregate = completion tokens / wall of the whole burst, prefill included, as localeval defines it); two cells per arm and config, "spread" = (max - min) / median of the two. Tables from `raw/scripts/summarize-speed.py` (`raw/speed-table.md`, `raw/speed-cells.json`; localeval result dirs `/opt/code/deepeval/results/*-adhoc-m27rt-*`).

| max-batch | prompt | streams | A baseline: cells, median | spread | B this lane: cells, median | spread | B / A |
|---|---|---|---|---|---|---|---|
| 4 | 4 K | 4 | 28.1, 28.2 -> 28.1 | 0.4 % | 29.7, 29.7 -> 29.7 | 0.0 % | **x1.055** |
| 4 | 16 K | 4 | 11.6, 11.7 -> 11.6 | 0.9 % | 12.6, 12.7 -> 12.6 | 0.8 % | **x1.086** |
| 8 | 4 K | 8 | 29.0, 29.1 -> 29.1 | 0.3 % | 31.0, 31.1 -> 31.1 | 0.3 % | **x1.069** |
| 8 | 4 K | 16 (8 slots) | 28.8, 28.9 -> 28.9 | 0.3 % | 30.8, 30.9 -> 30.9 | 0.3 % | **x1.069** |
| 8 | 16 K | 8 | 11.7, 11.5 -> 11.6 | 1.7 % | 12.5, 12.5 -> 12.5 | 0.0 % | **x1.078** |
| 8 | 16 K | 16 (8 slots) | 11.5, 11.4 -> 11.4 | 0.9 % | 12.5, 12.5 -> 12.5 | 0.0 % | **x1.092** |

Per-stream decode tok/s (median over the streams of a cell, includes interleaved prefill of the others) moves the same way: 16 K x 8 streams 2.64 -> 2.98, 16 K x 16 streams 1.78 -> 1.94, 4 K x 8 streams 4.82 -> 5.24. Single-stream decode in the same serves is unchanged (25.4 to 25.7 tok/s at 16 K, 27.6 to 27.7 at 4 K, in every arm), as expected since the 1-row launch was already bitwise-split.

**Stop rule:** at 16 streams and 16 K the aggregate gain is +9.2 % (>= 5 %): cleared. The gain is smaller than the single-stream +24 % at 16 K because the burst wall is about half prefill, which this change does not touch.

## 6. Gates

GATES_PLACEHOLDER

## 7. Recommendation

REC_PLACEHOLDER

## 8. Jobs and commands

All jobs `FLEETQ_OWNER=claude-m27-hsr`, `--host endor`, run from `/opt/code/intarsiaml/.worktrees/m27-hs-rowtable`. Step-1 A/B: #8239, #8241 (the `--max-batch 16` attempts #8242, #8243 failed at load); build #8244 (`raw/scripts/build.sh intarsia.rt`, `--load cpu`), #8249 (clean rebuild after the mutations), #8297 (`intarsia.rtm`, post-merge); kernel test #8245, #8250 (`raw/scripts/test-hsb.sh`: `cargo test --release -p intarsia-backend-metal --test hs_bitwise -- --nocapture`, `--load cpu`), mutations #8246 (did not apply), #8247, #8248; sweep #8258 (`HSB_SWEEP=300`); identity #8251 to #8259 (`chain-ident.sh`), flake hunt #8260, #8267, #8268, #8283 (`chain-flake.sh`, `ROUNDS=6`); speed #8299; workspace tests #8298; gates #8300. Speed command per arm: `zsh raw/scripts/conc-arm.sh <tag> <bin> raw/speed <max-batch> <conc>` with `SIZES=... GEN=512` (see `speed-driver.sh`), which runs `uv run localeval speed --url ... --sizes S --gen 512 --reps 1 --thinking off --concurrency C --recipe "<serve command>"` after starting `<bin> serve --backend metal --model ~/models/Qwen3.8-27B-Quark-AWQ-MXFP4 --max-seq 18000 --max-batch MB --spec off --profile-every 0 --no-supervisor --no-ui`.

Fleet notes: the speed slot waited about 2 h for another owner's `--load cpu` job (#8280, "held across steps") to end before the quiet slot could start; I cancelled my own queued chain once (#8293) and resubmitted the speed work as one prioritised slot; nothing of another owner's was touched, no `--force`, no `--no-queue`. The first 16-stream plan (`--max-batch 16`) and a first sequential chain (`chain-speed.sh`) were abandoned and are not in the results.

## 9. Not checked

* `--max-batch 16`: refused at load for the 27B on Metal (section 2). 16 streams were measured as 16 requests over 8 slots.
* MTP is off in every batched run (`--max-batch > 1` keeps plain decode); batched + MTP does not exist on main.
* The kernel-level register spill (no GPU counters); the decode-phase-only gain is derived from the burst wall minus an estimated prefill share, not measured.
* Aggregate speed has 2 cells per arm and config (4 for none); spreads under 2 % within an arm, but only one ABAB pair per config.
* Greedy ids were compared on 12 requests of 40 to 10600 prompt tokens, 200 tokens each; not 32 K, not W4A4, not the 35B or 8B end to end (the test pins their row-table output unchanged with HSB=3). The pre-existing q4 flip (section 4) was not root-caused.
* Prefill and the verify launch are unchanged by this lane and were not re-measured. CUDA and HIP were not run (footprint is Metal only).
