# RIP2: row-invariant GEMM at cuBLAS speed; row-invariant prefill on by default on CUDA (dense)

Lane `rip2`, branch `lane-rip2/inv-gemm`, base `origin/main` `bfbee3cb6` (#237, RIP), merged `origin/main` `664ea1718` (#236, pass
list). spark2 (spark-fa00, GB10 sm_121, cuBLASLt 13.1.1.3), `spark2-vllm` (retro) up throughout. Follows RIP
(`2026-10-29-rip-row-invariant-prefill.md`), whose `matmul_bf16_inv` pinned one cuBLASLt algorithm per shape and cost Qwen3-8B
1k-token prefill -10.4%. Raw: `2026-10-29-rip2-row-invariant-gemm/raw/`.

## Result

**Row invariance only needs every row count to run an algorithm with the primary's summation order, not the primary itself.**
On sm_121 most split-K-1 cuBLASLt bf16 kernels are bitwise equal to every other, whatever its tile: of the algorithm x
tile x stages configurations checked (164 per Qwen3-8B projection shape, custom option 0, split-K 1; the enumeration hit its time
limit before algo 67's custom options), **524 of 656** gave byte-identical 4352-row outputs to RIP's primary
(`raw/probe/all1.log`; algo 21 at every tile, algo 67 at most), and so did **every split-K-1 pick cuBLASLt's own
heuristic makes at M = 991 / 1024 / 2048 / 4096** (`raw/probe/pool1.log`; `raw/kernel/kernel1.log`: the first differing pick is wanted at 17-65 rows). The ones that differ (algo 16, algo 24, algo 67 at small tiles:
ids 3, 5, 6, 8-13, 15) are presumably other MMA shapes / sliced-K; split-K > 1 always differs. cuBLAS itself uses split-K only below ~512 rows.

`matmul_bf16_inv` now keeps RIP's primary (it defines the bits) and, per row count, runs the heuristic's first pick for that M
(32 MB workspace preference, as `cublasGemmEx` asks) that has split-K 1, needs no workspace, and is **proven bitwise equal to the
primary** by a one-time probe per (N, K, algorithm): 256 rows of fixed-seed wide-magnitude bf16 against the call's own weights,
every output compared. Anything else runs the primary. GEMM cost per 8B layer, inv / cuBLAS: **1.002 at 991 rows** (RIP: 1.241),
1.009 at 1024, 0.993 at 4096, 1.03-1.05 at 64-256 rows (cuBLAS's split-K picks there are not usable).

With that, row-invariant prefill is **on by default on CUDA** (`Backend::row_invariant_prefill` true; the load-time rule keeps it
off for hybrid / MoE / fp8 / NVFP4 / multi-rank models; `INTARSIA_ROW_INVARIANT_PREFILL=0` opts out). Prefix reuse uses RIP's rule
(`Model::reuse_len`: any prefix of the rows row-invariant passes wrote, `SequenceState::inv_len()` from #236's pass list), so
multi-turn chat on CUDA resumes the history instead of re-prefilling it.

## 1. Kernel (`kernels/cuda/matmul_lt.cu`)

- Primary per (N, K): unchanged from RIP (heuristic at M = 4096, no workspace, split-K pinned 1, checked at M = 1).
- `pick(M)`: cached per (N, K, M). Heuristic for M (top 8, 32 MB workspace preference); skip results with workspace > 0 or
  split-K != 1; each new algorithm (keyed by id, tile, stages, split-K, reduction, swizzle, custom option, inner / cluster shape)
  is probed once against the primary and remembered as equal or not; the first equal one that checks at M wins, else the
  primary. The probe synchronizes the call's stream, so a call inside a stream capture whose M is not decided runs the primary
  (not cached). Algo 13 (a GEMV-style kernel for tiny M) does not serve the 256-row probe and is never used.
- `INTARSIA_INV_SINGLE=1` runs the primary at every M (RIP's behaviour, A/B only; same bits).
- Probe evidence of what is equal is from the standalone `scripts/rip2/lt_probe.cu` (`heur`, `enum`, `pool` modes).

Picks seen on the 8B (`raw/kernel/kernel1.log`, `INTARSIA_LT_DEBUG=1`): 66 algorithms probed equal across the four shapes; DIFFERS
only for algo 67 tiles 11 / 13 / 15 (first wanted at 17-65 rows); algo 13 unusable.

GEMM timing, the 8B's seven projections of one layer (`gemm_inv_timing_at_8b_shapes`, `--ignored`), us per layer:

| M | cuBLAS | inv (rip2) | rip2 / cuBLAS | RIP single algorithm / cuBLAS |
|---:|---:|---:|---:|---:|
| 16 | 1724 | 1726 | 1.001 | 1.124 |
| 64 | 1814 | 1867 | 1.030 | 1.107 |
| 256 | 2056 | 2168 | 1.054 | 1.195 |
| 512 | 2954 | 2969 | 1.005 | 0.972 |
| 991 | 4135 | 4143 | **1.002** | 1.241 |
| 1024 | 4417 | 4456 | 1.009 | 1.116 |
| 1536 | 6922 | 6922 | 1.000 | 1.039 |
| 2048 | 8836 | 8938 | 1.011 | 0.956 |
| 3000 | 12845 | 12753 | 0.993 | 0.973 |
| 4096 | 17079 | 16952 | 0.993 | 0.959 |

At 2048-4096 rows the primary itself is ~3% faster than the heuristic's pick (it was chosen there); the heuristic is not
always the fastest. Not taken: preferring the primary from 2048 rows would be a fixed rule (no autotuning) worth ~3% GEMM
time at 4k / 16k prefill; left as a follow-up because it needs another speed window.

## 2. Equality evidence

| Check | Result |
|---|---|
| `lt_probe enum` / `pool`: every candidate's 4352-row output vs the primary, random wide-magnitude bf16, 4 shapes | enum: 524 / 656 equal; pool: split-K-1 heuristic picks at every M of 16..4096 equal, except algo 67 tiles 12 / 13 / 15 |
| New `gemm_inv_rows_equal_at_every_m` (`tests/row_invariance.rs`): **every M in 1..=4352**, 4 shapes, rows 0 / M/2 / M-1 (all rows up to M = 64) vs one 4352-row pass, byte for byte | **pass**, 0 of 4352 row counts differ per shape (4.4 / 1.3 / 11.7 / 12.3 s) |
| RIP's `gemm_rows_do_not_depend_on_m` (25-30 windows) + the rest of `row_invariance` (attention, row-wise ops, GEMV) | **pass** (9 passed, 2 ignored) |
| `rows_exact` | **pass** (5 passed) |

## 3. Gates (merged tree `bd905bb`, default on)

Raw: `raw/gates-w2/` (model + served), `raw/gates-w3/` (kernel tests, plans), `raw/diag/`.

| Gate | Result |
|---|---|
| 8B probe (cold reference, `INTARSIA_BI8_PROBE=1`): 16 prompts, self-reuse / every split / cross-prompt | **pass**: self-reuse 16 / 16 equal, 0 differing splits, 0 cross differences |
| RIP's 8B split test `real_qwen3_8b_reuse_and_busy_chunks_equal_the_cold_reference` (k = 1, 7, 255, 256, 1000, plen-1, 4095 / 4096 / 4097, busy chunks, reuse) + BI8 batched gates | **pass** (3 / 3) |
| Multi-turn: 3 turns (2802 / 2872 / 2936 tokens, cached 0 / 2802 / 2872), single-sequence and batched | **pass**: warm == cold every turn |
| Decode-written rows (bitwise) / ordinary short pass at 8176 | **pass** (2 / 2): resume at `inv_len` 555 + 16 reply tokens EQUAL |
| Served greedy vs cold reference, solo 1 / 4 / 16 x 3 reps | **9 / 9 identical** |
| Served greedy vs cold reference, `--max-batch 16` 1 / 4 / 8 / 16 x 3 reps | **12 / 12 identical** |
| equal.py solo vs `--max-batch 16` | **21 / 21 cells**, 9408 token ids |
| `plan show` 8B, knob off (`=0`) vs `origin/main` | **byte-identical** (json + txt); default: base + one `matmul_bf16_inv` row |
| `plan show` 35B / Gemma-4, default and `=0` vs `origin/main` | **byte-identical** (json + txt); Gemma-4 logs "row-invariant prefill off for this model" |
| `batch_cuda_real` 35B (`real_35b_batched*`), 35B 21-cell served identity base vs fix | **not run** (window cap, see 5) |
| Mac `cargo test --workspace --release` | **pass**: 1104 passed, 0 failed |

**A first gate run failed on a stale binary.** Window 1 (tree `13e81ef`, before the #236 merge) gave solo 6 / 9 and
`--max-batch 16` 6 / 12 (streams 4, 5, 13 differ from the cold reference at the same token every rep, `raw/gates-w1/`). Diagnosis
(`raw/diag/`, micro jobs): `INTARSIA_INV_SINGLE=1` on that binary differed the same way, `origin/main` with
`INTARSIA_ROW_INVARIANT_PREFILL=1` was identical, and so was that binary with the knob set explicitly: the server's default was
the ordinary prefill, and that binary's `plan show` 8B was byte-identical to base (no `matmul_bf16_inv` row), i.e. the
installed binary did not carry the default flip. The rebuilt merged binary's default and `=1` give the same plan (with the
row) and its default served n8 was identical twice; window 2 then passed every served gate.

## 4. Speed (knob off vs on, 3 interleaved rounds, localeval)

SPEED_TABLE

## 5. Not done / caveats

- 35B gates (`batch_cuda_real` `real_35b_batched*`, 35B served identity base vs fix): not run, the three GPU windows went to
  gates (two, the first spent on the stale binary) and speed. The 35B loads with row-invariant prefill off (hybrid: the load
  rule), its plan is byte-identical to `origin/main` default and `=0`, and nothing in it calls `matmul_bf16_inv`; the change
  that could reach it is only the default of `Backend::row_invariant_prefill`, which its load rule overrides.
- The probe is a sample, not a proof: an algorithm equal on 256 random rows with the call's weights and different on other data
  would have to share no summation-order difference that random data exposes; the every-M test and the served gates back it.
- Library drift: algorithm ids are cuBLASLt 13.1.1.3's. A different library re-probes everything at startup (no table is
  stored), so a changed kernel fails the probe and runs the primary.
- Small M (64-256 rows) costs 3-5% GEMM time vs cuBLAS (its split-K picks); short resumed suffixes run there.
- The GEMV / split-KV / decode-attention replacements are RIP's; their cost is in the 1-stream and 16-stream decode cells above.
