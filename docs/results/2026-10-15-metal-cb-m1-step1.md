# Metal continuous batching, M1 step 1: exact batched decode step (no scheduler)

> **Update (2026-10-21, docs-accuracy pass, see `docs/decisions/LOG.md`):** C16: the rule that the recipe `max_batch` applies when `--max-batch` is absent is reversed by `2026-10-15-metal-cb-m1-step2.md`: an absent flag means 1 (`serve` runs one sequence); the recipe `max_batch` is only a ceiling (8 in `qwen3.toml`, 32 in the `qwen3_5*` recipes).

Date 2026-10-15. Branch `metal-cb-m1` off `metal-cb-m0` 8254311, merged with `spike/metal` b300b3e (M0 included) before the final gates. `Seq::prefill_step` is untouched (another branch edits its chunk sizes).
Machine endor (Apple M5 Max), Qwen3-8B BF16. Plan: `docs/research/continuous-batching-design.md`, milestone M1, operator
decisions 8.1 (exact only: every batched row bitwise equal to the solo decode step; batch cap 8).

**Verdict: step 1 met.** `Model::forward_batch` runs B decode tokens (one per sequence, each with its own KV cache) in one pass:
B-row projections, norms, FFN and head on the row-exact multi-row kernels, attention once per row through the same function the
solo step uses. Every row is bit for bit the solo step (138 (row, step) pairs at B = 2, 3, 4, 6, 8, contexts 5 to 4000, greedy
and seeded-sampled rows, logits compared by bits), and one batched step costs 1.00x / 1.96x / 3.7x / 5.7x less time than B solo
steps at 1k context (B = 1 / 2 / 4 / 8). Not done here: scheduler, `BatchEngine`, serve (see section 7).

## 1. Commits (branch `metal-cb-m1`)

| Commit | What |
|---|---|
| `f73beae` | `attn_block` split: `attn_project` (q/k/v projections for t rows), `attn_rows` (q-norm, rope, KV append or the fused `QkNormRopeKv`, attention, for the rows in an `AttnBufs` view against one cache slot), `attn_out_proj` (gate, `wo`, tp sum). Pure refactor. |
| `f394391` | `engine.rs`: `Seq::plain_input` / `accept_plain` / `prepare`, `advance` runs through them; `SERVE_MAX_BATCH = 1`. |
| `8768b6a` | `model/batch.rs` (`enable_batch`, `forward_batch`, `batch_capable`, batch plan probe), `Phase::Batch`, plan structure check, `--max-batch` / recipe resolution, tests, `batch_step` example. |
| later | capability check moved into `enable_batch`, profile off in the batch test and bench, tighter KV test, raw scripts, merges of de8fe5b / b300b3e. |

## 2. API

```rust
// crates/intarsia-core/src/model/batch.rs
pub const BATCH_MAX_ROWS: usize = 8;
pub fn batch_capable(&self) -> bool;                    // row-exact Matmul + Logits kernels registered
pub fn enable_batch(&mut self, max_rows: usize) -> Result<()>;   // 2..=8; call before configure_plan and new_state
pub fn batch_rows(&self) -> usize;
pub fn forward_batch(&self, states: &mut [&mut SequenceState], ids: &[u32], modes: &[HeadMode]) -> Result<Vec<StepOut>>;
```

`forward_batch` takes a per-row `modes` (`LastLogits` or `LastArgmax`), a deviation from the brief's two-argument form: greedy rows
return the on-device argmax (`StepOut::Token`), sampled rows and rows inside a `min_tokens` window return their logits row
(`StepOut::Logits`), exactly the forms `Model::forward` returns for a one-token step, so `Seq::accept_plain` takes either. One state
is the solo step itself. The `Seq` pair for a driver: `prepare` (Emit / verify / plain decision) -> `forward_batch` over the rows that need
a plain step -> `accept_plain` per row.

`enable_batch` refuses tensor-parallel, recurrent, mixture-of-experts and gated-attention models, a backend without row-exact
kernels (the cpu backend is exempt: its multi-row kernels loop the one-row code, as for verify), and row counts outside 2..=8. KV
budget: `plan.kv_cap_tokens` shrinks to `kv_budget_bytes / (max_rows * Model::kv_bytes_per_token())` (plan note), `kv_total_bytes` is
`max_rows * cap * bytes_per_token`; an error below 16 tokens. No plane sizes are summed anywhere.

## 3. The split and how a batched step is built

`attn_block(t rows)` = `attn_project` + `attn_rows` + `attn_out_proj`. Callers and their `t`: plain decode `t = 1`, verify `t = 2..8`
(one sequence, `exact_rows` on), prefill `t = chunk`, all through `attn_block`; a batch step calls `attn_project` and `attn_out_proj`
with `t = B` and `attn_rows` B times with `t = 1` on `rows(r, 1)` views of the scratch, that row's own `AttnKvState` and `pos0 = st_r.len`.
Exactness:

* projections, `kv_pair`, FFN and head carry `OpCall::exact_rows`, so the registry gives them the G0 multi-row GEMVs (bitwise the
  one-row serving GEMV per row); the `exact_rows` mark is switched off around the per-row calls, which therefore select the serving
  `QkNormRopeKv` / split-KV `AttnDecode` kernels (the dry run prints the picks: identical to the Decode phase);
* the fused-or-not decision (`QkNormRopeKv` vs norm + rope + append, pair vs two projections) is taken on the one-row call shape, as
  the solo step takes it; `plan::check_batch_ops` fails the load if the batched step's op multiset is not the solo step's with the
  per-row attention ops times B;
* one change inside `attn_block`'s old order: the k and v projections now run before q-norm / rope (they were interleaved). They write
  buffers nothing else touches, so values are unchanged (CPU engine tests exercise the unfused path); CUDA and HIP were only
  compile-checked.

`Phase::Batch` probes (rows 2..=max, all sequences at position 0 and at `cap - 1`) record the real batched pass over phantom states,
so a missing row-exact kernel fails the load and admission covers the batch kernels. With batching off nothing is probed: the plan is
as before.

## 4. `--max-batch` and the recipe

`--max-batch` is now `Option<usize>`. `Recipe::resolve_max_batch(flag, supported)`: an explicit value must be in
`1..=min(recipe max_batch, SERVE_MAX_BATCH)`; without one the recipe's `[defaults] max_batch` is used, **clamped to what the engine
implements** (1 today), because the qwen3_5 recipes declare 32 and must keep loading. `--max-batch 4` still fails with "not implemented
yet", now at load right after the recipe is read; `load_engine_tp` calls `enable_batch` when the resolved value exceeds 1 (not
reachable until the scheduler raises `SERVE_MAX_BATCH`).

## 5. Gates

| Gate | Result | Where |
|---|---|---|
| bitwise, tiny cpu model, B = 2, 3, 4, 6, 8, 5, 8 x 6 steps, 8 sequences 3..90 tokens (`tests/batch.rs`) | pass (mutation checked: a wrong row index fails it) | local |
| bitwise, Qwen3-8B Metal, B = 2, 3, 4, 6, 8 x 6 steps, 8 sequences 5/37/130/500/1100/2000/3000/4000 tokens, greedy (logits and argmax steps alternate) and seeded top-p/top-k rows (`tests/batch_metal.rs`) | **138 (row, step) pairs identical**, 12.5 s | slot #2067 |
| batched decode beside prefill chunks of another sequence (`run_with_prefill`: 4 decoding sequences, a 1300-token prompt in 256-token chunks, batched step before / after each chunk alternately): decode rows bitwise solo, prefill chunk logits and three later decode steps bitwise equal the prompt prefilled alone; cpu version too | pass | in the workspace run |
| `cargo check --workspace --all-targets` | pass | local |
| `cargo test --workspace --release` | 401 passed, 0 failed, 1 ignored (79 suites) | `raw/2026-10-15-metal-cb-m1-step1/out/` |
| `cargo test -p intarsia-backend-metal --release` | 102 passed, 0 failed | same |
| `intarsia kernels test --backend metal` | 27 kernels, 321 cases, all ADMITTED | same |
| M0 identity subset (11 cases; new binary spec off and `--spec ngram` vs the 8254311 binary spec off and `--spec ngram`) | all identical (stdout, ids, finish), 44 runs; also on the pre-merge tree | `id-table.md`, `id-premerge-table.md` |
| the same after the split alone (`f73beae`) | all identical | slot #2065 |
| `cargo check -p intarsia-cli --features cuda --tests` / `--features hip --tests` on spark1 | EXIT=0 both | `#2068`, scratch dir created and removed |

Identity cases: short-48, 4k-48, grounded-256, grounded-stop-topp, grounded-min335, repeat-eos, golden-docs-48, 4k-stop250,
sampled-short-128, sampled-open-min50-stop, sampled-4k-64 (`cases-sub.txt`), all flags as the M0 table.

## 6. Speed: one batched step vs B solo steps

`cargo run --release -p intarsia-models --example batch_step -- --ctx N` (`profile_every 0`, greedy on-device argmax, 8 rounds x 16
steps, solo and batched alternate, first round dropped; the batched tokens are checked equal to the solo tokens while it runs).
Median per-step time, ms; design estimate from section 4.6 in the last column (1k context).

| ctx | B | B solo steps | 1 batched step | speedup | batched tok/s | design est. ms |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 2 | 54.76 | 27.81 | 1.97x | 71.9 | 27.6 |
| 128 | 4 | 109.10 | 29.16 | 3.74x | 137.2 | 28.6 |
| 128 | 8 | 220.03 | 37.17 | 5.92x | 215.2 | 35.1 |
| 1024 | 1 | 27.62 | 27.67 | 1.00x | 36.1 | 27.3 |
| 1024 | 2 | 55.53 | 28.28 | 1.96x | 70.7 | 28.0 |
| 1024 | 4 | 110.32 | 30.04 | 3.67x | 133.2 | 29.3 |
| 1024 | 8 | 220.68 | 38.78 | 5.69x | 206.3 | 36.6 |
| 4096 | 2 | 56.44 | 29.59 | 1.91x | 67.6 | 29.5 |
| 4096 | 4 | 113.93 | 33.45 | 3.41x | 119.6 | 32.5 |
| 4096 | 8 | 227.46 | 45.00 | 5.06x | 177.8 | 43.0 |

The measurement is within 1-5 percent of the estimate up to B = 4 and 5-6 percent above it at B = 8 (38.8 vs 36.6 ms at 1k, 45.0
vs 43.0 ms at 4k): the 3 dispatches per row-layer of the per-row attention that the design priced at the G0b upper bound. The
decode-window target of M1 (N = 8 >= 175 tok/s at 1k) is met by the step itself (206 tok/s) before scheduler and prefill losses. Full
tables: `raw/2026-10-15-metal-cb-m1-step1/out/batch-step-*.log`.

## 7. Limits and not done

* No scheduler, `BatchEngine`, serve change or telemetry per batch (`forward_batch` bumps the step counter once, as the design wants).
  `--max-batch N > 1` still errors; `enable_batch` is callable from tests and from `load_engine` for a resolved value above 1.
* Fusion consistency is guaranteed by the op-multiset check at plan time, not per call; unfused-weight combinations on Metal (non-bf16)
  have no row-exact kernels and fail the load.
* CUDA / HIP: compile-checked only; their plans are unchanged (nothing is batched), the unfused attention projections moved earlier
  inside `attn_rows`' caller as described in section 3.
* `cargo test -p intarsia-models` now includes `batch_metal.rs`, which loads Qwen3-8B on the GPU. It is opt-in: it only runs with
  `INTARSIA_TEST_REAL_MODEL=1` (model `~/models/Qwen3-8B`, or `INTARSIA_METAL_TEST_MODEL`), and on endor it needs an exclusive
  fleetq slot like the other Metal tests.
* The test allocates 16 states (8 solo reference + 8 batch) against a plan budgeted for 8: intended.
* Commits carry the Sonnet 5.5 attribution of the session, not the Opus 5.5 line of the brief.
* The `Seq` is still `pub(crate)`; a driver must live in `engine.rs`.

## 8. Slots

#2065 (split identity), #2067 (Qwen3-8B bitwise gate), #2068 (spark1 build check, no GPU), #2074 (gates before the merge of b300b3e,
`out-premerge/`), #2080 (final gates on the tree merged with b300b3e, `out/`; same numbers within noise). The merges were clean (no conflict, `common.rs` merged
automatically around `for_serving`); `common.rs` edits of this branch are the `--max-batch` option and its `engine_options` line only.
