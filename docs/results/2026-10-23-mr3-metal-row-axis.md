# mr3: row axis for the per-row kernels of Metal hybrid batching (35B)

Lane mr3, 2026-10-23, endor (Apple M5 Max, 128 GB), Qwen3.6-35B-A3B NVFP4, `serve --backend metal --max-batch 8`. Follows
[mb](2026-10-23-mb-metal-hybrid-batching.md) (PR #124), which batched the projections and the MoE but left the GatedDeltaNet
conv and delta rule, the attention prologue and the attention itself as one launch per row, and used the CUDA row-table technique of
[rb3](2026-10-23-rb3-batching-r3.md) (PR #121). Branch `lane-mr3/metal-row-axis` off `origin/main` 23a088cc. Machine-readable
record: [trace.json](2026-10-23-mr3-metal-row-axis/trace.json); step diagram: [step.svg](2026-10-23-mr3-metal-row-axis/step.svg).

## 1. Result

Aggregate decode tok/s over streams, localeval `speed --sizes 1024 --reps 3` (median of 3), 1k-token prompt, `--max-batch 8` on both
sides, both arms inside one `fleetq --quiet gpu,cpu` job each:

| generated | arm | 1 stream | 4 streams | 8 streams | fleetq | localeval runs |
|---|---|---|---|---|---|---|
| 128 | main 23a088cc | 121.1 | 100.9 | 108.3 | #4813 | `20261023-202652-adhoc-mr3-main-mb8-gen128` |
| 128 | branch | 122.4 | **107.1** | **117.0** | #4815 | `20261023-202838-adhoc-mr3-v1-mb8-gen128` |
| 512 | main 23a088cc | 122.5 | 154.4 | 161.6 | #4813 | `20261023-202717-adhoc-mr3-main-mb8-gen512` |
| 512 | branch | 122.4 | **171.5** | **199.1** | #4815 | `20261023-202900-adhoc-mr3-v1-mb8-gen512` |

- 512 generated: +11% at 4 streams, +23% at 8. 128 generated: +6% / +8%; short outputs are dominated by the serialized 1k-token
  prefill of each request (0.6 s, TTFT 1.5 / 2.8 s at 4 / 8 streams), which this change does not touch.
- Single stream with `--max-batch 8` (same window, same arm): 122.4 against 121.1 (128) and 122.4 against 122.5 (512), inside
  the spread of the reps. (The 124 tok/s in the brief is the mb doc's no-`--max-batch` arm from another session.)
- Single stream without `--max-batch` (the engine the batching change must not slow), main against branch, served 35B, 1k prompt,
  decode tok/s, median of the reps per arm:

  | check | main | branch | fleetq |
  |---|---|---|---|
  | 512 generated, 6 reps, ABABAB | 121.6, 122.0, 121.8 | 122.1, 121.2, 121.9 | #4893 |
  | 128 generated, 6 reps, ABABAB | 123.0 (pair 1) | 122.5 (pair 1) | #4896 |
  | 128 / 512 generated, 3 reps, branch first or second | 123.1 / 122.3, 122.7 / 122.2, 121.5 / 121.0 | 119.6 / 116.7, 119.5 / 121.9, 121.0 / 121.6 | #4883, #4887 |

  The 512 series has branch 121.7 and main 121.8 on average. The earlier 3-rep pairs put the branch 0 to 5% below main, mostly at
  128 generated, and the second and third pairs of the 128 ABABAB were slowed on both arms by foreign load on the machine (main 112.0
  and 113.0, branch 109.7 and 119.7), so they say nothing. The 1% bound is therefore met at 512 generated and not settled at
  128 generated: the kernels of the one-row path are unchanged (fp8 and nvfp4 M = 1 `timing_rows` 26.8 to 27.2 us before and
  after), and the in-process solo step is 31.35 against 31.30 ms. A quiet re-run of the 128 series is the open item.
- The main numbers today (161.6 at 8 streams, 512) are below the 173.7 recorded in the mb doc for the same arm; the machine is
  shared and the spread between sessions is about 5%. The comparison in the table is same-window, same-job-shape.

In-process step time (`real_35b_batch_step_timing`, `forward_batch`, all rows `LastArgmax`, 24 steps; fleetq #4760 main, #4870 branch):

| B | main ms/step | branch ms/step | main tok/s | branch tok/s |
|---|---|---|---|---|
| 1 (solo) | 31.35 | 31.30 | 31.9 | 31.9 |
| 2 | 12.86 | 12.34 | 155.5 | 162.1 |
| 4 | 20.59 | 18.42 | 194.2 | 217.1 |
| 8 | 35.75 | 30.23 | 223.8 | 264.6 |

(The solo step here is the one-sequence `forward`, not the served 124 tok/s path; the served single stream is in the table above.)

The step still grows about 2.9 ms per extra row (3.8 before). What is left is the multi-row GEMVs and the MoE, section 4.

## 2. What changed

Five kernels got a row axis; one dispatch covers all rows of the batched step. The rows of a batched step have different positions
and different state buffers, so the buffers come from a table of device addresses that the host rewrites each step (`OpCall::rows`,
as on CUDA) and the position of each row comes from an `i32 [rows]` vector (`OpCall::pos`).

| op | solo kernel | row-table kernel (`kernels/metal`) | launches per step at B = 8 |
|---|---|---|---|
| `GdnConv` | `gdn_conv_bf16` | `gdn_conv_bf16_rows` (`gdn.metal`) | 240 to 30 |
| `GdnDeltaRule` | `gdn_delta_rule_bf16` | `gdn_delta_rule_bf16_rows` (`gdn.metal`) | 240 to 30 |
| `QkNormRopeKv` (norm, rope, K / V append) | `qk_norm_rope_kv_bf16` | `qk_norm_rope_kv_bf16_rows` (`qk_rope_kv.metal`) | 80 to 10 |
| `AttnDecode` | `attn_decode_bf16_splitkv` | `attn_decode_bf16_splitkv_rows` (`attention_splitkv.metal`) | 80 to 10 |
| `KvAppend`, `Rope` (unfused chain, other models) | `kv_append_bf16`, `rope_bf16` | `kv_append_bf16_rows`, `rope_bf16` with one position per token | as before to 1 |

Bitwise by construction: every expression is the solo kernel's at T = 1, same operations in the same order. The one place where
the solo launch depends on the position is the attention: the host-position launch chooses the split count and the number of
simdgroups from the visible keys, and the device-position launch uses a different simdgroup count (the negative control below). The row-table attention derives both per row from that
row's own position on the GPU, so each row gets exactly the partition the solo launch of that row makes; the launch uses the
largest simdgroup count any row can have and the surplus simdgroups and threadgroups of a short row return at once.

Other changes:

- `Backend::supports_row_tables` and `Backend::device_address` on Metal. The kernels read state buffers through addresses, which
  Metal does not track, so every buffer whose address was handed out is declared resident (`useResources`, read and write) in each
  encoder that runs a row-table kernel (once per encoder and per change of the set; `MetalBackend::with_encoder_ind`). The set is
  held weakly.
- `Model::attn_block_batch` (shared code) runs the fused prologue (`QkNormRopeKv`) as one row-table call where the backend has the
  kernel (`fused_ok`), otherwise the chain it ran before. `OpCall::check` accepts `rows` and per-token `pos` for `QkNormRopeKv`.
  Both are no-ops on backends without a row-table `QkNormRopeKv` kernel (CUDA, HIP): section 3.
- Multi-row fp8 / nvfp4 GEMVs (`fp8.metal`, `nvfp4.metal`): the weight chunk is decoded once and applied to every activation row,
  instead of being decoded again for each row. Per row the fma chain is the same ops over the same values in the same order
  (`fp8_dec16` + `fp8_dot16d` against `fp8_dot16`; the M = 1 code path is untouched). Alone this is about 2.8 ms of the 5.5 ms step
  saving at B = 8 (in-process B = 8 step 34.0 ms with the row-table kernels only, 31.2 ms with this; main 35.75). G-B (`tests/rows_exact.rs`) still passes.
- The `INTARSIA_ROW_TABLES=0` switch (existing) turns the row-table path off per model; `tuning.md` rows added.

## 3. Gates

All on the final code (`cbb56a9d` kernels, plus test and pin updates). "Bitwise" means byte equality, no tolerance.

| gate | result | fleetq |
|---|---|---|
| G-B per kernel, `tests/row_tables.rs` (8 tests): conv (4 shapes), delta rule (dk 128 / 64 / 32), KV append (incl. a dropped out-of-range position), prologue (35B, 8B, tiny), attention (3 shapes, positions around the point where the host and device simdgroup rules differ), rope with a position per token; every M = 1..=16. The state tables carry a decoy buffer in the unused column, which must come back untouched (poison bytes) | 8 of 8, 2.7 GB of state / cache / output bytes bitwise equal | #4880 |
| Negative controls | swapping rows 0 and 1 in the table changes the conv state, the KV cache, the delta state and the attention output; the attention on cancelling data (V = +2^20, -2^20, small noise per key) differs from the DEVICE-position solo launch (a different simdgroup count from the host rule) in 43,824 / 81,920 and 44,720 / 81,920 output bytes, so the test can see a wrong simdgroup rule | #4880 |
| G-B multi-row GEMVs / MoE, `tests/rows_exact.rs` (fp8 672 cases, nvfp4 168, MoE 168, bf16 pair 84, row-parallel 98) | 6 of 6 after the decode-once change | #4794, #4835 |
| G-C 35B, `real_35b_batched_decode_is_bitwise_the_solo_step` (N = 2..8, 130 steps, growing states) | 4,550 (row, step) pairs bitwise identical, 0 failures | #4800 |
| G-C 35B, batched engine == `LocalEngine` (4 and 8 slots, 20 requests, queueing, slot reuse) | pass | #4800 |
| Long prompts, `real_35b_batched_engine_long_prompts_equal_the_local_engine` and the busy-chunk-64 control, read unfiltered | 14 of 14 `IDENTICAL` / `identical` lines, 0 `DIFFER`: 2.6k, 4k and 16k (sampled) prompts beside 200-token / 900-generated decodes; 2, 4 and 4 slots (two request orders); control chunk 64 | #4803 |
| Tiny hybrid on Metal (`batch_recurrent`, dense, MoE W8A16, MoE W8A8) | 10 of 10 | #4835 |
| Dense `batch_metal` (Qwen3-8B, N up to 8, beside prefill chunks) | 138 (row, step) pairs bitwise identical, prefill bitwise equal; the test now expects the row-table kernels for the prologue and attention | #4870 |
| `intarsia kernels test --backend metal` | exit 0, 44 kernels, 817 cases, 0 rejected (the five row-table kernels are exempt from the sweep, they have no case without a table; their gate is G-B) | #4835 |
| `cargo test --workspace --release --no-fail-fast` | 879 passed, 0 failed, 17 ignored (the opt-in real-model tests; the kernel-list pins gained the five row-table kernels) | #4873 |

No-op with batching off (no `--max-batch`), branch against `origin/main` 23a088cc, both `--release --features metal`, one exclusive
slot (`--exclusive --mem 24`, fleetq #4830):

| check | result |
|---|---|
| `plan show` text and `--json`, Qwen3-8B and the 35B | all four byte-identical (1,342 / 8,371 B and 2,404 / 10,605 B) |
| 256 greedy ids (`generate --temperature 0 --ignore-eos --show-ids`), short prompt (17 tokens) and a long prompt (6,659 tokens on the 35B, 6,544 on the 8B), both models | all four stdout files and all four id lines byte-identical |

The shared-code change only runs on the batched path (`attn_block_batch`), so a batching-off no-op cannot reach it; the argument for CUDA / HIP is the registry rule instead. `fused_ok` of the row-table prologue call selects a kernel, and `Registry::select` takes a row-table kernel only for a call with `OpCall::rows` and a non-row-table kernel only for a call without (`registry.rs`, `k.row_table != call.rows.is_some()`). CUDA registers `.row_table()` for conv, delta rule, KV append and attention only and HIP for none (`grep -n 'row_table()' crates/intarsia-backend-{cuda,hip}/src/kernels.rs`), so on both `fused_ok` is false for `QkNormRopeKv` and the step runs the chain it ran before. CI compiles both (`cargo check --features cuda` / `hip`). CUDA 8B and 35B (batching off, `--features cuda`, spark2 Qwen3-8B and Qwen3.6-35B-A3B-NVFP4): `plan show` 8B json (8,758 B) and 35B text (3,729 B) byte-identical; 256 greedy ids identical on all four runs (5,000-character and 14,000-character prompts, 8B and 35B). The CUDA batching tests on the branch: `intarsia-core` lib 114 passed, `batch_recurrent` on CUDA 10 of 10, the CUDA `row_tables` G-B 6 of 6. The spark2 trees are the PR head at commit 08301dd4 and `origin/main` 23a088cc (fleetq #4881 stage, #4882 build, #4892 run).

## 4. Cap and what is left of the step

**Cap 16: measured, not taken.** The multi-row GEMV and MoE kernels take 8 rows. Nine to sixteen rows can be served row-exactly as
two chunks of at most 8 rows (each row of a multi-row kernel is bitwise the one-row call whatever the other rows are), so the
call is split in two launches; G-B at M up to 16 passed (fp8 1,440 cases, nvfp4 360, MoE 360, bf16 pair 180, row-parallel 210).
Step time with that change (in process, branch `lane-mr3/cap16-experiment`, fleetq #4826):

| B | 8 | 12 | 16 |
|---|---|---|---|
| ms per step | 30.56 | 47.44 | 59.09 |
| aggregate tok/s | 261.8 | 253.0 | 270.8 |

Sixteen rows gain 3% over eight, twelve rows lose to eight: the chunks read the weights twice. A real 16-row kernel (MB = 16) was
not tried. The cap stays 8 and the experiment stays on its own branch, not in this PR.

**Where the step goes now** (B = 8, 30.2 ms; per-kernel GPU time from `INTARSIA_METAL_PROFILE=kernel`, fleetq #4798): fp8 GEMVs
13.4 ms (130 calls), MoE experts 7.2 (40), nvfp4 GEMVs 5.0 (121), the five row-table kernels together 2.7, the rest 1.9. In the
per-op profile (`real_35b_batch_step_profile`, a sync after every op, so inflated) the four per-row ops were 93.6 of 234.7 ms per
step at B = 8 (delta rule 35.7, conv 33.2, attention 12.9, prologue 11.8) and are 17.8 of 140.7 ms now. The remaining ~2.9 ms per
row is mostly the fp8 / nvfp4 GEMVs: an 8-row fp8 GEMV (8192x2048) takes 117 us against 27 us for one row, twice as fast as eight
separate launches (232 us) but still 4.3x one row (`timing_rows`, #4794).

**Single stream:** 122.4 against main 121.1 / 122.5. The per-step encoder and command-buffer overhead was not touched.

**Expert-grouped MoE kernel:** left alone. Profiling still puts the experts at 7.2 ms of 30.2 (24%) and the serial 0.81x
random-routing measurement of mb stands (`timing_rows` run for this lane: T = 8 random routing 0.82x, identical routing 1.08x).

## 5. Open questions

1. A 16-row GEMV tile (MB = 16 with its own register budget) could make the cap 16 pay; the chunked version is 3% at best. Worth a
   lane only if 9 to 16 concurrent streams on a Mac is a target.
2. Re-run mb's 4 / 8 stream numbers for the engine at `--busy-chunk` values below 2048: the prefill stall of each new request
   (section 1) is now the larger part of the short-output figure.
3. `useResources` of every buffer whose address was handed out is done once per encoder. That set is the recurrent-state and KV
   buffers of the loaded slots; a model with very many such buffers would want a residency set instead.

## 6. Reproduce

```
# G-B
cargo test -p intarsia-backend-metal --release --test row_tables -- --test-threads=1 --nocapture
cargo test -p intarsia-backend-metal --release --test rows_exact -- --test-threads=1
# G-C and the long prompts (35B)
INTARSIA_QWEN36_DIR=~/models/Qwen3.6-35B-A3B-NVFP4 INTARSIA_REQUIRE_GPU=1 cargo test -p intarsia-models --release --test batch_metal_real -- --test-threads=1
INTARSIA_QWEN36_DIR=... cargo test -p intarsia-models --release --test batch_metal_real -- --ignored --nocapture --test-threads=1 real_35b_batched_engine_long_prompts
# step timing and per-op profile
INTARSIA_QWEN36_DIR=... cargo test -p intarsia-models --release --test batch_metal_real -- --ignored --nocapture --test-threads=1 real_35b_batch_step_timing real_35b_batch_step_profile
# speed (inside one fleetq --quiet gpu,cpu job)
intarsia serve --backend metal --model ~/models/Qwen3.6-35B-A3B-NVFP4 --max-batch 8 --served-model-name intarsia --max-seq 8192
uv run --project /opt/code/deepeval localeval speed --url http://127.0.0.1:PORT/v1 --model intarsia --sizes 1024 --gen 512 --reps 3 --concurrency 4,8
```

## 7. Jobs

fleetq #4760 (main in-process timing and profile), #4774 / #4784 / #4794 / #4795 (G-B and timing iterations), #4798 (per-kernel
trace), #4800 (G-C 35B), #4803 (long prompts), #4813 / #4815 (localeval main / branch), #4822 / #4825 / #4826 (cap 16 experiment),
#4830 (Metal no-op, exclusive), #4835 and #4870 (gates, dense, timing), #4873 (workspace), #4881 / #4882 / #4892 (CUDA stage, build, no-op and tests on spark2), #4880 (final G-B), #4883 / #4887 / #4893 / #4896 (single-stream checks). Owner `claude-f31ccf91-mr3`. All
released.
