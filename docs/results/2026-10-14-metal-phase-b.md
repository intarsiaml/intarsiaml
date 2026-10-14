# Metal phase B (2026-10-14)

## Phase B part 1: cheap fixes, apples-to-apples baseline, decode profile

**Outcome.** The phase B decode exit is not met and nothing in this part was meant to meet it; this part measured and profiled. On the same harness as the mlx-lm and llama.cpp baselines (localeval `speed` through a server), intarsia-metal BF16 decodes **33.4 / 31.4 / 30.2 tok/s** at 128 / 1k / 4k (pre) and 32.7 / 32.5 / 30.1 (after the cheap fixes, which are off the decode hot path, so the difference is run-to-run noise). That is 93-97 percent of mlx-lm (meets the 90 percent criterion), 94-96 percent of llama.cpp (fails "not below llama.cpp"), and 82 / 77 / 74 percent of the 40.6 tok/s weight-byte floor (fails 85 percent). The decode profile says the step is **98 percent GPU-busy**, 88 percent of the busy time is the bf16 GEMV at 569 GB/s (92.7 percent of 614, already close to the GEMV-only ceiling), and the remaining levers are, in order: decode attention at long context (3.6 ms/token at 4k), the 364 small elementwise launches (about 1 ms/token), and GEMV launch-count/tail effects plus the 0.5 ms inter-step gap (about 1 ms together). `Backend::copy` and `zero` were never on the decode or prefill hot path of this model; they are now GPU dispatches anyway (doctor device copy 143 -> 307 GB/s).

Host endor (Apple M5 Max 40c, 614 GB/s, macOS 27.0), branch `spike/metal`. Model `/Users/josef/models/Qwen3-8B` BF16 (15.1368 GB of weights read per decode token; byte floor at 614 GB/s: 24.65 ms, 40.6 tok/s). Raw logs: `docs/results/raw/2026-10-14-metal-phase-b/`. fleetq: #1800 (localeval pre), #1801 and #1802 (doctor, identity, profile; #1802 is the one used), #1803 (cargo test, kernels test, engine op table, quick qualify), #1806 (localeval after). All exclusive on endor, all released; the servers were stopped by the run script.

Status of the task steps: 0, 1, 2, 3, 4, 5, 6 done; 7 is this document. Not done: nothing from the list; the caveats are under "Limits".

## 1. Apples-to-apples decode (localeval `speed`, step 0 and step 5)

Identical arguments to the baseline runs `20261014-072555-adhoc-llamacpp-bf16` and `20261014-072755-adhoc-mlx-lm-bf16`: `--sizes 128,1024,4096 --gen 256 --reps 3 --thinking off` (1 discarded warmup, median of 3), against `intarsia serve --backend metal --model /Users/josef/models/Qwen3-8B --max-seq 8192 --profile-every 0 --no-supervisor --no-ui` (prefill chunk 256, thinking switched per request by localeval). Server and localeval ran inside one exclusive fleetq slot (`fleetq acquire --host endor --exclusive --mem 24 --wait`, `FLEETQ_JOB=<id>` on both commands). Script: `docs/results/raw/2026-10-14-metal-phase-b/leval.sh`. Run IDs: pre `20261014-094205-adhoc-intarsia-metal-bf16-pre`, after `20261014-100122-adhoc-intarsia-metal-bf16-b1`.

Decode tok/s, median (min-max) of 3 reps, 256 forced tokens; percent of byte floor uses 40.6 tok/s (the weights-only floor, KV reads not included):

| Prompt tokens (target) | intarsia pre | intarsia after (b1) | mlx-lm 0.31.3 | llama.cpp b11146 | pre / mlx-lm | pre / llama.cpp | pre, % floor | after, % floor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ~128 (114 / 114 / 113 / 135) | 33.4 (32.6-33.5) | 32.7 (32.3-32.7) | 34.4 | 34.8 | 97% | 96% | 82% | 81% |
| ~1k (988 / 992 / 989 / 991) | 31.4 (31.0-31.7) | 32.5 (32.2-32.5) | 33.8 | 33.4 | 93% | 94% | 77% | 80% |
| ~4k (4089 all) | 30.2 (29.8-30.3) | 30.1 (29.7-30.3) | 31.0 | 31.7 | 97% | 95% | 74% | 74% |

Reading it: pre and after differ by -2.2, +3.5 and -0.3 percent in decode, and `localeval compare` (`compare-pre-vs-b1.txt`) marks every one of them noise (threshold 10 percent for speed); the changes do not alter any decode launch (section 2; the only decode-path difference is one extra uncontended mutex lock per launch to read the profile flag, about 616 per step). The baselines are single runs of the other engines. Exit criteria (scope doc section 4) on the pre numbers: at least 90 percent of mlx-lm **met** at all three sizes; at least 85 percent of floor **not met** (82 / 77 / 74; note that at 4k the weights-only floor is unreachable for any engine here: mlx-lm is at 76 and llama.cpp at 78 percent there, and 36 layers of KV add about 0.6 GB/token at 4k, so the 85 percent target is only meaningful at the short size); not below llama.cpp **not met** (94-96 percent). To meet "not below llama.cpp" intarsia needs about 28.7 / 29.9 / 31.5 ms per token at 128 / 1k / 4k against 29.9 / 31.8 / 33.1 now (1000 / the pre decode tok/s), i.e. 1.2 / 1.9 / 1.6 ms/token to save.

The qualify quick bench (same binary as the after run, floor taken from the new table, no `--bandwidth-gbs`): 33.5 / 32.2 / 30.4 tok/s = 83 / 79 / 75 percent of floor (`qualify-quick/quick-report.md`), consistent with phase A's 33.3 / 32.1 / 30.3.

## 2. `Backend::copy` and `zero` (step 1)

**Were they on the hot path? No.** Every call site in the workspace (excluding the CPU/CUDA/HIP backends themselves and tests):

| Call | Where | When it runs |
|---|---|---|
| `zero` | `intarsia-core/src/exec.rs:35` (`put`) | host-side op execution: kernel admission, `kernels test`, plan checks; once per test tensor |
| `zero` | `state.rs:109` | allocating a `SequenceState` with a recurrent slot (GatedDeltaNet models only, not Qwen3-8B), once per sequence |
| `copy`, `zero` | `state.rs:171`, `208`, `221` | recurrent-state snapshot and rollback (`snapshot`, `truncate`), once per request at most, GDN models only |
| `copy` | `intarsia-cli/src/cmd/doctor.rs:73` | doctor's bandwidth probe |

Nothing runs per decode or prefill step for a dense attention model. For Qwen3.5/3.6-style hybrid models (phase C) they run per request, not per token. So the fix does not move any speed number; it removes a host wait that is wrong in principle (each call flushed and waited for the whole stream) and makes doctor's device-copy figure meaningful.

**What was done.** Both are now compute dispatches appended to the stream's open serial encoder (`crates/intarsia-backend-metal/src/fill.rs`, kernels `zero_u4`, `zero_bytes`, `copy_bytes` plus the existing `copy_u4` in `kernels/metal/copy.metal`). No flush, no wait. Why a compute kernel and not a blit: the stream is one serial compute encoder; a blit needs that encoder ended and a new one opened around every call (an encoder boundary per call, and a second code path in `Stream`), while a compute dispatch inherits the serial-encoder ordering for free and `copy_u4` already is the bandwidth-ceiling kernel. 16-byte aligned ranges use the vector kernels; unaligned operands and tails under 16 bytes use one-thread-per-byte kernels (correctness path, not fast). Copies whose ranges overlap inside one buffer keep the old host path (flush, wait, memmove), since a parallel kernel cannot give memmove semantics. Read-only wrapped host memory is still refused as a target and works as a source. `upload` and `download` stay host-side with flush and wait. Also fixed: dropping a `MetalBackend` with an open encoder aborted in Metal (`Command encoder released without endEncoding`); `Drop` now synchronizes (a no-op when idle).

**Tests.** `crates/intarsia-backend-metal/tests/fill.rs`, 6 tests: byte-exact copy for 11 alignment/tail combinations plus out-of-bounds errors, a 96 MiB copy, memmove overlap in both directions, zero of 11 sizes (odd sizes, 1 byte to 8 MiB), ordering with launches in one stream (launch, copy, launch, zero, copy, with `pending_dispatches() == 6` proving nothing was waited on), and wrapped read-only memory as source but not target. All pass (`checks/cargo-test-metal.log`: 30 tests, 0 failed).

**Doctor.** Device copy (read plus write bytes, best of 4, 256 MiB, includes commit and wait): **143.3 GB/s -> 306.9 GB/s** (`doctor/doctor-pre.out`, `doctor/doctor-after.out`). It is still far below the 550 GB/s copy kernel because one 256 MiB copy is about 1 ms of GPU work and the timed region includes command-buffer commit, wait and wake-up; a larger buffer or several copies per wait would approach the kernel figure. Not changed here (doctor's method is out of scope); the qualify floor no longer depends on it (section 4).

## 3. Fingerprint covers launcher logic (step 2)

`build.rs` now hashes (FNV-1a 64, file name plus contents) `src/lib.rs` (GEMV variant choice by alignment and K, `plan_gemv`), `src/kernels.rs` (kernel table and predicates), `src/dense.rs` (launch geometry of the dense ops) and `src/fill.rs`, emits it as `INTARSIA_METAL_LAUNCHER_HASH`, and `fingerprint()` becomes `metal|<device>|<arch>|metallib:<hash>|launch:<hash>`. Any edit to those files, comments included, changes the fingerprint (deliberately conservative: it costs one re-admission). The list is `LAUNCHER_SOURCES` in `build.rs`; add a file there when it starts to influence selection or geometry. Cost: build.rs reruns on those edits (about 2 s incl. shader compile). Test: `fingerprint_names_device_and_metallib` asserts the `launch:` suffix. Gap: `fill.rs` takes its thread-group width and `copy_threads` from `bench.rs`, which is not in the list; copy and zero are not admitted kernels, so no cached admission depends on it, but a change there would not alter the fingerprint.

## 4. qualify floor table (step 3)

`floor_inputs` in `crates/intarsia-cli/src/cmd/qualify.rs` has an `applegpu_g17s` entry (614 GB/s, source string names ADR-0009); `docs/adding-a-model.md` updated. Verified by `qualify --quick-only` with no `--bandwidth-gbs`: the report says "bandwidth: 614 GB/s, the Apple M5 Max (40-core GPU) spec figure ..." and prints 83 / 79 / 75 percent of floor.

## 5. Decode profile (step 4)

**Method.** The engine's own per-op mode (`--profile-every 1 --op-table`) synchronizes the stream around every op and times the host wall clock; on this backend that is unusable for decode: the 64-step decode ran at 8.3 tok/s (121 ms/token against 30 unprofiled) and, e.g., `rope` shows 148 us per call against 6.3 us of GPU time (`engine-op-table/`; the table also contains the prefill). So the profile uses a new backend-level GPU-timestamp trace, `crates/intarsia-backend-metal/src/profile.rs`, enabled by `INTARSIA_METAL_PROFILE=<mode>:<file>` (off by default, diagnostic, JSON lines with each command buffer's `GPUStartTime`/`GPUEndTime`):

- `kernel`: every launch is its own command buffer and is waited on before the next is encoded, so records never overlap. Per-op times are the kernel plus Metal's fixed per-command-buffer cost. (A first version that did not wait let independent launches such as gate/up overlap across command buffers and double counted them; that is why the wait is there.)
- `cb`: the stream runs unchanged (one command buffer per step, 616 dispatches); records are whole steps, so step time and GPU idle between steps are exact.

Runs: `intarsia generate --backend metal --max-seq 8192 --profile-every 0 --temperature 0 --max-tokens 129 --ignore-eos`, greedy, prompts `prompt-short.txt` (122 tokens) and `prompt-4k.txt` (4062 tokens), the last 128 decode steps analysed by `analyze.py` (`profile/analysis-short.md`, `profile/analysis-4k.md`, `profile/*.jsonl*`). One run per mode and size. Plain `generate` decode speed for the same runs: 33.4-33.5 tok/s (short), 29.7-30.4 (4k).

**Per-op table, ~122 tokens of context** (616 launches per token; `kernel` mode, isolated kernel times; GB/s is weight bytes over time):

| kernel | n x k | calls/token | us/call | ms/token | share | GB/s |
|---|---|---:|---:|---:|---:|---:|
| matmul_bf16_gemv (gate, up) | 12288x4096 | 72 | 174.2 | 12.54 | 42.5% | 578 |
| matmul_bf16_gemv (down) | 4096x12288 | 36 | 176.6 | 6.36 | 21.5% | 570 |
| matmul_bf16_gemv (q, o) | 4096x4096 | 72 | 60.7 | 4.37 | 14.8% | 552 |
| logits_bf16_gemv (lm_head) | 151936x4096 | 1 | 2092.9 | 2.09 | 7.1% | 595 |
| matmul_bf16_gemv (k, v) | 1024x4096 | 72 | 17.4 | 1.25 | 4.2% | 483 (about 580 after removing the ~3 us fixed cost, below) |
| rmsnorm_add_bf16 | | 72 | 14.9 | 1.07 | 3.6% | |
| attn_decode_bf16 | | 36 | 18.2 | 0.66 | 2.2% | |
| rope_bf16 | | 72 | 6.3 | 0.46 | 1.5% | |
| qk_norm_bf16 | | 72 | 3.8 | 0.28 | 0.9% | |
| kv_append_bf16 | | 72 | 3.0 | 0.22 | 0.7% | |
| argmax_f32 | | 1 | 137.4 | 0.14 | 0.5% | |
| silu_glu_bf16 | | 36 | 2.8 | 0.10 | 0.3% | |
| rmsnorm_bf16, embedding_bf16 | | 2 | | 0.02 | 0.1% | |
| **sum** | | 616 | | **29.55** | | |

Every record carries about 3 us of fixed per-command-buffer cost (the near-empty `silu_glu` and `embedding` records are 2.8 and 3.0 us), so the small-kernel GB/s are understated: k/v is about 14.4 us, roughly 580 GB/s, once it is removed. GEMV total 26.62 ms/token = 15.136 GB at **569 GB/s** (92.7 percent of 614; the spike's GEMV-only ceiling was 94.6 percent); elementwise/row 2.06 ms; attention (decode + kv_append) 0.87 ms.

**At ~4k context** the same table differs in one row: `attn_decode_bf16` 130.0 us/call, 4.68 ms/token (13.3 percent), against 18.2 us at 122 tokens; GEMV 27.80 ms (544 GB/s), elementwise 2.35 ms, attention total 4.94 ms. Full table: `profile/analysis-4k.md`. The K/V of one layer at 4062 tokens is 16.6 MB, which is 29 us at 575 GB/s: the kernel is about 4.5 times its byte floor.

**Where the 29.8 ms/token (33.5 tok/s, ~122 tokens) goes** (`cb` mode: step period 29.84 ms, GPU busy 29.32 ms = 98.3 percent, GPU idle between consecutive steps 0.52 ms, min 0.43, max 0.78):

| Part | ms/token | share of period |
|---|---:|---:|
| GEMV (253 launches, 569 GB/s) | 26.6 | 89% |
| elementwise/row ops (363 launches) | 2.1 | 7% |
| attention (decode + kv append) | 0.9 | 3% |
| isolated-mode overhead not present in-stream (kernel sum 29.55 vs step busy 29.32) | -0.2 | -1% |
| gap between steps: sync return, argmax readback, sampling, next-step encode and commit | 0.5 | 1.7% |
| **period** | **29.84** | |

At 4k the step period is 32.94 ms (30.4 tok/s), busy 32.40 ms (98.4 percent), gap 0.54 ms. The isolated kernel sum at 4k is 35.09 ms, 2.7 ms above the in-stream busy time, so the per-op rows at 4k overstate by about 8 percent (isolation and the GPU clock state after an 18 s prefill are the likely causes; not separated). Read the 4k attention figure as 3.5-4.7 ms/token.

**Do host-side steps create GPU idle gaps?** Yes, but small: 0.5 ms per token (1.7 percent), the same at both sizes. It is the fixed cost of the per-step synchronize (argmax readback), the host bookkeeping and encoding 616 dispatches into a command buffer that is only committed once complete, so the GPU starts each step after the host has finished encoding it. Splitting that 0.5 ms into encode versus sync was not measured; the spike's 0.107 ms for 253 GEMV dispatches suggests encoding is most of it. Within a step there is no GPU idle (a single serial encoder).

## 6. Next levers, by expected ms/token saved

Estimates are from the byte floors and launch counts above, not from prototypes.

1. **Decode attention at long context: about 3.6 ms/token at 4k (11 percent), about 1.1 at 1k (interpolated, not measured), about 0.4 at 128.** 130 us/call against a 29 us byte floor at 4k. GQA-grouped decode (4 query heads share each KV head, so K/V are read once per group instead of four times) plus split-KV for parallelism over the sequence. This single item covers the 4k and most of the 1k gap to llama.cpp.
2. **The 363 small elementwise launches: about 1.0-1.2 ms/token at every size.** `rmsnorm_add` is 72 calls at 14.9 us for an 8 KB row, latency bound (one threadgroup reduction): a faster row kernel is about 0.6 ms; fusing `qk_norm` + `rope` + `kv_append` (216 launches to 72) is about 0.4-0.5 ms; `silu_glu` into the down projection or gate/up fusion saves more launches. Items 2 and 3 are both "cut the dispatch count": the isolated sum (29.55 ms) roughly equals the in-stream busy time (29.32 ms), which suggests the ~3 us per-dispatch cost exists in the single serial encoder too, i.e. about 1.5-1.8 ms/token over 616 dispatches. Together they are what closes the 128-token gap (1.2 ms/token to save).
3. **Cut the dispatch count (GEMV fusion): about 0.5-1 ms/token.** GEMV averages 569 GB/s; the bytes alone at the 595 GB/s the lm_head reaches would be 25.4 ms, 1.2 ms less. With the fixed per-record cost removed the small GEMVs are not slow per byte (k/v about 580 GB/s), so the shortfall is mostly per-dispatch ramp and tail, not GEMV inefficiency: fusing q/k/v into one dispatch (3 to 1 per layer) and gate/up into one removes 108 launches. The 0.5 ms inter-step gap (commit the step in chunks so the GPU starts while the host is still encoding; about 0.3 ms) rides along with this.

Not in the list because the profile does not support them: graph/ICB replay (GPU is already 98 percent busy, encode is not the limit), and a faster GEMV core (already at 93 percent of 614).

## 7. Checks (step 6)

| Check | Result |
|---|---|
| `cargo test -p intarsia-backend-metal --release` (inside fleetq #1803) | 30 tests passed, 0 failed (`checks/cargo-test-metal.log`) |
| `intarsia kernels test --backend metal -m /Users/josef/models/Qwen3-8B` | 16 kernels, 181 cases, all ADMITTED, 21.8 s (`checks/kernels-test.out`); every case's tensors go through the new GPU `zero` |
| greedy `generate`, Qwen3-8B, 48 tokens, 122-token and 4062-token prompts, pre binary (built at `0e25de9`) vs new binary | `.out` byte-identical (`cmp`) and the token ids equal, both prompts (`identity/`) |
| `intarsia qualify --quick-only` (no `--bandwidth-gbs`) | all stages PASS, decode 33.5 tok/s, 83% of floor (`qualify-quick/`) |
| `cargo check --workspace` | passes |

## Limits

- The two localeval runs are single runs per engine; pre/after differences of up to 5 percent are noise. The mlx-lm and llama.cpp numbers are the earlier single runs (same machine, same arguments, about two hours earlier), not rerun.
- Per-op times come from isolated command buffers (a few microseconds of fixed cost per record, GPU clock state after idle); they agree with the in-stream busy time to 1 percent at 122 tokens and overstate by 8 percent at 4k. Single run per mode and size.
- The decode profile was taken with the new binary; the changes in this part do not alter which kernels decode launches (one extra mutex lock per launch aside), and `plain` runs of the pre and new binaries match (32.8-33.5 tok/s).
- The first and third levers' ms/token values are estimates. The 1k attention figure is interpolated.
- Background load on endor (`endor-kev`, `endor-tensorfold`) and thermal state were not checked, as in phase A.
- `profile.rs` and the `INTARSIA_METAL_PROFILE` hooks in `lib.rs` go beyond the listed steps; they were added because the engine's own op table distorts decode (section 5) and per-op GPU time was the point of step 4. Off by default.
- GPU unit tests were first run outside a fleetq slot during development (small, seconds) and again inside slot #1803 for the recorded result.

## Phase B part 2b: launch reduction

Branch `metal-b2-fuse` (base `40996bb`), endor (Apple M5 Max, 614 GB/s), `/Users/josef/models/Qwen3-8B` BF16. Raw logs, scripts and traces:
`docs/results/raw/2026-10-14-metal-phase-b/b2b/`. Goal was to cut decode dispatches and per-launch fixed cost (target at least 1.5 ms/token).
**Result: 616 -> 544 dispatches per step, about 0.5 to 1.0 ms/token faster (1.7 to 3.3 percent); the 1.5 ms target is not met.** Output is
bitwise unchanged: the greedy 256-token ids and the accuracy numbers are identical to `40996bb`.

### What the engine fuses for Qwen3-8B (and what it does not)

The four optional fused core ops are `MatmulPair`, `MatmulFp8Pair`, `MatmulNvfp4Glu` and `GatedRmsnorm`. At `40996bb` the engine invoked **none of them for dense
bf16 Qwen3-8B**: `ffn_block` ran gate, up and `SiluGlu` as three launches without a `try_fused` probe, and the k/v probe in `attn_block` only matched
fp8 weights. `MatmulPair` was only probed for GatedDeltaNet b/a, `MatmulFp8Pair` for fp8 projections, `MatmulNvfp4Glu` and `GatedRmsnorm` for nvfp4 shared
experts and GDN (Qwen3.5/3.6). So registering a Metal `MatmulPair` alone would have changed nothing.

Two additive call sites in `crates/intarsia-core/src/model.rs` now probe `MatmulPair` for gate + up (`ffn_block`) and for bf16 k + v (`attn_block`, next to
the existing fp8 arm; v is computed with k, before the k norm / rope, as the fp8 path already does). No new core op. Not existing: a bf16 pair fused with
`SiluGlu` (`MatmulNvfp4Glu` is nvfp4 only), a q/k/v triple, and any norm/rope/KV-append fold (proposals below).

**Cross-backend effect (not tested here):** the cpu backend registers `MatmulPair` too, so a dense bf16 layer now plans 3 `matmul` + 2 `matmul_pair`
(`crates/intarsia-models/tests/plan.rs` updated, workspace tests pass). HIP and CUDA also register `matmul_pair_bf16_gemv` (`m == 1`, `k % 8 == 0`), so
their dense bf16 decode will now take it. Their tests claim bitwise equality with two GEMVs, but nobody ran them on those boxes in this task; check
before merging if HIP/CUDA dense bf16 matters. `INTARSIA_NO_FUSION=1` turns all of it off.

### Dispatch cost: serial vs concurrent encoder (`examples/dispatch_cost.rs`, fleetq #1810, #1818)

2000 chained dispatches in one command buffer, GPU timestamps, median of 15 (`dispatch-cost-before.log`, `dispatch-cost-after.log`):

| kernel | serial encoder | concurrent + barrier after every dispatch | concurrent, no barrier |
|---|---:|---:|---:|
| `add_bf16`, 32 elements (near-empty) | 0.83-1.05 us | 0.99-1.12 us | 0.10-0.13 us |
| `rmsnorm_add` H=4096, 256 thr (old) | 6.0 us | 6.3 us | 0.12 us |
| `rmsnorm_add` H=4096, 512 thr | 4.1 us | 4.5 us | 0.12 us |
| `rmsnorm_add_bf16_ex16`, 256 thr (new) | 2.7 us | 3.1 us | 0.12 us |

- The fixed cost of a dependent dispatch is about **0.9 us on the serial encoder**; concurrent plus a buffer barrier costs 0.2 us **more**. The stream stays
  on the serial encoder. The "no barrier" column is independent dispatches overlapping, not a per-dispatch cost, and is unsafe for the dependent chains of a decode step.
  A concurrent encoder only pays with dependency-tracked barriers (barrier only on read/write overlap of the tensor ranges): in the layer's op graph that
  frees at most q-branch/k-branch overlap, about 1-2 dispatches per layer (estimate 0.1-0.2 ms), minus about 0.1 ms of barrier cost. Not built.
- **Why `rmsnorm_add` took 15 us:** one threadgroup per row, latency bound, not bandwidth bound. With 2-byte strided loads, two passes (the second re-reads x and
  res, then loads w) and 256 threads it needs 6.0 us even with hot caches (19 us at 64 threads, 4.1 us at 512). The 14.9 us in the phase B1 table is that plus the roughly 3 us
  of a one-dispatch command buffer (kernel mode) plus cold weight loads. Dispatch overhead itself is the 0.9 us above.
- **Fix** (`rmsnorm_bf16_ex*`, `rmsnorm_add_bf16_ex*`, also used for `qk_norm`): same element-to-thread assignment and reduction order as the 256-thread kernels (so the result
  is **bitwise identical**, test `low_latency_norms_are_bitwise_the_256_thread_kernels`), but all loads (x, res, w) are issued before the reduction, values stay in registers for
  the second pass, one barrier instead of two, none for rows of at most 32 elements. Rows up to 8192 elements; wider rows use the old kernel. A first version with 8-element vector loads
  was 2.5 us hot but changed the summation order and flipped greedy tokens after 22 and 49 tokens, so it was dropped for the exact-order form. `rope` is already at the dispatch floor (2.1 us hot); a
  one-thread-per-pair form measured 2.0 us and was removed. `argmax_f32` over 151936 logits: `argmax_f32_v4` (float4 loads, 1024 threads) 64 -> 12 us hot, 143 -> 15 us in the trace; same result,
  including ties and NaN.

### What changed

| change | launches/step | where |
|---|---:|---|
| `MatmulPair` for gate + up and bf16 k + v: `gemv_pair_bf16` (16-byte loads) and `gemv_pair_bf16_a8` (8-byte loads: 15 of the 72 pairs sit in a shard with an 8 mod 16 header), one dispatch, bitwise two `matmul_bf16_gemv` | -72 | `src/fused.rs`, `gemv_bf16.metal`, `model.rs` |
| low-latency norms, argmax | 0 | `dense_ops.metal`, `dense.rs` |
| command buffer committed every 128 dispatches (was 4096, i.e. only at the readback) | 0 | `lib.rs` `MAX_DISPATCHES`; `INTARSIA_METAL_COMMIT_EVERY` overrides |

Registered with HIP's predicate and tolerance (`Tol::new(1.5e-2, 1e-3)`); the launcher falls back to two ordinary GEMV dispatches for any other alignment (tests cover 16-byte, 8-byte, mixed, 2-byte, K % 4 != 0).

### Results

Per category, GPU-timestamp kernel trace (one command buffer per launch, 128 decode steps, `profile/analysis2-*.md`; fleetq #1827, #1833):

| | ctx 128 before | ctx 128 after | ctx 4k before | ctx 4k after |
|---|---:|---:|---:|---:|
| launches per step | 616 | 544 | 616 | 544 |
| GEMV: launches / ms per token | 253 / 27.22 | 181 / 26.38 | 253 / 26.73 | 181 / 26.40 |
| elementwise/row: launches / ms | 255 / 2.19 | 255 / 1.47 | 255 / 2.00 | 255 / 1.48 |
| attention + kv_append: launches / ms | 108 / 0.96 | 108 / 0.84 | 108 / 4.20 | 108 / 4.18 |
| `rmsnorm_add` us per call | 16.1 | 9.4 | 14.7 | 9.5 |
| `argmax_f32` us | 143 | 14 | 134 | 15 |
| gate+up GEMV GB/s | 564 | 585 (pair) | 576 | 585 (pair) |
| k+v GEMV GB/s | 463 | 522 (pair) | 477 | 520 (pair) |
| stream as normal (cb trace): period / GPU busy / idle, ms per token | 30.02 / 29.50 / 0.52 | 29.37 / 29.06 / 0.31 | 32.83 / 32.30 / 0.53 | 32.01 / 31.69 / 0.31 |

Isolated-buffer kernel sums overstate the gain (they drop 1.7 ms because every removed launch also removes about 3 us of per-command-buffer overhead). The in-stream figures are the ones that count, and they are noisy:
interleaved 5-rep runs (`reps-*.log`, period from the cb trace, same slot, base and new alternating) give medians of 30.01 -> 29.54 ms (ctx 128) and 33.21 -> 32.75 ms (4k), i.e. -0.46 ms, while
the commit-cadence sweep (5 reps, binary before the a8 pair) gave 29.95 -> 28.94 ms (-1.0 ms). Run-to-run spread of one variant is about 0.5 ms, so the in-stream gain is **0.5 to 1.0 ms**. Commit cadence alone: idle between steps
0.52 ms (4096) -> 0.37 (256) -> 0.31 (128) -> 0.28 (64); 128 chosen. The remaining 0.3 ms is the host side of the step boundary (wake-up after the readback, sampling, embedding upload, encoding the first 128 dispatches).

localeval `speed` (`--sizes 128,1024,4096 --gen 256 --reps 3 --thinking off`, same arguments as the b1 run, `docs/results/raw/2026-10-14-metal-phase-b/b2b/localeval/`, fleetq #1835):

| ctx | b1 decode tok/s | b2fuse | mlx-lm | llama.cpp |
|---|---:|---:|---:|---:|
| ~128 | 32.7 | **33.8** (33.2-33.9) | 34.4 | 34.8 |
| ~1k | 32.5 | **33.5** (33.2-33.8) | 33.8 | 33.4 |
| ~4k | 30.1 | **30.8** (30.7-30.8) | 31.0 | 31.7 |

Run `20261014-135214-adhoc-intarsia-metal-bf16-b2fuse` (unanimis e9c3c2d6), `compare` against `20261014-100122-adhoc-intarsia-metal-bf16-b1`: decode +2.4 to +3.3 percent, all inside the tool's 10 percent noise band
(1.0 / 0.9 / 0.75 ms/token). Against the other engines: 98 / 99 / 99 percent of mlx-lm and 97 / 100 / 97 percent of llama.cpp. The 1k marginal prefill is flagged "slower" (343 -> 284 tok/s); an interleaved check at 4k (`prefill-check.log`, fleetq #1836) shows no
prefill difference between base and new for any commit cadence (about 220 tok/s), so that figure is the known mid-length prefill noise (up to +-39 percent in fleet history).

### Correctness

- `cargo test -p intarsia-backend-metal --release`: 40 passed, 0 failed (`checks/cargo-test-metal.log`; new: `tests/fused.rs` 5 tests, `tests/lowlat.rs` 5 tests).
- `intarsia kernels test --backend metal -m /Users/josef/models/Qwen3-8B`: 17 kernels, 187 cases, all ADMITTED, including `matmul_pair_bf16_gemv` (6 cases, zero error against the cpu reference) (`kernels-test.out`).
- Greedy `generate` (256 tokens, 122-token and 4062-token prompts) against the `40996bb` binary: stdout and token ids identical (`identity/`).
- `intarsia accuracy --gate-class bf16` on `golden/qwen3-8b-fp32.safetensors` and `qwen3-8b-long-fp32.safetensors`: both binaries give the same numbers (short: top-1 100 percent, mean KL 3.639e-4, PASS; long: top-1 100 percent, KL 3.637e-4, PASS) (`accuracy/`).
- `cargo check --workspace` and `cargo test --release --workspace --exclude intarsia-backend-metal` pass.

### Why the target was missed, and what would get closer

The dispatch floor is about 1 us per dependent launch, so 72 fewer launches is worth about 0.1 ms; the rest of the gain is faster small kernels (0.3 ms), argmax (0.13 ms), pair-GEMV tail effects (about 0.1 ms) and the earlier GPU start (0.2 ms).
GEMV is 26.4 of 29 ms at 570-585 GB/s; reaching the 614 GB/s ceiling is worth 1.5 ms on its own and is a kernel-efficiency job, not a launch-count job.

Proposals (new core ops, not added; savings are per decode step at 1-3 us per removed small launch):
1. `MatmulPairGlu` (x, w_gate, w_up -> silu(gate) * up, bf16-rounding gate and up first so it equals `SiluGlu(Matmul, Matmul)`): -36 launches and the g/u round trip, about 0.1 ms.
2. `QkNormRope` (x `[T,N,D]`, w `[D]` -> roped out; eps, w_offset, rope attrs) for q and k: -72 launches. Fold the k result into the cache as well (`KvAppend` of k after rope as one op, or a `KvAppendPair` writing k and v rows): -36 to -72 more. Together about 0.3-0.45 ms.
3. `MatmulTriple` for q/k/v (one x, three weights): -36 launches and one big dispatch instead of 4096 + 2048 rows, about 0.1-0.2 ms; needs the q projection to move after the k/v ones in `attn_block`.
4. Device-side greedy feedback (argmax output read directly by the next step's embedding, tokens read back asynchronously): removes most of the 0.3 ms step-boundary idle. Engine change.

I did not fold these inside the backend by looking at consecutive launches: it cannot know that an intermediate tensor is dead, and the engine has no hook for it.

### Not done

Dependency-tracked concurrent encoder (estimate above); GEMV tuning; the fold ops; rope, kv_append and silu_glu are at the dispatch floor and untouched. HIP/CUDA behaviour with the new `model.rs` probes is untested.
fleetq jobs: #1810, #1811, #1815-#1818, #1820, #1824, #1826, #1827, #1830-#1836 (the attention agent's jobs in between are not mine).
