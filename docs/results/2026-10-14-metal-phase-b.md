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

## Phase B part 2a: decode attention

Branch `metal-b2-attn` (from `40996bb`). Host endor (Apple M5 Max 40c, 614 GB/s), Qwen3-8B BF16: 32 q heads, 8 kv heads, head_dim 128, bf16 KV cache, 36 layers. Raw logs: `docs/results/raw/2026-10-14-metal-phase-b/b2a/`. fleetq: #1812 #1813 #1817 #1819 #1821 (microbench and tuning sweeps), #1825 (kernels test, identity, traces, final microbench), #1828 (localeval).

**Outcome.** Attention per call (in-stream, GPU timestamps): **105 -> 46 us at 4k, 393 -> 135 us at 16k, 32.5 -> 20.7 us at 1k** (2.3x / 2.9x / 1.6x). At 4k that is 59 percent of the 614 GB/s byte floor (365 GB/s effective), at 16k 81 percent (498 GB/s). Decode (localeval `20261014-133256-adhoc-intarsia-metal-bf16-b2attn`) is 32.6 / 32.8 / 31.7 tok/s at 128 / 1k / 4k against b1 32.7 / 32.5 / 30.1: 4k +5 percent (the attention saving is about 1.9 ms of the 33 ms step), 128 and 1k inside the noise. 4k now matches llama.cpp (31.7) and is 2 percent above mlx-lm (31.0); 128 and 1k stay 2-6 percent below mlx-lm (34.4 / 33.8) and llama.cpp (34.8 / 33.4).

### Design

* New registry kernel `attn_decode_bf16_splitkv` (priority 10, needs scratch, `Deterministic`, `BF16_TOL` as HIP's AttnDecode). `attn_decode_bf16` stays registered as the no-scratch kernel for every shape the new one rejects (head_dim not a power of two in 8..256, group over 8, `m != 1`); the new kernel's launcher itself falls back to it for K/V caches that are not 16-byte aligned. Predicate: `m == 1`, head_dim in {8,...,256} power of two, `n_heads % n_kv_heads == 0`, group <= 8.
* **Stage 1** (`kernels/metal/attention_splitkv.metal`, launcher `src/attn.rs`): grid `nkv x nsplit` threadgroups. A threadgroup reads its slice of one kv head's K/V rows once and applies them to all `G = N / Nkv` query heads (GP = G rounded up to 1/2/4/8 heads in registers; q is pre-scaled by `scale * log2(e)`, softmax in the log2 domain). Each lane loads 16 bytes of a row (head_dim/8 lanes per key, 256/head_dim keys per simdgroup step, two steps of loads in flight). The keys-per-step slots of a simdgroup are merged with `simd_shuffle_xor`, the simdgroups through 16 KB of threadgroup memory. 4 simdgroups (128 threads) per threadgroup, 8 when the grid has at most 64 threadgroups. With one split, stage 1 writes the bf16 output itself and there is no second pass.
* **Stage 2** (`attn_decode_bf16_combine`): one threadgroup per q head, one thread per output dim, combines the partials in split order. No atomics, so results are run-to-run identical. Both stages are encoded in one `with_encoder` call: one launch is one profiled record.
* **Split policy** (the same function in launcher and shader): `nsplit = clamp(min(ceil(visible / 128), ceil(120 / Nkv)), 1, 64)`; chunk = ceil(visible / nsplit) rounded up to 32 keys, trailing splits may be empty (they publish max = -inf). For Qwen3-8B: 1 split up to 128 keys, 8 splits at 1k, 15 splits from about 1800 keys on. 120 threadgroups is the sweep optimum on this 40-core GPU (3 resident threadgroups of 4 simdgroups per core); 25 splits = 200 threadgroups was 20 percent slower at 4k than 15-20, i.e. more than one resident wave costs (`sweep/policy-sg4.log`). `INTARSIA_METAL_SPLITKV=<min_keys>,<target>` overrides the policy for sweeps (changes rounding order; unset in production).
* **Scratch** (`KernelImpl::scratch` -> `OpCall::scratch`, nothing allocated by the backend): `float part[N][max_split][D]`, then `float2 ml[N][max_split]`. Declared bound is context-independent, `N * 64 * (D+2) * 4` (1.06 MB for Qwen3-8B); the 15-split steady state touches 250 KB. The launcher returns `Error::invalid` for a missing, too small or misaligned scratch.
* **Device-side position** (`OpCall::pos`): the visible-key count is clamped as in the old kernel; the grid is sized for a full cache and the shader derives the real split count from the position it reads (surplus threadgroups return), so the tensor decides and the host `pos0` only bounds the call. A host-position launch uses the same split count but may use another simdgroup count (4 vs 8), so host and device results agree to rounding order (tests: 1 bf16 ulp), while two device-position launches are bit-identical.
* Shared files touched minimally: `dense.rs` (`Bind`, `bind`, `pos_bind` made `pub(crate)`, one match arm), `kernels.rs` (constants, TABLE entry, registration, shader names), `lib.rs` (`mod attn;`), `build.rs` (`src/attn.rs` added to `LAUNCHER_SOURCES`). The shader is a new file next to `attention.metal`.

### Per-call attention time

In-stream microbench (`crates/intarsia-backend-metal/examples/attn_bw.rs`, `bench/attn_bw.txt`): 36 launches per command buffer, one distinct KV cache per layer so reads come from DRAM like a decode step's; median of 30 reps from the stream's command-buffer GPU timestamps. Bytes are the K+V rows read once, `2 * 8 * visible * 128 * 2`; "GB/s" is that over the time (the old kernel really reads them 4 times).

| visible keys | K+V MB | floor at 614 GB/s us | before us (GB/s) | after us (GB/s, % of 614) | speedup |
|---:|---:|---:|---:|---:|---:|
| 64 | 0.3 | 0.4 | 11.5 (23) | 10.5 (25, 4%) | 1.10x |
| 128 | 0.5 | 0.9 | 10.3 (51) | 11.8 (44, 7%) | 0.87x |
| 256 | 1.0 | 1.7 | 16.7 (63) | 16.5 (63, 10%) | 1.01x |
| 512 | 2.1 | 3.4 | 20.3 (103) | 17.4 (120, 20%) | 1.17x |
| 1024 | 4.2 | 6.8 | 32.5 (129) | 20.7 (203, 33%) | 1.57x |
| 2048 | 8.4 | 13.7 | 56.6 (148) | 31.2 (269, 44%) | 1.81x |
| 4096 | 16.8 | 27.3 | 104.9 (160) | 45.9 (365, 59%) | 2.29x |
| 8192 | 33.6 | 54.6 | 200.6 (167) | 76.4 (439, 71%) | 2.63x |
| 16384 | 67.1 | 109.3 | 393.1 (171) | 134.7 (498, 81%) | 2.92x |

Per-launch GPU-timestamp trace of real decode (`INTARSIA_METAL_PROFILE=kernel`, last 128 steps, `trace/attn-trace.md`; each launch is its own command buffer, so Metal's fixed per-buffer cost and idle-GPU clock state are in the figure, more for the two-dispatch launch than for the old single dispatch):

| context | build | us/call (median, p10-p90) | ms/token (36 calls) | GB/s |
|---|---|---:|---:|---:|
| 1k (968 tok) | before | 37.6 (36.3-38.8) | 1.35 | 112 |
| 1k (968 tok) | after | 35.6 (32.7-36.6) | 1.28 | 119 |
| 4k (4062 tok) | before | 110.0 (108.5-111.8) | 3.96 | 154 |
| 4k (4062 tok) | after | 57.1 (56.3-58.8) | 2.06 | 296 |
| 16k (16578 tok) | before | 404.9 (402.5-410.5) | 14.58 | 168 |
| 16k (16578 tok) | after | 145.8 (143.7-152.6) | 5.25 | 468 |

At 4k and 16k the trace agrees with the microbench within the fixed per-buffer cost; at 1k it hides most of the gain (35.6 vs 20.7 us). The localeval figures are the end-to-end truth.

### localeval (run `20261014-133256-adhoc-intarsia-metal-bf16-b2attn`, same arguments as b1)

`--sizes 128,1024,4096 --gen 256 --reps 3 --thinking off`, `intarsia serve --backend metal --max-seq 8192 --profile-every 0 --no-supervisor --no-ui`, fleetq #1828. Decode tok/s, median (min-max):

| prompt (tok) | b1 (`20261014-100122`) | b2a | change | mlx-lm | llama.cpp |
|---|---:|---:|---:|---:|---:|
| ~128 (114) | 32.7 | 32.6 (32.4-32.8) | -0.3% | 34.4 | 34.8 |
| ~1k (991) | 32.5 | 32.8 (32.7-33.0) | +0.9% | 33.8 | 33.4 |
| ~4k (4110) | 30.1 | 31.7 (31.6-31.8) | +5.3% | 31.0 | 31.7 |

Prefill is unchanged by design (298 / 341 / 223 tok/s; prefill attention is out of scope). Decode at 4k is now about 78 percent of the weights-plus-KV floor (b1: 74 percent).

### Correctness

* `cargo test -p intarsia-backend-metal --release`: all pass (unit 5, attention 10, dense 8, fill 6, gemv 6, wrap_host 3). New attention tests: split-KV against the f64 reference over GQA groups 1/2/3/4/5/6/8 and 9, 16 (fallback), head_dim 8/16/32/64/128/256 and non-power-of-two (96, 30: fallback), contexts 1, 17, 128, 129, 1k, 4k, ~16k, 16-byte-aligned offsets and odd element offsets (fallback); worst error 0.73-0.95 of the half-bf16-ulp tolerance, as for the old kernel. Determinism and agreement with the single-pass kernel (< 2 percent), device-position equivalence at positions 0..4095 with 1 to 15 splits (60 for the 2-kv-head shape), clamping, missing/short scratch rejected, predicate and scratch declaration (independent of the context length).
* `intarsia kernels test --backend metal -m /Users/josef/models/Qwen3-8B`: 17 kernels, 196 cases, all ADMITTED, including `attn_decode_bf16_splitkv` (15 cases, max_abs 3.356e-3, rel_l2 2.358e-4: same as `attn_decode_bf16`) (`checks/kernels-test.out`).
* Greedy `generate` (48 tokens, temperature 0) on Qwen3-8B with the 122-token, 968-token and 4062-token prompts: token ids identical between the `40996bb` binary and the new one (`identity/`, the `ids:` line of the `.err` files; `.out` byte-identical). These prompts are repetitive, so 48 identical tokens are weak evidence on their own; the kernel-level error above is the stronger one, and a different split count changes rounding order only.
* `cargo check --workspace` passes.

### Limits and left undone

* Up to about 224 visible keys the old single-pass kernel is 0-3 us per call faster (11.8 vs 10.3 us at 128 keys, 36 calls per token, at most 0.1 ms/token or 0.3 percent of decode). Not worth a second selection path for now: the registry cannot see the context length (`kv_len` in a plan probe would break graph validity) and a launcher-side switch would leave the one-split direct path without test coverage.
* At 4k the call is 59 percent of the byte floor. Skipping the score shuffles or the exps changes the time by under 5 percent (`sweep/alu-experiments.log`), so it is not ALU-bound; U (loads in flight per lane) 4 and 8 and 2 or 8 simdgroups per threadgroup were slower than the shipped 2 / 4 at 4k and 16k (`sweep/usg-v2.log`). What is left is the fixed ramp/tail of two serial dispatches (about 10 us at 4k) and DRAM efficiency; folding the combine into the next op or a one-dispatch last-block-combines variant (needs an atomic counter) are the next ideas.
* The head_dim 8 path (one lane per key) and group 5/6 paths are tested for correctness only, not tuned.
* The 16k trace needs `--max-seq 17000` (KV about 2.4 GB at 16.6k).
* Bit-for-bit equality of host and device-position launches is not provided (see Design); `OpCall::pos` is not used by the Metal engine path today (no graph replay).
* Tuning sweeps used `INTARSIA_METAL_SPLITKV_U` / `_X` experiment knobs; they were removed before the final build (the sweep logs remain as raw data; `sweep/usg-v2.log` and `sweep/alu-experiments.log` come from the intermediate commit "template U variants and tuning knobs (WIP)").
