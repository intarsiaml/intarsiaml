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

**Outcome.** Attention per call (in-stream, GPU timestamps): **105 -> 46 us at 4k, 393 -> 135 us at 16k, 32.5 -> 20.7 us at 1k** (2.3x / 2.9x / 1.6x). At 4k that is 59 percent of the 614 GB/s byte floor (365 GB/s effective), at 16k 81 percent (498 GB/s). Decode (localeval `20261014-133256-adhoc-intarsia-metal-bf16-b2attn`) is 32.6 / 32.8 / 31.7 tok/s at 128 / 1k / 4k against b1 32.7 / 32.5 / 30.1. `localeval compare` marks all three decode deltas as noise (4k is +5.3 percent, under its 10 percent threshold; `leval/compare-b1-vs-b2a.txt`), so the localeval pair alone does not prove a gain. The evidence is the per-call saving, which is far outside noise and consistent with the 4k step: 36 calls x (110 - 57 us) = 1.9 ms of the 33 ms step in the kernel trace (microbench: 2.1 ms), i.e. 30.1 -> about 31.9 tok/s predicted, 31.7 measured. The attention-floor gap to mlx-lm / llama.cpp that phase B1 identified at 4k (31.0 / 31.7 tok/s, single earlier runs, not rerun) is closed on paper; 128 and 1k stay 2-6 percent below them (34.4 / 33.8 and 34.8 / 33.4).

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

Verdict of `localeval compare` (b1 vs b2a): every row noise or overhead-dominated, including decode 4k +5.3 percent and peak decode +6.2 percent. Prefill is unchanged by design (298 / 341 / 223 tok/s; prefill attention is out of scope). Decode at 4k is now about 78 percent of the weights-plus-KV floor (b1: 74 percent).

### Correctness

* `cargo test -p intarsia-backend-metal --release`: all pass (unit 5, attention 10, dense 8, fill 6, gemv 6, wrap_host 3). New attention tests: split-KV against the f64 reference over GQA groups 1/2/3/4/5/6/8 and 9, 16 (fallback), head_dim 8/16/32/64/128/256 and non-power-of-two (96, 30: fallback), contexts 1, 17, 128, 129, 1k, 4k, ~16k, 16-byte-aligned offsets and odd element offsets (fallback); worst error 0.73-0.95 of the half-bf16-ulp tolerance, as for the old kernel. Determinism and agreement with the single-pass kernel (< 2 percent), device-position equivalence at positions 0..4095 with 1 to 15 splits (60 for the 2-kv-head shape), clamping, missing/short scratch rejected, predicate and scratch declaration (independent of the context length).
* `intarsia kernels test --backend metal -m /Users/josef/models/Qwen3-8B`: 17 kernels, 196 cases, all ADMITTED, including `attn_decode_bf16_splitkv` (15 cases = 3 head configs x positions 0, 7, 40, 2000, 5000 from `intarsia-core/src/admission.rs`, so the 2-pass combine path is compared with the cpu oracle; max_abs 3.356e-3, rel_l2 2.358e-4, identical to `attn_decode_bf16` because the worst element is the output's bf16 rounding) (`checks/kernels-test.out`).
* The registry picks the new kernel in real decode: the `new` kernel traces hold 4608 `attn_decode_bf16_splitkv` records (36 layers x 128 steps) and no `attn_decode_bf16` one.
* Greedy `generate` (48 tokens, temperature 0) on Qwen3-8B with the 122-token, 968-token and 4062-token prompts: token ids identical between the `40996bb` binary and the new one (`identity/`, the `ids:` line of the `.err` files; `.out` byte-identical). These prompts are repetitive, so 48 identical tokens are weak evidence on their own; the kernel-level error above is the stronger one, and a different split count changes rounding order only.
* `cargo check --workspace` passes.

### Limits and left undone

* Up to about 224 visible keys the old single-pass kernel is 0-3 us per call faster (11.8 vs 10.3 us at 128 keys, 36 calls per token, at most 0.1 ms/token or 0.3 percent of decode). Not worth a second selection path for now: the registry cannot see the context length (`kv_len` in a plan probe would break graph validity) and a launcher-side switch would leave the one-split direct path without test coverage.
* At 4k the call is 59 percent of the byte floor. Skipping the score shuffles or the exps changes the time by under 5 percent (`sweep/alu-experiments.log`), so it is not ALU-bound; U (loads in flight per lane) 4 and 8 and 2 or 8 simdgroups per threadgroup were slower than the shipped 2 / 4 at 4k and 16k (`sweep/usg-v2.log`). What is left is the fixed ramp/tail of two serial dispatches (about 10 us at 4k) and DRAM efficiency; folding the combine into the next op or a one-dispatch last-block-combines variant (needs an atomic counter) are the next ideas.
* The head_dim 8 path (one lane per key) and group 5/6 paths are tested for correctness only, not tuned.
* The 16k trace needs `--max-seq 17000` (KV about 2.4 GB at 16.6k).
* Bit-for-bit equality of host and device-position launches is not provided (see Design); `OpCall::pos` is not used by the Metal engine path today (no graph replay).
* Tuning sweeps used `INTARSIA_METAL_SPLITKV_U` / `_X` experiment knobs; they were removed before the final build (the sweep logs remain as raw data; `sweep/usg-v2.log` and `sweep/alu-experiments.log` come from the intermediate commit "template U variants and tuning knobs (WIP)").

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

## Phase B part 2, merged (2a + 2b)

`spike/metal` @ `f1131f1` merges `metal-b2-attn` (`4fa52e1`) and `metal-b2-fuse` (`5d24762`). The conflicts were all additive: kernel tables, `LAUNCHER_SOURCES`, `bind` visibility and this doc. Raw logs are in `raw/2026-10-14-metal-phase-b/b2-merged/`.

**Decode, localeval** (same args as b1: `--sizes 128,1024,4096 --gen 256 --reps 3 --thinking off`), run `intarsia-metal-bf16-b2merged-r2`, fleetq #1844, unanimis `b3111ddb`:

| prompt | b1 | 2a | 2b | merged | mlx-lm | llama.cpp | merged % of 40.6 floor |
|---|---:|---:|---:|---:|---:|---:|---:|
| ~128 | 32.7 | 32.6 | 33.8 | **34.3** | 34.4 | 34.8 | 84.5 |
| ~1k | 32.5 | 32.8 | 33.5 | **33.2** | 33.8 | 33.4 | 81.8 |
| ~4k | 30.1 | 31.7 | 30.8 | **33.1** | 31.0 | 31.7 | 81.5 |

- **Phase B exit status:**
  - ≥ 90% of mlx-lm: met everywhere (98-107%).
  - Not below llama.cpp: indicative only. Merged is +4% at 4k, -0.6% at 1k and -1.4% at 128, but that is one retry run against a baseline taken 7 hours earlier in a different machine state. Closing this criterion needs llama.cpp and intarsia measured back to back in one fleetq slot, with nothing else on the GPU.
  - ≥ 85% of byte floor: not met (81.5-84.5%).
- **Prefill** (marginal) is 347 tok/s at 1k and 203 at 4k, against mlx-lm 3189 at 4k. Unchanged by part 2.
- **The first merged run was discarded.** Run `intarsia-metal-bf16-b2merged`, fleetq #1840, unanimis `e97945ab`, gave 26-27 tok/s. A `generate` probe with the same binary and command gave 33.6 tok/s at 14:04, 28.3 tok/s at 14:12 and 33.9 tok/s at 14:15, so the slowdown came from the machine, not the code. The likely source is Zoom. `zoom.us` had been running since about 11:46, and video encode/decode contends for the GPU intermittently. There was no thermal warning, and the GPU was 2-4% busy between runs. This also splits today's series: the baselines (07:25) and b1 (10:01) ran before Zoom started, while 2a (13:32), 2b (13:52) and both merged runs ran during it. Cross-run deltas therefore span a change in machine state; for example, merged is about 0.5 ms/token faster than 2a and 2b combined would predict. The retry (#1844) is the result above; all numbers are single runs.

**Correctness of the merge:**
- `cargo test -p intarsia-backend-metal --release` and `cargo test --workspace --release --exclude intarsia-backend-metal` pass.
- `intarsia kernels test --backend metal -m Qwen3-8B`: 18 kernels, 202 cases, all ADMITTED (#1838).
- Greedy token ids are identical to `40996bb` for 48 tokens on the short, 1k and 4k prompts, and for 256 tokens on the 4k prompt. The 256-token short prompt first diverges at generated token 49 (id 26471 vs 29994, from the `--show-ids` lists), just past the 48-token window. This is expected: 2a's split-KV kernel replaces the old decode kernel at every context length (the traces show no plain `attn_decode_bf16`), its reduction order differs, so rounding moves and a near-tie eventually flips. The evidence is the decode-only KL shift below.
- Golden accuracy (`--gate-class bf16`, #1839), merged vs `40996bb`: both PASS. The short golden's decode KL rises from 3.639e-4 to 3.890e-4 (top-1 100%), and the long golden goes from 3.637e-4 to 3.627e-4. Prefill is identical. The decode-only shift fits split-KV's different summation order (2a). 2b alone was bitwise identical.

**Before merging to main:**
- The 2b change in `crates/intarsia-core/src/model.rs` routes bf16 gate/up and k/v through `Op::MatmulPair` on every backend that registers it. On CUDA and HIP this switches dense bf16 decode onto their existing `matmul_pair_bf16_gemv`, which is admitted but was never used on this path before. That change has not been run on CUDA or HIP. spark1 has no Qwen3-8B and tatooine has no cargo for this user. `INTARSIA_NO_FUSION=1` turns it off.

**Next levers:**
- GEMV efficiency: 26.4 of about 29 ms is at 570-585 GB/s, against a 614 ceiling. Closing that gap is worth about 1.5 ms.
- The proposed core ops `QkNormRope` plus a KV-append fold (0.3-0.45 ms), and device-side greedy token feedback (about 0.3 ms of step-boundary idle).

## Phase B part 3: GEMV efficiency

Branch `spike/metal` on `649cc37`, code in `7b491e1`. Host endor (Apple M5 Max 40c, 614 GB/s nominal, macOS 27.0), `/Users/josef/models/Qwen3-8B` BF16. Raw logs, scripts and traces: `docs/results/raw/2026-10-14-metal-phase-b/b3/`.

**Outcome.** The decode GEMV got about 1.0 ms per token faster (29.34 to 28.39 ms stream period at a 128-token context, 4 interleaved pairs; 0.3 to 0.8 ms at 4k, 3 pairs). localeval decode, pre and after run back to back in one slot: **34.9 / 34.4 / 33.4 tok/s at 128 / 1k / 4k, against 33.6 / 33.3 / 32.7 for the 649cc37 binary** (+4.0 / +3.3 / +2.1 percent; the tool calls all three noise, below its 10 percent band, but the 128-token stream-period gain is outside the run-to-run spread). That is 86.0 / 84.7 / 82.3 percent of the 40.6 tok/s byte floor. The 85 percent target is met at 128, missed by 0.3 points at 1k and by 2.7 at 4k. The 4k gap is no longer GEMV: the GEMVs now run at 545 to 600 GB/s in the stream and the rest of the 4k shortfall is attention and the small ops (last section).

The change is one idea: **one threadgroup of 8 simdgroups per output row** (`gemv_bf16_rt<2, 8>`, 256 lanes, two 16-byte loads each) instead of one simdgroup per row with 4 rows per threadgroup (`r1u4s4`). The lanes of a threadgroup split K between them and the 8 simdgroup partials are summed in a fixed order through threadgroup memory. A K=4096 row is then one load round trip, and a threadgroup, which is the unit whose tail ends a serial dispatch, is short (8 KB per row instead of 32 KB for four rows, one round trip instead of four). The same form exists for the 8-byte-load kernels (`gemv_bf16_rt_a8`, 4 loads of 8 bytes per lane) and for both `MatmulPair` shaders. The dispatch count (544 per step) is unchanged.

**Accumulation order changes.** K is no longer summed per simdgroup lane in the old order, so GEMV and pair outputs are no longer bitwise those of the old kernels: in the bench's random data (`gemv_b3`, compared against `r1u4s4` on one pool entry) q/o, down and the k+v pair came out bitwise identical and 2 of 24576 gate/up outputs flipped by one bf16 ulp (9.8e-4); other `rt` shapes (`u4s4`, `u2s4`) flipped 1 to 10 outputs of 4096 to 24576. The unit tests check against the f64 reference and the bf16 tolerance, not against the old kernels. The pair kernel is still bitwise two single GEMV launches (the existing `tests/fused.rs` assertions pass unchanged). Accuracy is under Correctness below.

### 1. Where the ms were (real stream, before)

Kernel-mode GPU trace of the 649cc37 binary, 128 decode steps on `prompt-short.txt`, two runs, idle GPU reading 0 to 10 percent before each (`profile/*-kernel-short.jsonl.gz`, `shapes.py`). Kernel mode inflates small ops by a per-command-buffer cost, so the stream-period numbers in section 4 are the totals; the per-call numbers compare before and after under identical conditions.

| shape (calls/token) | before us/call (GB/s) | after us/call (GB/s) | ms/token before | after | saved |
|---|---:|---:|---:|---:|---:|
| q, o 4096x4096 (72) | 59.8 (561) | 58.5 (574) | 4.306 | 4.208 | 0.097 |
| k+v pair 2x1024x4096 (36) | 31.8 (528) | 30.9 (544) | 1.143 | 1.111 | 0.032 |
| gate+up pair 2x12288x4096 (36) | 341.1 (590) | 335.7 (600) | 12.278 | 12.085 | 0.193 |
| down 4096x12288 (36) | 175.2 (575) | 170.2 (591) | 6.309 | 6.129 | 0.180 |
| lm_head 151936x4096 (1) | 2080.6 (598) | 2075.9 (600) | 2.081 | 2.076 | 0.005 |
| **GEMV total** | | | **26.12** | **25.61** | **0.51** |

Per-call figures are kernel-trace medians of the last 100 steps, averaged over the two runs of each binary. The kernel-trace saving is 0.51 ms; the stream saving is about twice that, because the per-dispatch drain between serially dependent dispatches (the tail of one dispatch before the next can start) is part of the stream period and not of an isolated command buffer's time. Shorter threadgroup tails shrink it. Kernel-sum per token: 29.20 / 29.86 ms before (two runs), 28.36 / 28.44 after.

The spike's synthetic chain-serial numbers for the four shapes were 556.7 (q/o) / 588.9 (gate or up alone) / 576.0 (down) / 589.6 (lm_head) GB/s; the real stream before this phase gave 561 / 590 (pair) / 575 / 598, i.e. the stream matched the synthetic chain and the per-dispatch loss it contains was already in those numbers. The loss against 614 GB/s was not uniform. The big shapes (gate/up, lm_head, 100 MB and more per dispatch) were already at 590 to 600 GB/s, which is the practical ceiling of this memory system (the lm_head GEMV never exceeds about 600, the read-only probe in the spike gave 552 to 568). The losses were in the shapes with a short dispatch or a long per-row loop: q/o (60 us per dispatch), k+v (32 us) and down (K=12288, 24.6 KB per row, 4 rows per threadgroup, so a threadgroup runs about 100 KB and its tail is long).

### 2. Levers

All numbers are from `examples/gemv_b3.rs`: the shape's chain of distinct weights (3 GiB pool, 16 to 192 dispatches) in one command buffer with serial dispatch, i.e. what the decode stream does; candidates are measured interleaved round by round with the order rotated each round, and a round starts only after the GPU's idle `Device Utilization %` read at most 10 (the readings per round are in each log). Medians of 12 to 24 rounds, GB/s of weight + x + y bytes. Differences under about 2 percent are inside the spread.

| lever | result | kept |
|---|---|---|
| **Row per threadgroup, SG simdgroups per row (`rt`)**, sweep of UNROLL (1..8) and SG (2, 4, 8, 16) | `rt_u2s8` best or tied everywhere: q/o 583.6 vs 567.8 for `r1u4s4` (+2.8 percent; +4.9 in the first sweep), down 591.6 vs 580.2 (+2.0), k+v pair 565.2 vs 540.2 (+4.6; +10.4 in the first sweep), gate/up pair 599.8 vs 593.8 (+1.0). Neighbours (`u2s4`, `u4s4`, `u4s8`, `u3s8`, `u6s8`, `u2s16`) land within 1 to 2 percent of it; SG=2 variants and `u1s4` are 1 to 4 percent below it (k+v pair: `rt_u2s2` 534.9, `rt_u4s2` 545.4 vs 565.2). | yes, `rt_u2s8` |
| Fewer, smaller threadgroups with the old one-row-per-simdgroup shape (`r1u4s2`, `r1u4s1`, pair `u4s2`, `u4s1`) | +1.5 / +1.1 percent on q/o, -0.1 / -0.5 on down, +3.5 / +5.1 on k+v pair, 0 on gate/up: smaller threadgroups help the short dispatches a little, the per-row serial loop still dominates down. | no (subsumed) |
| Loads in flight per lane (`r1u2s4`, `r1u8s4`, `r1u8s2`, `r1u2s2`; `rt` u1..u8) | u2 on the old shape is 1.4 to 2.8 percent slower, u8 equal to u4. In `rt`, 2 loads per lane at 256 lanes already put a whole K=4096 row in flight at once, and more loads per lane (u3 to u6, fewer lanes) did not help. | no |
| 2 or 4 rows per threadgroup sharing the x load (`rtr2`, `rtr4`, tried, then removed from the tree; fewer x loads per weight byte) | -1.9 to +3.2 percent vs `r1u4s4` on q/o, -3.1 to +1.5 on down, always at or below `rt_u2s8` (`sweep3-rtr.log`). | no |
| x traffic: x staged in threadgroup memory / x loads removed | Staging cannot help `rt` (each lane reads each x chunk once per threadgroup, the threadgroup memory copy would be read once too). A probe kernel with the x loads removed (wrong results on purpose, `gemv_bf16_rtx_u2s8`, removed again) was inconclusive: both runs fell in a degraded stretch (see Limits), the medians favoured the no-x kernel by 1 to 4 percent on q/o and down, the p90s were mixed (run 2: q/o 573.6 vs 580.9, down 594.5 vs 597.3; run 1: down 582.5 vs 596.2 the other way). The evidence that x re-reads are not the limiter is `rtr2`/`rtr4` (row above), which share the x load across rows and never beat `rt_u2s8`. | no |
| **a8 path (shard 1, 8 mod 16)**: 8-byte-load kernel vs 16-byte kernel on the same data | The existing a8 kernel is as fast as the aligned one, in the synthetic chain (q/o 565.5 vs 567.5, down 582.8 vs 579.0, k+v 545.0 vs 543.8, gate/up 592.1 vs 593.3) and in the real stream (layers 0-6, which are entirely in shard 1: q/o 59.7 us vs 59.7 for layers 8-35, k+v pair 31.7 vs 31.7, gate/up pair 340.2 vs 340.6, down 174.9 vs 175.1). Shard 1 holds layers 0-6 and the q, k, v projections of layer 7: 2.75 GB of 2-D bf16 weights outside the embedding (read from the shard header). | a8 `rt` form added so those layers get the same gain (`rt_a8u4s8`: +2.1 / +2.6 / +5.0 / +0.9 percent on q/o, down, k+v, gate/up). |
| **Realigning shard-1 weights at load** (one-time copy to 16-byte-aligned buffers, 2.75 GB extra) | Not implemented: it buys nothing, the a8 kernel is as fast. The memory cost would be real (2.75 GB of 15.1) for zero speed. Every k+v and gate+up pair still runs as one dispatch (the kernel trace has 36 pair records of each shape per step). | no |
| Down projection K=12288: split-K | Splitting K inside the threadgroup is what `rt` does (down +2.0 percent). A split across threadgroups needs a second pass or atomics (changes the dispatch count or determinism); down is already at 591 GB/s, 96 percent of 614 and the same as the gate/up pair, so there is nothing left for it. | not tried across threadgroups |
| Persistent grid sized to the 40 cores | Not tried: with 4096 short threadgroups (q/o, down) the tail is already a fraction of a microsecond of load latency; the remaining gap to the 600 GB/s ceiling is 1 to 3 percent for gate/up and down and 3 to 5 percent for q/o and k+v. | no |
| lm_head (`logits_bf16_gemv`, f32 out) | Left alone, 598 to 600 GB/s. | n/a |
| Concurrent dispatch for independent GEMVs | Not redone. | n/a |

The first and second sweeps are `sweep1-*.log` and `sweep2-qo-down-kv.log`, the a8 comparison `sweep4-a8.log` (offset 8 bound pools for the a8 candidates; the "diff_elems" columns of that file compare different data and are meaningless), the x probe `sweep5-xprobe*.log`. fleetq #1845 held all of them.

### 3. Remaining gap in the stream

After: gate/up 600 GB/s (97.7 percent of 614), lm_head 600, down 591 (96.3), q/o 574 (93.5), k+v pair 544 (88.6). Taking q/o, k+v and down to 600 GB/s would save about 0.4 ms per token more (0.18 + 0.10 + 0.09), roughly +0.5 tok/s, and the per-dispatch drain is what is left of it. The structural way to get the rest is fewer, larger dispatches (a q+k+v triple GEMV, the proposed `MatmulTriple`, 36 fewer drains, 0.1 to 0.2 ms), not a faster GEMV kernel.

### 4. Interleaved A/B in the real stream

`b3/ab.sh`: the engine's `generate` with `INTARSIA_METAL_PROFILE=cb:<file>` (stream as normal), 129 forced tokens, step period / GPU busy / idle from the last 128 steps (`b2b/period.py`), A and B alternated each pair (A B, B A, ...) in the same slot (fleetq #1846). Before each run the idle `Device Utilization %` is read and the run waits until it is at most 10 (the value used is printed). Binaries: `pre` = 649cc37 built into its own target dir, `v1` = this branch (`7b491e1`, byte-identical to the localeval binary).

| context | pair | pre period / busy (ms) | pre tok/s | idle % | after period / busy (ms) | after tok/s | idle % |
|---|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1 | 29.423 / 29.117 | 33.98 | 2 | 28.318 / 28.012 | 35.31 | 2 |
| 128 | 2 | 29.382 / 29.115 | 34.03 | 4 | 28.323 / 28.033 | 35.30 | 2 |
| 128 | 3 | 29.333 / 29.057 | 34.09 | 2 | 28.527 / 28.236 | 35.05 | 5 |
| 128 | 4 | 29.209 / 28.909 | 34.23 | 3 | 28.371 / 28.065 | 35.24 | 3 |
| 128 | mean | **29.337** / 29.050 | 34.08 | | **28.385** / 28.087 | 35.23 | |
| 4k | 1 | 29.808 / 29.533 | 33.06 | 1 | 29.477 / 29.175 | 33.40 | 0 |
| 4k | 2 | 30.354 / 30.056 | 32.48 | 2 | 29.519 / 29.229 | 33.36 | 2 |
| 4k | 3 | 29.981 / 29.678 | 32.89 | 5 | 29.141 / 28.838 | 33.83 | 0 |
| 4k | mean | **30.048** / 29.756 | 32.81 | | **29.379** / 29.081 | 33.53 | |

Saving: 0.95 ms (3.2 percent) at 128, 0.67 ms (2.2 percent) at 4k. Every after run is faster than every pre run at 128; at 4k each pair is faster after (0.33, 0.84, 0.84 ms). The idle gap between steps is unchanged (0.27 to 0.31 ms). Logs `ab-short-1.log`, `ab-4k-1.log`, traces in `ab/`.

### 5. localeval, pre and after back to back

`b3/leval2.sh`: one fleetq slot, the server of one binary up, `localeval speed --sizes 128,1024,4096 --gen 256 --reps 3 --thinking off`, server down, the next binary, same arguments (`intarsia serve ... --max-seq 8192 --profile-every 0 --no-supervisor --no-ui`). Decode tok/s median (min-max):

| prompt | pre r2, `20261014-151912-adhoc-intarsia-metal-bf16-b3pre-r2` | after r2, `20261014-151601-adhoc-intarsia-metal-bf16-b3gemv-r2` | change | after, % of 40.6 floor | mlx-lm / llama.cpp (earlier today) | after vs mlx-lm / llama.cpp |
|---|---:|---:|---:|---:|---:|---:|
| ~128 | 33.6 (33.2-33.6) | **34.9** (34.8-35.0) | +4.0% | 86.0% | 34.4 / 34.8 | 101% / 100% |
| ~1k | 33.3 (33.2-33.6) | **34.4** (34.4-34.5) | +3.3% | 84.7% | 33.8 / 33.4 | 102% / 103% |
| ~4k | 32.7 (32.6-32.7) | **33.4** (33.4-33.6) | +2.1% | 82.3% | 31.0 / 31.7 | 108% / 105% |

Marginal prefill is unchanged (4k: 207 vs 205 tok/s), as it should be: nothing in prefill changed (`localeval compare`: all deltas "noise", `localeval-r2/compare-pre-vs-b3gemv-r2.txt`). This r2 pair ran after / pre (order reversed) in fleetq #1849; unanimis records `c68ae5c9` (after r2) and `2d46d493` (pre r2). The first pair, run pre / after in #1848 (`20261014-150822-adhoc-intarsia-metal-bf16-b3pre`: 33.6 / 32.5 / 32.6; `20261014-151140-adhoc-intarsia-metal-bf16-b3gemv`: 34.7 / 33.3 / 30.2; unanimis `cb17b310` and `6795cd94`), is contaminated at 4k: the after run's 4k prefill dropped 12 percent (197 vs 223 tok/s) and its decode range is 29.8 to 33.7, with nothing changed in prefill, which is the GPU contender below; 128 and 1k of that pair agree with r2 (+3.3 and +2.2 percent). mlx-lm and llama.cpp are single runs from this morning (`20261014-072555-adhoc-llamacpp-bf16`, `20261014-072755-adhoc-mlx-lm-bf16`), a different machine state, indicative only. The merged part 2 section measured the same 649cc37 code at 34.3 / 33.2 / 33.1 earlier today; the pre binary here read 33.6 / 33.3 / 32.7, so the machine state differs by up to 2 percent between those two sessions, which is why the comparison above is pre against after in one slot and not against the merged table.

### 6. Correctness

- `cargo test -p intarsia-backend-metal --release`: 48 passed, 0 failed (`checks/cargo-test-metal.log`). New coverage comes for free from the variant tables: `every_variant_matches_reference` (K=4104, N=1031) runs the new `rt` and `rt_a8` variants against the f64 reference, the `tests/fused.rs` pair tests assert the pair is bitwise two single GEMVs (16-byte and 8-byte aligned, odd N, N1 = 1, K below one block, output offsets).
- `cargo test --workspace --release --exclude intarsia-backend-metal`: 266 passed, 0 failed. `cargo check --workspace`: ok (`checks/`).
- `intarsia kernels test --backend metal -m /Users/josef/models/Qwen3-8B`: 18 kernels, 202 cases, all ADMITTED, `matmul_pair_bf16_gemv` 6 cases max error 0 (`kernels-test.out`). The metallib hash changed, so the admission was redone. That run binds weights at 16-byte alignment under the registered kernel names, so it admits the 16-byte `rt` GEMV and pair kernels; the `rt_a8` kernels are covered by the cargo tests (`unaligned_operand_offsets`, `eight_byte_aligned_weights_are_one_dispatch_of_the_a8_pair`, `every_variant_matches_reference`) and end to end by accuracy and greedy identity, since layers 0-6 and the q, k, v of layer 7 run on them.
- `intarsia accuracy --gate-class bf16`: short golden PASS, prefill KL 2.608e-4 (unchanged), decode KL **4.042e-4** (merged 3.890e-4), top-1 100 percent; long golden PASS, prefill 3.738e-4 (unchanged), decode **4.071e-4** (merged 3.627e-4), top-1 100 percent (`accuracy/`). Decode KL rose by 4 percent (short) and 12 percent (long, 3.627e-4 to 4.071e-4); the long golden had not moved between earlier tree states (3.637e-4 for 2b alone and 3.627e-4 merged), so this rise is real, though it is still about 2.5 times under the 1.0e-3 gate with top-1 at 100 percent. The reorder of the K sum is the only decode change; I did not investigate which prompts carry the rise.
- Greedy ids against the 649cc37 binary, `--show-ids` (`identity/`): 48 tokens on the short and the 4k prompt: ids and stdout identical. 256 tokens: 4k prompt identical; short prompt first differs at generated token 49 (ids 26471 in 649cc37, 29994 here), the near-tie that 649cc37 itself flipped relative to `40996bb` (part 2 merged section); this tree's 256 ids are identical to `40996bb`'s and to 2b-alone's.

### Limits

- **GPU contender.** Zoom (`zoom.us`, started 11:46, a new process again this afternoon) uses the GPU intermittently; the idle utilization read 10 to 30 percent for stretches during this phase and a few of my bench rounds (the x-probe medians, one localeval 4k run) were visibly degraded (q/o `r1u4s4` fell from about 567 to 505 GB/s in `sweep5-xprobe.log`). Idle-gating before a round does not exclude the contender starting mid-round. Medians over interleaved rounds, the p90 where the medians were unusable, and a reversed second localeval pair are the mitigation; the pre/after A/B pairs above all read 0 to 5 percent idle and agree with each other.
- The 1.0 ms stream saving is about twice what the isolated kernel times account for (0.51 ms). The extra half millisecond is attributed to the shorter tails between serial dependent dispatches, which an isolated command buffer cannot show; I did not measure the tails directly.
- 4k shows 2.2 percent and localeval's own noise band is 10 percent, so only the 128 and the paired-stream numbers are outside noise individually.
- The `rt` kernels differ from the old ones in accumulation order (see Outcome); GPU kernels on CUDA/HIP are untouched. The old `r1` GEMV and pair kernels stay compiled in (`GEMV_VARIANTS`, `gemv_pair_bf16`, `gemv_pair_bf16_a8`) for comparison; the registry always picks `rt`.
- Experiment shaders listed in `EXPERIMENT_SHADERS` (kernels.rs) are compiled into the metallib only for `examples/gemv_b3.rs`.

### Next levers (from this phase)

1. 4k is attention and small ops: 4k attention is 2.06 ms/token (59 percent of its byte floor) and the elementwise/row ops about 1.5 ms. Reaching 85 percent at 4k (34.5 tok/s, 29.0 ms) from 29.4 ms needs about 0.4 ms from there: the `QkNormRope` + KV-append fold (0.3 to 0.45 ms) or device-side greedy token feedback (about 0.3 ms of step-boundary idle).
2. `MatmulTriple` (q, k, v as one dispatch): 36 fewer dispatch drains, 0.1 to 0.2 ms.
3. Not done or not tried: a persistent grid, cross-threadgroup split-K, 32-byte (u16) loads per lane, realignment (rejected on the data above), the ADR-0009 text still names `r1u4s4` as the default (historical, left as written). fleetq jobs of this part: #1845 (microbench sweeps), #1846 (A/B, kernel traces, correctness checks), #1847 (aborted: bad output path), #1848 and #1849 (localeval pairs).

## Phase B part 4: qk-norm / rope / KV fold

Branch `spike/metal` on `48fd9ca`, code in `2ed47c2`. Host endor (Apple M5 Max 40c, 614 GB/s nominal, macOS 27.0), `/Users/josef/models/Qwen3-8B` BF16. Raw logs and scripts: `docs/results/raw/2026-10-14-metal-phase-b/b4/`.

**Outcome.** One new optional fused op, `QkNormRopeKv`, replaces the six per-layer launches qk_norm(q), rope(q), qk_norm(k), rope(k), kv_append(k), kv_append(v) with one. **Decode dispatches per step 544 -> 364** (-180, 36 layers x 5). The output is **bitwise unchanged**: same greedy ids at 48 and 256 tokens on both prompts, and the accuracy numbers equal B3's to the printed digit. Stream time per token falls **0.66 ms at 128 context and 0.46 ms at 4k** (interleaved A/B, 4 pairs each). localeval decode, pre and after back to back in one slot, then a second pair in reversed order: **35.7 / 35.3 / 34.3 tok/s at 128 / 1k / 4k = 87.9 / 86.9 / 84.5 percent of the 40.6 tok/s byte floor** (B3: 86.0 / 84.7 / 82.3). 128 and 1k clear the 85 percent exit; **4k misses it by 0.5 points** (34.3 against 34.5 tok/s). The fold was the 0.3-0.45 ms lever from part 2b and delivered more than that (its estimate was per removed dispatch, the dependent-launch tails add to it); what is left at 4k is attention (2.06 ms/token in part 3), not launches.

### 1. Op design and why

`Op::QkNormRopeKv` (`crates/intarsia-core/src/ops.rs`, appended last in `Op` and `ALL` so every existing op keeps its admission-sweep seed):

| inputs | outputs | attrs |
|---|---|---|
| q `[T,N,D]`, k `[T,Nkv,D]`, v `[T,Nkv,D]`, q_w `[D]`, k_w `[D]` (one float dtype) | q_out `[T,N,D]`, k cache `[Nkv,cap,D]`, v cache (both **in place**, like `KvAppend`) | eps, w_offset, theta, rotary_dim, pos0 or `OpCall::pos` (device-side position), kv_cap, n_kv_heads, head_dim |

Defined as exactly `q_out = Rope(QkNorm(q, q_w))`, `KvAppend(Rope(QkNorm(k, k_w)), k cache)`, `KvAppend(v, v cache)`, including the rounding of every intermediate to the input dtype. The cpu implementation is that composition (`fused_ref.rs`, used by both `ref_` and `cpu_f32_`, so the admission reference is the unfused ops by construction).

**One op, not two.** The two-op split from part 2b (`QkNormRope` + `KvAppendPair`) removes 108 launches; the single op removes 180, and it needs no extra intermediate tensors: k never needs a workspace copy because only attention reads it, and attention reads the cache. The cost is a wider op (5 inputs, 3 outputs) and a requirement that both norms exist; models without q/k norms keep the unfused ops. Prefill is not special-cased: the same grid covers T rows (row t sits at `pos0 + t`), so the prefill chunk and both goldens exercise the fold.

**Core change** (additive; the unfused path is untouched):
- `ops.rs`: variant, arity (5,3), validation in `OpCall::check` (shapes, one dtype, caches agree, `pos0 + T <= cap`, even rotary dim), `ShapeInfo` (cache dtype in `weight`, as `AttnDecode`), device position allowed for the op.
- `model.rs`: `qkv_fold` runs right after the q projection. It builds the call, asks `fused_ok` (the same fusion-off / `check` / `Registry::select` test `try_fused` does, now factored out and shared) and returns `false` before emitting anything if the op is not selectable; then the sequence continues exactly as before. If selectable it runs the k/v projections (the existing `MatmulPair` / `MatmulFp8Pair` via `kv_pair`, else two `proj`), then the fused op through `exec_pos` with the decode graph's device position. The order of the unfused fallback is unchanged. `qc` flips once and q_out lands in `ws.q[1 - qc]`, so attention and the output-gate buffer follow as before.
- `admission.rs`: sweep cases (T 1/5/32, pos0 0/37, three head shapes including D=256, rotary 0 and D/4, w_offset 0 and 1, bf16 and f32; fp8 caches are not swept because an f32 versus f64 difference re-rounded into an fp8 step would fail for a reason that is not a kernel bug; `KvAppend`'s own cases cover fp8), plus the exact-shape label and case builder used by plan-time admission.
- Tests that enumerate ops: the CUDA and HIP kernel-table coverage tests now list the op as not registered there (`UNIMPLEMENTED` in HIP, an exclusion in CUDA); `intarsia-models/tests/graphs.rs` counts the fused op for the cpu backend, which registers every op in `Op::ALL` (so the cpu now takes the fold too; its recording shows 2 `QkNormRopeKv` + 2 `AttnDecode` in place of 4 rope + 4 kv_append).

**Metal kernel** `qk_norm_rope_kv_bf16` (`kernels/metal/qk_rope_kv.metal`, launcher in `dense.rs`): one threadgroup per (token, head) over N + 2 Nkv heads; q heads norm, rope and store to q_out; k heads norm, rope and store into cache row `pos0 + t`; v heads copy into the v cache row. `roundup32(D)` threads with element i on thread i, the layout `rmsnorm_bf16_ex1` (what `qk_norm_bf16` runs at D <= 256) uses, with the same fma / `simd_sum` / serial-partials reduction. The normed value is rounded to bf16 before the rotation (the unfused norm stores bf16 and rope reads it back); the rotation partner comes from threadgroup memory. The rope and cache-drop expressions are copied from `rope_bf16` / `kv_append_bf16`. Own parameter struct (`QkRopeKvP`, 80 bytes) because `OpP` has 5 skips and eight bf16 operands need eight. Predicate: bf16 activations and bf16 caches, even head_dim <= 256, bf16 q_out, any T; priority 10. Registered with `BF16_TOL` (1.5e-2 / 6e-3, the budget of the ops it fuses); the admission worst case is 1.5e-3 max_abs / 8.7e-5 rel_l2.

### 2. Dispatches per step

GPU-timestamp trace (`INTARSIA_METAL_PROFILE=cb:<file>`, `period.py`): **544 -> 364 dispatches per decode step** at both 128 and 4k context (the A/B logs print it in the fourth column). This is the only proof the fold is active rather than silently falling back (a fallback would also be bitwise); the engine takes it on all 36 layers.

### 3. Interleaved A/B (fleetq #1854)

`b4/ab.sh`: `pre` (a binary built from `48fd9ca` in its own target dir) against `fold` (`2ed47c2`), A B / B A alternating, 4 pairs per context, cb-mode trace of a 129-token forced decode (128 steps), GPU `Device Utilization %` read before each run (the script waits and resamples above 10; every run started at 10 or below). Step period / GPU busy, ms per token:

| ctx | pair | pre idle % | pre period / busy | fold idle % | fold period / busy |
|---|---:|---:|---|---:|---|
| 128 | 1 | 10 | 28.862 / 28.541 | 10 | 28.128 / 27.816 |
| 128 | 2 | 0 | 28.880 / 28.568 | 0 | 28.158 / 27.845 |
| 128 | 3 | 10 | 28.665 / 28.383 | 10 | 28.105 / 27.820 |
| 128 | 4 | 8 | 28.751 / 28.447 | 0 | 28.143 / 27.843 |
| 128 | mean | | **28.790** / 28.485 | | **28.134** / 27.831 |
| 4k | 1 | 0 | 29.624 / 29.325 | 0 | 29.166 / 28.851 |
| 4k | 2 | 10 | 29.586 / 29.291 | 0 | 29.143 / 28.808 |
| 4k | 3 | 0 | 29.586 / 29.301 | 10 | 29.117 / 28.821 |
| 4k | 4 | 0 | 29.588 / 29.290 | 10 | 29.124 / 28.816 |
| 4k | mean | | **29.596** / 29.302 | | **29.138** / 28.824 |

Gain **0.656 ms (2.3 percent) at 128 and 0.458 ms (1.5 percent) at 4k** in the step period; every pair agrees within 0.1 ms. GPU idle per step is unchanged (0.3 ms: step boundary). That is 3.6 us per removed dispatch at 128 (180 removed) and 2.5 us at 4k; the 4k figure is smaller because the same fixed cost is paid while the longer attention shares less of the gap. Stream period against the 24.65 ms byte floor: 87.6 percent at 128, 84.6 percent at 4k. Logs: `b4/ab-short-1.log`, `b4/ab-4k-1.log`.

### 4. localeval (fleetq #1857 pair 1, #1858 pair 2)

`b4/leval4.sh` (the B3 `leval2.sh` with absolute output paths): `localeval speed --sizes 128,1024,4096 --gen 256 --reps 3 --thinking off` against `intarsia serve --backend metal --max-seq 8192 --profile-every 0 --no-supervisor --no-ui`, one slot per pair, idle utilization 8-10 percent before each server. Decode tok/s, median (min-max); % is of the 40.6 tok/s floor.

| prompt tok | pair 1 pre | pair 1 fold | pair 2 fold (first) | pair 2 pre | fold, % floor | B3 after (649cc37 -> 7b491e1), % floor |
|---|---:|---:|---:|---:|---:|---:|
| ~128 | 34.9 (34.8-34.9) | **35.7** (35.6-35.7) | **35.6** (35.6-35.7) | 34.9 (34.5-35.0) | 87.9 / 87.7 | 34.9, 86.0 |
| ~1k | 34.4 (34.3-34.5) | **35.3** (35.3-35.3) | **35.3** (34.6-35.3) | 34.4 | 86.9 / 86.9 | 34.4, 84.7 |
| ~4k | 33.4 (32.6-33.6) | **34.3** (34.3-34.3) | **34.3** (34.3-34.3) | 33.3 (33.1-33.7) | 84.5 / 84.5 | 33.4, 82.3 |

Gains +2.2 / +2.7 / +2.6 percent (pair 1) and +2.3 / +2.4 / +3.1 percent (pair 2); `localeval compare` marks them noise under its 10 percent speed threshold, but they repeat in both orders and match the stream A/B (-0.66 and -0.46 ms is +2.3 and +1.6 percent). The pre numbers reproduce B3's after numbers (34.9 / 34.4 / 33.4 against 34.9 / 34.4 / 33.4), so the machine state matches. Run ids (results under `/opt/code/deepeval/results/`, summaries copied to `b4/localeval*/`):
- pair 1: `20261014-161833-adhoc-intarsia-metal-bf16-b4pre`, `20261014-162143-adhoc-intarsia-metal-bf16-b4fold`
- pair 2 (reversed): `20261014-162453-adhoc-intarsia-metal-bf16-b4fold-r2`, `20261014-162800-adhoc-intarsia-metal-bf16-b4pre-r2`

Two earlier attempts (#1855, #1856) produced no run: a relative output path failed after the servers were up, so no number comes from them.

**Against the 85 percent exit:** 128 and 1k pass with margin (87.9 and 86.9); 4k is 84.5 percent, 0.2 tok/s (about 0.35 ms/token) short. Not widened: the remaining lever at 4k is the attention kernel (part 3: 2.06 ms/token, 59 percent of its byte floor).

### 5. Correctness

- `cargo test -p intarsia-backend-metal --release`: all pass, including `tests/qkv_fold.rs` (5 tests): the fused kernel against the unfused Metal sequence on identical data, comparing q_out and **the whole caches byte for byte** (untouched rows included; poisoned q_out). Cases: decode at pos 0 / 133 / 4000 / 8100 / 8191 with host and device positions, prefill T = 5 / 64 / 256 / 300 at several `pos0`, D = 64 / 96 / 128 / 256, rotary 0 and D/4, w_offset 0 and 1, theta up to 1e7, all bf16 operands at a 2-byte offset, and a device position at or past the cache end (rows dropped as `kv_append_bf16` does). One dispatch per launch is asserted. **The fold is bitwise the unfused Metal kernels in every case.**
- `cargo test --workspace --release --exclude intarsia-backend-metal`: all pass (the one failure after adding the op, `graphs.rs`, is the updated recording count above). `cargo check --workspace`: ok on endor; `cargo check --workspace --all-targets` on Linux (spark1, git bundle, fleetq #1853): ok (`b4/linux-check.log`; the warnings in it are existing ones).
- `intarsia kernels test --backend metal -m /Users/josef/models/Qwen3-8B` (#1852, `b4/kernels-test.out`): all kernels ADMITTED, including `qk_norm_rope_kv_bf16` (36 cases, max_abs 1.479e-3, rel_l2 8.704e-5).
- `intarsia accuracy --gate-class bf16` (#1852, `b4/accuracy/`): short golden prefill KL **2.608e-4**, decode **4.042e-4**; long golden (8192) prefill **3.738e-4**, decode **4.071e-4**; all PASS. Identical to B3 (2.608e-4 / 4.042e-4 / 3.738e-4 / 4.071e-4), as a bitwise fold must give.
- Greedy ids against the `48fd9ca` binary, `--show-ids`, 48 and 256 tokens on `prompt-short.txt` and `prompt-4k.txt`: stdout byte-identical and `ids:` lines identical in all four (`b4/identity/`, `b4/checks4.log`).

### 6. CUDA / HIP

Unchanged. Neither registers `QkNormRopeKv`, so `fused_ok` finds no kernel and returns before anything is emitted; the unfused sequence then runs in its original order (the fold decision sits after the q projection and before the q norm, the `kv_pair` helper is the old inline match moved verbatim, and the fallback body is the old code re-indented). `try_fused`'s behaviour is the same code path as before. The Linux all-targets check passes and `intarsia-models/tests/plan.rs` (HIP / CUDA op counts) still passes. The only edits in those crates are two test exclusions that list the op as not registered there. Untested on a CUDA or HIP device (none here); the claim rests on the registry having no kernel for the op.

### 7. Not done / limits

- 4k is 84.5 percent, below the 85 percent exit; no attention work was attempted.
- fp8 and f32 KV caches never take the Metal kernel (predicate); they would need the unfused path, which falls to the cpu on Metal as before.
- No GPU test on models with partial rotary / zero-centred norms (Qwen3.5/3.6); the kernel tests cover those parameters synthetically.
- Contender: GPU idle readings were 0-10 percent before every timed run; the A/B pairs, both localeval orders and B3's repeat agree, so no run was discarded.
- fleetq jobs: #1850 (qkv_fold test), #1851 (smoke), #1852 (kernels test, identity, accuracy), #1853 (Linux check), #1854 (A/B), #1855-#1856 (failed localeval attempts, no data), #1857 and #1858 (localeval).
