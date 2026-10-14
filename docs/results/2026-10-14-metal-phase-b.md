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

Reading it: pre and after differ by -2 percent, +3.5 percent and -0.3 percent, inside the harness's documented decode noise (about 5 percent); neither is a result of the changes (see section 2, no decode-path change). The baselines are single runs of the other engines. Exit criteria (scope doc section 4) on the pre numbers: at least 90 percent of mlx-lm **met** at all three sizes; at least 85 percent of floor **not met** (82 / 77 / 74; note that at 4k the weights-only floor is unreachable for any engine here: mlx-lm is at 76 and llama.cpp at 78 percent there, and 36 layers of KV add about 0.6 GB/token at 4k, so the 85 percent target is only meaningful at the short size); not below llama.cpp **not met** (94-96 percent). To meet "not below llama.cpp" intarsia needs about 28.7 / 29.9 / 31.5 ms per token at 128 / 1k / 4k against 29.8 / 31.8 / 33.1 now, i.e. 1.1 / 1.9 / 1.6 ms/token to save.

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

`build.rs` now hashes (FNV-1a 64, file name plus contents) `src/lib.rs` (GEMV variant choice by alignment and K, `plan_gemv`), `src/kernels.rs` (kernel table and predicates), `src/dense.rs` (launch geometry of the dense ops) and `src/fill.rs`, emits it as `INTARSIA_METAL_LAUNCHER_HASH`, and `fingerprint()` becomes `metal|<device>|<arch>|metallib:<hash>|launch:<hash>`. Any edit to those files, comments included, changes the fingerprint (deliberately conservative: it costs one re-admission). The list is `LAUNCHER_SOURCES` in `build.rs`; add a file there when it starts to influence selection or geometry. Cost: build.rs reruns on those edits (about 2 s incl. shader compile). Test: `fingerprint_names_device_and_metallib` asserts the `launch:` suffix.

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
| matmul_bf16_gemv (k, v) | 1024x4096 | 72 | 17.4 | 1.25 | 4.2% | 483 |
| rmsnorm_add_bf16 | | 72 | 14.9 | 1.07 | 3.6% | |
| attn_decode_bf16 | | 36 | 18.2 | 0.66 | 2.2% | |
| rope_bf16 | | 72 | 6.3 | 0.46 | 1.5% | |
| qk_norm_bf16 | | 72 | 3.8 | 0.28 | 0.9% | |
| kv_append_bf16 | | 72 | 3.0 | 0.22 | 0.7% | |
| argmax_f32 | | 1 | 137.4 | 0.14 | 0.5% | |
| silu_glu_bf16 | | 36 | 2.8 | 0.10 | 0.3% | |
| rmsnorm_bf16, embedding_bf16 | | 2 | | 0.02 | 0.1% | |
| **sum** | | 616 | | **29.55** | | |

GEMV total 26.62 ms/token = 15.136 GB at **569 GB/s** (92.7 percent of 614; the spike's GEMV-only ceiling was 94.6 percent); elementwise/row 2.06 ms; attention (decode + kv_append) 0.87 ms.

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
2. **The 363 small elementwise launches: about 1.0-1.2 ms/token at every size.** `rmsnorm_add` is 72 calls at 14.9 us for an 8 KB row, latency bound (one threadgroup reduction): a faster row kernel is about 0.6 ms; fusing `qk_norm` + `rope` + `kv_append` (216 launches to 72) is about 0.4-0.5 ms; `silu_glu` into the down projection or gate/up fusion saves more launches. Together with item 3 this is what closes the 128-token gap (1.1 ms/token to save).
3. **GEMV launch count and tail: about 0.5-1 ms/token.** GEMV averages 569 GB/s; the bytes alone at the 595 GB/s the lm_head reaches would be 25.4 ms, 1.2 ms less. The shortfall is concentrated in the small kernels (k/v 483 GB/s, q/o 552): fusing q/k/v into one dispatch (3 to 1 per layer) and gate/up into one removes 108 launches and their ramp/tail. The 0.5 ms inter-step gap (commit the step in chunks so the GPU starts while the host is still encoding; about 0.3 ms) rides along with this.

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
- The decode profile was taken with the new binary; the changes in this part do not touch decode launches, and `plain` runs of the pre and new binaries match (32.8-33.5 tok/s).
- The first and third levers' ms/token values are estimates. The 1k attention figure is interpolated.
- Background load on endor (`endor-kev`, `endor-tensorfold`) and thermal state were not checked, as in phase A.
- GPU unit tests were first run outside a fleetq slot during development (small, seconds) and again inside slot #1803 for the recorded result.
