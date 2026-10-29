# conv16: bf16 GDN conv state on CUDA, bitwise; B = 16 step -0.9 %

Lane conv16, 2026-10-29, branch `lane-conv16/conv-bf16` from main 07f178dad. spark1 (spark-4e77, GB10 sm_121),
`nvidia/Qwen3.6-35B-A3B-NVFP4`. Follow-up of GDNX (docs/results/2026-10-29-gdnx-decode-kernels.md, "Follow-ups"): vLLM's conv
state is bf16, ours was f32, and the conv bucket was bytes, not kernel quality. **Status: PR (not merged).**

## Change

Every value the GDN conv state holds is a past bf16 input (`ik_b2f(x)`) or zero, so storing it as bf16 rounds nothing and
every value read back is the f32 one.

- `Backend::gdn_conv_dtype()` (`crates/intarsia-core/src/backend.rs`), default **f32**. Only the CUDA backend overrides it
  (bf16; `INTARSIA_CUDA_CONV_BF16=0` opts out, docs/tuning.md). Metal, HIP and CPU keep today's layout byte for byte.
- The recurrent slot of a GDN layer is `[conv state C*(K-1) elements of that dtype | pad to 256 | delta state f32]`:
  `gdn_block::state_bytes` / `state_s_offset` take the dtype, and the solo, segment, speculative-verify and batched
  (row-table) paths build the conv `TensorRef` with it and the delta state at the matching offset (the row table's `byte_off`).
  On the 35B (C = 8192, K = 4) the conv region goes from 96 KiB to 48 KiB per layer and sequence (30 layers: 1.5 MiB per
  sequence). Snapshots (prefix store, spec rollback) copy whole slots, so they follow the layout with no change.
- `kernels/cuda/gdn.cu`: `gdn_conv_kernel`, `gdn_conv_state_kernel` and `gdn_conv_rows_kernel` are templated on the state
  type (`float` or `ik_bf16`); the launchers pick by the state tensor's dtype, so the f32 layout still works (opt-out, the
  admission sweeps and every older test).
- `gdn::check` accepts a bf16 conv state (`ShortConv` and the delta state stay f32).

## Exactness

| gate | result |
|---|---|
| `gdn_conv16` (new, `crates/intarsia-backend-cuda/tests/gdn_conv16.rs`): bf16-state `gdn_conv_bf16` vs f32-state, chunked T = 1..100 on 4 shapes, fresh and non-zero states, and `gdn_conv_bf16_rows` bf16 vs the f32 solo conv, M 1..16, 6 steps | y bytes and state values identical, 72 + 60 calls |
| `batch_cuda_real` 35B (default = bf16 state) | 4290 (row, step) pairs bitwise identical (`raw/batch-real.txt`) |
| `rows_exact` / `row_tables` (incl. `gdn_conv_rows_are_the_solo_conv`) | 10 + 6 passed (`raw/gates-*.txt`); the row_tables conv test still runs the f32 layout |
| `prefix_store_restore_is_bitwise_cold_cuda` (35B, snapshot restore) | restored vs cold (store on): bitwise true (`raw/gates-prefix_store.txt`) |
| served identity main (f32) vs dev (bf16), `scripts/htk/equal.py`, 7 sampling settings x 1 / 4 / 16 streams, 64 tokens | **21/21 cells identical, 9408 token ids** (`raw/ident/`) |
| `plan show` 8B / Gemma-4 / 35B, base vs dev | all byte-identical. The 35B too: `plan show` prints the kernel selection, not slot sizes, and the conv kernel row is the same (`gdn_conv_bf16`). That the bf16 layout is live is shown by the nsys kernel names below. |
| Mac `cargo test --workspace --release --no-fail-fast` (Metal, CPU, HIP host tests) | green |

`ops::tests::check_phase2_folds` asserted that a bf16 conv state is rejected; it now asserts the bf16 state is accepted and an
i32 one is not. No row-table fallback line ("row-table path unavailable") appeared in any gate or timing run.

## Speed

nsys, `serve --max-batch 16`, 16-stream burst, decode window (`raw/dev-b16`, `raw/dev-off-b16`; the same dev binary, the
second with `INTARSIA_CUDA_CONV_BF16=0`):

| kernel | f32 state | bf16 state |
|---|---:|---:|
| `gdn_conv_rows_kernel` (decode, per layer) | 17.2 us (`<float>`) | **14.1 us** (`<__nv_bfloat16>`), -3.1 us |
| `gdn_conv_kernel` (prefill, per call) | 100.1 us | 100.1 us |

That is -0.09 ms per step from the conv kernel alone; the other kernels of the two profiles differ by up to 3 % in
either direction (profile-to-profile noise, e.g. `moe_dec_gate_up` 347 vs 359 us), so the buckets are not a step delta.
The kernel does not reach vLLM's 7.6 us: halving the bytes does not halve the time of a 14 us, latency-bound launch.

`timing_35b_batched_step` (dev binary, env A/B in one window, two interleaved rounds, the second in reverse; the middle of
the test's three 60-step runs, ms; `raw/steptime-w1.txt`):

| arm | B = 16 r1 | r2 | mean | B = 8 r1 | r2 | mean |
|---|---:|---:|---:|---:|---:|---:|
| f32 state (`INTARSIA_CUDA_CONV_BF16=0`) | 44.30 | 44.44 | 44.37 | 27.83 | 27.80 | 27.82 |
| **bf16 state (default)** | **43.98** | **44.00** | **43.99** | **27.71** | **27.56** | **27.64** |

B = 16: -0.38 ms (-0.9 %); B = 8: -0.18 ms (-0.65 %). Both arms ran in the same window and both rounds agree. The step
gain is larger than the conv kernel's -0.09 ms in nsys, and this lane did not separate the rest (smaller state traffic
around the delta kernel is a guess, not measured).

Serve A/B (`scripts/conv16/le-window.sh`, localeval speed, ~1k-token prompts, 256 tokens, 3 interleaved rounds, aggregate
decode tok/s, base = main 07f178dad, dev = this branch, both `--max-batch 16`; raw `raw/w2/`):

| streams | base r1 / r2 / r3 | dev r1 / r2 / r3 | base mean | dev mean | delta |
|---:|---|---|---:|---:|---:|
| 1 | 80.9 / 80.8 / 81.0 | 81.1 / 81.7 / 80.8 | 80.90 | 81.20 | +0.4 % |
| 4 | 172.1 / 173.3 / 173.6 | 173.5 / 175.6 / 170.9 | 173.00 | 173.33 | +0.2 % |
| 8 | 229.0 / 230.1 / 230.7 | 231.0 / 228.0 / 230.8 | 229.93 | 229.93 | 0.0 % |
| 16 | 294.3 / 295.5 / 292.7 | 295.3 / 292.0 / 293.0 | 294.17 | 293.43 | -0.25 % |

No cell is slower than -0.5 %, but the expected +0.9 % at 16 streams is below this A/B's spread (single rounds move by
+-0.5 %), so the serve A/B neither shows nor refutes the step gain.

## Gates

- Exactness (default on): all met, see above.
- Step B = 16 / B = 8 not slower and conv bucket down: met (-0.9 % / -0.65 %; conv 17.2 -> 14.1 us per layer).
- Serve A/B no cell below -0.5 %: met (+0.4 / +0.2 / 0.0 / -0.25 %).

## Process

Builds (`--load cpu`): #6940 (base main 07f178dad + dev), plan show #6947. GPU: #6953 (gates; the script's abort test
looked for `... ok` on the test line, which `--nocapture` output breaks, so it stopped after the gates, 3.3 min), #6957 (W1:
batch_cuda_real, 4 step-timing runs, 2 nsys profiles, 11 min), #6964 (W2: served identity + 3-round serve A/B, 12 min).
`spark1-vllm` was not touched. `~/intarsiaml-conv16` on spark1 was deleted at the end. Scripts: `scripts/conv16/` (GDNX's,
with `gates.sh` and the `c16-` / `ps-` / `chk` build targets; `chk` counts `INTARSIA_CUDA_CONV_BF16` in each binary so a stale
build shows).

## Follow-ups

- The decode conv kernel is at 14.1 us against vLLM's 7.6: latency-bound at 16 rows; a different thread mapping (several
  channels per thread was +11 % slower in GDNX for the f32 kernel) might move it.
- The prefill `gdn_conv_kernel` (100 us per call) is unchanged: it reads x, not the state.
