# Inkling-2-Mini end to end on Metal (lane ink-metal, 2026-11-21)

Status: runs end to end on Metal (endor, M5 Max); G1, G2 and G3 short / long pass; first speed numbers and a decode profile
below. Branch `ink/metal` (not pushed, not merged). Op contracts unchanged.

## Briefs received

- 2026-11-21 (coordinator, SendMessage): ink/skel is done at 46349d879 (G1, G2, G3 short / long / 4k pass on cpu; new: ring-safe prompt reuse, faster cpu experts bitwise the same, the accuracy turn format, recipe coverage, results doc; op contracts unchanged). Merge ink/skel at the next clean point, rerun the Metal gates on the merged tip, and message main with the first endor decode tok/s and the per-op profile.

## Integrated

| Branch | Head | What | Conflicts |
|---|---|---|---|
| `ink/skel` | b5e283927 | base: cpu port, model block, loader, recipe, tiny tests | - |
| `ink/m-attn` | 9410a9b67 | InkKvPrologue (70), RelSinkAttn (71, split-KV + NAX prefill), InkAttnDecode (75) | none |
| `ink/m-moe` | 98317825d | MoeRoute, NVFP4 SiTU experts, MXFP8 experts (decode + NAX prefill) | `build.rs` (metal4 sources, launcher sources), `kernels.rs` (metal4 list, TABLE, the count test): kept both |
| `ink/m-mix` | d99d2c0fc (in progress) | InkInProj (73), ConvMix (74), AttnResMix (67), LoopBlend (68), SconvRes (69) | `build.rs`, `kernels.rs`, `lib.rs` (module, launch dispatch): kept all three lanes' entries |

`intarsia kernels test --backend metal --op <op>`, every Inkling op (fleetq #10681): all ADMITTED.

| op | kernel | cases | fail | result |
|---|---|---:|---:|---|
| ink_kv_prologue | ink_kv_prologue_bf16 | 117 | 0 | ADMITTED |
| rel_sink_attn | rel_sink_attn_bf16 / _nax | 63 / 7 | 0 | ADMITTED |
| ink_attn_decode | ink_attn_decode_bf16 | 70 | 0 | ADMITTED |
| moe_route | moe_route_f32 | 19 | 0 | ADMITTED |
| moe_experts | moe_experts_situ_bf16 / _rows | 10 / 7 | 0 | ADMITTED |
| moe_experts_mxfp8 | moe_experts_mxfp8_bf16 / _rows | 19 / 14 | 0 | ADMITTED |
| ink_in_proj | _gemv / _gemv_rows / _nax | 6 / 24 / 23 | 0 | ADMITTED |
| conv_mix | conv_mix_bf16 | 532 | 0 | ADMITTED |
| attn_res_mix | attn_res_mix_bf16 | 97 | 0 | ADMITTED |
| sconv_res | sconv_res_bf16 | 64 | 0 | ADMITTED |
| loop_blend | loop_blend_bf16 | 27 | 0 | ADMITTED |

## Changes in this lane

1. **State plumbing (step 2): nothing Metal-specific was missing.** `SlotSpec::Ring` slots are allocated and zeroed through
   `Backend::alloc` / `Backend::zero` in `SequenceState::new` (backend-generic, `crates/intarsia-core/src/state.rs`); rollback by
   length within the margin (`check_ring_cut`), the ring reset on a cut to 0, the failed-pass reset (`model.rs`, a failed
   Inkling-2 pass starts the sequence over), the `reuse_len` margin rule and the prefix-store refusal of ring slots are all
   backend-generic too. The Metal state tests below (rollback, interleaved, failed pass) exercise them on Metal.
2. **Op 75 selection** (`model/inkling2.rs`, `ink_attn_fused_rule(rows, local, ctx, window)`): per visit, on a pass of up to 16
   rows, the fused `InkAttnDecode` where the rule says so and the backend admits the call (`Registry::select`), else the
   `InkKvPrologue` + `RelSinkAttn` pair. Rule (Metal; the pair elsewhere), keyed on the keys a row reads (min(ctx, window) on
   SWA visits): 1 row fused up to 4096 keys (so every SWA visit and full visits to 4k; 8k / 16k unmeasured, the pair there);
   2..=4 rows up to 512 keys; more rows the pair. Override `INTARSIA_INK_ATTN_FUSED=0|1` (docs/tuning.md). A dump
   (`ink_dump`, the G1 tests) always takes the pair (it needs q / k / v / rel). Both paths give the same bits.
3. **Router on Metal**: Metal has no `Matmul` with an f32 output; where the registry has no kernel for it the router runs as
   `Logits` (same math, `ops.rs`: "same math as `Matmul`, f32 output"). cpu keeps `Matmul`.
4. **Rows-exact small passes on Metal** (`ink_rows_exact`): an Inkling-2 pass of 2..=16 rows, and a head group of 2..=16 rows,
   runs under `exact_rows` (the row-exact kernels: `matmul_bf16_gemv_rows`, `logits_bf16_gemv_rows`,
   `ink_in_proj_bf16_gemv_rows`, `moe_experts_*_rows`), so token-by-token decode, small chunks and a rolled-back
   verify-sized pass leave the same state and the same logits, bitwise. Larger passes (prefill chunks) take the fast kernels
   (NAX). `INTARSIA_INK_ROWS_EXACT=0` turns it off. The head marking is guarded by `self.ink.is_some()`: no other model sees it.
5. **Host `pos0` for prefill**: already the case (Inkling-2 refuses device-position passes), so the NAX prefill attention is
   taken for every chunk above 16 rows (`rel_sink_attn_bf16_nax` in the plan; no fallback to split-KV).
6. **NoKernel refusals**: Metal never had an explicit Inkling refusal; the load-time plan check now resolves every op (plan
   below). The tiny-test assertion "Metal registers none of the Inkling ops" is flipped to "registers every one", and the cpu
   NoKernel test no longer expects a refusal for `InkAttnDecode` (optional: the pair covers it).
7. Kept refused: batched / packed decode, TP, MTP / DSpark, the prefix store, device-position passes.
8. **Decode graphs / command-buffer replay**: not available on Metal for any model. `MetalBackend` does not override
   `Backend::supports_graphs` (false), so `graph_step.rs` never records on Metal; `crates/intarsia-backend-metal/src/decode.rs`
   holds an indirect-command-buffer spike over synthetic Qwen3-8B GEMVs (examples `icb_check`, `gemv_bw`), not wired to the
   model. `generate` reports `graph replays=0`. Metal decode is eager: launches are encoded into command buffers committed
   every `INTARSIA_METAL_COMMIT_EVERY` (default 128) dispatches. Enabling replay for Inkling needs (a) a Metal graph backend
   (ICB recording of arbitrary compute pipelines, or encoder reuse) and (b) device-position variants of every
   position-dependent Inkling op (InkKvPrologue / RelSinkAttn / InkAttnDecode take `OpCall::pos` already; ConvMix and the
   conv rings read `pos0`), plus removing the Inkling refusal of device-position passes. That is not a small step, so it is
   left to a later lane; the commit cadence A/B is below.
9. Tests: `crates/intarsia-models/tests/inkling2.rs` takes `INTARSIA_INKLING2_BACKEND=metal`; the test wrapper backend now
   forwards the Metal scheduling hints and `wrap_host` (without it the uploaded weights took another GEMV variant than the
   zero-copy ones, a 5e-8 logits difference). docs/tuning.md rows for the new variables and for the m-* lanes' knobs that
   `docs_env_vars` flagged (INTARSIA_INK_MOE, _MOE_NAX, _SPLIT_KEYS, _NAX_SG, _MIX_NE, _PROJ_CFG / _P / _RB / _GRID).

`plan show --model ~/models/Qwen3-8B --backend metal`: byte-identical stdout and stderr, base (b5e283927, built with
`--features metal`) vs this branch.

## Gates

GATES_PLACEHOLDER

## Speed

SPEED_PLACEHOLDER
