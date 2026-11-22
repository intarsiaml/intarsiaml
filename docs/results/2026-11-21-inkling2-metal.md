# Inkling-2-Mini end to end on Metal (lane ink-metal, 2026-11-21)

Status: runs end to end on Metal (endor, M5 Max); G1, G2 and G3 short / long pass; first speed numbers and a decode profile
below. Branch `ink/metal` (not pushed, not merged). Op contracts unchanged.

## Briefs received

- 2026-11-21 (coordinator, SendMessage): ink/skel is done at 46349d879 (G1, G2, G3 short / long / 4k pass on cpu; new: ring-safe prompt reuse, faster cpu experts bitwise the same, the accuracy turn format, recipe coverage, results doc; op contracts unchanged). Merge ink/skel at the next clean point, rerun the Metal gates on the merged tip, and message main with the first endor decode tok/s and the per-op profile.

- 2026-11-21 (coordinator, SendMessage): scope from here: (1) finish the per-dispatch trace (#10756) and the localeval run
  (speed-plain, conc-plain-1k vs oMLX 20261120-222536); write the trace path into this doc as soon as it lands (the Fable
  reviewer reads it here). (2) Do not start controller / router fusion or host-gap / command-buffer work (decode graphs
  included): lanes ink-dfuse (`.worktrees/ink-dfuse`: controllers, router fusion, <= 350 dispatches) and ink-hostgap
  (`.worktrees/ink-hostgap`: cb pipelining, GPU sampling, encode-ahead) own them, both branched from a88db4620. (3) This
  lane stays the integration owner: merge lane branches when they report done with passing gates and rerun G1-G3 after each
  merge; next in line are m-mix's final tip and ink/serve aaad4c9b6 (chat / reasoning / tools).

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

Real-model runs on endor through fleetq (#10690, rerun on the final code #10721, same numbers to the last digit), checkpoint
b85c3cfa, W4A16, bf16, `intarsia accuracy --backend metal --max-seq 4096 --golden ... --write-self-golden`; KL21 = the tool's
"KL21 scored" column; confident-row misses from the skel lane's side script (`p1 - p2 >= 0.5` on the golden) over the
self-golden's prefill rows.

| Gate | Golden | KL21 (limit) | confident misses (allowance) | top-1 | cpu skel |
|---|---|---|---|---|---|
| G1(a) | tiny f64 | cpu only (f32 activations; Metal kernels are bf16) | | | pass |
| G1(b) | tiny `ref_dump_bf16_tf`, `tf_check(factor=3)` | 0 failures, all 16 visits; every tensor at about 1/3 of its budget | | | 0 failures |
| G1 state | tiny | chunks 7 / 13 == 16-row passes == prefill 9 + decode, bitwise; rollback by length, reuse at 30, refusal past the margin, reset to 0; interleaved == solo; failed pass resets | | | pass |
| G2 short | oracle W4A16 short | **2.380e-3** (4e-3) | 0 / 363 (1) | 98.95 % | 2.994e-3, 0 |
| G2 long | oracle W4A16 long | **3.418e-3** (6e-3) | 1 / 1206 (2) | 99.17 % | 3.879e-3, 1 |
| G3 short | vLLM short | **8.391e-3** (1.05e-2) | 0 / 367 (1) | 97.89 % | 9.218e-3, 0 |
| G3 long | vLLM long | **1.248e-2** (1.65e-2) | 0 / 1211 (2) | 97.64 % | 1.416e-2, 0 |

Decode mode (teacher-forced decode rows, which exercise the 1-row kernels and the fused attention; reported, not gated):
G2 short 2.589e-3 over 128 rows (cpu: 9.275e-3, one chat_reasoning row at 0.61 nats; Metal's worst row 5.3e-2), G2 long
2.474e-3 over 160 rows. The 4k G3 set was not run on Metal (out of this lane's gate list; cpu passed it).

On the tiny fixture the 40-row single prefill (NAX / prefill kernels) against the 16-row passes free-running is rel l2 0.67,
top-1 12/40: the tiny random net is chaotic under bf16 (spec section i), so that comparison is printed, not gated; the prefill
kernels are gated teacher-forced (G1(b) is one 40-row pass).

Bitwise on real weights (19-token raw prompt, 64 greedy tokens, #10721): `INTARSIA_INK_ATTN_FUSED` unset / `0` / `1` and
`INTARSIA_INK_ROWS_EXACT=0` give the same text (md5 identical). Peak memory footprint of `generate` at `--max-seq 4096`:
16.0 GB (`/usr/bin/time -l`; RSS 27.3 GB counts the mmapped checkpoint).

Greedy text on Metal (W4A16): raw "The capital of France is" -> " Paris.\n\nA. Noun\n\nB. Pronoun\n\nC. Adjective ..." (the
oracle's continuation shape); chat "What is the capital of France?" -> thinking "The user asks a simple factual question:
capital of France. Need answer accurately. France's capital is Paris. ..." then "Paris." (the cpu / oracle text); the
lighthouse prompt continues into a coherent story.

## Speed

### Engine timing (`intarsia generate`, greedy, fleetq #10730 `--quiet gpu,cpu`)

19-token raw prompt, 256 tokens; decode = (tokens - 1) / decode wall time, prefill includes the first-call warm-up.

| Run | prefill | decode tok/s |
|---|---|---:|
| default (`INTARSIA_METAL_COMMIT_EVERY` 128), two runs | 467 / 797 ms | 49.9 / 45.5 |
| `COMMIT_EVERY=512`, two runs | 408 / 416 ms | 48.8 / 51.8 |
| `COMMIT_EVERY=1024`, two runs | 806 / 432 ms | 39.9 / 51.7 |
| `INTARSIA_INK_ATTN_FUSED=0` (pair everywhere) | 405 ms | 50.0 |
| `INTARSIA_INK_ATTN_FUSED=1` | 403 ms | 51.8 |
| 5051-token prompt, chunk 256 (default), 128 tokens | 4144 ms = **1219 tok/s** | **48.3** |
| 5051-token prompt, `--prefill-chunk 2048` | 2274 ms = **2221 tok/s** | 48.7 |

Run-to-run spread is about +-5 % (the two default runs: 49.9 and 45.5), so the commit cadence A/B (128 / 512 / 1024) and the
fused-vs-pair choice are inside the noise at short context; the fused rule saves ~0.35 ms / token by lane ink-m-attn's
harness, which is ~2 %. A larger prefill chunk is a clear prefill win (1.8x at 5k): the recipe default 256 is a candidate
for a later lane (memory: the workspace grows with the chunk; not changed here).

SPEED_REST
