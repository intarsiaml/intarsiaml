# Inkling-2-Mini on HIP: tensor-core MoE prefill, flash prefill attention and prefill chunk 2048 default-on (lane ink-hip-pfflip)

Status: done; branch `ink/hip-pfflip`, PR opened against main. Three radiance windows, 6 min 19 s down in all.

## Brief and later messages (kept here because compaction loses messages)

- Brief: user decision 2026-11-26 ~07:30 EST ("Yes, flip it"): make the tensor-core MoE prefill (`INTARSIA_HIP_INK_MOE_WMMA`), flash
  prefill attention (`INTARSIA_HIP_INK_ATTN_FLASH`) and prefill chunk 2048 (recipe `[tuning.hip] prefill_chunk`) the Inkling-2-Mini
  HIP default, accepting that greedy output is no longer bit-identical to the old default. The coordinator merges.
- Coordinator message, 2026-11-26 ~11:20 EST: **Phase B is approved** (user: radiance windows now are fine); no further go needed.
  Limits as in the brief: at most 5 min down per window and 10 min in total, radiance back up and answering `/v1/models` before each
  job ends, `HIP_VISIBLE_DEVICES=0`. The coordinator is deleting old folders and cargo volumes on tatooine (a `--load disk` slot) and
  will not touch `~/intarsiaml-ink-hip-pfflip`, the volume `intarsia-cargo-target-ink-hip-pfflip`, `intarsia-cargo-target-registry`,
  `intarsia-cargo-target-67e66c624a8f`, any image, or `/data/models`.
- Coordinator message, 2026-11-26 ~11:45 EST: three more HIP lanes started at 11:45 (ink-hip-spec, ink-hip-batch, ink-hip-pf2; shared rules in
  `/opt/code/intarsiaml/.worktrees/hip-swarm/COMMON-1126.md`). They queue radiance windows on `--host tatooine-r9700` as this lane does, so
  expect to wait in line; use that host only. This lane owns `kWmDefault`, `INK_ATTN_FLASH_DEFAULT` and the new `[tuning.hip]` section; the other
  lanes merge this PR after it lands. The final report starts with a speed table: prefill 4k / 16k and decode short / 4k, main default vs
  this branch's default, with % change.

## Speed: main default vs branch default (one R9700 per run; 3 runs each, mean tok/s)

| | main default | branch default | change |
|---|---|---|---|
| Prefill 4k (4228 tokens) | 613.4 (615.5 / 613.0 / 611.7) | 2729.4 (2722.6 / 2731.1 / 2734.4) | +345.0 % (4.45x) |
| Prefill 16k (16915 tokens) | 539.8 (544.5 / 539.2 / 535.7) | 2843.1 (2846.9 / 2838.9 / 2843.5) | +426.7 % (5.27x) |
| Decode short (31-token prompt, 256 tokens) | 76.93 (76.93 / 76.93 / 76.92) | 76.90 (76.92 / 76.88 / 76.90) | -0.04 % (noise) |
| Decode after the 4k prompt (64 tokens) | 73.19 (72.82 / 73.44 / 73.30) | 73.18 (72.74 / 73.44 / 73.37) | -0.01 % (noise) |

Main default = `intarsia-main` (`87c9745c0`, md5 `3e70c45a`) with nothing set: chunk 256, both knobs off. Branch default = `intarsia-pf` (md5
`492211cf`) with nothing set: chunk 2048 from the recipe, tensor-core experts, flash attention. The 16k prefill against main's
chunk-2048-off number from `docs/results/2026-11-25-inkling2-hip-moe.md` (579 tok/s) is 4.91x. Main's arm ran in window 2 and the
branch's in window 3 (the branch arm of window 2 died on a script bug, `intarsia-branch`, see raw log), not interleaved; main's three
runs agree within 0.6 %, so the order cannot account for the gap. Prefill 4k and 16k runs: 64 and 8 tokens generated.

## Greedy md5 (64 tokens, window 1, `--spec off --ignore-eos`)

| Prompt | M0: main, nothing set | M1: main + `MOE_WMMA=1 ATTN_FLASH=1 --prefill-chunk 2048` | P0: branch, nothing set | P1: branch + `=0 =0 --prefill-chunk 256` |
|---|---|---|---|---|
| 4k | `a7c6a2ac` | `2252566a` | `2252566a` (= M1) | `a7c6a2ac` (= M0) |
| short | `04cdb28a` | `04cdb28a` | `04cdb28a` | `04cdb28a` |

Full md5s: 4k `a7c6a2acedecfac21c77486444810c38` / `2252566a5c54cf52cb564bd214c9af2b`; short `04cdb28a6849728e8aa5be39bf65c07e`. So the
defaults are wired as intended, and `=0 =0 --prefill-chunk 256` is bit for bit the old default. Longer greedy runs are not the old
default's: 256 tokens after the 31-token prompt give md5 `9a27c21b` on the branch against `b4a4f492` on main (flash attention on a
prompt of more than 16 rows), and the 4k prompt diverges within 64 tokens (the md5 above). 16k, 8 tokens: equal on both (`b9852772`).

## G2 at the branch defaults (window 1)

| Golden | 21-way KL | limit | top-1 | verdict | before (docs/results/2026-11-26-inkling2-hip-dec3 window 4, defaults off) |
|---|---|---|---|---|---|
| short (n = 569) | 4.491e-3 | 3.0e-2 | 99.47 % | PASS | 2.566e-3 |
| long (n = 1569) | 4.083e-3 | 3.0e-2 | 98.98 % | PASS | 3.251e-3 |

G3 (the vLLM goldens) was not run: there is no HIP G3 baseline (`docs/results/2026-11-25-inkling2-hip.md`: "G3 ... not run") and the
brief made it conditional on the CUDA flip having run it.

## Admission at the branch defaults (`intarsia kernels test --backend hip --op ...`, window 1), 0 FAIL

| Op | Kernel: cases, max_abs, rel_l2 |
|---|---|
| `moe_experts` | `moe_experts_situ_bf16` 10, 3.472e-3, 2.084e-4; `_rows` 7, 1.748e-3, 1.266e-4; the 35B kernels `moe_experts_bf16` (+ `_v2`, `_v3`, `_rows`) 12 / 4, 4.149e-3, 3.160e-4: all ADMITTED |
| `moe_experts_mxfp8` | `moe_experts_mxfp8_bf16` 19, 3.378e-3, 2.428e-4; `_rows` 14, 2.058e-3, 1.634e-4: ADMITTED |
| `rel_sink_attn` | `rel_sink_attn_bf16` 63, 5.952e-3, 1.581e-3: ADMITTED (flash form at the default) |
| `ink_kv_prologue` | `ink_kv_prologue_bf16` 117, 5.464e-3, 4.346e-4: ADMITTED |

## Tests

| Where | Command | Result |
|---|---|---|
| endor (CPU), fleetq slot | `cargo test --release -p intarsia-backend-hip --lib -p intarsia-models --test recipe_ink_hip_prefill --test recipe_ink_busy_chunk --test recipes_doc --test inkling2` | hip lib 34 passed (incl. `flash_prefill_is_on_by_default_and_zero_reverts`, `set_flash_pins_this_thread_only`), models lib 80, `inkling2` 17, `recipe_ink_busy_chunk` 1, `recipe_ink_hip_prefill` 2, `recipes_doc` 3: all pass |
| tatooine R9700, window 1 | `cargo test --release -p intarsia-backend-hip --test ink_moe -- --test-threads=1` | `ink_moe_matches_cpu_reference`, `ink_moe_rows_exact` (pinned to the decode kernels), `ink_moe_wave_grouping_odd_ids`, `ink_moe_wmma_prefill` pass; the new `ink_moe_wmma_default_and_revert` FAILED: the shape I picked (T 140, H 128, I 64) gives the same bits on both paths, so the test could not tell them apart (its own last assertion). Shape changed to T 100, E 32, K 8, H 192, I 320 |
| tatooine R9700, window 2 | `... --test ink_moe -- ink_moe_wmma_default_and_revert` | passes: resolution (unset 64-pair tile, `0` off, `1` / `32` / `64` / garbage), unset = pinned tensor-core bits, `=0` = decode kernels = `_rows` twin |
| tatooine R9700, window 1 | `INK_ATTN_QUICK=1 cargo test --release -p intarsia-backend-hip --test ink_attn -- --test-threads=1` | 3 passed (`admission_full_sweep` for both attention forms, quick sweep; `flash_prefill_default_and_revert`; `rows_exact_real_shapes` with the split-KV form pinned: 63 cases bitwise), 2 ignored (benches) |

The tiny-model tests of `crates/intarsia-models/tests/inkling2.rs` with `INTARSIA_INKLING2_BACKEND=hip` were not run on the GPU (the
CPU run above passes): that file's `backend()` now pins `set_flash(0)` and `set_wmma(0)` for HIP, because its chunked-vs-whole logits
test is bitwise and one 40-row pass would otherwise take the flash form.

What changed in the tests (the "silently relied on the off default" list): `ink_moe_rows_exact` (set_wmma(0)); all tests of `ink_moe.rs`
now hold a file-level lock because the knob is process-wide and the new test sets the env var; `ink_attn::rows_exact_real_shapes` (set_flash(0));
`ink_attn::admission_full_sweep` (grades both forms); `ink_attn::bench_prefill` (set_flash instead of the env); the HIP arm of
`inkling2.rs`. `set_flash` is per thread so tests that need different forms do not race; `inkmoe::wmma_pairs` (new) exposes the
kernel library's resolution for the default test.

## Notes

- Windows used both R9700 cards at once (card 0: md5, G2, admission, 4k and short speed; card 1: hip tests, 16k speed): radiance spans
  both cards and is down either way, and two streams fit the 5-minute limit. `HIP_VISIBLE_DEVICES` was set per stream (0 / 1), not 0 for all.
- Radiance down: window 1 (#12029) 11:34:01 - 11:37:11 (3 min 10 s), window 2 (#12032) 11:39:02 - 11:41:06 (2 min 4 s), window 3 (#12034)
  11:43:30 - 11:44:35 (1 min 5 s): 6 min 19 s in all, `/v1/models` answered within 20 s after each.
- The lane's first branch build was a no-op (cargo saw the synced files as older than main's build in the same volume and called it
  fresh): `build.sh` now touches the sources first. The `intarsia-pf` md5 above is the real branch build.
- HIP busy chunk today: not applicable. `serve --max-batch` is refused for Inkling-2 on HIP (no row-table kernels; `docs/results/2026-11-25-inkling2-hip.md`),
  so nothing prefills beside decoding streams, and the loader's rule (`model.recurrent_batch()`) never applies to it. The recipe sets none. If batched
  decode lands (lane ink-hip-batch), the loader falls back to 256 unless that lane adds `[tuning.hip] busy_chunk`; Metal's 2048 busy chunk was measured on
  Metal only.
- Not measured: serve-level TTFT and the chunk-2048 workspace at `--max-seq` above 20480 (the 16k run used `--max-seq 20480`).

## Files

`raw/`: `build.sh`, `sync.sh`, `run.sh`, `window.sh`, `phaseA.sh`, `phaseA2.sh`, `stage_gate.sh`, `stage_speed.sh`, logs of fleetq #12026 / #12028
(builds), #12029 (gate), #12032 (speed, main arm only: the branch arm failed on a script bug), #12034 (speed, branch arm).
