# Lane mpf: the macro prefill pass runs with the MTP head; `--spec mtp` becomes the default for the Qwen3.6-35B-A3B on CUDA (tp=1)

Date 2026-10-26 evening (EDT; file dated 2026-10-27 as the lane was named). Box spark1 (GB10), model `nvidia/Qwen3.6-35B-A3B-NVFP4`, single-sequence engine, tp=1. Follows lane cmtp
(`2026-10-26-cmtp-cuda-mtp-default.md`): every decode gate passed there but a loaded MTP head disabled the macro prefill pass (prefill -20 / -37 / -51 % at 4k / 16k / 34k), so the default was left off.
Raw logs: `2026-10-27-mpf-macro-prefill-mtp/raw/`. Scripts `scripts/mpf/`, tables `tools/mpf/lc_table.py`, `tools/cmtp/round_table.py`, `tools/dfh/cat_table.py`.

## 0. Result

**All five gates pass; `[tuning.cuda] mtp_default_k = 2` is set in `recipes/qwen3_5_moe.toml`**: `--spec mtp --spec-k 2` is the default for the Qwen3.6-35B-A3B NVFP4 on CUDA at tp=1 (single sequence, no `--spec`;
`--spec off` / `INTARSIA_SPEC_DEFAULT=0` opt out; a configured prefix store, `--max-batch` above 1 and checkpoints without the head keep plain decode). The prefill cost that kept it off in lane cmtp is gone.

| gate | result |
| --- | --- |
| 1 ids: 6 short, 6 ~4k, 6 ~16k, 3 ~34k | 21/21 identical to `--spec off`, with explicit `--spec mtp` and with the shipped default (no flag) |
| 2 soak, 40 greedy requests (one prompt ~3.9k tokens, so the macro pass runs) | 40/40 identical, equal to spec-off |
| 3 mutants | 3 CUDA mutants of lane cmtp + 1 new in the macro head-ingest path (`macro-head-skip`) caught on CUDA (hashes differ from clean); 3 more head-ingest mutants and the two older shared-logic mutants caught on CPU |
| 4 speed | prefill tok/s 0.995 / 0.994 / 0.995 of spec-off at 4k / 16k / 34k (worst round 0.990), TTFT +0.4 / +0.6 / +0.5 %; decode 1.17-1.53x in all 7 categories (worst round 1.163x) |
| 5 G0 | CUDA 8B, Gemma-4, 35B opt-outs byte-identical to main (also with the default on); Metal 8B and 35B byte-identical; HIP untouched |

Two attempts closed the prefill gap from 0.80 / 0.63 / 0.49 (lane cmtp) to 0.995: the macro pass with the head (this lane, first attempt at 0.97-0.96), then KV-only head ingest of prompt rows (0.990), then skipping the q projection in it (0.995).
The remaining 0.5 % is the head's fc projections, norms and K / V projections over every prompt row. Prefill measured with a 1 % gate is within it, but not far inside: the worst single round is exactly 0.990 at 4k.

## 1. What changed (code)

1. `Model::load` no longer zeroes the macro rows when the MTP head is loaded (`crates/intarsia-core/src/model.rs`), and the macro group no longer falls back to chunk-by-chunk when the sequence has MTP state.
   After a macro pass the drafter's queue is fed with the pass's hidden rows exactly as after a plain chunk (`mtp_after_forward(st, &hn, total, true, ..)`), and the plan probe of the head is capped at one chunk
   (the head ingests a macro pass in chunk-sized pieces).
2. The head's catch-up over a prompt's rows keeps only what the cache needs. A catch-up piece whose output nobody reads (every piece but the one that reaches the last queued row of a drafting catch-up) runs
   `mtp_layer_kv_only`: embedding and norms, the two fc projections, pre-norm, then the attention block's K / V projections, k-norm, rope and cache append (`attn_block_kv_only`, the dead-tail machinery),
   and nothing else (no attention, no q projection, no output projection, no MoE FFN, no final norm). The cache is the full layer's (same projection and rope calls).
   Attempt 1 (KV-only, q projection still run): prefill 0.990-0.991x of `--spec off`. Attempt 2 (also skip the q projection, `kv_only_skip_q`): 0.994-0.995x (section 5).
3. A prompt's rows (more rows than any verify pass) take the ordinary multi-row attention in the head; the per-row decode calls of the short post-verify catch-up (bitwise the captured step) are kept for short catch-ups only.
   Per-row decode over a whole prompt was quadratic.
4. Default: `[tuning.cuda] mtp_default_k = 2` in `recipes/qwen3_5_moe.toml` (mechanism, `INTARSIA_SPEC_DEFAULT=0` / `--spec off` opt-outs, `--max-batch`, prefix-store rule and lone `--spec-k` from lane cmtp, #175, #179).
5. Tests: `mtp_model_ignores_the_macro_prefill_knob` is replaced by `mtp_model_takes_the_macro_prefill_pass_and_the_head_follows` (`crates/intarsia-models/tests/mtp.rs`): the MTP model runs a macro pass
   (`macro_passes()` grows), its logits and the head's K / V cache equal the chunked schedule's bit for bit, the head's cache built in chunk-sized pieces equals the one-pass build (tolerance 1e-3; a row at a wrong
   position differs by O(1)), and the engine's greedy ids equal `--spec off`, the chunked MTP engine and the one-pass MTP engine, with the same proposals and acceptances. `tp_mtp.rs`' recipe-pair test now expects
   `qwen3_5_moe cuda tp1 2` and `qwen3_5_vl hip tp 2`.

Behaviour change outside the default: an explicit `--spec mtp` now takes the macro pass wherever a recipe sets `prefill_macro` (CUDA 35B: 8192). It did not before.

## 2. Gate 1: ids (21 prompts), `raw/ids-g1.log`, `raw/ids-auto.log`

256 greedy ids (`--ignore-eos`, `--temperature 0`, chat template, thinking off), `generate`, `--max-seq` 4096 / 8192 / 20480 / 36864. Prompts `scripts/mpf/ids-prompts`: 6 short (34-69 tokens), 6 ~4k (3.8-4.3k),
6 ~16k (15.5-16.3k), 3 ~34k (33.1-34.3k). The first run of the 34k prompts used the default 20480 context and failed in both arms (`raw/ids-g1-prekvonly-34k-invalid.log`); `ids.sh` now sets 36864 and the run below is the valid one.

| arm | prompts | result |
| --- | --- | --- |
| explicit `--spec mtp --spec-k 2` vs `--spec off`, final code (target-dev) | 21 | 21/21 identical (`ids-g1.log`) |
| no flag (the shipped default, target-flip) vs `--spec off` | 21 | 21/21 identical, each with the default line and `spec:` stats (`ids-auto.log`, off ids reused from the first row) |
| earlier code (macro pass, head ingest not yet KV-only), prompts short to 16k | 18 | 18/18 identical (`raw/ids-g1-prekvonly-34k-invalid.log`) |

Acceptance 50-94 %, 2.0-2.9 tokens per verify pass, as in lane cmtp. Every default run prints the `default of recipe qwen3_5_moe on cuda at tp=1` line and the `spec:` stats line.

## 3. Gate 2: soak, `raw/soak/`

One server per arm (`serve --max-batch 1`, `--max-seq 8192`), greedy chat, 256 forced tokens; two prompts alternating, prompt 1 a ~3.9k-token document (two prefill chunks, so the macro pass runs with the head).
`--spec mtp --spec-k 2`: 20 + 20 requests, one distinct answer per prompt (`2664ec19db205d54`, `613987a838862642`), equal to the 2 + 2 answers of the `--spec off` server: 40/40.

## 4. Gate 3: mutants

| mutant | where | caught by | result |
| --- | --- | --- | --- |
| `macro-head-skip` (macro pass does not hand its hidden rows to the drafter) | `model.rs` | CPU `mtp_model_takes_the_macro_prefill_pass_and_the_head_follows` | caught; CUDA: 2/2 prompts (4k, 34k) fail at the first request with `MTP state at 0 + 1 queued rows ... sequence at N` |
| `head-kvonly-pos` (KV-only ingest writes at position 0) | `model/mtp.rs` | CPU, same test (the K / V comparison against the one-pass build) | caught (survived until that comparison was added) |
| `head-kvonly-last` (KV-only used for the piece the first draft needs) | `model/mtp.rs` | CPU `captured_spec_step_drafts_equal_the_eager_drafts`, `eager_head_catch_up_is_per_row_attention`, `reduced_draft_vocab_keeps_output_and_drafts_inside_the_prefix` | caught |
| `head-skipq-sticky` (the skip-q flag stays set after the KV-only ingest) | `model/mtp.rs` | CPU `tests/mtp.rs` (several) | caught |
| `rollback-skip`, `accept-off-by-one` (lane cmtp's) | `state.rs`, `spec.rs` | CPU `tests/mtp.rs`; CUDA ids | caught; CUDA: ids differ on 3/3 prompts each (short, 4k, 16k) |
| `cuda-perrow-off` (CUDA verify attention back to multi-row) | `device.rs` | CUDA ids at ~16k | ids differ on 3/3 16k prompts |

CUDA binaries are built in a copy of the tree (`target-mut`, never the measured binary); sha256 of the clean build and of each mutant differ (`raw/mutant-build-*.log`: clean `0b0eb31807e378ef`; `rollback-skip` `8b0d53cea2b3dbed`, `accept-off-by-one` `685387f511647b48`, `cuda-perrow-off` `3fe385ad06e38dc0`, `macro-head-skip` `cb1ad1fe3afb5094`; ids `raw/mutant-ids-*.log`). CPU: `raw/mutants-cpu*.log`.

## 5. Gate 4: speed

### Long context (localeval speed, 3 interleaved rounds x 5 reps, `raw/longctx/`, `scripts/mpf/longctx.sh`), `--spec off` vs `--spec mtp --spec-k 2`

Median over the 3 rounds; TTFT is prompt tokens / prefill tok/s (localeval's definition; its printed TTFT has two decimals):

| prompt tokens | prefill tok/s off | mtp | x off | TTFT off | mtp | x off | decode off | mtp | x off | worst round, prefill x off |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4.1k | 6980 | 6948 | 0.995 | 0.586 s | 0.588 s | 1.004 | 80.1 | 102.3 | 1.28 | 0.990 |
| 16.7k | 6645 | 6605 | 0.994 | 2.507 s | 2.522 s | 1.006 | 72.7 | 77.9 | 1.07 | 0.992 |
| 33.9k | 5893 | 5862 | 0.995 | 5.753 s | 5.784 s | 1.005 | 65.7 | 69.1 | 1.05 | 0.991 |

Before this lane (lane cmtp, same method): prefill 0.80 / 0.63 / 0.49 of off, TTFT +0.15 s / +1.5 s / +6.0 s. Attempt 1 (KV-only ingest): 0.990-0.991 (`raw/longctx-b`); attempt 2 (skip q): 0.994-0.995 (`raw/longctx`, `raw/longctx-c`).
The remaining 0.5 % is the head's two fc projections, norms, and K / V projections over every prompt row. The decode column is the localeval filler-text sweep (acceptance not representative of chat).

### Decode per category (`raw/s1/`, `scripts/mpf/window.sh`: one `serve --max-batch 1` per arm, `localeval quality --judge none`, 7 category files, 512 forced tokens, 3 rounds, order alternating)

| category | off tok/s | mtp k=2 | x off | worst round | tokens / pass |
| --- | --- | --- | --- | --- | --- |
| ho-chat | 84.1 | 110.9 | 1.32 | 1.308 | 2.35 |
| ho-code | 83.9 | 128.4 | 1.53 | 1.516 | 2.74 |
| ho-prose | 83.9 | 98.5 | 1.17 | 1.163 | 2.08 |
| ho-recall | 84.1 | 116.1 | 1.38 | 1.370 | 2.47 |
| rcp-code | 84.0 | 127.6 | 1.52 | 1.506 | 2.72 |
| rcp-prose | 84.0 | 100.2 | 1.19 | 1.182 | 2.12 |
| rcp-recall | 83.9 | 119.4 | 1.42 | 1.411 | 2.53 |

## 6. Gate 5: G0

- CUDA, `raw/cuda-g0.log` (`scripts/mpf/cuda-g0.sh`, target-dev with the key unset vs origin/main `bd8a19397`, 256 ids after a short and a ~3.5k prompt, `plan show` JSON and text): Qwen3-8B and Gemma-4-26B-A4B no flag: ids and plans
  byte-identical; 35B no flag, `--spec off`, `INTARSIA_SPEC_DEFAULT=0`: ids identical, plans byte-identical for the two opt-outs.
- CUDA with the default on, `raw/cuda-g0-flip.log` (target-flip): 35B base (plain decode) vs dev `--spec off`, dev `INTARSIA_SPEC_DEFAULT=0`: ids and plans byte-identical (md5 `a6889e87` short, `51b7fcfd` 4k); dev no flag (the default, MTP): ids identical, and its `plan show` differs from main's only by the five extra row-exact verify kernels (`add_bf16`, `matmul_bf16_gemv_rows`, `matmul_fp8_gemv_rows`, `matmul_nvfp4_gemv_rows`, `moe_experts_decode_v2_rows_bf16`); the scratch line no longer differs (it did in lane cmtp, when the macro workspace was dropped).
- Metal (endor), `raw/metal-g0.log`: 8B and 35B ids (short, 4k) and `plan show` byte-identical to main; kernels test 44 kernels / 817 cases ADMITTED.
- HIP: no HIP code or recipe key changed; the recipe-pair test shows `qwen3_5_vl hip tp 2` as the only HIP default. HIP at tp=1 with an explicit `--spec mtp` takes the macro pass only where its recipe sets `prefill_macro` (not set).
  No GPU run on tatooine.

## 7. Tests

`cargo test --release -p intarsia-core -p intarsia-models -p intarsia-cli`: 625 passed, 0 failed; `scripts/prepush.sh` (docs-sync) passes.

## 8. Reproduce

`scripts/mpf/spark1.sh` (build), `ids.sh`, `longctx.sh`, `window.sh` (speed: `W=s1 ARMS="off mtp2" ROUNDS=3`; soak: `CATS=none SOAK=40`), `cuda-g0.sh`, `mutant.sh` + `mutate.py`, `metal-noop.sh`, `chain*.sh` (the order they ran in),
`TEST=mtp scripts/mpf/mutants.sh <names>` for the CPU mutants.
