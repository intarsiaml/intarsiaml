Go-ahead 2026-11-08 ~13:05 EST (user, verbatim): "yes, make mix the 27B CUDA default"

# Qwen3.8-27B on CUDA: `mix` MTP draft head as the default (lane mxd)

## Change

New recipe tuning key `[tuning.<backend>] mtp_quant` (`"fp8"`, `"mxfp4"` or `"mix"`), read per backend like `lm_head_quant`:

* `crates/intarsia-core/src/recipe.rs`: `BackendTuning::mtp_quant`, `Recipe::mtp_quant(backend)`, value checked when the recipe is read.
* `crates/intarsia-core/src/quant/head.rs`: `parse_mtp_quant`, `resolve_mtp_quant(env, recipe)` (the variable `INTARSIA_MTP_QUANT` wins; every off spelling of it, `off` `none` `0` `bf16` or empty, beats the recipe key; unset on both: off).
* `crates/intarsia-core/src/model/mtp.rs`: `mtp_quant(recipe key)` replaces the env-only parse; the existing limits stay (a head sharded over in-process ranks and an MoE head keep bf16; the dense head whole on rank 0 of a remote group is quantised, as q2s measured).
* `recipes/qwen3_5_vl.toml`: `mtp_quant = "mix"` under `[tuning.cuda]` ONLY (comment cites the user decision and the two evidence docs). Metal and HIP set nothing.
* Docs: `docs/tuning.md` (`INTARSIA_MTP_QUANT` row), `docs/recipes.md` (key table), `docs/decisions/LOG.md` (decision row and defaults row).
* Test: `crates/intarsia-models/tests/recipe_27b_chunk.rs::shipped_27b_recipe_mtp_quant` (CUDA resolves to `mix` = fc + attention fp8, feed-forward mxfp4; Metal, HIP, cpu and the 8B / 35B / Llama / qwen3_5 recipes set nothing; six off spellings beat the key; a bad value is refused).

Evidence for the choice (not re-measured here): lane c27 `docs/results/2026-11-07-cuda27-decode.md` (+6.1 % at 4k tp1) and lane q2s `docs/results/2026-11-07-27b-tp2-mtp-quant.md` (+9.0 / +9.4 / +7.0 % at tp2 over the Sparks, sampled acceptance 67.5 % against 68.2 % unset). Drafts only: every emitted token is a verify row's argmax.

## Gates

Binary: spark1 (GB10, vLLM up, `avail` 63 to 83 GB throughout, nothing near 10 GB), CUDA release built from this branch (`dev`) and from origin/main `c01d61442` (`main`), `CARGO_BUILD_JOBS=4 MAX_JOBS=4 FLASHINFER_NVCC_THREADS=1` (fleetq #9118). Scripts and raw output: `docs/results/2026-11-08-27b-cuda-mix-default/raw/` (`g2.sh`, `g3.sh`, `g2/`, `g3/`, `metal-noop/`).

1. Local (endor, fleetq #9112): `cargo test -p intarsia-core` ok; `cargo test -p intarsia-models --test recipe_27b_chunk --test mtp --test recipes_doc` ok (4 / 36 / 3 passed); `cargo test -p intarsia-cli --test docs_env_vars` ok; `cargo build --locked --workspace` ok; `scripts/prepush.sh` ok.
2. CUDA 27B tp1, vLLM up, default recipe `--spec mtp --spec-k 2`, greedy 256 ids (fleetq #9121, `--quiet gpu,cpu`):
   * No env set: `mtp head quantised (INTARSIA_MTP_QUANT / recipe mtp_quant): fc and attention fp8, feed-forward mxfp4` printed exactly once per run; ids identical to `--spec off` on short-0 (md5 b594fdcc7955) and 4k-1 (md5 56eefb4d9f43), PASS.
   * `INTARSIA_MTP_QUANT=off`: no `mtp head quantised` line (head stays bf16); ids identical to `--spec off` on both prompts, PASS.
   * Quick 4k decode (14035-char corpus prompt, 192 tokens, order default / off / off / default, one process per visit): default 30.73 and 30.67 tok/s (mean 30.70), off 29.04 and 29.01 (mean 29.03), **+5.8 %** (c27 measured +6.1 % at 4k). Acceptance 74.5 % in all four runs (same 77 passes). Not a full ABAB, a sanity check that the gain shows.
3. CUDA plan JSON, `plan show --backend cuda --max-seq 8192 --mem-budget 30 --json` (fleetq #9120):
   * 35B NVFP4: main vs dev byte-identical (15570 B). 8B: byte-identical (8973 B).
   * 27B: differs, and only through the MTP head quantisation: with `INTARSIA_MTP_QUANT=off` dev is byte-identical to main (15461 B). Default dev (15748 B) vs main, `diff`: the kernel list drops `matmul_bf16` and `matmul_bf16_gemv` (the plan no longer uses them once the head projections are fp8 / mxfp4; both were `admitted: false`, 0 calls per decode step) and the pair kernel `matmul_pair_bf16_gemv`, gains `matmul_fp8_gemm`, `matmul_fp8_gemm_reg`, `matmul_fp8_pair_gemm_reg`, `matmul_fp8_pair_gemv` entries, and the plan-level `exactness` aggregate moves from `fast` (the removed `matmul_bf16`) to `deterministic`. Full diff: `raw/g3/` (`plan-27b-main.json`, `plan-27b-dev.json`).
   * `kernels test --quick --backend cuda` on dev: exit 0, 0 FAIL lines (106 kernels, 954 cases).
4. Metal no-op on endor (`scripts/dm/noop.sh`, fleetq #9119, exclusive, builds #9116 and #9117): all SAME (plan 8B / 27B / 35B x default / `--spec off` / `--spec mtp`; `kernels list` SAME; `kernels test` 55 kernels, 1232 cases, exit 0, SAME). HIP plan no-op on tatooine: skipped. The key lives only in `[tuning.cuda]`, so the HIP and Metal recipes resolve nothing (the test covers `hip`, `metal`, `cpu`), and a HIP build on tatooine would have needed a fresh ROCm build on a box whose radiance server leaves 1 GB of memory free.
5. CI and merge: see the PR.

Cleanup: `~/intarsia-mxd` removed on spark1; no fleetq slot held; vLLM answers on both Sparks (`/v1/models` 200 on spark-4e77 and spark-fa00).

Limits: one ids pair (short-0, 4k-1), greedy only; the c27 and q2s results (20 of 20 and tp2 ids) carry the rest. The 4k number is two runs per arm.
