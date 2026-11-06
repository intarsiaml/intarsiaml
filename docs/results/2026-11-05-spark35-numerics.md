# spark35 numerics: bf16 GDN recurrent state (C) and fp8 KV as the default (D), Qwen3.6-35B-A3B NVFP4 on GB10

Lane `lane-sn/spark35-numerics`, 2026-11-05, branch `lane-sn/spark35-numerics` from `main` 5390e65b4. CUDA only, `nvidia/Qwen3.6-35B-A3B-NVFP4`,
the model both Sparks serve through vLLM for Mission Control (spark1 at 32K context, spark2 at 128K). Raw: `docs/results/2026-11-05-spark35-numerics/raw/`.

**Status (2026-11-05 19:30 EST): code, harness and gates scripts are done and pushed; NO quality or speed data exists yet.** spark1 and spark2 are held
exclusively by the dsv41 windows (#8609, #8624, #8636), so the GPU jobs below are queued (fleetq estimated start for the first one: about 22:00 on
spark2). The pass rules below were written before any data was collected and are not to be edited after data arrives.

## Authority

The user said at 2026-11-05 17:06 EST:

> ignoring what I deferred previously what are the largest possible wins across dense and moe models?

and at 18:00 EST scoped it:

> only work on the items with models that the current fleet uses

Both levers ship **opt-in, default off**. Only the user flips a default. D's default flip is a separate commit on the branch.

## What is measured

- **D** (`--kv-dtype fp8_e4m3`, exists since fp8kv, `docs/results/2026-10-29-fp8kv-opt-in.md`): quality at 32K and 64K+, where the fleet runs.
  Attention kernels have not changed since 2026-10-29 (`git log` of `kernels/cuda/attention_splitkv.cu`, `attention_flash.cu` is empty
  since then), so fp8kv's MMLU / GSM8K stay valid; they are re-run here in the same window anyway.
- **C** (bf16 delta-rule state, today f32): quality first, by emulation. The kernels still hold the state in f32 memory, but every launch of the
  decode and split (prefill) kernels rounds it to bf16 (round to nearest even) at its final store (`INTARSIA_CUDA_GDN_STATE_BF16_EMU=1`). The
  kernels load and compute in f32 in both cases, so the emulated outputs are the outputs of a bf16-stored state. The real layout (half the bytes,
  the speed) is only built if the emulation passes.
- Arms: **base** (bf16 KV, f32 state), **noise** (base with `--prefill-chunk 1024`: the same-window benign perturbation, bf16 against bf16),
  **fp8** (D), **cemu** (C), **cdemu** (C+D).

### Harness

- KL: `intarsia accuracy` self-golden. The base arm writes its own top-64 rows (`--write-self-golden`); the other arms are scored against them
  (`--self-golden`, loose budgets, the numbers are read from `SELF-STATS`). A *carrier golden* (`scripts/spark35num/mkcarrier.py`) supplies only
  the prompt ids, because the vLLM goldens were made with an fp8 KV cache and would confound D, and `goldenlib` caps prompts at 16384 tokens.
  - `std` carrier, 5 prompts, prefill mode over every position plus a 2048-token teacher-forced decode window:
    `cmtp-4k-0` (3858 tokens) and `cmtp-16k-0` (15469), the existing `scripts/cmtp/ids-prompts`; `code-16k` (16384, `crates/**/*.rs`);
    `docs-32k` (32768) and `docs-64k` (65536), `docs/results/*.md` (no 32k or 64k ids-prompt existed; distinct text, nothing repeated).
  - `dec8k` carrier, `code-16k`, decode mode, an 8192-token teacher-forced window (state rounded at every token, 8192 times; the 2048-token
    windows at the end of the 32k and 64k prompts cover decode over a long KV cache).
  - `INTARSIA_ACC_POS_DUMP` gives KL against position.
- Needle: vault-register rows (`scripts/spark35num/mkneedle.py`, the format of deepeval's `m27-hs3-longctx`, which is only ~12k tokens): 12 rows of
  32.8k+ tokens and 8 rows of 65.5k+ tokens, the asked record at depth 5 % to 95 %; `localeval quality --dataset ... --judge none` (Contains expected).
- Greedy ids: `scripts/fp8kv/greedy.py`, 1k / 4k / 16k prompts, 1024 greedy tokens (base == main is the "env unset" gate).
- MMLU (10 per task, n = 570) and GSM8K (100), thinking off, as lane fp8kv ran them.

## Pass rules (fixed before data)

Notation: `K_L(arm)` is the mean 21-way KL (nats) of an arm against base over all scored positions of the prompts of length class L
(prefill rows and decode-window rows reported separately); `N_L` is the same for the **noise** arm (same prompts, same positions); flips% is the
top-1 flip rate of the same rows.

| # | rule | applies to |
|---|---|---|
| Q1 | `K_L <= max(5 x N_L, 2e-3)` for L in {4k, 16k, 32k, 64k}, prefill rows | D, C, C+D |
| Q2 | growth with length: `K_64k <= 3 x max(K_4k, N_4k)` | D, C, C+D |
| Q3 | top-1: `flips% <= max(3 x flips%(noise), 1.5)` per length class | D, C, C+D |
| Q4 | decode windows (2048 and 8192): the same bound as Q1 on the window's mean KL, and **no drift**: mean KL of the last quarter of the window `<= 2 x` the first quarter's `+ 5e-4` | C, C+D (D too, reported) |
| Q5 | needle recall at 32k and at 64k: arm correct `>=` base correct `- 1` (of 12 and of 8) | D, C, C+D |
| Q6 | MMLU and GSM8K: `|arm - base|` inside 2 SE of the difference of the two scores (fp8kv: +-0.043 and +-0.141) | D, C, C+D |
| Q7 | determinism and the env-unset gate: base == `origin/main` greedy ids (every prompt), `plan show` and `kernels test --quick` unchanged | the branch |
| Q8 | worth shipping C: 16-stream aggregate gain `>= +3 %` over base in the ABAB (else C is reported as not worth its numerics risk, whatever the quality) | C |

An arm passes a lever if it passes Q1 to Q6. "Reported" rows (greedy first-divergence token, per-prompt KL) have no rule: greedy output diverges
under any numerics change, and fp8kv could not clear its divergence for want of a same-window reference; the **noise** arm is that reference here.

## What was built

- **C, real layout** (commit `3ed867cfd`): `INTARSIA_GDN_STATE=bf16` or recipe `[tuning.<backend>] gdn_state = "bf16"` stores the delta-rule state `[Hv, dv, dk]`
  bf16 in the recurrent slot (`Model::gdn_state_dt`, slot bytes follow, `Backend::gdn_state_supported`: CUDA only, a backend that cannot refuses at load).
  `kernels/cuda/gdn.cu`: the decode kernel and the split (prefill) kernel, solo and row-table, are templated on the state type; they load 4 elements as one
  8-byte access, compute in f32 and round to nearest even once per launch (after each decode token, at the end of each prefill chunk). The opt-in
  `gdn_delta_rule_bf16_lanes` and the dk != 128 paths keep f32 and reject a bf16 state. Test: `crates/intarsia-backend-cuda/tests/gdn_state16.rs` (bf16 state ==
  f32 state rounded after every launch, bitwise outputs, T = 1 and T > 1, four shapes). **Not run yet** (needs the GPU; queued).
- **C, emulation** (dev only, `INTARSIA_CUDA_GDN_STATE_BF16_EMU=1`): the same rounding on the f32 layout. Arm `cemu` and arm `creal` must give the same numbers.
- **D patch** (commit `4bbb338e2`, its own commit, merge it or not): new per-backend recipe key `[tuning.<backend>] kv_dtype` (`Recipe::resolve_kv_dtype`,
  `--kv-dtype` still wins), `recipes/qwen3_5_moe.toml` sets `fp8_e4m3` under `[tuning.cuda]` only, test `recipe_kv_dtype.rs`. Reverting that commit alone restores bf16.
  The doc rows (`docs/tuning.md`, `docs/recipes.md`) mention it. **The default stays bf16 until the user decides.**
- No default changed in the lever commits: `INTARSIA_GDN_STATE` unset gives the f32 layout and the same kernel instantiations (the CUDA gates below check it).

## Queue (fleetq, all owner claude-1943a70c, all pinned to commit `4845dabe6` for the `dev` binary so the flip commit can never be the base arm)

| job | what | host | state at 19:30 |
|---|---|---|---|
| #8659 J1 | CUDA builds (main = `5390e65b4`, dev), gates (plan show 35B main vs dev, `kernels test --quick`, `gdn_state16`), KL arms base, cemu, fp8, noise, cdemu, creal | spark2, `--quiet gpu,cpu --mem 70 --expect 130` | queued |
| #8660 J2 | serve arms main, base, fp8, cemu, cdemu, creal: greedy ids 1k/4k/16k, needle 32k and 64k, MMLU, GSM8K | spark2, `--mem 70 --expect 120` | queued |
| #8661 J3 | speed ABAB, 2 rounds (a b c d / d c b a): base, C, D, C+D; single stream 1k/4k/16k, 4/8/16 streams at 1k | spark2, `--mem 55 --expect 120` | queued |
| #8632 | Metal no-op (plan show, 256 greedy ids 8B and 35B, kernels test) main vs branch (shared files changed) | endor | queued |
| #8662 | `cargo test --workspace --no-fail-fast` on the final head | endor, `--load cpu` | queued |

Earlier jobs: #8646 (main CUDA build, 3m37s, OK; dev from HEAD failed in nvcc: `gdn_dec_run<true,...>` could not deduce the state type from `nullptr`, fixed in
`6699ec3eb`), #8653 (the diagnosing rebuild), #8643 (`cargo test --workspace --no-fail-fast` on `1016c7e5e`: 231 binaries, 1324 passed, 0 failed, 29 ignored;
`raw/cargo-test-workspace-endor.log`). #8619, #8630, #8641, #8642, #8645, #8658: cancelled by me and resubmitted (a queued job of mine was blocking the
backfill of another). Harness checks done on the Mac: the carrier goldens load in `intarsia accuracy` (Qwen3.5-0.8B on the CPU backend: self-golden write, noise-arm
compare with `--prefill-chunk`, position dump, `analyze.py` tables), the needle rows are 33.1k to 33.2k and 66.1k to 66.2k tokens.

## Results

None yet. Fill the tables with `python3 scripts/spark35num/analyze.py docs/results/2026-11-05-spark35-numerics/raw` after J1 (KL), read needle, MMLU and GSM8K from
`raw/needle-*.out`, `raw/quality-*.out`, the greedy first-divergence from `scripts/fp8kv/greedy.py compare`, and the speed tables from `raw/speed/`.

Existing evidence for D (not re-measured): `docs/results/2026-10-29-fp8kv-opt-in.md`: +9.9 % single-stream decode at 16k, +1.4 % at 16 streams, MMLU 0.846 to 0.851,
GSM8K 0.500 to 0.500, both inside noise; the long-context (32k, 64k) quality is the gap this lane fills. vLLM at 16 streams: 340 tok/s; the engine: 301 tok/s (0.88x).
