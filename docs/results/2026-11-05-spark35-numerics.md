# spark35 numerics: bf16 GDN recurrent state (C) and fp8 KV as the default (D), Qwen3.6-35B-A3B NVFP4 on GB10

Lane `lane-sn/spark35-numerics`, 2026-11-05, branch `lane-sn/spark35-numerics` from `main` 5390e65b4. CUDA only, `nvidia/Qwen3.6-35B-A3B-NVFP4`,
the model both Sparks serve through vLLM for Mission Control (spark1 at 32K context, spark2 at 128K). Raw: `docs/results/2026-11-05-spark35-numerics/raw/`.

**Status (2026-11-06 08:45 EST): all rules are now measured. Round 1 (J1 to J3, 2026-11-05 evening, dev `7a7aa47b2`) gave Q5 to Q8 but no KL data (kl.sh bug, fixed). Round 2
(2026-11-06 morning, branch merged with `origin/main`, dev `1f4281d4c`) re-ran the gates and the KL arms: Q1 to Q4 PASS for D, C and C+D. By the rule below ("An arm passes a lever
if it passes Q1 to Q6") D, C and C+D each pass, and C passes Q8. Only the user flips a default; none is flipped on this branch (the D flip commit `4d9003f67` is reverted).**
The pass rules below were written before any data was collected and were not edited after data arrived.

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

- **C, real layout** (`3ed867cfd`, hardened in `7a7aa47b2`): `INTARSIA_GDN_STATE=bf16` or recipe `[tuning.<backend>] gdn_state = "bf16"` stores the delta-rule state
  `[Hv, dv, dk]` bf16 in the recurrent slot (`Model::gdn_state_dt`; slot bytes, the speculative-rollback copies and the prefix-store key follow it;
  `Backend::gdn_state_supported`: CUDA only). `kernels/cuda/gdn.cu`: the decode kernel and the split (prefill) kernel, solo and row-table (the four launchers the 35B
  plan binds: `gdn_delta_rule_bf16_split` for T >= 1 and `_split_rows` for the batched step), are templated on the state type; they load 4 elements as one 8-byte
  access, compute in f32 and round to nearest even once per launch (after each decode token, at the end of each prefill chunk). **Refused at load** (an opt-in may
  refuse) where a kernel cannot take it: `INTARSIA_CUDA_GDN_SPLIT=0`, `INTARSIA_CUDA_GDN_LANES=1`, `INTARSIA_CUDA_FAST_PREFILL=1` (they select the f32-only
  kernels), a key head dim other than 64 or 128, any non-CUDA backend. The fused `GdnConvDelta` snapshotting path is HIP-only, so CUDA speculative verify goes
  row by row and its rollback copies whole slots (sized from the same `state_bytes`). Tests: `crates/intarsia-backend-cuda/tests/gdn_state16.rs` (bf16 state == f32 state
  rounded after every launch, bitwise outputs, T = 1 and T > 1, four shapes; compiles under `--cfg ik_cuda` on the Mac, **not run yet**).
- **C, emulation** (dev only, `INTARSIA_CUDA_GDN_STATE_BF16_EMU=1`): the same rounding on the f32 layout. `creal` must equal `cemu` (idgate: greedy ids, MTP on and
  `--spec off`, are an early hard gate in J1).
- **D patch** (commit `4d9003f67`, **reverted on the branch by `8942c0764`** on 2026-11-06 so the branch merges without D; `git revert 8942c0764` brings it back. Original text: merge the branch with or without it, or `git revert` it alone): new per-backend recipe key
  `[tuning.<backend>] kv_dtype` (`Recipe::resolve_kv_dtype`, `--kv-dtype` still wins), `recipes/qwen3_5_moe.toml` sets `fp8_e4m3` under `[tuning.cuda]` only, test
  `recipe_kv_dtype.rs`. **The default stays bf16 until the user decides.** Merging it changes: (1) the 35B CUDA `plan show` (the KV cache bytes and dtype), nothing on
  Metal, HIP, CPU or any other model; (2) CUDA tests and `intarsia accuracy` baselines recorded with a bf16 KV cache of this model (they now load fp8; a recorded
  baseline refuses on the config mismatch, pass `--kv-dtype bf16` to keep comparing against it); (3) the `INTARSIA_TEST_KV_DTYPE` row in `docs/tuning.md` says
  "the recipe's (`bf16`)", true on CPU only after the flip. Smoke test to run at that commit once a GPU slot is free: `/metrics` `intarsia_info{kv_dtype=...}` says
  `fp8_e4m3` with no flag and `bf16` with `--kv-dtype bf16`. Not run.
- No default changed in the lever commits: `INTARSIA_GDN_STATE` unset gives the f32 layout and the same kernel instantiations (gates below).

## Queue (fleetq, owner claude-1943a70c, spark2 because it was earlier in the queue than spark1; the `dev` binary is built from the sha in `scripts/spark35num/DEV_REF`, never from the flip commit)

| job | what | host | state (2026-11-06) |
|---|---|---|---|
| #8666 J1 | CUDA builds (main = `5390e65b4`, dev) and test binaries; GPU bitwise guards `gdn_state16 gdn_split gdn_conv16 gdn_lanes rows_exact row_tables` with `INTARSIA_REQUIRE_GPU=1`; plan show 35B main vs dev; `kernels test --quick`; idgate (main == base, creal == cemu, MTP on and off); KL arms base, cemu, fp8, noise, cdemu, creal | spark2, `--quiet gpu,cpu --mem 70 --expect 160` | done (raw files up to 20:36); builds, GPU guards, plan, kernels test, idgate OK; **KL arms invalid (CLI refusal)** |
| #8667 J2 | serve arms main, base, fp8, cemu, cdemu, creal: greedy 1024 tokens at 1k/4k/16k, needle 32k and 64k, MMLU, GSM8K | spark2, `--mem 70 --expect 120` | done 20:36 to 21:27; all arms served (creal: greedy and needle only) |
| #8668 J3 | speed ABAB, 2 rounds (a b c d / d c b a): base, C, D, C+D; single stream 1k/4k/16k, 1 and 4/8/16 streams at 1k | spark2, `--mem 55 --expect 120` | done 21:28 to 21:48; 2 rounds, 4 arms |
| #8632 | Metal no-op (plan show, 256 greedy ids 8B and 35B, kernels test) main vs branch (shared files changed) | endor | raw in `raw/metal/`: all identical (ids 8B and 35B short and 4k, plan json and txt); fleetq record not read here |
| #8662 | `cargo test --workspace --no-fail-fast` on the final head | endor, `--load cpu` | **no result in raw** (the only cargo log is #8643 on `1016c7e5e`) |

Round 2 (2026-11-06, owner claude-1943a70c, after the merge of `origin/main` into the branch; results in `raw-merged/`):

| job | what | host | state |
|---|---|---|---|
| #8735 | `cargo test --workspace --no-fail-fast` on the merged head (`1f4281d4c` + scripts), `--load cpu --expect 30` | endor | done, exit 0, about 9 min: 234 test binaries, 1362 passed, 0 failed, 32 ignored (`raw-merged/cargo-test-workspace-endor.log`) |
| #8736 | builds: main (`af6aa5226`, origin/main tip) and dev (`1f4281d4c`, merged head, `DEV_REF` env) and the six GPU test binaries, `--load cpu --mem 40 --expect 30` | spark2 | done, exit 0, 07:14 to 07:23 (main 3m21s, dev 3m27s) |
| #8741 | gates (plan show, `kernels test --quick`, six GPU bitwise guards, idgate main == base and cemu == creal MTP on and off) then the KL rerun, arms `base noise fp8 cemu cdemu creal` (base rerun because the merged binary differs from the one that wrote the old dumps), `--quiet gpu,cpu --mem 70 --expect 90` | spark2 | done, exit 0, started 07:32 (waited for #8737), gates 07:32 to 07:44, KL 07:44 to 08:42; `gates fails=0` |


Earlier jobs: #8646 (main CUDA build, 3m37s, OK; dev from HEAD failed in nvcc: `gdn_dec_run<true,...>` could not deduce the state type from `nullptr`, fixed in
`6699ec3eb`; the lane head has therefore **never been through nvcc yet**), #8653 (the diagnosing rebuild), #8643 (`cargo test --workspace --no-fail-fast` on `1016c7e5e`: 231 binaries,
1324 passed, 0 failed, 29 ignored; `raw/cargo-test-workspace-endor.log`). #8619, #8630, #8641, #8642, #8645, #8658, #8659 to #8661: cancelled by me and resubmitted (a queued job of
mine blocked the backfill of another; the harness changed). Harness checks done on the Mac: the carrier goldens load in `intarsia accuracy` (Qwen3.5-0.8B on the CPU backend:
self-golden write, noise-arm compare with `--prefill-chunk`, position dump, `analyze.py` tables), the needle rows are 33.1k to 33.2k and 66.1k to 66.2k tokens,
`gdn_state16.rs` type-checks with `--cfg ik_cuda`.

## Results

Round 1 (J1 to J3) used dev `7a7aa47b2`; Q5, Q6, Q8 (needle, MMLU / GSM8K, speed) and the greedy ids are from it and were not re-run after the merge. Round 2 re-ran the gates and the
whole KL set on the merged dev `1f4281d4c`; Q1 to Q4 are from it. Arms: base (bf16 KV, f32 state), noise (base, `--prefill-chunk 1024`), fp8 (D), cemu (C emulated), cdemu (C+D),
creal (C real layout). The D commit `4d9003f67` is reverted on the branch; D was measured with `--kv-dtype fp8` on the dev binary.

### Verdict per rule

| rule | D (fp8 KV) | C (bf16 state) | C+D | numbers |
|---|---|---|---|---|
| Q1 KL, prefill, per length class | PASS | PASS | PASS | K / noise: D 1.12x to 1.23x, C 0.51x to 1.09x, C+D 1.21x to 1.35x (bound 5x; table below) |
| Q2 growth 64k vs 4k | PASS | PASS | PASS | K_64k D 1.22e-2, C 1.15e-2, C+D 1.27e-2 against bounds 2.70e-2, 2.20e-2, 2.97e-2 |
| Q3 top-1 flips, prefill | PASS | PASS | PASS | flips at most +0.6 percentage points above noise per class (noise 2.5 to 4.8 %; bound 3x noise); C at or below noise except 64k (+0.1) |
| Q4 decode windows 2048 and 8192, no drift | PASS | PASS | PASS | K / noise 1.01x to 1.59x (largest: C+D on the 16k window); last / first quarter at most 1.56x (C+D, 16k), under the 2x + 5e-4 bound; the 8192 window falls (e.g. D 8.6e-3 to 5.5e-3) |
| Q5 needle recall | PASS 12/12, 8/8 | PASS 12/12, 8/8 | PASS 12/12, 8/8 | base 12/12 and 8/8: at the ceiling, cannot resolve a small loss |
| Q6 MMLU / GSM8K | PASS 0.851 / 0.500 | PASS 0.839 / 0.500 | PASS 0.844 / 0.490 | base 0.846 / 0.500; bounds +-0.043 / +-0.141 |
| Q7 env-unset gate (the branch) | n/a | PASS | n/a | round 1 and round 2 (merged binary): see the Q7 table |
| Q8 C worth shipping | n/a | PASS +8.3 % at 16 streams | +10.8 % (reported) | rounds +9.0 % and +7.5 %; measured on the pre-merge dev |
| **Lever passes (Q1 to Q6)** | **YES** | **YES** | **YES** | |

The noise reference is large: a benign change of the prefill chunk schedule (1024 vs 2048, bf16 against bf16) already moves the mean 21-way KL to 9.4e-3 (prefill, 134015 positions) with
4.2 % top-1 flips and 8.5e-3 / 3.5 % on the 2048-token decode windows. The pre-registered bound (5 x noise) is therefore loose, and a pass means "no larger than a few
times a benign perturbation". The more useful reading is the ratio column: on prefill no lever adds more than 35 % to the noise-level KL at any length (D at 64k: 1.16x, C at 64k: 1.09x, C+D 1.21x to 1.35x),
on the decode windows up to 59 % (C+D, 16k window), and the 8192-token decode window shows no drift for any arm (last quarter below the first).

### Q1 to Q4 by length class (`scripts/spark35num/classes.py raw-merged`; per-prompt tables: `raw-merged/analyze.txt`)

#### Prefill rows by length class: K (mean 21-way KL, nats) / flips %, bound = max(5 x noise, 2e-3) and max(3 x noise flips, 1.5)

| class | prompts | positions | noise K / flips | fp8 K / flips (K/noise) | cemu K / flips (K/noise) | cdemu K / flips (K/noise) | creal K / flips (K/noise) |
|---|---|---:|---|---|---|---|---|
| 4k | cmtp-4k-0 | 3858 | 7.33e-03 / 4.15 | 9.01e-03 / 4.77 (1.23x) | 3.72e-03 / 1.19 (0.51x) | 9.89e-03 / 4.67 (1.35x) | 3.72e-03 / 1.19 (0.51x) |
| 16k | cmtp-16k-0, code-16k | 31853 | 6.28e-03 / 2.50 | 7.66e-03 / 2.89 (1.22x) | 5.48e-03 / 2.11 (0.87x) | 8.10e-03 / 2.87 (1.29x) | 5.48e-03 / 2.11 (0.87x) |
| 32k | docs-32k | 32768 | 1.06e-02 / 4.82 | 1.18e-02 / 5.10 (1.12x) | 1.04e-02 / 4.63 (0.99x) | 1.31e-02 / 5.37 (1.24x) | 1.04e-02 / 4.63 (0.99x) |
| 64k | docs-64k | 65536 | 1.05e-02 / 4.67 | 1.22e-02 / 5.09 (1.16x) | 1.15e-02 / 4.77 (1.09x) | 1.27e-02 / 5.26 (1.21x) | 1.15e-02 / 4.77 (1.09x) |

#### Q2: K_64k <= 3 x max(K_4k, N_4k)

- fp8: K_64k 1.22e-02 <= 3 x max(9.01e-03, 7.33e-03) = 2.70e-02: PASS
- cemu: K_64k 1.15e-02 <= 3 x max(3.72e-03, 7.33e-03) = 2.20e-02: PASS
- cdemu: K_64k 1.27e-02 <= 3 x max(9.89e-03, 7.33e-03) = 2.97e-02: PASS

#### Q4: decode windows (teacher-forced), K bound = max(5 x noise, 2e-3), no drift: last quarter <= 2 x first quarter + 5e-4

| window | arm | K (noise K) | flips % (noise) | first / last quarter | verdict |
|---|---|---|---|---|---|
| std 4k | fp8 | 1.17e-02 (8.88e-03) | 3.52 (3.37) | 1.35e-02 / 8.00e-03 | PASS |
| std 4k | cemu | 1.00e-02 (8.88e-03) | 3.71 (3.37) | 1.13e-02 / 6.61e-03 | PASS |
| std 4k | cdemu | 1.14e-02 (8.88e-03) | 3.76 (3.37) | 1.22e-02 / 9.35e-03 | PASS |
| std 4k | creal | 1.00e-02 (8.88e-03) | 3.71 (3.37) | 1.13e-02 / 6.61e-03 | PASS |
| std 16k | fp8 | 8.33e-03 (6.75e-03) | 3.05 (2.73) | 7.42e-03 / 9.08e-03 | PASS |
| std 16k | cemu | 8.88e-03 (6.75e-03) | 3.35 (2.73) | 6.88e-03 / 8.88e-03 | PASS |
| std 16k | cdemu | 1.07e-02 (6.75e-03) | 2.98 (2.73) | 6.11e-03 / 9.54e-03 | PASS |
| std 16k | creal | 8.88e-03 (6.75e-03) | 3.35 (2.73) | 6.88e-03 / 8.88e-03 | PASS |
| std 32k | fp8 | 1.04e-02 (9.86e-03) | 3.71 (4.10) | 9.26e-03 / 1.01e-02 | PASS |
| std 32k | cemu | 9.98e-03 (9.86e-03) | 4.15 (4.10) | 9.02e-03 / 8.43e-03 | PASS |
| std 32k | cdemu | 1.06e-02 (9.86e-03) | 4.20 (4.10) | 1.16e-02 / 6.92e-03 | PASS |
| std 32k | creal | 9.98e-03 (9.86e-03) | 4.15 (4.10) | 9.02e-03 / 8.43e-03 | PASS |
| std 64k | fp8 | 1.12e-02 (1.01e-02) | 5.03 (4.79) | 1.31e-02 / 1.05e-02 | PASS |
| std 64k | cemu | 1.12e-02 (1.01e-02) | 5.62 (4.79) | 1.18e-02 / 1.18e-02 | PASS |
| std 64k | cdemu | 1.28e-02 (1.01e-02) | 6.01 (4.79) | 1.31e-02 / 1.29e-02 | PASS |
| std 64k | creal | 1.12e-02 (1.01e-02) | 5.62 (4.79) | 1.18e-02 / 1.18e-02 | PASS |
| dec8k code-16k 8192 | fp8 | 6.12e-03 (5.11e-03) | 1.88 (1.86) | 8.62e-03 / 5.47e-03 | PASS |
| dec8k code-16k 8192 | cemu | 6.03e-03 (5.11e-03) | 1.92 (1.86) | 7.58e-03 / 5.55e-03 | PASS |
| dec8k code-16k 8192 | cdemu | 7.04e-03 (5.11e-03) | 2.00 (1.86) | 8.60e-03 / 6.27e-03 | PASS |

#### Verdict by arm (all classes and windows)

- fp8: Q1 PASS; Q2 PASS; Q3 PASS; Q4 PASS
- cemu: Q1 PASS; Q2 PASS; Q3 PASS; Q4 PASS
- cdemu: Q1 PASS; Q2 PASS; Q3 PASS; Q4 PASS

`creal` (the real bf16 layout) gives exactly the same SELF-STATS as `cemu` (prefill mean KL 9.554824e-3, 5364 flips; decode 9.794583e-3, 413 flips; identical to the last digit),
so the emulation is the real layout bit for bit on the std carrier as well (the creal rows in the tables above equal the cemu rows). The first run's `analyze.py` output ("no prefill data") was the empty
result of the invalid round-1 logs, kept in `raw/`.

### How round 1 failed and what round 2 changed

- Round 1: `kl.sh` passed `--golden` together with `--self-golden`; the CLI refuses that (`accuracy.rs:2662`), so every non-base KL run stopped at startup. The base logs ran against
  the carrier's dummy reference rows (KL about 5.5, `accuracy gate FAILED`, expected, and the dumps were written anyway). Kept in `raw/` as `acc-*.log`, not used.
- Round 2: `kl.sh` drops `--golden` for every arm except base (`a7a93ea91`); `DRY=1` prints the command lines. Checked on the Mac CPU backend with the same flag shapes
  (Qwen3.5-0.8B-Base, a 300-token carrier): base writes the dump, a `--prefill-chunk` arm reads it with `--self-golden` only and prints SELF-STATS, `analyze.py` parses the log.
  `w1.sh` got `SKIP_DATA=1` (the data stage hashes the current `docs/` and `crates/`, which the merge changed; the committed carriers are kept).
- Base was re-run with the merged binary (the 2026-11-05 dumps on spark2 were written by `7a7aa47b2` and would have mixed two builds into the comparison).
- The Q1 to Q3 checks of `analyze.py` are per prompt; `classes.py` pools the 16k class (cmtp-16k-0 and code-16k) as the rule says. Both give PASS for every arm.

### Q5: needle recall (`localeval quality --judge none`, Contains expected)

| arm | 32k (12 rows) | 64k (8 rows) |
|---|---|---|
| base | 12 | 8 |
| fp8 (D) | 12 | 8 |
| cemu | 12 | 8 |
| cdemu | 12 | 8 |
| creal | 12 | 8 |

### Q6: MMLU (570) and GSM8K (100), thinking off

| arm | MMLU | diff vs base | GSM8K | diff vs base |
|---|---:|---:|---:|---:|
| base | 0.846 | - | 0.500 | - |
| fp8 (D) | 0.851 | +0.005 | 0.500 | 0.000 |
| cemu | 0.839 | -0.007 | 0.500 | 0.000 |
| cdemu | 0.844 | -0.002 | 0.490 | -0.010 |

Bounds from the rule (2 SE of the difference): MMLU +-0.043, GSM8K +-0.141. All inside. creal has no MMLU / GSM8K run: it is bitwise equal to cemu (idgate below), so it adds
nothing. GSM8K scores 0.500 with a mean output of 8 to 12 tokens (thinking off, no chain of thought), so it has little power; MMLU is the more useful one.

### Q7: the env-unset gate and the other safety checks (round 1: dev `7a7aa47b2`, main `5390e65b4`; round 2 rows below)

| check | result |
|---|---|
| greedy ids base vs `origin/main` (1024 tokens, 6 prompts: 1k/4k/16k x 2) | IDENTICAL, all 6 (`greedy-main-vs-base.txt`) |
| idgate, 256 ids, 6 prompts: main == base, cemu == creal (MTP on), cemu == creal (`--spec off`) | IDENTICAL, all (`idgate/cmp-*.txt`) |
| greedy ids cemu vs creal, 1024 tokens | IDENTICAL, all 6 (so the real layout equals the emulation) |
| CUDA `plan show` 35B main vs dev | byte-identical |
| `kernels test --quick` main vs dev | 106 kernels, 918 cases, same rows and numbers (only the wall time differs) |
| GPU bitwise guards (`INTARSIA_REQUIRE_GPU=1`): gdn_state16 2, gdn_split 2, gdn_conv16 2, gdn_lanes 2, rows_exact 12 (2 ignored), row_tables 7 (2 ignored) | all passed; gdn_state16 relative state error S16 vs S32 2.7e-3 to 2.8e-3 after 1000 steps |
| Metal no-op, main vs branch: plan 8B and 35B (json + txt), 256 ids 8B and 35B (short, 4k) | all identical |
| **Round 2, merged binary** (main `af6aa5226` vs dev `1f4281d4c`): plan show 35B json + txt | byte-identical |
| round 2: `kernels test --quick` main vs dev | same status lines |
| round 2: six GPU bitwise guards | all passed (gdn_state16 2, gdn_split 2, gdn_conv16 2, gdn_lanes 2, rows_exact 12 + 2 ignored, row_tables 7 + 2 ignored) |
| round 2: idgate 256 ids x 6 prompts: main == base, cemu == creal (MTP on), cemu == creal (`--spec off`) | PASS all three (`raw-merged/idgate/`, `gates fails=0`) |
| round 2: `cargo test --workspace --no-fail-fast` on `1f4281d4c` + scripts (endor, #8735) | 234 binaries, 1362 passed, 0 failed, 32 ignored (the two "panicked" lines in the log are the dsv41 scripted-panic tests) |
| round 2: Metal no-op after the merge | not re-run (the workspace tests, which include the Metal crate, pass; the merge touches no Metal kernel of this lane) |
| base-vs-base determinism rerun | not run separately; the idgate main == base and the 1024-token main == base agreement is the evidence |

### Greedy first-divergence (reported only; no noise arm, so it cannot be read against a reference)

First differing token index vs base, 1024 greedy tokens (positional agreement after divergence is 0.7 to 14 % for all arms, i.e. all diverge and then wander):

| prompt | fp8 | cemu = creal | cdemu |
|---|---:|---:|---:|
| 1k-0 | 14 | 27 | 132 |
| 4k-0 | 22 | 22 | 22 |
| 16k-0 | 2 | 2 | 2 |
| 1k-1 | 55 | 49 | 51 |
| 4k-1 | 27 | 27 | 61 |
| 16k-1 | 17 | 26 | 17 |

No arm is closer to base than another in any consistent way; as the doc said, this cannot be judged without the noise reference.

### Speed (J3, 2 rounds ABAB: round 1 base c d cd, round 2 cd d c base; `raw/speed/`)

Single stream, MTP on (recipe default), gen 128, 3 reps; decode tok/s, mean of the two rounds (r1 / r2):

| prompt | base | C | D | C+D |
|---|---|---|---|---|
| 1k | 106.1 (105.5 / 106.7) | 110.9 (110.0 / 111.8) | 117.8 (121.0 / 114.6) | 113.6 (113.7 / 113.4) |
| 4k | 112.4 (113.3 / 111.6) | 113.5 (112.7 / 114.2) | 111.0 (107.5 / 114.5) | 116.2 (113.7 / 118.8) |
| 16k | 91.4 (91.7 / 91.1) | 90.5 (91.6 / 89.5) | 94.4 (97.1 / 91.8) | 99.0 (94.7 / 103.2) |

D at 16k single stream is +3.3 % (+5.9 % in round 1, +0.8 % in round 2), against +9.9 % in the fp8kv lane; the round-to-round spread of the same arm (C+D 16k: 94.7 vs 103.2) is as
large as the effect, so D is not distinguishable from noise on single-stream decode here. Prefill at 4k: base 6107 tok/s against 6840 to 6950 for C, D and C+D (the base arm
is the slow one in both rounds and D does not touch the state, so this is not attributable to C; unexplained, reported only). The 1k prefill of d-single-r1 (2016 tok/s, TTFT 0.49 s) is a
cold-start outlier and is excluded from any mean.

Burst (1k prompts, gen 256, `--max-batch 16`), aggregate tok/s, mean of the two rounds and gain over base (r1 / r2 per round):

| streams | base | C | D | C+D |
|---:|---:|---:|---:|---:|
| 4 | 174.2 | 182.7 (+4.9 %: +5.9 / +4.0) | 177.7 (+2.0 %) | 186.5 (+7.1 %) |
| 8 | 235.4 | 248.1 (+5.4 %: +7.0 / +3.8) | 236.2 (+0.4 %) | 254.1 (+7.9 %) |
| 16 | 305.3 | 330.5 (**+8.3 %**: +9.0 / +7.5) | 307.4 (+0.7 %: +4.2 / -2.7) | 338.4 (+10.8 %: +12.7 / +9.0) |

Q8: C passes (+8.3 %, both rounds above +3 %). D at 16 streams is +0.7 % with the two rounds on opposite sides of zero: no measurable speed effect. C+D is 2.4 % above C alone at 16
streams. The engine's 16-stream number against vLLM's 340 tok/s: base 305 (0.90x), C 331 (0.97x), C+D 338 (0.995x). Memory: `MemAvailable` 50 to 51 GB with each server up (`speed/mem.txt`); D's KV
memory saving at 32k / 128k context was not measured in this lane.

### Decisions (the user decides; no default is flipped on this branch)

- **C (bf16 delta-rule state default on CUDA 35B): the doc's rules are met.** Q1 to Q6 PASS (Q1 to Q4 on the merged binary, K / noise 0.51x to 1.09x on prefill and 1.01x to 1.32x on the decode windows, no drift over 8192
  decode tokens), Q7 PASS, Q8 PASS (+8.3 % at 16 streams, both rounds above +3 %; +4.9 % at 4 and +5.4 % at 8 streams). The real layout equals the emulation bit for bit
  (greedy ids with MTP on and off, and KL to the last digit). Caveats a flip should weigh: (1) the noise reference is large, so the pre-registered bound would also pass a lever that is several times
  worse than what was observed; (2) needle is at its ceiling and GSM8K has little power; (3) the speed gain was measured before the merge (the merge brings the batched-MTP default
  and other main changes) and was not re-measured; (4) the longest decode window is 8192 tokens, and the state is rounded at every token, so a 30k-token generation is not covered.
  Recommendation: flip it, as its own small commit (`[tuning.cuda] gdn_state = "bf16"` in `recipes/qwen3_5_moe.toml`), after one cheap re-measure of the 16-stream gain on the merged
  binary (J3 shape, base vs C, about 25 min on spark2) if (3) matters; the quality rules do not stand in the way. Not done: the doc says only the user flips a default.
- **D (fp8 KV default on CUDA 35B): the doc's rules are met as well** (Q1 to Q6 PASS; K / noise 1.12x to 1.23x on prefill, 64k 1.16x; no drift), but D buys no speed here: +0.7 % at 16 streams
  (rounds +4.2 % and -2.7 %) and +3.3 % at 16k single stream (+5.9 % and +0.8 %), against +9.9 % in the fp8kv lane; and, combined with C, it adds up to +35 % (prefill) and +59 % (16k decode window) to the noise-level KL.
  Recommendation: leave bf16 the default unless the halved KV memory at 128K context (spark2) is wanted; then D is allowed by the rules. Its flip commit `4d9003f67` is reverted
  on the branch (`8942c0764`), and it was never built or run on a GPU as a default (smoke test of `/metrics` `kv_dtype` not run).
- Nothing else in the doc asks for a decision.

### Merge readiness

- Branch head = lane commits + revert of D + merge of `origin/main` (`1f4281d4c`) + scripts and this write-up. Without D: merge the branch as it is. With D: `git revert 8942c0764` first
  (it re-applies the D patch, which then needs its own build and the `/metrics` smoke test).
- Checks on the merged tree: workspace `cargo test` passes (#8735), CUDA builds of main and dev (#8736), plan show byte-identical, `kernels test --quick` same, six GPU guards pass,
  idgate PASS (#8741). Metal no-op not re-run (see Q7).

Existing evidence for D (not re-measured): `docs/results/2026-10-29-fp8kv-opt-in.md`: +9.9 % single-stream decode at 16k, +1.4 % at 16 streams, MMLU 0.846 to 0.851,
GSM8K 0.500 to 0.500, both inside noise; the long-context (32k, 64k) quality is the gap this lane fills. vLLM at 16 streams: 340 tok/s; the engine: 301 tok/s (0.88x).
