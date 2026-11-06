# spark35 numerics: bf16 GDN recurrent state (C) and fp8 KV as the default (D), Qwen3.6-35B-A3B NVFP4 on GB10

Lane `lane-sn/spark35-numerics`, 2026-11-05, branch `lane-sn/spark35-numerics` from `main` 5390e65b4. CUDA only, `nvidia/Qwen3.6-35B-A3B-NVFP4`,
the model both Sparks serve through vLLM for Mission Control (spark1 at 32K context, spark2 at 128K). Raw: `docs/results/2026-11-05-spark35-numerics/raw/`.

**Status (2026-11-06, written up from the raw data after the lane agent died): all queued GPU jobs have run (J1 #8666, J2 #8667, J3 #8668 on spark2). The
KL arms (rules Q1 to Q4) produced NO data: `kl.sh` passed `--golden` together with `--self-golden` and the CLI refuses that, so every non-base KL run stopped at
startup (see Results, section "KL"). Q5 to Q8 are measured. By the rule below ("An arm passes a lever if it passes Q1 to Q6") neither D nor C has passed yet.**
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
- **D patch** (the LAST commit of the branch, `git log -1`; merge the branch with or without it, or `git revert` it alone): new per-backend recipe key
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

Earlier jobs: #8646 (main CUDA build, 3m37s, OK; dev from HEAD failed in nvcc: `gdn_dec_run<true,...>` could not deduce the state type from `nullptr`, fixed in
`6699ec3eb`; the lane head has therefore **never been through nvcc yet**), #8653 (the diagnosing rebuild), #8643 (`cargo test --workspace --no-fail-fast` on `1016c7e5e`: 231 binaries,
1324 passed, 0 failed, 29 ignored; `raw/cargo-test-workspace-endor.log`). #8619, #8630, #8641, #8642, #8645, #8658, #8659 to #8661: cancelled by me and resubmitted (a queued job of
mine blocked the backfill of another; the harness changed). Harness checks done on the Mac: the carrier goldens load in `intarsia accuracy` (Qwen3.5-0.8B on the CPU backend:
self-golden write, noise-arm compare with `--prefill-chunk`, position dump, `analyze.py` tables), the needle rows are 33.1k to 33.2k and 66.1k to 66.2k tokens,
`gdn_state16.rs` type-checks with `--cfg ik_cuda`.

## Results

Written 2026-11-06 from `raw/` only (no new runs). Arms: base (bf16 KV, f32 state), fp8 (D), cemu (C emulated), cdemu (C+D), creal (C real layout). The `dev` binary is
`7a7aa47b2` (`scripts/spark35num/DEV_REF`); `7a7aa47b2..3860aba95` changes only scripts and this doc, so the measured code is the lane code up to `3860aba95`. The D commit
`4d9003f67` was never built or run on a GPU (D was measured with `--kv-dtype fp8` on the `dev` binary).

### Verdict per rule

| rule | D (fp8 KV) | C (bf16 state) | C+D | what the data is |
|---|---|---|---|---|
| Q1 KL vs noise, prefill 4k to 64k | **not measured** | **not measured** | **not measured** | no KL data (section KL) |
| Q2 growth with length | **not measured** | **not measured** | **not measured** | same |
| Q3 top-1 flips | **not measured** | **not measured** | **not measured** | same |
| Q4 decode windows 2048 / 8192, no drift | **not measured** | **not measured** | **not measured** | same |
| Q5 needle recall (arm >= base - 1) | PASS 12/12, 8/8 | PASS 12/12, 8/8 | PASS 12/12, 8/8 | base 12/12 and 8/8; at ceiling, so it cannot resolve a small loss |
| Q6 MMLU / GSM8K inside 2 SE | PASS | PASS | PASS | table below |
| Q7 env-unset gate (the branch) | PASS | PASS | n/a | table below |
| Q8 C worth shipping (16-stream gain >= +3 %) | n/a | **PASS, +8.3 %** (+9.0 %, +7.5 %) | n/a | C+D +10.8 %, reported only |
| **Lever passes (Q1 to Q6)** | **NO: Q1 to Q4 open** | **NO: Q1 to Q4 open** | **NO** | |

Nothing here fails a rule; the open rules are the ones that carry the long-context quality question this lane was written to answer (D at 32k and 64k, C over thousands of
decode steps). Needle at 12 and 8 rows is at its ceiling for every arm, so it is weak evidence next to KL.

### KL (Q1 to Q4): no data, and why

`python3 scripts/spark35num/analyze.py docs/results/2026-11-05-spark35-numerics/raw` was run on the final raw data: every table is empty and the verdict section prints
`no prefill data` for fp8, cemu, cdemu and creal. That output is correct, not an analyzer fault:

- The 10 non-base logs (`acc-{std,dec8k}-{noise,fp8,cemu,cdemu}.log` and `acc-std-creal.log`) end at startup with
  `Error: --golden and --self-golden are separate runs (the self-golden carries its own prompts)` (`crates/intarsia-cli/src/cmd/accuracy.rs:2662`).
  `scripts/spark35num/kl.sh` (`run_acc`) passes `--golden $carrier` on every arm and adds `--self-golden` for the non-base ones.
- The two base logs ran (`acc-std-base.log`, `acc-dec8k-base.log`) and wrote the self-golden dumps (`wrote self-golden .../dump-std-base.st`, `dump-dec8k-base.st` on
  spark2, `~/intarsiaml-s35/out/`, not copied to `raw/`; whether they still exist was not checked). Their printed KL (about 5.5 nats) and the `pos-*-base.tsv.gz` rows are
  against the carrier's dummy reference rows, so they say nothing about numerics. The two base runs exit with `Error: accuracy gate FAILED` for that reason (expected, and the
  script does not stop on it).
- Consequence: the noise arm, the fp8 / C / C+D arms and every number of Q1 to Q4 are missing. No inference is drawn from the unrelated numbers below.

### What fills it (not run; no fleetq or GPU job was started for this write-up)

1. `scripts/spark35num/kl.sh`: leave out `--golden $L/carriers/$gold.safetensors` when `--self-golden` is given. One line in `run_acc`. Not changed here (no code changes in this task).
2. Re-run on spark2: `fleetq run --host spark2 --quiet gpu,cpu --mem 70 --expect 80 --label "spark35: KL arms" -- env ARMS="noise fp8 cemu cdemu creal" scripts/spark35num/kl.sh`
   (add `base` first if the two base dumps are gone). Expected time: from the file times in J1, one std arm (5 prompts, 134 k prefill positions + 5 x 2048 decode) took
   about 9 min (20:18 to 20:27) and the dec8k base about 3 min (20:27 to 20:30); the five arms are 4 x (9 + 3) + 9 = 57 min, plus 12 for base if its dump must be redone;
   about 60 to 70 min, `--expect 80`. Then `python3 scripts/spark35num/analyze.py <raw>` fills Q1 to Q4.
3. Also not run and not in the raw data: `#8662` (`cargo test --workspace --no-fail-fast` on the final head, endor, `--load cpu`; the committed log `cargo-test-workspace-endor.log`
   is #8643 on `1016c7e5e`, 1324 passed, 0 failed, 29 ignored, and `1016c7e5e..` changes `gdn.cu`, `device.rs`, `model.rs`, `prefix_store.rs`), and the D smoke test
   (`/metrics` `intarsia_info{kv_dtype=...}` with the D commit built). Expected time is a guess, not read from a log: cargo about 20 min on endor with `--load cpu` (#8643's wall time is not recorded in `raw/`); smoke test about 10 min on a spark.

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

### Q7: the env-unset gate and the other safety checks (dev `7a7aa47b2`, main `5390e65b4`)

| check | result |
|---|---|
| greedy ids base vs `origin/main` (1024 tokens, 6 prompts: 1k/4k/16k x 2) | IDENTICAL, all 6 (`greedy-main-vs-base.txt`) |
| idgate, 256 ids, 6 prompts: main == base, cemu == creal (MTP on), cemu == creal (`--spec off`) | IDENTICAL, all (`idgate/cmp-*.txt`) |
| greedy ids cemu vs creal, 1024 tokens | IDENTICAL, all 6 (so the real layout equals the emulation) |
| CUDA `plan show` 35B main vs dev | byte-identical |
| `kernels test --quick` main vs dev | 106 kernels, 918 cases, same rows and numbers (only the wall time differs) |
| GPU bitwise guards (`INTARSIA_REQUIRE_GPU=1`): gdn_state16 2, gdn_split 2, gdn_conv16 2, gdn_lanes 2, rows_exact 12 (2 ignored), row_tables 7 (2 ignored) | all passed; gdn_state16 relative state error S16 vs S32 2.7e-3 to 2.8e-3 after 1000 steps |
| Metal no-op, main vs branch: plan 8B and 35B (json + txt), 256 ids 8B and 35B (short, 4k) | all identical |
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

### Decisions

- **D (fp8 KV default on CUDA 35B): do not flip yet; merge without `4d9003f67`.** D has not passed (Q1 to Q4 unmeasured), and the speed it would buy is within noise at 16 streams
  (+0.7 %) and about +3 % at 16k single stream. What is known: needle 12/12 and 8/8, MMLU +0.005, GSM8K 0.000, which are the same ceiling-level checks that fp8kv already passed
  on 2026-10-29. The 32k and 64k KL, which is the gap this lane was to close, is missing. Reconsider after the KL rerun above if the KV memory at 128K (spark2) is the reason to
  flip, since the speed is not. The D commit itself was never built or run on a GPU.
- **C (bf16 delta-rule state): ship as opt-in (default off), as the lane said; no default decision is asked for or made.** Q5 to Q8 and the equality gates pass, +8.3 % at 16
  streams. It is not "passed" by the doc's rule until Q1 to Q4 exist; those are the numbers to read before anyone proposes a default.
- Nothing else in the doc asks for a decision.

### Merge readiness

- Without D: lane commits up to `3860aba95` (code unchanged since `7a7aa47b2`, the measured dev), plus this write-up. GPU-tested at `7a7aa47b2`: bitwise guards, idgate, plan, kernels test,
  Metal no-op. Not tested on the final head: the workspace `cargo test` (#8662). The gates were measured against `main` `5390e65b4`; the branch is behind `origin/main` and was not rebased.
- With D: also `4d9003f67` (changes `model.rs`, `recipe.rs`, `recipes/qwen3_5_moe.toml`, two tests, two docs). Never built on a GPU; the `plan show` change and the `/metrics` smoke test are not run.
  Revert it alone (`git revert 4d9003f67`) to keep bf16; it is no longer the last commit because this write-up is committed after it.

Existing evidence for D (not re-measured): `docs/results/2026-10-29-fp8kv-opt-in.md`: +9.9 % single-stream decode at 16k, +1.4 % at 16 streams, MMLU 0.846 to 0.851,
GSM8K 0.500 to 0.500, both inside noise; the long-context (32k, 64k) quality is the gap this lane fills. vLLM at 16 streams: 340 tok/s; the engine: 301 tok/s (0.88x).
